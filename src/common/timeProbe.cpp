#define _CRT_SECURE_NO_WARNINGS // NOLINT: fopen and getenv are fine for this local diagnostic.

#include "common/timeProbe.h"

#include "common/config.h"
#include "common/hostException.h"

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep
#include <intrin.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Common::TimeProbe {

namespace {

constexpr int REPORT_SECONDS = 5;
constexpr int REPORT_ROWS    = 40;

struct Key {
	uint64_t address;
	uint64_t detail;
	uint32_t source;
	// With a rate band, the two guest frames above the caller of a clock read (callers are often
	// generic timer helpers).
	uint64_t parent1 = 0;
	uint64_t parent2 = 0;

	bool operator==(const Key& other) const = default;
};

struct KeyHash {
	size_t operator()(const Key& key) const noexcept {
		return static_cast<size_t>((key.address * 0x9E3779B97F4A7C15ull) ^ (key.detail << 8u) ^
		                           key.source ^ (key.parent1 * 0xC2B2AE3D27D4EB4Full) ^
		                           (key.parent2 * 0x165667B19E3779F9ull));
	}
};

using Counts = std::unordered_map<Key, uint64_t, KeyHash>;

// One per thread that records, kept after the thread exits.
struct ThreadTable {
	std::mutex  mutex; // Taken by its thread on each call, and by the reporter.
	Counts      counts;
	Counts      reported; // The reporter's copy of `counts` at its last report.
	std::string name;
	uint32_t    host_id = 0;
	bool        traced  = false;
};

struct State {
	std::mutex                                tables_mutex;
	std::vector<std::unique_ptr<ThreadTable>> tables;
	std::mutex                                trace_mutex;
	std::FILE*                                trace_file   = nullptr;
	const char*                               trace_target = nullptr;
	uint64_t                                  trace_parent = 0;
	// KYTY_TIME_PROBE_DUMP=<caller>:<offset>:<size> (hex): at a clock read from that guest
	// address, at most every 250 ms, the bytes at [guest r15 + offset] go to the trace.
	uint64_t                              dump_caller = 0;
	uint64_t                              dump_offset = 0;
	uint64_t                              dump_size   = 0;
	std::chrono::steady_clock::time_point last_dump {};
};

State& GetState() {
	static auto* state = new State; // Never destroyed: threads may record during exit.
	return *state;
}

const char* SourceName(uint32_t source) {
	static constexpr const char* NAMES[] = {"ProcessTime",  "ReadTsc",       "ProcessTimeCounter",
	                                        "ClockGettime", "Gettimeofday",  "FlipStatus",
	                                        "VblankStatus", "IsFlipPending", "WaitVblank",
	                                        "WaitEqueue",   "Usleep",        "Nanosleep",
	                                        "Submit"};
	static_assert(std::size(NAMES) == static_cast<size_t>(Source::Count));
	return source < std::size(NAMES) ? NAMES[source] : "?";
}

uint32_t CurrentThreadId() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	return GetCurrentThreadId();
#else
	return 0;
#endif
}

uint64_t Ticks() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	return __rdtsc();
#else
	return static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
#endif
}

constexpr int GUEST_FRAMES = 6;

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
// Follows the guest's frame-pointer chain (guest code keeps rbp frames). A bad pointer faults
// into the __except, so the emulator's handler must let the fault through (ScopedPassThrough).
int WalkGuestFramesGuarded(uint64_t rbp, uint64_t stack_low, uint64_t* frames, int count) {
	__try {
		while (count < GUEST_FRAMES && rbp > stack_low && rbp < stack_low + (64u << 20u) &&
		       (rbp & 7u) == 0) {
			frames[count++] = *reinterpret_cast<const uint64_t*>(rbp + 8);
			const auto next = *reinterpret_cast<const uint64_t*>(rbp);
			if (next <= rbp) {
				break;
			}
			rbp = next;
		}
	} __except (EXCEPTION_EXECUTE_HANDLER) {
		// Keep the frames found so far.
	}
	return count;
}

// The return addresses of the guest frames that led to the probed function, innermost first:
// the host frames are unwound with their tables, which restores the guest's callee-saved
// registers (rbp for the walk; r15 is returned for KYTY_TIME_PROBE_DUMP).
int GuestFrames(uint64_t* frames, uint64_t* guest_r15 = nullptr) {
	CONTEXT context {};
	RtlCaptureContext(&context);
	for (int depth = 0; depth < 32; depth++) {
		DWORD64 image    = 0;
		auto*   function = RtlLookupFunctionEntry(context.Rip, &image, nullptr);
		if (function == nullptr) {
			break;
		}
		PVOID   handler_data = nullptr;
		DWORD64 establisher  = 0;
		RtlVirtualUnwind(UNW_FLAG_NHANDLER, image, context.Rip, function, &context, &handler_data,
		                 &establisher, nullptr);
	}
	frames[0] = context.Rip;
	if (guest_r15 != nullptr) {
		*guest_r15 = context.R15;
	}
	const HostException::ScopedPassThrough pass_through;
	return WalkGuestFramesGuarded(context.Rbp, context.Rsp, frames, 1);
}
#else
int GuestFrames(uint64_t* /*frames*/, uint64_t* guest_r15 = nullptr) {
	(void)guest_r15;
	return 0;
}
#endif

// The thread's description (guest threads get their guest name), else its id.
std::string ThreadName(uint32_t host_id) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (HANDLE thread = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, host_id);
	    thread != nullptr) {
		PWSTR       description = nullptr;
		std::string name;
		if (SUCCEEDED(GetThreadDescription(thread, &description)) && description != nullptr) {
			char buffer[128] {};
			WideCharToMultiByte(CP_UTF8, 0, description, -1, buffer,
			                    static_cast<int>(sizeof(buffer)), nullptr, nullptr);
			LocalFree(description);
			name = buffer;
		}
		CloseHandle(thread);
		if (!name.empty()) {
			return name;
		}
	}
#endif
	return "thread " + std::to_string(host_id);
}

// KYTY_TIME_PROBE_BAND=<low>:<high> also lists every call site whose rate, summed over all
// threads, is in that band (e.g. once per frame), and keys clock reads by their guest call stack.
const std::pair<double, double>& Band() {
	static const auto band = [] {
		std::pair<double, double> limits {0.0, 0.0};
		if (const char* value = std::getenv("KYTY_TIME_PROBE_BAND"); value != nullptr) {
			char* end    = nullptr;
			limits.first = std::strtod(value, &end);
			if (end != nullptr && *end == ':') {
				limits.second = std::strtod(end + 1, nullptr);
			}
		}
		return limits;
	}();
	return band;
}

void Report() {
	auto& state = GetState();
	struct Row {
		std::string name;
		Key         key;
		uint64_t    delta;
	};
	std::vector<Row> rows;
	{
		std::lock_guard tables_lock(state.tables_mutex);
		for (auto& table: state.tables) {
			Counts counts;
			{
				std::lock_guard lock(table->mutex);
				counts = table->counts;
			}
			const auto name = ThreadName(table->host_id);
			if (!name.starts_with("thread ")) {
				table->name = name;
			}
			for (const auto& [key, count]: counts) {
				const auto previous = table->reported.find(key);
				const auto delta =
				    count - (previous != table->reported.end() ? previous->second : 0);
				if (delta != 0) {
					rows.push_back({table->name, key, delta});
				}
			}
			table->reported = std::move(counts);
		}
	}
	std::sort(rows.begin(), rows.end(),
	          [](const Row& a, const Row& b) { return a.delta > b.delta; });
	const auto& band = Band();
	std::printf("time probe: calls per second over the last %d s\n", REPORT_SECONDS);
	for (size_t i = 0; i < rows.size() && i < REPORT_ROWS; i++) {
		const auto& row = rows[i];
		std::printf("time probe: %9.1f/s  %-18s detail 0x%-12llx caller 0x%012llx  %s\n",
		            static_cast<double>(row.delta) / REPORT_SECONDS, SourceName(row.key.source),
		            static_cast<unsigned long long>(row.key.detail),
		            static_cast<unsigned long long>(row.key.address), row.name.c_str());
	}
	if (band.second > 0.0) {
		// The same call site summed over all threads: a once-per-frame job moves between workers.
		std::unordered_map<Key, std::pair<uint64_t, int>, KeyHash> sites;
		for (const auto& row: rows) {
			auto& site = sites[row.key];
			site.first += row.delta;
			site.second++;
		}
		std::vector<std::pair<Key, std::pair<uint64_t, int>>> listed(sites.begin(), sites.end());
		std::sort(listed.begin(), listed.end(),
		          [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
		for (const auto& [key, site]: listed) {
			const auto rate = static_cast<double>(site.first) / REPORT_SECONDS;
			if (rate >= band.first && rate <= band.second) {
				std::printf("time probe band: %7.1f/s  %-18s detail 0x%-12llx caller 0x%012llx  "
				            "parents 0x%012llx 0x%012llx  %d threads\n",
				            rate, SourceName(key.source),
				            static_cast<unsigned long long>(key.detail),
				            static_cast<unsigned long long>(key.address),
				            static_cast<unsigned long long>(key.parent1),
				            static_cast<unsigned long long>(key.parent2), site.second);
			}
		}
	}
	std::fflush(stdout);
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
bool ReadGuarded(uint64_t address, void* out, size_t size) {
	__try {
		std::memcpy(out, reinterpret_cast<const void*>(address), size);
		return true;
	} __except (EXCEPTION_EXECUTE_HANDLER) {
		return false;
	}
}

// KYTY_TIME_PROBE_PEEK=<address of a guest pointer>: prints the first 64 bytes of the object it
// points at, as 32-bit words and floats (e.g. the game's frame settings).
void Peek() {
	static const uint64_t slot = [] {
		const char* value = std::getenv("KYTY_TIME_PROBE_PEEK");
		return value != nullptr ? std::strtoull(value, nullptr, 0) : 0ull;
	}();
	if (slot == 0) {
		return;
	}
	const HostException::ScopedPassThrough pass_through;
	uint64_t                               object = 0;
	uint32_t                               words[16] {};
	if (!ReadGuarded(slot, &object, sizeof(object)) || object == 0 ||
	    !ReadGuarded(object, words, sizeof(words))) {
		std::printf("time probe peek: 0x%llx unreadable\n", static_cast<unsigned long long>(slot));
		return;
	}
	for (int i = 0; i < 16; i++) {
		float value = 0.0f;
		std::memcpy(&value, &words[i], sizeof(value));
		std::printf("time probe peek: [0x%02x] 0x%08x %g\n", i * 4, words[i],
		            static_cast<double>(value));
	}
}
#else
void Peek() {}
#endif

void ReportLoop() {
	for (;;) {
		std::this_thread::sleep_for(std::chrono::seconds(REPORT_SECONDS));
		Report();
		Peek();
		auto&           state = GetState();
		std::lock_guard lock(state.trace_mutex);
		if (state.trace_file != nullptr) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			// Pairs the trace's ticks with the performance counter, for their frequency.
			LARGE_INTEGER counter {};
			LARGE_INTEGER frequency {};
			QueryPerformanceCounter(&counter);
			QueryPerformanceFrequency(&frequency);
			std::fprintf(state.trace_file, "# sync %llu %lld %lld\n",
			             static_cast<unsigned long long>(Ticks()), counter.QuadPart,
			             frequency.QuadPart);
#endif
			std::fflush(state.trace_file);
		}
	}
}

// KYTY_TIME_PROBE_TRACE holds name fragments separated by '|'.
bool IsTraced(const std::string& name, const char* targets) {
	if (targets == nullptr) {
		return false;
	}
	for (const char* begin = targets; *begin != '\0';) {
		const char* end      = std::strchr(begin, '|');
		const auto  fragment = end != nullptr ? std::string(begin, end) : std::string(begin);
		if (!fragment.empty() && name.find(fragment) != std::string::npos) {
			return true;
		}
		if (end == nullptr) {
			break;
		}
		begin = end + 1;
	}
	return false;
}

ThreadTable& CurrentTable() {
	thread_local ThreadTable* table = [] {
		auto& state    = GetState();
		auto  owned    = std::make_unique<ThreadTable>();
		owned->host_id = CurrentThreadId();
		owned->name    = ThreadName(owned->host_id);
		owned->traced  = IsTraced(owned->name, state.trace_target);
		if (owned->traced) {
			std::lock_guard trace_lock(state.trace_mutex);
			if (state.trace_file != nullptr) {
				std::fprintf(state.trace_file, "# thread %u %s\n", owned->host_id,
				             owned->name.c_str());
			}
		}
		std::lock_guard lock(state.tables_mutex);
		state.tables.push_back(std::move(owned));
		return state.tables.back().get();
	}();
	return *table;
}

} // namespace

bool Enabled() noexcept {
	static const bool enabled = [] {
		const char* path = std::getenv("KYTY_TIME_PROBE");
		if (path == nullptr || *path == '\0') {
			return false;
		}
		auto&       state  = GetState();
		const char* target = std::getenv("KYTY_TIME_PROBE_TRACE");
		if (target != nullptr && *target != '\0') {
			state.trace_target = target;
		}
		// KYTY_TIME_PROBE_TRACE_PARENT=<guest address>: a thread is traced from its first clock
		// read with that address in its call stack (e.g. a game loop that runs on a worker thread).
		if (const char* parent = std::getenv("KYTY_TIME_PROBE_TRACE_PARENT"); parent != nullptr) {
			state.trace_parent = std::strtoull(parent, nullptr, 16);
		}
		if (const char* dump = std::getenv("KYTY_TIME_PROBE_DUMP"); dump != nullptr) {
			char* end         = nullptr;
			state.dump_caller = std::strtoull(dump, &end, 16);
			if (end != nullptr && *end == ':') {
				state.dump_offset = std::strtoull(end + 1, &end, 16);
			}
			if (end != nullptr && *end == ':') {
				state.dump_size = std::min<uint64_t>(std::strtoull(end + 1, nullptr, 16), 0x2000);
			}
		}
		if (state.trace_target != nullptr || state.trace_parent != 0 || state.dump_size != 0) {
			state.trace_file = std::fopen(path, "w");
			if (state.trace_file != nullptr) {
				std::fprintf(state.trace_file, "# ticks source detail caller thread\n");
			}
		}
		std::thread(ReportLoop).detach();
		return true;
	}();
	return enabled;
}

void Record(Source source, uint64_t detail, const void* return_address) noexcept {
	auto&      table      = CurrentTable();
	const auto ticks      = Ticks();
	auto&      state      = GetState();
	const bool band_stack = Band().second > 0.0 && source == Source::ProcessTimeCounter;
	const bool find_parent =
	    !table.traced && state.trace_parent != 0 && source == Source::ProcessTimeCounter;
	uint64_t  frames[GUEST_FRAMES] {};
	const int count = table.traced || band_stack || find_parent ? GuestFrames(frames) : 0;
	if (find_parent && std::find(frames, frames + count, state.trace_parent) != frames + count) {
		table.traced = true;
		std::lock_guard lock(state.trace_mutex);
		if (state.trace_file != nullptr) {
			std::fprintf(state.trace_file, "# thread %u %s\n", table.host_id, table.name.c_str());
		}
	}
	Key key {reinterpret_cast<uint64_t>(return_address), detail, static_cast<uint32_t>(source)};
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (state.dump_size != 0 && key.address == state.dump_caller) {
		// KYTY_TIME_PROBE_DUMP_MS: the least time between dumps (default 250 ms; 0 = every call).
		static const auto interval = std::chrono::milliseconds([] {
			const char* value = std::getenv("KYTY_TIME_PROBE_DUMP_MS");
			return value != nullptr ? std::strtoll(value, nullptr, 10) : 250;
		}());
		const auto        now      = std::chrono::steady_clock::now();
		std::lock_guard   lock(state.trace_mutex);
		if (now - state.last_dump >= interval && state.trace_file != nullptr) {
			state.last_dump = now;
			uint64_t frames_unused[GUEST_FRAMES] {};
			uint64_t object = 0;
			GuestFrames(frames_unused, &object);
			std::vector<uint8_t>                   bytes(state.dump_size);
			const HostException::ScopedPassThrough pass_through;
			if (ReadGuarded(object + state.dump_offset, bytes.data(), bytes.size())) {
				std::fprintf(state.trace_file, "# dump %llu %llx ",
				             static_cast<unsigned long long>(ticks),
				             static_cast<unsigned long long>(object));
				for (const auto byte: bytes) {
					std::fprintf(state.trace_file, "%02x", byte);
				}
				std::fputc('\n', state.trace_file);
			}
		}
	}
#endif
	if (band_stack) {
		key.parent1 = count > 1 ? frames[1] : 0;
		key.parent2 = count > 2 ? frames[2] : 0;
	}
	{
		std::lock_guard lock(table.mutex);
		table.counts[key]++;
	}
	if (table.traced) {
		char   stack[GUEST_FRAMES * 20] {};
		size_t used = 0;
		for (int i = 1; i < count && used < sizeof(stack); i++) {
			used += static_cast<size_t>(std::snprintf(stack + used, sizeof(stack) - used, " %llx",
			                                          static_cast<unsigned long long>(frames[i])));
		}
		std::lock_guard lock(state.trace_mutex);
		if (state.trace_file != nullptr) {
			std::fprintf(state.trace_file, "%llu %s %llx %llx %u%s\n",
			             static_cast<unsigned long long>(ticks), SourceName(key.source),
			             static_cast<unsigned long long>(detail),
			             static_cast<unsigned long long>(key.address), table.host_id, stack);
		}
	}
}

} // namespace Common::TimeProbe
