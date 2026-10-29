#include "common/corePlacement.h"

#include "common/config.h"

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep
#include <tlhelp32.h>
#endif

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace Common::CorePlacement {

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

namespace {

std::atomic_int g_reserved_processor {-1};

int ConfiguredProcessor() {
	static const int processor = [] {
		const char* value  = std::getenv("KYTY_GPU_CORE");
		const int   parsed = value != nullptr && value[0] != '\0' ? std::atoi(value) : 2;
		return parsed >= 0 && parsed < 64 ? parsed : 2;
	}();
	return processor;
}

DWORD_PTR ProcessMask() {
	DWORD_PTR process_mask = 0;
	DWORD_PTR system_mask  = 0;
	return GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask) != 0
	           ? process_mask
	           : 0;
}

// Windows numbers the two logical processors of an SMT core next to each other.
DWORD_PTR CoreMask(int processor) {
	return DWORD_PTR {3} << (static_cast<unsigned>(processor) & ~1u);
}

// Every thread of the process except the caller.
template <typename Func>
void ForEachOtherThread(Func&& func) {
	const auto snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
	if (snapshot == INVALID_HANDLE_VALUE) {
		return;
	}
	THREADENTRY32 entry {};
	entry.dwSize          = sizeof(entry);
	const auto process_id = GetCurrentProcessId();
	const auto self_id    = GetCurrentThreadId();
	for (BOOL more = Thread32First(snapshot, &entry); more != FALSE;
	     more      = Thread32Next(snapshot, &entry)) {
		if (entry.th32OwnerProcessID != process_id || entry.th32ThreadID == self_id) {
			continue;
		}
		if (const auto thread = OpenThread(THREAD_SET_INFORMATION | THREAD_QUERY_INFORMATION, FALSE,
		                                   entry.th32ThreadID);
		    thread != nullptr) {
			func(thread);
			CloseHandle(thread);
		}
	}
	CloseHandle(snapshot);
}

} // namespace

void SetReserved(bool reserved) {
	const auto process_mask = ProcessMask();
	if (process_mask == 0) {
		return;
	}
	const int  processor = ConfiguredProcessor();
	const auto others    = reserved ? process_mask & ~CoreMask(processor) : process_mask;
	if (others == 0 || (reserved && (process_mask & (DWORD_PTR {1} << processor)) == 0)) {
		return;
	}
	g_reserved_processor.store(reserved ? processor : -1, std::memory_order_relaxed);
	SetThreadAffinityMask(GetCurrentThread(), reserved ? DWORD_PTR {1} << processor : process_mask);
	SetThreadPriority(GetCurrentThread(),
	                  reserved ? THREAD_PRIORITY_ABOVE_NORMAL : THREAD_PRIORITY_NORMAL);
	ForEachOtherThread([others](HANDLE thread) { SetThreadAffinityMask(thread, others); });
	std::printf("core placement: %s logical processor %d for the GPU thread\n",
	            reserved ? "reserved" : "released", processor);
	std::fflush(stdout);
}

void KeepOffReservedCore() {
	const int processor = g_reserved_processor.load(std::memory_order_relaxed);
	if (processor < 0) {
		return;
	}
	if (const auto mask = ProcessMask() & ~CoreMask(processor); mask != 0) {
		SetThreadAffinityMask(GetCurrentThread(), mask);
	}
}

void SetCurrentThreadAboveNormal(bool above_normal) {
	SetThreadPriority(GetCurrentThread(),
	                  above_normal ? THREAD_PRIORITY_ABOVE_NORMAL : THREAD_PRIORITY_NORMAL);
}

#else

void SetReserved(bool /*reserved*/) {}
void KeepOffReservedCore() {}
void SetCurrentThreadAboveNormal(bool /*above_normal*/) {}

#endif

} // namespace Common::CorePlacement
