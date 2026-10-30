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
std::atomic_int g_record_processor {-1};

int ConfiguredProcessor() {
	static const int processor = [] {
		const char* value  = std::getenv("KYTY_GPU_CORE");
		const int   parsed = value != nullptr && value[0] != '\0' ? std::atoi(value) : 2;
		return parsed >= 0 && parsed < 64 ? parsed : 2;
	}();
	return processor;
}

// The record-stream consumer's core (AbSwitch OWN_CORES). 0 means none: logical processor 0 is
// the main thread's core and never a good own core.
int ConfiguredRecordProcessor() {
	static const int processor = [] {
		const char* value  = std::getenv("KYTY_RECORD_CORE");
		const int   parsed = value != nullptr && value[0] != '\0' ? std::atoi(value) : 4;
		return parsed > 0 && parsed < 64 ? parsed : 0;
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

// Every thread of the process except the caller and `skip_id`.
template <typename Func>
void ForEachOtherThread(Func&& func, DWORD skip_id = 0) {
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
		if (entry.th32OwnerProcessID != process_id || entry.th32ThreadID == self_id ||
		    entry.th32ThreadID == skip_id) {
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
	DWORD_PTR reserved = CoreMask(processor);
	if (const int record = g_record_processor.load(std::memory_order_relaxed); record > 0) {
		reserved |= CoreMask(record);
	}
	if (const auto mask = ProcessMask() & ~reserved; mask != 0) {
		SetThreadAffinityMask(GetCurrentThread(), mask);
	}
}

void SetOwnCores(bool own, unsigned record_thread_id) {
	const auto process_mask = ProcessMask();
	if (process_mask == 0) {
		return;
	}
	const auto restore_all = [&] {
		SetThreadAffinityMask(GetCurrentThread(), process_mask);
		ForEachOtherThread(
		    [process_mask](HANDLE thread) { SetThreadAffinityMask(thread, process_mask); });
	};
	if (!own) {
		restore_all();
		g_reserved_processor.store(-1, std::memory_order_relaxed);
		g_record_processor.store(-1, std::memory_order_relaxed);
		std::printf("core placement: own cores released\n");
		std::fflush(stdout);
		return;
	}

	const int  gpu      = ConfiguredProcessor();
	const auto gpu_bit  = DWORD_PTR {1} << gpu;
	if ((process_mask & gpu_bit) == 0) {
		return;
	}
	// The consumer's core, when configured and different from the GPU thread's physical core.
	int       record       = ConfiguredRecordProcessor();
	DWORD_PTR record_bit   = 0;
	DWORD_PTR reserved     = CoreMask(gpu);
	if (record > 0 && (record & ~1) != (gpu & ~1) &&
	    (process_mask & (DWORD_PTR {1} << record)) != 0) {
		record_bit = DWORD_PTR {1} << record;
		reserved |= CoreMask(record);
	} else {
		record = 0;
	}
	const auto others = process_mask & ~reserved;
	// The guest needs its threads somewhere: at least six logical processors left.
	int remaining = 0;
	for (auto bits = others; bits != 0; bits &= bits - 1) {
		remaining++;
	}
	if (others == 0 || remaining < 6) {
		return;
	}

	// Entering from RESERVE_CORE or from a previous placement: widen every mask first, then pin.
	restore_all();
	SetThreadAffinityMask(GetCurrentThread(), gpu_bit);
	if (record_bit != 0 && record_thread_id != 0) {
		if (const auto thread = OpenThread(THREAD_SET_INFORMATION | THREAD_QUERY_INFORMATION, FALSE,
		                                   record_thread_id);
		    thread != nullptr) {
			SetThreadAffinityMask(thread, record_bit);
			CloseHandle(thread);
		} else {
			record     = 0;
			record_bit = 0;
		}
	}
	ForEachOtherThread([others](HANDLE thread) { SetThreadAffinityMask(thread, others); },
	                   record_thread_id);
	g_reserved_processor.store(gpu, std::memory_order_relaxed);
	g_record_processor.store(record, std::memory_order_relaxed);
	std::printf("core placement: own cores logical %d (GPU) and %d (record)%s\n", gpu, record,
	            record == 0 ? " - no consumer yet" : "");
	std::fflush(stdout);
}

void SetCurrentThreadAboveNormal(bool above_normal) {
	SetThreadPriority(GetCurrentThread(),
	                  above_normal ? THREAD_PRIORITY_ABOVE_NORMAL : THREAD_PRIORITY_NORMAL);
}

#else

void SetReserved(bool /*reserved*/) {}
void SetOwnCores(bool /*own*/, unsigned /*record_thread_id*/) {}
void KeepOffReservedCore() {}
void SetCurrentThreadAboveNormal(bool /*above_normal*/) {}

#endif

} // namespace Common::CorePlacement
