#define _CRT_SECURE_NO_WARNINGS // NOLINT: fopen and getenv are fine for this local profiling aid.

#include "common/threadSampler.h"

#include "common/config.h"

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep

// clang-format off
#include <psapi.h>
#include <tlhelp32.h>
// clang-format on

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#endif

namespace Common::ThreadSampler {

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

namespace {

constexpr int MAX_FRAMES = 64;

using Frames = std::array<uint64_t, MAX_FRAMES>;

// Walks the suspended thread's stack with the image unwind tables. Only the innermost frame may
// lack them (a leaf function). Nothing here allocates or locks, since the thread is suspended.
int Unwind(CONTEXT context, Frames& frames) {
	int count = 0;
	__try {
		while (count < MAX_FRAMES && context.Rip != 0) {
			frames[count++]  = context.Rip;
			DWORD64 image    = 0;
			auto*   function = RtlLookupFunctionEntry(context.Rip, &image, nullptr);
			if (function == nullptr) {
				if (count > 1) {
					break;
				}
				context.Rip = *reinterpret_cast<const DWORD64*>(context.Rsp);
				context.Rsp += 8;
				continue;
			}
			PVOID   handler_data = nullptr;
			DWORD64 establisher  = 0;
			RtlVirtualUnwind(UNW_FLAG_NHANDLER, image, context.Rip, function, &context,
			                 &handler_data, &establisher, nullptr);
		}
	} __except (EXCEPTION_EXECUTE_HANDLER) {
		// Keep the frames walked so far.
	}
	return count;
}

bool SamplingEnabled(const char* control) {
	if (control == nullptr) {
		return true;
	}
	std::FILE* file = std::fopen(control, "r");
	if (file == nullptr) {
		return true;
	}
	const int first = std::fgetc(file);
	std::fclose(file);
	return first != '0';
}

void Write(const std::string& path, const std::unordered_map<std::string, uint64_t>& stacks,
           uint64_t samples, uint64_t interval_us) {
	const auto temp = path + ".tmp";
	std::FILE* file = std::fopen(temp.c_str(), "w");
	if (file == nullptr) {
		return;
	}
	std::fprintf(file, "# samples %llu interval_us %llu\n", static_cast<unsigned long long>(samples),
	             static_cast<unsigned long long>(interval_us));
	std::array<HMODULE, 1024> modules {};
	DWORD                     needed = 0;
	if (K32EnumProcessModules(GetCurrentProcess(), modules.data(),
	                          static_cast<DWORD>(sizeof(modules)), &needed) != 0) {
		const auto count = std::min<size_t>(needed / sizeof(HMODULE), modules.size());
		for (size_t i = 0; i < count; i++) {
			MODULEINFO info {};
			char       name[MAX_PATH] {};
			if (K32GetModuleInformation(GetCurrentProcess(), modules[i], &info, sizeof(info)) != 0 &&
			    K32GetModuleFileNameExA(GetCurrentProcess(), modules[i], name, MAX_PATH) != 0) {
				std::fprintf(file, "module %llx %lx %s\n",
				             static_cast<unsigned long long>(
				                 reinterpret_cast<uintptr_t>(info.lpBaseOfDll)),
				             info.SizeOfImage, name);
			}
		}
	}
	for (const auto& [key, count]: stacks) {
		std::fprintf(file, "%llu", static_cast<unsigned long long>(count));
		const auto* frames = reinterpret_cast<const uint64_t*>(key.data());
		for (size_t i = 0; i < key.size() / sizeof(uint64_t); i++) {
			std::fprintf(file, " %llx", static_cast<unsigned long long>(frames[i]));
		}
		std::fputc('\n', file);
	}
	std::fclose(file);
	MoveFileExA(temp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING);
}

// The first thread of this process whose description (SetThreadDescription) contains `fragment`.
HANDLE FindThread(const char* fragment) {
	HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
	if (snapshot == INVALID_HANDLE_VALUE) {
		return nullptr;
	}
	HANDLE        found = nullptr;
	THREADENTRY32 entry {};
	entry.dwSize = sizeof(entry);
	for (BOOL next = Thread32First(snapshot, &entry); next != 0 && found == nullptr;
	     next      = Thread32Next(snapshot, &entry)) {
		if (entry.th32OwnerProcessID != GetCurrentProcessId()) {
			continue;
		}
		HANDLE thread = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT |
		                               THREAD_QUERY_LIMITED_INFORMATION,
		                           FALSE, entry.th32ThreadID);
		if (thread == nullptr) {
			continue;
		}
		PWSTR description = nullptr;
		if (SUCCEEDED(GetThreadDescription(thread, &description)) && description != nullptr) {
			char name[128] {};
			WideCharToMultiByte(CP_UTF8, 0, description, -1, name, static_cast<int>(sizeof(name)),
			                    nullptr, nullptr);
			LocalFree(description);
			if (std::strstr(name, fragment) != nullptr) {
				found = thread;
				continue;
			}
		}
		CloseHandle(thread);
	}
	CloseHandle(snapshot);
	return found;
}

// With a `target` name the sampled thread is looked up by name, again whenever it goes away.
void Run(HANDLE thread, std::string path, const char* control, const char* target,
         uint64_t interval_us) {
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
	HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
	                                      TIMER_ALL_ACCESS);
	std::unordered_map<std::string, uint64_t> stacks;
	uint64_t                                  samples      = 0;
	uint64_t                                  last_write   = GetTickCount64();
	uint64_t                                  last_control = 0;
	bool                                      enabled      = true;
	for (;;) {
		if (timer != nullptr) {
			LARGE_INTEGER due {};
			due.QuadPart = -static_cast<LONGLONG>(interval_us * 10);
			SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE);
			WaitForSingleObject(timer, INFINITE);
		} else {
			Sleep(1);
		}
		const uint64_t now = GetTickCount64();
		if (now - last_control >= 1000) {
			enabled      = SamplingEnabled(control);
			last_control = now;
			if (thread == nullptr && target != nullptr) {
				thread = FindThread(target);
			}
		}
		if (now - last_write >= 10000) {
			Write(path, stacks, samples, interval_us);
			last_write = now;
		}
		if (!enabled || thread == nullptr) {
			continue;
		}
		Frames frames {};
		int    count = 0;
		if (SuspendThread(thread) == static_cast<DWORD>(-1)) {
			if (target == nullptr) {
				break; // The thread is gone.
			}
			CloseHandle(thread);
			thread = nullptr;
			continue;
		}
		CONTEXT context {};
		context.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
		if (GetThreadContext(thread, &context) != 0) {
			count = Unwind(context, frames);
		}
		ResumeThread(thread);
		if (count == 0) {
			continue;
		}
		stacks[std::string(reinterpret_cast<const char*>(frames.data()),
		                   static_cast<size_t>(count) * sizeof(uint64_t))]++;
		samples++;
	}
	Write(path, stacks, samples, interval_us);
}

} // namespace

void StartForCurrentThread() {
	static std::once_flag once;
	std::call_once(once, [] {
		const char* path = std::getenv("KYTY_SAMPLE_OUT");
		if (path == nullptr || *path == '\0') {
			return;
		}
		// KYTY_SAMPLE_THREAD=<name fragment> samples that thread instead, e.g. a guest thread.
		const char* target = std::getenv("KYTY_SAMPLE_THREAD");
		if (target != nullptr && *target == '\0') {
			target = nullptr;
		}
		HANDLE thread = nullptr;
		if (target == nullptr) {
			thread = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
			                    FALSE, GetCurrentThreadId());
			if (thread == nullptr) {
				return;
			}
		}
		uint64_t interval_us = 1000;
		if (const char* value = std::getenv("KYTY_SAMPLE_INTERVAL_US"); value != nullptr) {
			interval_us = std::max<uint64_t>(std::strtoull(value, nullptr, 10), 100);
		}
		std::thread(Run, thread, std::string(path), std::getenv("KYTY_SAMPLE_CONTROL"), target,
		            interval_us)
		    .detach();
	});
}

#else

void StartForCurrentThread() {}

#endif

} // namespace Common::ThreadSampler
