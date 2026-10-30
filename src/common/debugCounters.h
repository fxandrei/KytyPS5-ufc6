#ifndef KYTY_COMMON_DEBUG_COUNTERS_H_
#define KYTY_COMMON_DEBUG_COUNTERS_H_

#include <atomic>
#include <chrono>
#include <cstdint>

namespace Common::DebugCounters {

static_assert(std::atomic<uint64_t>::is_always_lock_free && std::atomic<bool>::is_always_lock_free);

// Collect events only while the panel is visible. Relaxed, lock-free atomics keep updates
// cheap and safe in signal handlers. Rates come from comparing two snapshots.
enum class Counter {
	// GPU work.
	Draws,
	DrawInstances,
	IndirectDraws,
	Dispatches,
	QueueSubmits,
	GpuWaits, // CPU blocked on a GPU timeline semaphore.
	GpuWaitNs,
	GpuThreadWaits, // The same, on the guest GPU thread only.
	GpuThreadWaitNs,
	UnmapWaits, // A guest unmap waited until the GPU was idle.
	UnmapWaitNs,
	UnmapWaitsUnregistered, // Only because priority operations were pending.
	FlipWaits,              // The command processor suspended on WaitFlipDone.
	ShRegTableReads,        // SET_SH_REG_INDIRECT tables the command processor read.
	ShRegTableReadsClean,   // On a GPU-written page, but none of their bytes GPU-written.
	ShRegTableReadsDirty,   // GPU-written bytes: the read waits for the GPU.
	DispatchArgReads,       // Thread-dimension DispatchIndirect arguments read on the CPU.
	DispatchArgReadsClean,
	DispatchArgReadsDirty,
	CpReadsFromBacking, // Served from the backing store instead (A/B CP_READ_BACKING).
	SrtWalkerFaults,    // SrtWalker JIT loads that faulted on GPU-tracked memory.
	SrtReadsFromBacking,
	LabelWrites, // End-of-pipe label writes by the command processor.
	LabelWritesClean,
	LabelWritesDirty, // Over GPU-written bytes: downloads them first unless written in order.
	LabelWritesInOrder,
	WrittenBindings, // Shader-written buffer bindings, each marking its whole range GPU-written.
	WrittenBindingBytes,
	DirtyReadsUnchanged, // A read that waited for GPU-dirty bytes found guest memory unchanged.
	DirtyReadsChanged,   // ... found that the GPU had changed them.
	TextureRefreshesGpu, // Bytes of images refreshed after a GPU buffer write over them.
	TextureRefreshesCpu, // Bytes of images refreshed after guest CPU writes.
	GpuBusyNs,           // GPU execution time of command buffers, from timestamps.
	GpuTimedBuffers,
	GpuBusyUnionNs,     // GPU busy time without double-counting overlapping command buffers.
	GpuIdleGapNs,       // GPU time between command buffers with nothing queued (under 1 s each).
	TextureUploadGpuNs, // GPU time of texture refreshes (detile and copy), from timestamps.
	PartialImageWritesSkipped, // Image bytes not refreshed after a partial buffer write (bit 4).
	GuestFlushBarriers,        // Barriers at guest synchronization points (bit 5).
	EagerFlushes,              // Submits without a wait because the GPU ran dry (bit 6).
	GpuFaults,                 // Guest writes to GPU-tracked memory.
	ShadersCompiled,
	ShaderCompileNs,
	PipelinesCreated,
	PipelineCreateNs,
	// Caches.
	TextureUploadBytes,
	TextureDownloadBytes,
	TextureEvictions,
	ColorMetadataReadbacks, // CPU blocked reading back GPU-written DCC/CMASK metadata.
	ColorMetadataReadbackNs,
	ColorMetadataChecks, // GPU clear-key checks of DCC/CMASK metadata.
	BufferUploadBytes,
	BufferDownloadBytes,
	BufferEvictions,
	// Presentation.
	FlipsRequested,
	FlipsRejected, // Guest flip queue was full.
	LateVblanks,   // The present thread fell behind the vblank rate.
	SwapchainRecreations,
	DroppedPresents,
	// Emulator.
	EmulatedInstructions, // Host CPU lacked an instruction the guest used.
	UnresolvedImportCalls,
	GuestFileReadBytes,
	GuestFileWriteBytes,
	AudioUnderruns,
	// The record stream (AbSwitch RECORD_THREAD).
	RecordStreamPackets,     // Packets appended by the GPU thread.
	RecordStreamBytes,       // Bytes appended by the GPU thread.
	RecordStreamDrains,      // Drains (a caller waited for the consumer).
	RecordStreamDirectCalls, // Calls that drained and ran on the caller: a missing hook.
	RecordStreamSleeps,      // Times the consumer slept.
	// The draw chain (AbSwitch DRAW_CHAIN).
	DrawChainTaken, // Draws that took the previous draw's render targets.
	Count
};

enum class Peak { ShaderCompileNs, PipelineCreateNs, Count };

namespace Detail {
inline std::atomic<bool>     g_enabled {false};
inline std::atomic<uint64_t> g_counters[static_cast<int>(Counter::Count)] {};
inline std::atomic<uint64_t> g_peaks[static_cast<int>(Peak::Count)] {};
// A submission's counters, accumulated on the submitting thread and added with one atomic per
// counter (BatchScope, used around the command processor's packet loop).
inline thread_local uint64_t g_batch[static_cast<int>(Counter::Count)] {};
inline thread_local uint32_t g_batch_depth = 0;
} // namespace Detail

[[nodiscard]] inline bool Enabled() noexcept {
	return Detail::g_enabled.load(std::memory_order_relaxed);
}

inline void SetEnabled(bool enabled) noexcept {
	Detail::g_enabled.store(enabled, std::memory_order_relaxed);
}

inline void Add(Counter counter, uint64_t value = 1) noexcept {
	if (!Enabled()) {
		return;
	}
	const auto index = static_cast<uint32_t>(counter);
	if (Detail::g_batch_depth != 0) {
		Detail::g_batch[index] += value;
		return;
	}
	Detail::g_counters[index].fetch_add(value, std::memory_order_relaxed);
}

// Adds the calling thread's pending counters. Counters read from another thread see a batch only
// after the submitting thread flushes it; rate samples compare whole seconds, so this is invisible.
inline void FlushBatched() noexcept {
	auto& batch = Detail::g_batch;
	for (uint32_t i = 0; i < static_cast<uint32_t>(Counter::Count); i++) {
		const auto value = batch[i];
		if (value != 0) {
			batch[i] = 0;
			Detail::g_counters[i].fetch_add(value, std::memory_order_relaxed);
		}
	}
}

// Accumulates every Add() on this thread until the outermost scope ends (the command processor
// wraps its packet loop: a draw adds several counters, each was an atomic RMW on a shared line).
class BatchScope final {
public:
	BatchScope() noexcept {
		if (Enabled()) {
			Detail::g_batch_depth++;
		}
	}
	~BatchScope() {
		if (Detail::g_batch_depth != 0 && --Detail::g_batch_depth == 0) {
			FlushBatched();
		}
	}
	BatchScope(const BatchScope&)            = delete;
	BatchScope& operator=(const BatchScope&) = delete;
};

class ScopedTimer final {
public:
	ScopedTimer(Counter count, Counter duration, Peak peak = Peak::Count)
	    : m_count(count), m_duration(duration), m_peak(peak),
	      m_start(Enabled() ? std::chrono::steady_clock::now()
	                        : std::chrono::steady_clock::time_point {}) {}
	~ScopedTimer() {
		if (m_start == std::chrono::steady_clock::time_point {} || !Enabled()) {
			return;
		}
		const auto elapsed =
		    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                              std::chrono::steady_clock::now() - m_start)
		                              .count());
		Add(m_count);
		Add(m_duration, elapsed);
		if (m_peak != Peak::Count) {
			auto& peak     = Detail::g_peaks[static_cast<int>(m_peak)];
			auto  previous = peak.load(std::memory_order_relaxed);
			while (previous < elapsed &&
			       !peak.compare_exchange_weak(previous, elapsed, std::memory_order_relaxed)) {
			}
		}
	}
	ScopedTimer(const ScopedTimer&)            = delete;
	ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
	Counter                               m_count;
	Counter                               m_duration;
	Peak                                  m_peak;
	std::chrono::steady_clock::time_point m_start;
};

[[nodiscard]] inline uint64_t Get(Counter counter) noexcept {
	const auto index = static_cast<uint32_t>(counter);
	const auto value = Detail::g_counters[index].load(std::memory_order_relaxed);
	return Detail::g_batch_depth != 0 ? value + Detail::g_batch[index] : value;
}

[[nodiscard]] inline uint64_t Get(Peak peak) noexcept {
	return Detail::g_peaks[static_cast<int>(peak)].load(std::memory_order_relaxed);
}

} // namespace Common::DebugCounters

#endif /* KYTY_COMMON_DEBUG_COUNTERS_H_ */
