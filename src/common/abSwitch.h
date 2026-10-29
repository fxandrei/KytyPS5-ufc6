#ifndef KYTY_COMMON_AB_SWITCH_H_
#define KYTY_COMMON_AB_SWITCH_H_

#include <atomic>
#include <cstdint>
#include <cstdlib>

// Switches for A/B performance experiments, one bit each. KYTY_AB=<mask> sets them at startup.
// With KYTY_PERF_CSV, the file named by KYTY_AB_FILE is re-read once per second, so a test
// harness can alternate variants inside one run and compare them on the same scene.
namespace Common::AbSwitch {

inline std::atomic<uint32_t> g_mask {0};

// The switches that measured faster and showed no problem: FLIP_RUNAHEAD, CP_READ_BACKING,
// SRT_READ_BACKING, LABEL_IN_ORDER, EAGER_FLUSH, SKIP_DRAW_CHECKS, EAGER_FLUSH_COARSE, IMAGE_POOL,
// QUEUED_SUBMIT, DIRECT_READBACK, BDA_SYNC_PER_SUBMIT, LAZY_GPU_POLL, BDA_SYNC_WRITE_RANGES,
// PIPELINE_BIND_DEDUPE, TARGET_LOOKUP_MEMO, SPIN_YIELD, SRT_LOAD_CHECK, SPIN_GPU_WAIT,
// DIRECT_URGENT_SUBMIT, EARLY_FLUSH_WRITERS, UPLOAD_PRECHECK, LRU_TOUCH_MEMO and
// TEXTURE_INFO_MEMO. KYTY_AB=0 turns them all off.
inline constexpr uint32_t DEFAULT_MASK =
    (1u << 0) | (1u << 1) | (1u << 2) | (1u << 3) | (1u << 6) | (1u << 7) | (1u << 8) | (1u << 9) |
    (1u << 10) | (1u << 11) | (1u << 13) | (1u << 14) | (1u << 15) | (1u << 16) | (1u << 17) |
    (1u << 19) | (1u << 20) | (1u << 21) | (1u << 22) | (1u << 23) | (1u << 24) | (1u << 27) |
    (1u << 28);

[[nodiscard]] inline uint32_t Mask() noexcept {
	static const bool initialized = [] {
		const char* value = std::getenv("KYTY_AB");
		g_mask.store(value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 0))
		                              : DEFAULT_MASK,
		             std::memory_order_relaxed);
		return true;
	}();
	(void)initialized;
	return g_mask.load(std::memory_order_relaxed);
}

[[nodiscard]] inline bool Enabled(uint32_t bit) noexcept {
	return (Mask() & (1u << bit)) != 0;
}

// Bits in use.
inline constexpr uint32_t FLIP_RUNAHEAD = 0; // Render into a display buffer once its flip is prepared.
// Command-processor reads of guest memory on a GPU-written page whose own bytes the GPU did not
// write come from the backing store, without the fault and the wait for the GPU.
inline constexpr uint32_t CP_READ_BACKING = 1;
// The same for the shader resource-table walker (SrtWalker JIT) loads: the fault handler completes
// a load of clean bytes from the backing store.
inline constexpr uint32_t SRT_READ_BACKING = 2;
// End-of-pipe labels over GPU-written bytes are written to guest memory and, in command order, to
// the cached GPU copy, instead of first downloading the GPU bytes under them.
inline constexpr uint32_t LABEL_IN_ORDER = 3;
// Experiment: a shader-written buffer that covers less than a quarter of an image starting at the
// same address does not make the whole image refresh from the buffer copy. Frostbite reuses
// render-target memory for small transient buffers; the bytes written there are lost to the image.
inline constexpr uint32_t PARTIAL_IMAGE_WRITES = 4;
// Compute dispatches are ordered by the guest's own synchronization (partial flushes, end-of-pipe
// and acquire events, submission starts), as on the hardware, instead of a full barrier around
// each dispatch. Dispatches with emulated LDS or GDS keep their barriers.
inline constexpr uint32_t GUEST_BARRIERS = 5;
// The open command buffer is submitted, without waiting, as soon as the GPU has run out of work
// (checked every 16 recorded draws or dispatches) or after 256 of them, so recording and GPU
// execution overlap instead of alternating at the sync points.
inline constexpr uint32_t EAGER_FLUSH = 6;
// CommitBindings skips its per-draw proof that the scalar reads behind a shader specialization do
// not overlap image, attachment or shader-buffer writes. The proof only exits on a violation.
inline constexpr uint32_t SKIP_DRAW_CHECKS = 7;
// EAGER_FLUSH checks every 64 recorded draws or dispatches instead of 16: fewer and larger
// submits when the GPU is waiting for the command processor (each submit costs the driver
// about 13 us of CPU in the fight). Together with EARLY_FLUSH_WRITERS, which spends those submits
// on the writers that readbacks wait for, the fight ran ~3-5% faster.
inline constexpr uint32_t EAGER_FLUSH_COARSE = 8;
// Images that DeleteImage frees are kept, up to 512 MiB, for a CreateImage with the same
// parameters: Frostbite recreates its transient render targets every frame, and creating and
// freeing their dedicated memory costs the command processor about 3% in the fight.
inline constexpr uint32_t IMAGE_POOL = 9;
// Command buffers without semaphores are submitted by a scheduler thread, in order, instead of
// by the command processor: vkQueueSubmit costs the driver about 13 us of CPU each, 5,000 times a
// second in the fight.
inline constexpr uint32_t QUEUED_SUBMIT = 10;
// A read of GPU-written guest memory copies it straight from the guest buffer, mapped through the
// resizable BAR, once the commands that last wrote the buffer have completed, instead of
// recording a copy, submitting and waiting for all the work recorded so far.
inline constexpr uint32_t DIRECT_READBACK = 11;
// A physical core is kept for the GPU command-processor thread (see common/corePlacement.h).
inline constexpr uint32_t RESERVE_CORE = 12;
// Guest memory that game threads write reaches the buffers that shaders address directly (BDA)
// once per slice of a guest submission, and after each command-processor write, instead of before
// every such draw or dispatch: the CPU writes what a submission reads before submitting it.
inline constexpr uint32_t BDA_SYNC_PER_SUBMIT = 13;
// The timeline semaphore that tells which GPU work has completed is read (a driver call) on every
// 64th draw or dispatch instead of on each, and not again by the eager-flush check.
inline constexpr uint32_t LAZY_GPU_POLL = 14;
// With BDA_SYNC_PER_SUBMIT, a command-processor write synchronizes only the bytes it wrote (an
// end-of-pipe label, about 450 per frame in the fight), not everything game threads wrote since
// the slice began.
inline constexpr uint32_t BDA_SYNC_WRITE_RANGES = 15;
// A draw does not bind the graphics pipeline that the command buffer already has bound.
inline constexpr uint32_t PIPELINE_BIND_DEDUPE = 16;
// A render-target slot whose registers have not changed (HW::Context stamps) reuses its last
// resolved descriptor and image while the texture cache's image set is the same.
inline constexpr uint32_t TARGET_LOOKUP_MEMO = 17;
// Experiment: guest GPU timestamps run slower within each frame (Sync::ReadReferenceClock), so
// dynamic resolution stops reading command-processor time as GPU time.
inline constexpr uint32_t GPU_TIME_COMPRESS = 18;
// A guest thread that reads the clock in a tight loop sleeps 1 ms after every 4096 reads made
// within 2 ms (kernel/pthread.cpp, KernelClockGettime): UFC 6's audio watchdog spins a core.
inline constexpr uint32_t SPIN_YIELD = 19;
// The resource-table walker looks a page up before each guest load and reads GPU-written pages
// through a call instead of taking a page fault (SrtWalker.cpp, LoadGuest).
inline constexpr uint32_t SRT_LOAD_CHECK = 20;
// The command-processor thread polls the GPU timeline for up to 2 ms before a blocking wait
// (MasterSemaphore::Wait): its waits are mostly short readbacks.
inline constexpr uint32_t SPIN_GPU_WAIT = 21;
// A flush that the command processor waits for next (a readback of GPU-written bytes) is
// submitted on its own thread instead of through the submission thread.
inline constexpr uint32_t DIRECT_URGENT_SUBMIT = 22;
// A draw or dispatch that writes into a buffer whose readback once had to wait for the GPU is
// submitted as soon as it is recorded (CommandScheduler::RequestFlushAfterRecord).
inline constexpr uint32_t EARLY_FLUSH_WRITERS = 23;
// A read-only buffer binding first checks, without the region locks, whether the CPU modified
// any of its pages (MemoryTracker::ForEachUploadRange, and BufferCache::ObtainBuffer before its
// small-range stream copy): most bindings have nothing to upload.
inline constexpr uint32_t UPLOAD_PRECHECK = 24;
// The command-processor thread runs above normal priority (without RESERVE_CORE's pinning).
inline constexpr uint32_t GPU_THREAD_PRIORITY = 25;
// A readback whose writer is still being recorded copies through cached host memory (the copy
// rides the flush it needs anyway) instead of reading the resizable BAR afterwards.
inline constexpr uint32_t READBACK_COPY_WHEN_FLUSHING = 26;
// A buffer or image remembers the tick of its LRU entry, so touching it again in the same frame
// does not look the entry up (an allocation of its own in the LRU's deque) only to find it current.
inline constexpr uint32_t LRU_TOUCH_MEMO = 27;
// A texture binding reuses the image layout (ImageInfo, after UpdateSize) it decoded the last
// time it saw the same descriptor with the same resource flags (RenderExecutor::ResolveTexture).
inline constexpr uint32_t TEXTURE_INFO_MEMO = 28;

} // namespace Common::AbSwitch

#endif /* KYTY_COMMON_AB_SWITCH_H_ */
