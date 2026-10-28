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
// SRT_READ_BACKING, LABEL_IN_ORDER, EAGER_FLUSH and SKIP_DRAW_CHECKS. KYTY_AB=0 turns them all off.
inline constexpr uint32_t DEFAULT_MASK =
    (1u << 0) | (1u << 1) | (1u << 2) | (1u << 3) | (1u << 6) | (1u << 7);

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
// about 13 us of CPU in the fight).
inline constexpr uint32_t EAGER_FLUSH_COARSE = 8;
// Images that DeleteImage frees are kept, up to 512 MiB, for a CreateImage with the same
// parameters: Frostbite recreates its transient render targets every frame, and creating and
// freeing their dedicated memory costs the command processor about 3% in the fight.
inline constexpr uint32_t IMAGE_POOL = 9;

} // namespace Common::AbSwitch

#endif /* KYTY_COMMON_AB_SWITCH_H_ */
