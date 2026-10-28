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

[[nodiscard]] inline uint32_t Mask() noexcept {
	static const bool initialized = [] {
		if (const char* value = std::getenv("KYTY_AB"); value != nullptr) {
			g_mask.store(static_cast<uint32_t>(std::strtoul(value, nullptr, 0)),
			             std::memory_order_relaxed);
		}
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

} // namespace Common::AbSwitch

#endif /* KYTY_COMMON_AB_SWITCH_H_ */
