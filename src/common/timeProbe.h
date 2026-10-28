#ifndef KYTY_COMMON_TIME_PROBE_H_
#define KYTY_COMMON_TIME_PROBE_H_

#include <cstdint>

// Diagnostic for game timing: with KYTY_TIME_PROBE set, calls to the guest's clock, video-out
// status and wait functions are counted per thread and calling guest address, and the busiest
// entries are printed every 5 seconds. With KYTY_TIME_PROBE_TRACE=<thread name fragment>, the
// calls of the matching threads are also written, with their times, to KYTY_TIME_PROBE.
namespace Common::TimeProbe {

enum class Source : uint32_t {
	ProcessTime,
	ReadTsc,
	ProcessTimeCounter,
	ClockGettime,
	Gettimeofday,
	FlipStatus,
	VblankStatus,
	IsFlipPending,
	WaitVblank,
	WaitEqueue,
	Usleep,
	Nanosleep,
	Submit,
	Count
};

[[nodiscard]] bool Enabled() noexcept;

// `detail` tells calls of one source apart, e.g. the clock id.
void Record(Source source, uint64_t detail, const void* return_address) noexcept;

} // namespace Common::TimeProbe

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define KYTY_TIME_PROBE(source, detail)                                                            \
	do {                                                                                           \
		if (Common::TimeProbe::Enabled()) {                                                        \
			Common::TimeProbe::Record(Common::TimeProbe::Source::source,                           \
			                          static_cast<uint64_t>(detail), __builtin_return_address(0)); \
		}                                                                                          \
	} while (false)

#endif /* KYTY_COMMON_TIME_PROBE_H_ */
