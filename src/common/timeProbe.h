#ifndef KYTY_COMMON_TIME_PROBE_H_
#define KYTY_COMMON_TIME_PROBE_H_

#include <cstdint>

// Diagnostic for game timing: with KYTY_TIME_PROBE set, calls to the guest's clock, video-out
// status and wait functions are counted per thread and calling guest address, and the busiest
// entries are printed every 5 seconds. With KYTY_TIME_PROBE_TRACE=<thread name fragment>, the
// calls of the matching threads are also written, with their times, to KYTY_TIME_PROBE.
// KYTY_TIME_PROBE_CALLER=<hex guest return address> optionally restricts all recording to one
// caller before thread registration, timestamps or stack walking; unset or zero keeps all callers.
// KYTY_TIME_PROBE_DUMP=<caller>:<offset>:<size> (hex) reads guest register + offset at that caller.
// KYTY_TIME_PROBE_DUMP_REGISTER=r14|r15 selects the base register (default r15).
// KYTY_TIME_PROBE_DUMP_PARENT=<hex guest return address> optionally requires a matching stack
// frame. DUMP_REGISTER=stack-r14 reads the saved R14 at [rbp-8] of the frame returning to
// DUMP_PARENT. This opt-in mode requires the guest wrapper's push-rbp/mov-rbp-rsp/push-r14 prolog
// to be verified.
// Each dump ends with thread=<host thread id>, so analysis can select the same thread as its events.
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
