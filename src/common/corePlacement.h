#ifndef KYTY_COMMON_CORE_PLACEMENT_H_
#define KYTY_COMMON_CORE_PLACEMENT_H_

namespace Common::CorePlacement {

// Windows only: with AbSwitch RESERVE_CORE, a physical core is kept for the GPU command-processor
// thread, which limits UFC 6's fight: a game thread on the same core (the busy-waiting audio
// watchdog, the job workers) would take half of its execution resources. KYTY_GPU_CORE=<logical
// processor> picks the core (default 2, the second core).

// Called by the GPU thread whenever the switch changes: it moves to the reserved logical processor
// above normal priority and every other thread of the process off that core, or all back.
void SetReserved(bool reserved);

// Called when a guest thread starts: it stays off the reserved core while one is reserved.
void KeepOffReservedCore();

} // namespace Common::CorePlacement

#endif /* KYTY_COMMON_CORE_PLACEMENT_H_ */
