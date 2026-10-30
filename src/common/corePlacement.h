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

// AbSwitch OWN_CORES (bit 48): one physical core for the GPU thread and a second for the
// record-stream consumer, everything else off both (KYTY_GPU_CORE / KYTY_RECORD_CORE pick the
// logical processors; a record id of 0 means the consumer is not known yet or absent). Called by
// the GPU thread when the switch or the consumer's thread id changes.
void SetOwnCores(bool own, unsigned record_thread_id);

// Called when a guest thread starts: it stays off the reserved core(s) while any are reserved.
void KeepOffReservedCore();

// The calling thread runs above normal priority (AbSwitch GPU_THREAD_PRIORITY), or back at normal.
void SetCurrentThreadAboveNormal(bool above_normal);

} // namespace Common::CorePlacement

#endif /* KYTY_COMMON_CORE_PLACEMENT_H_ */
