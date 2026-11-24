#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_

#include "common/abSwitch.h"
#include "common/common.h"
#include "common/debugCounters.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/commandHooks.h"
#include "graphics/host_gpu/renderer/dynamicState.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/render.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>

#include <queue>

#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

// With KYTY_GPU_CHECKPOINTS: marks the GPU position in a command buffer of the queue before an
// operation of the renderer's own (copies, tiling, blits), for the stall and lost-device report.
[[nodiscard]] bool GpuCheckpointsEnabled() noexcept;
void               MarkGpuCheckpoint(vk::CommandBuffer command, std::string_view name);
[[nodiscard]] bool GpuBufferWordProbeEnabled() noexcept;

class CommandScheduler {
public:
	CommandScheduler(RenderContext& context, GraphicContext& graphics);
	~CommandScheduler();
	KYTY_CLASS_NO_COPY(CommandScheduler);

	void Begin(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders);
	void EndRendering() { m_command.EndRendering(); }
	void Flush();
	void Flush(SubmitInfo& submit);
	// Flush for a caller that waits for this work next: with AbSwitch DIRECT_URGENT_SUBMIT the
	// command buffer is submitted on this thread, after the queued ones, so the submission
	// thread's wake-up does not add to the wait.
	void           FlushUrgent();
	void           Finish();
	CommandBuffer& BeginCommand();
	// A compute dispatch deferred its post-dispatch ShaderAccessBarrier (AbSwitch COMPUTE_CHAIN):
	// consecutive dispatches are ordered by one barrier at the end of the run.
	void DeferShaderAccessBarrier();
	// Records the pending deferred barrier on the current command buffer, if any. `full` selects a
	// GuestFlushBarrier (AbSwitch BARRIER_LITE off) instead of the narrow compute one.
	void               FlushDeferredShaderAccessBarrier(bool full);
	[[nodiscard]] bool HasDeferredShaderAccessBarrier() const noexcept {
		return m_deferred_shader_access;
	}
	uint64_t Submit(SubmitInfo submit = {});
	// Deferred callbacks can observe an externally owned drain, but cannot initiate shutdown:
	// the priority runner cannot join itself.
	void Shutdown();
	void Wait(uint64_t tick);
	void PopPendingOperations();
	void DrainPriorityOperations();
	void WaitPriorityOperations(uint64_t tick);
	// Guest-memory completions use the priority queue; normal callbacks maintain GPU resources.
	void                      DeferOperation(Common::UniqueFunction<void>&& operation);
	// A priority operation that writes guest memory (or changes its page protection) names that
	// guest range, so an unmap waits only for the publications into its own range
	// (KYTY_UNMAP_TARGETED). Operations without a range must not touch guest memory.
	void                      DeferPriorityOperation(Common::UniqueFunction<void>&& operation,
	                                                 uint64_t guest_vaddr = 0,
	                                                 uint64_t guest_size  = 0);
	[[nodiscard]] bool        HasPendingPriorityOperations();
	// The newest tick of a queued or running priority operation whose guest range overlaps
	// [vaddr, vaddr + size), or 0 when none does.
	[[nodiscard]] uint64_t NewestGuestPublicationTick(uint64_t vaddr, uint64_t size);
	// Queued or running priority operations that carry a guest range (KYTY_VERIFY_UNMAP_TARGETED).
	[[nodiscard]] uint64_t PendingGuestPublications();
	[[nodiscard]] static bool InDeferredOperation() noexcept;
	// Installs the Vulkan command hooks and, with AbSwitch RECORD_THREAD (or KYTY_AB_FILE, so the
	// switch can be toggled inside a run), starts the submit thread that records the GPU thread's
	// commands through a CommandStream and submits them in order. The device dispatcher must be
	// initialized (RenderContext construction). Enable before the first Submit().
	void EnableRecordThread();
	// Waits until the record stream's consumer has recorded and submitted everything appended so
	// far (a no-op without a stream). Call from a thread that does not record before taking the
	// queue lock, like the presenter's RenderDoc capture.
	void DrainGpuCommands();
	// The record stream's consumer thread id, or 0 when no stream runs (AbSwitch OWN_CORES).
	[[nodiscard]] uint32_t RecordThreadId() const noexcept {
		return m_stream != nullptr ? m_record_thread_id.load(std::memory_order_relaxed) : 0;
	}

	// Diagnostics while the counters are on: GPU time of the commands recorded between the two
	// calls, added to `counter` once their tick completes. Ends rendering first.
	[[nodiscard]] uint32_t BeginGpuScope();
	void                   EndGpuScope(uint32_t scope, Common::DebugCounters::Counter counter);
	// With KYTY_GPU_MARKERS=<path>: GPU time of each guest debug marker (PUSH_MARKER to
	// POP_MARKER, children included), appended to <path> every 5 seconds as running totals.
	[[nodiscard]] uint32_t BeginGpuMarker();
	void                   EndGpuMarker(uint32_t scope, std::string_view name);
	// With KYTY_GPU_SHADERS (and KYTY_GPU_MARKERS for the output): the same per draw or dispatch,
	// keyed by its shader addresses.
	[[nodiscard]] uint32_t BeginGpuShaderTimer();
	// Opt-in diagnostics: snapshot one native dword before a suspect operation, without
	// submitting or waiting. Query availability permits reporting completed copies on a stall.
	void ProbeBufferWord(vk::Buffer source, uint64_t offset, std::string_view label);
	void ReportBufferWordProbes();
	// With KYTY_GPU_CHECKPOINTS: remembers the argument range of the latest indirect draws. A
	// stall report then prints, for draws still in flight, the argument words as the GPU sees
	// them (read through the buffer's host mapping) next to the guest's CPU backing. Adds no GPU
	// commands, so it does not change the timing of the run.
	void NoteIndirectDraw(const Buffer* args, uint64_t offset, uint32_t count, uint32_t stride,
	                      bool indexed, uint64_t vs_hash, uint64_t ps_hash);
	[[nodiscard]] bool CheckpointsEnabled() const noexcept { return m_graphics_checkpoints; }
	// Indirect-draw notes are on with KYTY_GPU_CHECKPOINTS or, without the checkpoint labels'
	// cost (~12% of the GPU thread), with KYTY_INDIRECT_NOTES=1.
	[[nodiscard]] static bool IndirectNotesEnabled() noexcept;
	// KYTY_HANG_DIAG=1 (diagnostic, default off, adds no GPU commands): a ring of the latest draws
	// and dispatches (programs, size, indirect) and of the GPU-visible loop-bound word of the two
	// screen-space filter pixel shaders seen at the arena-transition hang. A stall report prints,
	// for the command buffer the GPU started and has not finished (tick finished + 1), a summary
	// of its work and the bound word as the GPU sees it (host mapping) and as the CPU backing has
	// it. Also turns on the indirect-draw notes.
	[[nodiscard]] static bool HangDiagEnabled() noexcept;
	void NoteHangWork(char kind, uint64_t hash_a, uint64_t hash_b, uint32_t x, uint32_t y,
	                  uint32_t z, const Buffer* args = nullptr, uint64_t args_offset = 0,
	                  bool indexed = false, bool indirect = false);
	void NoteHangBoundWord(const Buffer* buffer, uint64_t offset, uint64_t guest,
	                       uint64_t shader_hash, uint32_t cpu_word_at_record, bool gpu_dirty);
	// KYTY_READBACK_TIMING=1 (diagnostic): every direct readback wait is split into the time until
	// the writer's command buffer was submitted (record stream lag included) and the time the GPU
	// then needed; a summary line is printed every 5 s.
	[[nodiscard]] static bool ReadbackTimingEnabled() noexcept;
	void NoteReadbackWait(uint64_t write_tick, bool writer_in_open_buffer, uint64_t in_flight,
	                      int64_t start_ns, int64_t flushed_ns, int64_t done_ns, int64_t end_ns);
	[[nodiscard]] static int64_t NowNs() noexcept;

	// With KYTY_GPU_CHECKPOINTS: marks the GPU position before a draw or dispatch, named by
	// make_name() (only called when checkpoints are on), for the stall and lost-device report.
	template <typename MakeName>
	void GpuCheckpoint(MakeName&& make_name) {
		if (m_graphics_checkpoints && !m_command.IsInvalid()) {
			SetGpuCheckpoint(make_name());
		}
	}

	// After each recorded draw or dispatch; see AbSwitch EAGER_FLUSH.
	void RecordedWork();
	// The draw or dispatch being recorded writes bytes that the command processor has had to wait
	// for before: RecordedWork submits it right away (AbSwitch EARLY_FLUSH_WRITERS).
	void RequestFlushAfterRecord() noexcept {
		m_flush_after_record = Common::AbSwitch::Enabled(Common::AbSwitch::EARLY_FLUSH_WRITERS);
	}
	// Every submitted command buffer then ends with a barrier that makes its writes visible to the
	// host, for guest buffers read in place (AbSwitch DIRECT_READBACK).
	void SetHostReadBarrier(bool enabled, bool diagnostic_ab = false) noexcept {
		m_host_read_barrier = enabled;
		m_host_read_barrier_ab = diagnostic_ab;
	}
	// The tick clock that image ages use (TextureCache, ImageHeap): eager submits do not count,
	// so they do not make cached images look older than before.
	[[nodiscard]] uint64_t AgeTick() const noexcept { return CurrentTick() - m_eager_submits; }

	[[nodiscard]] bool     Active() const noexcept { return m_command.m_registers != nullptr; }
	void                   CheckActive() const;
	CommandBuffer&         Current();
	[[nodiscard]] uint64_t CurrentTick() const noexcept { return m_master.CurrentTick(); }
	[[nodiscard]] bool     IsFree(uint64_t tick);
	[[nodiscard]] MasterSemaphore& GetMasterSemaphore() noexcept { return m_master; }
	[[nodiscard]] DynamicState&    GetDynamicState() noexcept { return m_dynamic_state; }
	[[nodiscard]] RenderContext&   Context() const noexcept { return m_context; }
	[[nodiscard]] GraphicContext&  Graphics() const noexcept { return m_graphics; }

private:
	struct BufferWordProbe;
	std::unique_ptr<BufferWordProbe> m_buffer_word_probe;
	struct IndirectDrawNote {
		const Buffer* args     = nullptr;
		uint64_t      offset   = 0;
		uint64_t      guest    = 0;
		uint64_t      tick     = 0;
		uint64_t      vs_hash  = 0;
		uint64_t      ps_hash  = 0;
		uint32_t      count    = 0;
		uint32_t      stride   = 0;
		bool          indexed  = false;
	};
	static constexpr uint32_t                   INDIRECT_NOTES = 256;
	std::mutex                                  m_indirect_notes_mutex;
	std::array<IndirectDrawNote, INDIRECT_NOTES> m_indirect_notes {};
	uint64_t                                    m_indirect_notes_next = 0;
	void                                        ReportIndirectDraws(uint64_t finished);
	struct HangWorkNote {
		const Buffer* args        = nullptr;
		uint64_t      args_offset = 0;
		uint64_t      tick        = 0;
		uint64_t      hash_a      = 0;
		uint64_t      hash_b      = 0;
		uint32_t      x           = 0;
		uint32_t      y           = 0;
		uint32_t      z           = 0;
		char          kind        = 0;
		bool          indexed     = false;
		bool          indirect    = false;
	};
	struct HangBoundNote {
		const Buffer* buffer      = nullptr;
		uint64_t      offset      = 0;
		uint64_t      guest       = 0;
		uint64_t      tick        = 0;
		uint64_t      shader_hash = 0;
		uint32_t      cpu_word    = 0;
		bool          gpu_dirty   = false;
	};
	static constexpr uint32_t                   HANG_WORK_NOTES  = 8192;
	static constexpr uint32_t                   HANG_BOUND_NOTES = 64;
	std::mutex                                  m_hang_mutex;
	std::vector<HangWorkNote>                   m_hang_work;
	std::array<HangBoundNote, HANG_BOUND_NOTES> m_hang_bounds {};
	uint64_t                                    m_hang_work_next  = 0;
	uint64_t                                    m_hang_bound_next = 0;
	void                                        ReportHangNotes(uint64_t finished);

	class CommandPool {
	public:
		CommandPool(GraphicContext& graphics, MasterSemaphore& master);
		~CommandPool();
		KYTY_CLASS_NO_COPY(CommandPool);

		vk::CommandBuffer Commit();

	private:
		static constexpr size_t GrowStep = 4;

		size_t Grow();

		GraphicContext&                m_graphics;
		MasterSemaphore&               m_master;
		vk::CommandPool                m_pool = nullptr;
		std::vector<vk::CommandBuffer> m_buffers;
		std::vector<uint64_t>          m_ticks;
		size_t                         m_hint = 0;
	};

	enum class OperationState { Open, Draining, Closed };

	struct PendingOperation {
		Common::UniqueFunction<void> callback;
		uint64_t                     tick = 0;
		// The guest range a priority operation writes, [guest_begin, guest_end); empty for none.
		uint64_t guest_begin = 0;
		uint64_t guest_end   = 0;
	};

	void                   BeginNext();
	void                   StartGpuTimer();
	void                   EndGpuTimer(uint64_t tick);
	void                   ReadGpuTimers();
	void                   ReportStall(uint64_t tick);
	void                   SetGpuCheckpoint(std::string_view name);
	[[nodiscard]] uint32_t MarkerId(std::string_view name);
	void                   PriorityOperationsThread(std::stop_token stop);
	// AbSwitch QUEUED_SUBMIT: m_submit_thread submits these in order.
	struct QueuedSubmit {
		vk::CommandBuffer buffer;
		uint64_t          tick         = 0;
		uint32_t          debug_op     = 0;
		uint64_t          debug_submit = 0;
	};
	void SubmitThread(std::stop_token stop);
	// The submit itself: a timeline signal on the master semaphore and the vkQueueSubmit under
	// the queue lock, with the lost-device report. Runs on the submit thread (both the deque and
	// the record stream).
	void SubmitQueuedJob(const QueuedSubmit& job);
	// Returns once every queued submit is in the queue, so a direct submit cannot overtake one.
	void DrainQueuedSubmits();
	void QueueOperation(Common::UniqueFunction<void>&& operation, bool priority,
	                    uint64_t guest_begin = 0, uint64_t guest_end = 0);
	void RunOperation(Common::UniqueFunction<void>&& operation);

	MasterSemaphore              m_master;
	RenderContext&               m_context;
	GraphicContext&              m_graphics;
	CommandPool                  m_command_pool;
	CommandBuffer                m_command;
	DynamicState                 m_dynamic_state;
	std::queue<PendingOperation> m_pending_operations;
	// A deque, not a queue: an unmap looks for the publications into its range (KYTY_UNMAP_TARGETED).
	std::deque<PendingOperation> m_priority_operations;
	std::mutex                   m_operation_mutex;
	std::condition_variable      m_operation_available;
	std::jthread                 m_priority_thread;
	std::mutex                   m_submit_mutex;
	std::condition_variable_any  m_submit_changed;
	std::deque<QueuedSubmit>     m_submit_queue;
	bool                         m_submit_busy = false;
	std::jthread                 m_submit_thread;        // Started by the first queued submit.
	std::atomic<uint32_t>        m_record_thread_id {0}; // The stream consumer (OWN_CORES).
	// Deferred recording (AbSwitch RECORD_THREAD): the GPU thread's commands and submits,
	// recorded and submitted in order by the submit thread (see CommandStream).
	std::unique_ptr<CommandStream> m_stream;
	uint64_t                       m_priority_active_tick = 0; // 0 while none runs
	// The running priority operation's guest range (m_operation_mutex).
	uint64_t                       m_priority_active_begin = 0;
	uint64_t                       m_priority_active_end   = 0;
	OperationState                m_operation_state      = OperationState::Open;

	// Diagnostics while the counters are on: GPU execution time of each command buffer, from
	// timestamps at its start and end, read once its tick completes (Counter::GpuBusyNs).
	struct GpuTimerQuery {
		uint64_t                       tick;
		uint32_t                       query; // The first of a pair.
		Common::DebugCounters::Counter counter;
		uint32_t                       marker = UINT32_MAX; // Index in m_marker_names.
	};
	[[nodiscard]] uint32_t AllocateGpuTimer();
	void                   DumpGpuMarkers();
	// The last submitted command buffers, for the stall report: a command buffer that started and
	// never finished is named by its tick, and this says what it was (draw, dispatch, submit).
	struct RecentSubmit {
		uint64_t tick      = 0;
		uint32_t op        = 0; // CommandBufferDebugOp
		uint64_t submit_id = 0;
	};
	void RecordRecentSubmit(uint64_t tick, uint32_t op, uint64_t submit_id);
	[[nodiscard]] std::optional<RecentSubmit> FindRecentSubmit(uint64_t tick) const;
	mutable std::mutex                        m_recent_submit_mutex;
	std::deque<RecentSubmit>                  m_recent_submits;

	static constexpr uint32_t GPU_TIMER_QUERIES = 16384;
	vk::QueryPool             m_gpu_timer_pool;
	uint32_t                  m_gpu_timer_next  = 0;
	uint32_t                  m_gpu_timer_query = UINT32_MAX;
	std::deque<GpuTimerQuery> m_gpu_timer_pending;
	// With KYTY_GPU_MARKERS: query pairs still owned by a timer (1 per pair). Guest markers stay
	// open across many command buffers and are not in m_gpu_timer_pending until they close.
	std::vector<uint8_t>      m_gpu_pair_busy;
	void                      ReleaseGpuTimerPair(uint32_t query);

	std::unordered_map<std::string, uint32_t> m_marker_ids;
	std::vector<std::string>                  m_marker_names;
	std::vector<uint64_t>                     m_marker_ns;
	std::vector<uint64_t>                     m_marker_counts;
	uint64_t                                  m_marker_dump_ms        = 0;
	uint64_t                                  m_gpu_busy_end          = 0; // ns on the GPU clock
	uint32_t                                  m_recorded_since_submit = 0;
	uint64_t                                  m_eager_submits         = 0;
	bool                                      m_graphics_checkpoints  = false;
	bool                                      m_host_read_barrier     = false;
	bool                                      m_host_read_barrier_ab  = false;
	bool                                      m_direct_submit         = false; // FlushUrgent
	bool                                      m_flush_after_record    = false;
	// KYTY_READBACK_TIMING: steady-clock ns of each tick's vkQueueSubmit call (tick % 4096).
	std::array<std::atomic<int64_t>, 4096> m_submit_ns {};
	uint32_t                                  m_pop_calls             = 0; // AbSwitch LAZY_GPU_POLL
	uint64_t                                  m_last_verify_tick = UINT64_MAX; // KYTY_VERIFY_RECORD
	// A chained compute dispatch left its access barrier for the next sync point (AbSwitch
	// COMPUTE_CHAIN): submitted before the command buffer ends.
	bool m_deferred_shader_access = false;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
