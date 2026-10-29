#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_

#include "common/common.h"
#include "common/debugCounters.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/dynamicState.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/render.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
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

class CommandScheduler {
public:
	CommandScheduler(RenderContext& context, GraphicContext& graphics);
	~CommandScheduler();
	KYTY_CLASS_NO_COPY(CommandScheduler);

	void           Begin(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders);
	void           BeginRendering(const RenderState& state);
	void           EndRendering();
	void           Flush();
	void           Flush(SubmitInfo& submit);
	// Flush for a caller that waits for this work next: with AbSwitch DIRECT_URGENT_SUBMIT the
	// command buffer is submitted on this thread, after the queued ones, so the submission
	// thread's wake-up does not add to the wait.
	void           FlushUrgent();
	void           FlushAndWait();
	void           Finish();
	CommandBuffer& BeginCommand();
	uint64_t       Submit(SubmitInfo submit = {});
	// Deferred callbacks can observe an externally owned drain, but cannot initiate shutdown:
	// the priority runner cannot join itself.
	void                      Shutdown();
	void                      Wait(uint64_t tick);
	void                      PopPendingOperations();
	void                      DrainPriorityOperations();
	void                      WaitPriorityOperations(uint64_t tick);
	// Guest-memory completions use the priority queue; normal callbacks maintain GPU resources.
	void                      DeferOperation(Common::UniqueFunction<void>&& operation);
	void                      DeferPriorityOperation(Common::UniqueFunction<void>&& operation);
	[[nodiscard]] bool        HasPendingPriorityOperations();
	[[nodiscard]] static bool InDeferredOperation() noexcept;

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
	// Every submitted command buffer then ends with a barrier that makes its writes visible to the
	// host, for guest buffers read in place (AbSwitch DIRECT_READBACK).
	void SetHostReadBarrier(bool enabled) noexcept { m_host_read_barrier = enabled; }
	// The tick clock that image ages use (TextureCache, ImageHeap): eager submits do not count,
	// so they do not make cached images look older than before.
	[[nodiscard]] uint64_t AgeTick() const noexcept { return CurrentTick() - m_eager_submits; }

	[[nodiscard]] bool Active() const noexcept { return m_command.m_registers != nullptr; }
	void                           CheckActive() const;
	CommandBuffer&                 Current();
	[[nodiscard]] uint64_t         CurrentTick() const noexcept { return m_master.CurrentTick(); }
	[[nodiscard]] bool             IsFree(uint64_t tick);
	[[nodiscard]] MasterSemaphore& GetMasterSemaphore() noexcept { return m_master; }
	[[nodiscard]] DynamicState&    GetDynamicState() noexcept { return m_dynamic_state; }
	[[nodiscard]] RenderContext&   Context() const noexcept { return m_context; }
	[[nodiscard]] GraphicContext&  Graphics() const noexcept { return m_graphics; }

private:
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
	};

	void BeginNext();
	void StartGpuTimer();
	void EndGpuTimer(uint64_t tick);
	void ReadGpuTimers();
	void                   ReportStall(uint64_t tick);
	void                   SetGpuCheckpoint(std::string_view name);
	[[nodiscard]] uint32_t MarkerId(std::string_view name);
	void PriorityOperationsThread(std::stop_token stop);
	// AbSwitch QUEUED_SUBMIT: m_submit_thread submits these in order.
	struct QueuedSubmit {
		vk::CommandBuffer buffer;
		uint64_t          tick         = 0;
		uint32_t          debug_op     = 0;
		uint64_t          debug_submit = 0;
	};
	void SubmitThread(std::stop_token stop);
	// Returns once every queued submit is in the queue, so a direct submit cannot overtake one.
	void DrainQueuedSubmits();
	void QueueOperation(Common::UniqueFunction<void>&& operation, bool priority);
	void RunOperation(Common::UniqueFunction<void>&& operation);

	MasterSemaphore              m_master;
	RenderContext&               m_context;
	GraphicContext&              m_graphics;
	CommandPool                  m_command_pool;
	CommandBuffer                m_command;
	DynamicState                 m_dynamic_state;
	std::queue<PendingOperation> m_pending_operations;
	std::queue<PendingOperation> m_priority_operations;
	std::mutex                   m_operation_mutex;
	std::condition_variable      m_operation_available;
	std::jthread                 m_priority_thread;
	std::mutex                   m_submit_mutex;
	std::condition_variable_any  m_submit_changed;
	std::deque<QueuedSubmit>     m_submit_queue;
	bool                         m_submit_busy = false;
	std::jthread                 m_submit_thread; // Started by the first queued submit.
	bool                         m_priority_active      = false;
	uint64_t                     m_priority_active_tick = 0;
	OperationState               m_operation_state      = OperationState::Open;

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

	static constexpr uint32_t GPU_TIMER_QUERIES = 16384;
	vk::QueryPool             m_gpu_timer_pool;
	uint32_t                  m_gpu_timer_next  = 0;
	uint32_t                  m_gpu_timer_query = UINT32_MAX;
	std::deque<GpuTimerQuery> m_gpu_timer_pending;

	std::unordered_map<std::string, uint32_t> m_marker_ids;
	std::vector<std::string>                  m_marker_names;
	std::vector<uint64_t>                     m_marker_ns;
	std::vector<uint64_t>                     m_marker_counts;
	uint64_t                                  m_marker_dump_ms = 0;
	uint64_t                                  m_gpu_busy_end   = 0; // ns on the GPU clock
	uint32_t                                  m_recorded_since_submit = 0;
	uint64_t                                  m_eager_submits         = 0;
	bool                                      m_graphics_checkpoints  = false;
	bool                                      m_host_read_barrier     = false;
	bool                                      m_direct_submit         = false; // FlushUrgent
	uint32_t                                  m_pop_calls             = 0; // AbSwitch LAZY_GPU_POLL
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
