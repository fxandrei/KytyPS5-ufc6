#include "graphics/host_gpu/renderer/commandScheduler.h"

#include "common/abSwitch.h"
#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace Libs::Graphics {

static thread_local CommandScheduler* g_deferred_callback_scheduler = nullptr;

namespace {

void ReportVulkanFatal(const char* what, vk::Result result, uint64_t tick, uint32_t debug_op,
                       uint64_t debug_submit, uint32_t arg0, uint32_t arg1, uint32_t arg2,
                       uint32_t arg3, uint64_t arg4) {
	LOGF("%s failed: %s (%d), tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	     " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	     what, vk::to_string(result).c_str(), static_cast<int>(result), tick, debug_op,
	     debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::printf("%s failed: %s (%d), tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	            " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	            what, vk::to_string(result).c_str(), static_cast<int>(result), tick, debug_op,
	            debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::fflush(stdout);
}

// Checkpoint names for every scheduler of the queue: a lost device can surface in another
// scheduler's submit (the presenter's) than the one that recorded the checkpoint.
struct CheckpointNames {
	std::mutex                                mutex;
	std::unordered_map<std::string, uint32_t> ids;
	std::vector<std::string>                  names;
};

CheckpointNames& GetCheckpointNames() {
	static CheckpointNames names;
	return names;
}

std::atomic_bool g_checkpoints_enabled {false};

} // namespace

bool GpuCheckpointsEnabled() noexcept {
	return g_checkpoints_enabled.load(std::memory_order_relaxed);
}

void MarkGpuCheckpoint(vk::CommandBuffer command, std::string_view name) {
	auto&           checkpoints = GetCheckpointNames();
	std::lock_guard lock(checkpoints.mutex);
	auto [it, inserted] = checkpoints.ids.try_emplace(
	    std::string(name), static_cast<uint32_t>(checkpoints.names.size()));
	if (inserted) {
		checkpoints.names.emplace_back(name);
	}
	// The marker is the name's index plus one, so that no marker is null.
	command.setCheckpointNV(reinterpret_cast<const void*>(static_cast<uintptr_t>(it->second) + 1));
}

CommandScheduler::CommandPool::CommandPool(GraphicContext& graphics, MasterSemaphore& master)
    : m_graphics(graphics), m_master(master) {
	EXIT_IF(graphics.queue_family == static_cast<uint32_t>(-1));
	vk::CommandPoolCreateInfo create {};
	create.queueFamilyIndex = graphics.queue_family;
	create.flags            = vk::CommandPoolCreateFlagBits::eTransient |
	                          vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	const auto result       = graphics.device.createCommandPool(&create, nullptr, &m_pool);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_pool == nullptr);
}

CommandScheduler::CommandPool::~CommandPool() {
	m_graphics.device.destroyCommandPool(m_pool, nullptr);
}

size_t CommandScheduler::CommandPool::Grow() {
	const auto first = m_ticks.size();
	m_ticks.resize(first + GrowStep);
	m_buffers.resize(first + GrowStep);

	vk::CommandBufferAllocateInfo allocate {};
	allocate.commandPool        = m_pool;
	allocate.level              = vk::CommandBufferLevel::ePrimary;
	allocate.commandBufferCount = static_cast<uint32_t>(GrowStep);
	EXIT_IF(m_graphics.device.allocateCommandBuffers(&allocate, m_buffers.data() + first) !=
	        vk::Result::eSuccess);
	return first;
}

vk::CommandBuffer CommandScheduler::CommandPool::Commit() {
	auto       gpu_tick = m_master.KnownGpuTick();
	const auto search   = [this, &gpu_tick](size_t begin, size_t end) -> std::optional<size_t> {
		for (size_t index = begin; index < end; ++index) {
			if (gpu_tick >= m_ticks[index]) {
				m_ticks[index] = m_master.CurrentTick();
				return index;
			}
		}
		return std::nullopt;
	};

	auto found = search(m_hint, m_ticks.size());
	if (!found) {
		m_master.Refresh();
		gpu_tick = m_master.KnownGpuTick();
		found    = search(m_hint, m_ticks.size());
	}
	if (!found) {
		found = search(0, m_hint);
	}
	if (!found) {
		found           = Grow();
		m_ticks[*found] = m_master.CurrentTick();
	}

	m_hint = (*found + 1) % m_ticks.size();
	return m_buffers[*found];
}

bool CommandScheduler::InDeferredOperation() noexcept {
	return g_deferred_callback_scheduler != nullptr;
}

CommandScheduler::CommandScheduler(RenderContext& context, GraphicContext& graphics)
    : m_master(graphics), m_context(context), m_graphics(graphics),
      m_command_pool(graphics, m_master), m_command(*this),
      m_priority_thread([this](std::stop_token stop) { PriorityOperationsThread(stop); }),
      m_graphics_checkpoints(graphics.checkpoints_enabled) {
	m_master.SetStallHandler([this](uint64_t tick) { ReportStall(tick); });
	if (graphics.checkpoints_enabled) {
		g_checkpoints_enabled.store(true, std::memory_order_relaxed);
	}
}

CommandScheduler::~CommandScheduler() {
	Shutdown();
	if (m_gpu_timer_pool) {
		m_graphics.device.destroyQueryPool(m_gpu_timer_pool);
	}
}

void CommandScheduler::Shutdown() {
	{
		std::unique_lock lock(m_operation_mutex);
		if (m_operation_state == OperationState::Closed) {
			return;
		}
		if (g_deferred_callback_scheduler == this) {
			EXIT_IF(m_operation_state == OperationState::Open);
			// A priority callback cannot join its own runner, while a normal callback can be
			// executing inside the shutdown owner's final PopPendingOperations. The owning
			// thread will finish shutdown after this callback returns.
			return;
		}
		if (m_operation_state == OperationState::Draining) {
			m_operation_available.wait(
			    lock, [this] { return m_operation_state == OperationState::Closed; });
			return;
		}
		m_operation_state = OperationState::Draining;
	}
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	PopPendingOperations();
	DrainPriorityOperations();
	m_priority_thread.request_stop();
	m_operation_available.notify_all();
	if (m_priority_thread.joinable()) {
		m_priority_thread.join();
	}
	if (m_submit_thread.joinable()) {
		DrainQueuedSubmits();
		m_submit_thread.request_stop();
		m_submit_thread.join();
	}
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(!m_pending_operations.empty() || !m_priority_operations.empty() ||
		        m_priority_active);
		m_operation_state = OperationState::Closed;
	}
	m_operation_available.notify_all();
}

void CommandScheduler::Begin(HW::Context& registers, HW::UserConfig& user_config,
                             HW::Shader& shaders) {
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(m_operation_state != OperationState::Open);
	}
	m_command.Bind(registers, user_config, shaders);

	if (m_command.IsInvalid()) {
		BeginNext();
	}
}

void CommandScheduler::BeginRendering(const RenderState& state) {
	Current().BeginRendering(state);
}

void CommandScheduler::EndRendering() {
	if (Active() && !m_command.IsInvalid()) {
		Current().EndRendering();
	}
}

void CommandScheduler::Flush() {
	SubmitInfo submit;
	Flush(submit);
}

void CommandScheduler::RecordedWork() {
	const uint32_t interval =
	    Common::AbSwitch::Enabled(Common::AbSwitch::EAGER_FLUSH_COARSE) ? 64 : 16;
	if (!Common::AbSwitch::Enabled(Common::AbSwitch::EAGER_FLUSH) || m_command.IsInvalid() ||
	    ++m_recorded_since_submit % interval != 0) {
		return;
	}
	m_master.Refresh();
	if (m_recorded_since_submit >= 256 || m_master.IsFree(CurrentTick() - 1)) {
		Common::DebugCounters::Add(Common::DebugCounters::Counter::EagerFlushes);
		Flush();
		m_eager_submits++;
	}
}

void CommandScheduler::Flush(SubmitInfo& submit) {
	Submit(submit);
	BeginNext();
}

void CommandScheduler::FlushAndWait() {
	const auto tick = Submit();
	m_master.Wait(tick);
	BeginNext();
}

void CommandScheduler::Finish() {
	CheckActive();
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	BeginNext();
	PopPendingOperations();
}

void CommandScheduler::Wait(uint64_t tick) {
	EXIT_IF(tick > CurrentTick());
	if (tick == CurrentTick()) {
		CheckActive();
		// A stream-buffer wrap can wait while a draw is being prepared through a reference to
		// Current(). The wrapper stays stable while its pooled Vulkan buffer is retired. Deferred
		// resources are released only at the next GPU operation boundary.
		const auto submitted_tick = Submit();
		EXIT_IF(submitted_tick != tick);
		m_master.Wait(tick);
		BeginNext();
	} else {
		m_master.Wait(tick);
	}
}

void CommandScheduler::PopPendingOperations() {
	m_master.Refresh();
	ReadGpuTimers();
	for (;;) {
		PendingOperation operation;
		{
			std::lock_guard lock(m_operation_mutex);
			if (m_pending_operations.empty() ||
			    !m_master.IsFree(m_pending_operations.front().tick)) {
				return;
			}
			operation = std::move(m_pending_operations.front());
			m_pending_operations.pop();
		}
		WaitPriorityOperations(operation.tick);
		RunOperation(std::move(operation.callback));
	}
}

void CommandScheduler::DeferOperation(Common::UniqueFunction<void>&& operation) {
	QueueOperation(std::move(operation), false);
}

void CommandScheduler::DeferPriorityOperation(Common::UniqueFunction<void>&& operation) {
	QueueOperation(std::move(operation), true);
}

void CommandScheduler::QueueOperation(Common::UniqueFunction<void>&& operation, bool priority) {
	CheckActive();
	EXIT_IF(!operation);
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		auto& queue = priority ? m_priority_operations : m_pending_operations;
		queue.push({std::move(operation), CurrentTick()});
		lock.unlock();
		if (priority) {
			m_operation_available.notify_one();
		}
		return;
	}
	if (g_deferred_callback_scheduler != this) {
		m_operation_available.wait(lock,
		                           [this] { return m_operation_state == OperationState::Closed; });
	}
	lock.unlock();
	operation();
}

bool CommandScheduler::HasPendingPriorityOperations() {
	std::lock_guard lock(m_operation_mutex);
	return !m_priority_operations.empty() || m_priority_active;
}

void CommandScheduler::PriorityOperationsThread(std::stop_token stop) {
	while (!stop.stop_requested()) {
		PendingOperation operation;
		{
			std::unique_lock lock(m_operation_mutex);
			m_operation_available.wait(lock, [this, &stop] {
				return stop.stop_requested() || !m_priority_operations.empty();
			});
			if (stop.stop_requested()) {
				return;
			}
			operation = std::move(m_priority_operations.front());
			m_priority_operations.pop();
			m_priority_active      = true;
			m_priority_active_tick = operation.tick;
		}
		m_master.Wait(operation.tick);
		if (!stop.stop_requested()) {
			RunOperation(std::move(operation.callback));
		}
		{
			std::lock_guard lock(m_operation_mutex);
			m_priority_active      = false;
			m_priority_active_tick = 0;
		}
		m_operation_available.notify_all();
	}
}

void CommandScheduler::DrainPriorityOperations() {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(
	    lock, [this] { return m_priority_operations.empty() && !m_priority_active; });
}

void CommandScheduler::WaitPriorityOperations(uint64_t tick) {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(lock, [this, tick] {
		const bool active_before_or_at = m_priority_active && m_priority_active_tick <= tick;
		const bool queued_before_or_at =
		    !m_priority_operations.empty() && m_priority_operations.front().tick <= tick;
		return !active_before_or_at && !queued_before_or_at;
	});
}

void CommandScheduler::RunOperation(Common::UniqueFunction<void>&& operation) {
	auto* previous                = g_deferred_callback_scheduler;
	g_deferred_callback_scheduler = this;
	operation();
	g_deferred_callback_scheduler = previous;
}

bool CommandScheduler::IsFree(uint64_t tick) {
	if (m_master.IsFree(tick)) {
		return true;
	}
	m_master.Refresh();
	return m_master.IsFree(tick);
}

void CommandScheduler::CheckActive() const {
	EXIT_IF(!Active());
}

CommandBuffer& CommandScheduler::Current() {
	EXIT_IF(m_command.IsInvalid());
	return m_command;
}

CommandBuffer& CommandScheduler::BeginCommand() {
	EXIT_IF(!m_command.IsInvalid());
	m_command.m_buffer = m_command_pool.Commit();
	m_command.Begin();
	m_dynamic_state.Invalidate();
	StartGpuTimer();
	return m_command;
}

// A free query pair, reset in the recording command buffer (outside rendering), or UINT32_MAX.
uint32_t CommandScheduler::AllocateGpuTimer() {
	if (!Common::DebugCounters::Enabled() || m_command.IsInvalid()) {
		return UINT32_MAX;
	}
	if (!m_gpu_timer_pool) {
		vk::QueryPoolCreateInfo info {};
		info.queryType  = vk::QueryType::eTimestamp;
		info.queryCount = GPU_TIMER_QUERIES;
		if (m_graphics.device.createQueryPool(&info, nullptr, &m_gpu_timer_pool) !=
		    vk::Result::eSuccess) {
			m_gpu_timer_pool = nullptr;
			return UINT32_MAX;
		}
	}
	// Pairs are reused in order; skip timing while every pair is in flight.
	if ((m_gpu_timer_pending.size() + 1) * 2 >= GPU_TIMER_QUERIES) {
		return UINT32_MAX;
	}
	const auto query = m_gpu_timer_next;
	m_gpu_timer_next = (m_gpu_timer_next + 2) % GPU_TIMER_QUERIES;
	if (m_graphics.host_query_reset_enabled) {
		m_graphics.device.resetQueryPool(m_gpu_timer_pool, query, 2);
	} else {
		m_command.Handle().resetQueryPool(m_gpu_timer_pool, query, 2);
	}
	return query;
}

uint32_t CommandScheduler::BeginGpuShaderTimer() {
	static const bool enabled = std::getenv("KYTY_GPU_SHADERS") != nullptr;
	return enabled ? BeginGpuMarker() : UINT32_MAX;
}

uint32_t CommandScheduler::BeginGpuMarker() {
	static const bool enabled = std::getenv("KYTY_GPU_MARKERS") != nullptr;
	// Without a host reset the queries would end rendering at every marker.
	if (!enabled || !m_graphics.host_query_reset_enabled) {
		return UINT32_MAX;
	}
	const auto query = AllocateGpuTimer();
	if (query != UINT32_MAX) {
		m_command.Handle().writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe,
		                                  m_gpu_timer_pool, query);
	}
	return query;
}

void CommandScheduler::EndGpuMarker(uint32_t scope, std::string_view name) {
	if (scope == UINT32_MAX || m_command.IsInvalid()) {
		return;
	}
	m_command.Handle().writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, m_gpu_timer_pool,
	                                  scope + 1);
	m_gpu_timer_pending.push_back(
	    {CurrentTick(), scope, Common::DebugCounters::Counter::Count, MarkerId(name)});
}

uint32_t CommandScheduler::MarkerId(std::string_view name) {
	auto [it, inserted] = m_marker_ids.try_emplace(std::string(name),
	                                               static_cast<uint32_t>(m_marker_names.size()));
	if (inserted) {
		m_marker_names.emplace_back(name);
		m_marker_ns.push_back(0);
		m_marker_counts.push_back(0);
	}
	return it->second;
}

void CommandScheduler::SetGpuCheckpoint(std::string_view name) {
	MarkGpuCheckpoint(m_command.Handle(), name);
}

void CommandScheduler::DumpGpuMarkers() {
	static const char* path = std::getenv("KYTY_GPU_MARKERS");
	const auto         now  = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
	if (path == nullptr || m_marker_names.empty() || now - m_marker_dump_ms < 5000) {
		return;
	}
	m_marker_dump_ms = now;
	std::vector<uint32_t> order(m_marker_names.size());
	for (uint32_t i = 0; i < order.size(); i++) {
		order[i] = i;
	}
	std::sort(order.begin(), order.end(),
	          [this](uint32_t a, uint32_t b) { return m_marker_ns[a] > m_marker_ns[b]; });
	std::FILE* file = std::fopen(path, "a");
	if (file == nullptr) {
		return;
	}
	const auto unix_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
	                         std::chrono::system_clock::now().time_since_epoch())
	                         .count();
	std::fprintf(file, "# unix_ms %lld\n", static_cast<long long>(unix_ms));
	for (size_t i = 0; i < order.size() && i < 300; i++) {
		const auto id = order[i];
		std::fprintf(file, "%s\t%llu\t%.3f\n", m_marker_names[id].c_str(),
		             static_cast<unsigned long long>(m_marker_counts[id]),
		             static_cast<double>(m_marker_ns[id]) / 1e6);
	}
	std::fclose(file);
}

void CommandScheduler::StartGpuTimer() {
	m_gpu_timer_query = AllocateGpuTimer();
	if (m_gpu_timer_query != UINT32_MAX) {
		m_command.Handle().writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe,
		                                  m_gpu_timer_pool, m_gpu_timer_query);
	}
}

void CommandScheduler::EndGpuTimer(uint64_t tick) {
	if (m_gpu_timer_query != UINT32_MAX) {
		m_gpu_timer_pending.push_back(
		    {tick, m_gpu_timer_query, Common::DebugCounters::Counter::GpuBusyNs});
		m_gpu_timer_query = UINT32_MAX;
	}
}

uint32_t CommandScheduler::BeginGpuScope() {
	if (!Common::DebugCounters::Enabled() || m_command.IsInvalid()) {
		return UINT32_MAX;
	}
	if (!m_graphics.host_query_reset_enabled) {
		EndRendering();
	}
	const auto query = AllocateGpuTimer();
	if (query != UINT32_MAX) {
		m_command.Handle().writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe,
		                                  m_gpu_timer_pool, query);
	}
	return query;
}

void CommandScheduler::EndGpuScope(uint32_t scope, Common::DebugCounters::Counter counter) {
	if (scope == UINT32_MAX || m_command.IsInvalid()) {
		return;
	}
	m_command.Handle().writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, m_gpu_timer_pool,
	                                  scope + 1);
	m_gpu_timer_pending.push_back({CurrentTick(), scope, counter});
}

// A wait of several seconds is a GPU hang or a deadlock. With KYTY_GPU_SHADERS the timer pairs
// around each draw and dispatch show which one the GPU started and has not finished.
void CommandScheduler::ReportStall(uint64_t tick) {
	// Not Refresh(): it exits on a lost device, which is when this report matters most.
	uint64_t finished = 0;
	(void)m_graphics.device.getSemaphoreCounterValue(m_master.Handle(), &finished);
	std::printf("GPU stall: still waiting for tick %" PRIu64 "; the GPU finished tick %" PRIu64
	            ", recording tick %" PRIu64 "\n",
	            tick, finished, CurrentTick());
	if (m_graphics.checkpoints_enabled) {
		std::vector<vk::CheckpointDataNV> checkpoints;
		{
			Common::LockGuard lock(m_graphics.queue_mutex);
			checkpoints = m_graphics.queue.getCheckpointDataNV();
		}
		auto&           names = GetCheckpointNames();
		std::lock_guard lock(names.mutex);
		for (const auto& checkpoint: checkpoints) {
			const auto id = reinterpret_cast<uintptr_t>(checkpoint.pCheckpointMarker);
			std::printf("GPU stall: checkpoint at %s: %s\n",
			            vk::to_string(checkpoint.stage).c_str(),
			            id != 0 && id <= names.names.size() ? names.names[id - 1].c_str() : "?");
		}
	}
	// The pending timers belong to the GPU thread, which records here.
	if (!m_gpu_timer_pool || !GuestGpu::IsGpuThread()) {
		std::fflush(stdout);
		return;
	}
	const char* last_done = nullptr;
	int         reported  = 0;
	for (const auto& timer: m_gpu_timer_pending) {
		if (m_master.IsFree(timer.tick)) continue;
		std::array<uint64_t, 4> results {}; // Value and availability of the start and end stamps.
		(void)m_graphics.device.getQueryPoolResults(
		    m_gpu_timer_pool, timer.query, 2, sizeof(results), results.data(), 2 * sizeof(uint64_t),
		    vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability);
		const bool  started = results[1] != 0;
		const bool  ended   = results[3] != 0;
		const char* name    = timer.marker != UINT32_MAX ? m_marker_names[timer.marker].c_str()
		                      : timer.counter == Common::DebugCounters::Counter::GpuBusyNs
		                          ? "command buffer"
		                          : "scope";
		if (started && ended) {
			last_done = name;
		} else if (started) {
			std::printf("GPU stall: started, not finished: %s (tick %" PRIu64 ")\n", name,
			            timer.tick);
			if (++reported == 8) break;
		}
	}
	std::printf("GPU stall: last finished timer: %s\n", last_done != nullptr ? last_done : "none");
	std::fflush(stdout);
}

void CommandScheduler::ReadGpuTimers() {
	const double period = m_graphics.physical_device_properties.limits.timestampPeriod;
	while (!m_gpu_timer_pending.empty() && m_master.IsFree(m_gpu_timer_pending.front().tick)) {
		const auto timer = m_gpu_timer_pending.front();
		m_gpu_timer_pending.pop_front();
		std::array<uint64_t, 2> stamps {};
		if (m_graphics.device.getQueryPoolResults(m_gpu_timer_pool, timer.query, 2, sizeof(stamps),
		                                          stamps.data(), sizeof(uint64_t),
		                                          vk::QueryResultFlagBits::e64) ==
		        vk::Result::eSuccess &&
		    stamps[1] >= stamps[0]) {
			const auto ns =
			    static_cast<uint64_t>(static_cast<double>(stamps[1] - stamps[0]) * period);
			if (timer.marker != UINT32_MAX) {
				m_marker_ns[timer.marker] += ns;
				m_marker_counts[timer.marker]++;
				continue;
			}
			if (timer.counter == Common::DebugCounters::Counter::GpuBusyNs) {
				// Command buffers of one queue can overlap at their edges: count the union, and
				// the gaps in which the GPU had no command buffer at all.
				const auto start = static_cast<uint64_t>(static_cast<double>(stamps[0]) * period);
				const auto end   = static_cast<uint64_t>(static_cast<double>(stamps[1]) * period);
				if (m_gpu_busy_end != 0 && start > m_gpu_busy_end) {
					const auto gap = start - m_gpu_busy_end;
					// Gaps longer than a second are pauses (loading, the menu idle), not bubbles.
					if (gap < 1'000'000'000) {
						Common::DebugCounters::Add(Common::DebugCounters::Counter::GpuIdleGapNs, gap);
					}
				}
				const auto from = std::max(start, m_gpu_busy_end);
				if (end > from) {
					Common::DebugCounters::Add(Common::DebugCounters::Counter::GpuBusyUnionNs,
					                           end - from);
				}
				m_gpu_busy_end = std::max(m_gpu_busy_end, end);
				Common::DebugCounters::Add(Common::DebugCounters::Counter::GpuTimedBuffers);
			}
			Common::DebugCounters::Add(timer.counter, ns);
		}
	}
	DumpGpuMarkers();
}

uint64_t CommandScheduler::Submit(SubmitInfo submit) {
	EXIT_IF(m_command.IsInvalid());
	EXIT_IF(submit.num_wait_semaphores > SubmitInfo::MaxSemaphores ||
	        submit.num_signal_semaphores >= SubmitInfo::MaxSemaphores);

	if (m_gpu_timer_query != UINT32_MAX) {
		m_command.Handle().writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe,
		                                  m_gpu_timer_pool, m_gpu_timer_query + 1);
	}
	m_command.End();
	const auto buffer   = m_command.m_buffer;
	auto&      graphics = m_graphics;
	EXIT_IF(graphics.queue == nullptr);

	vk::Result result;
	uint64_t   tick;
	if (submit.num_wait_semaphores == 0 && submit.num_signal_semaphores == 0 &&
	    Common::AbSwitch::Enabled(Common::AbSwitch::QUEUED_SUBMIT)) {
		{
			std::lock_guard lock(m_submit_mutex);
			tick = m_master.NextTick();
			m_submit_queue.push_back(
			    {buffer, tick, m_command.m_debug_op, m_command.m_debug_submit_id});
			if (!m_submit_thread.joinable()) {
				m_submit_thread =
				    std::jthread([this](std::stop_token stop) { SubmitThread(stop); });
			}
		}
		m_submit_changed.notify_all();
		Common::DebugCounters::Add(Common::DebugCounters::Counter::QueueSubmits);
		EndGpuTimer(tick);
		m_recorded_since_submit = 0;
		m_command.m_buffer      = nullptr;
		return tick;
	}
	DrainQueuedSubmits();
	{
		Common::LockGuard lock(graphics.queue_mutex);
		tick = m_master.NextTick();
		submit.AddSignal(m_master.Handle(), tick);

		vk::TimelineSemaphoreSubmitInfo timeline_info {};
		timeline_info.waitSemaphoreValueCount   = submit.num_wait_semaphores;
		timeline_info.pWaitSemaphoreValues      = submit.wait_ticks.data();
		timeline_info.signalSemaphoreValueCount = submit.num_signal_semaphores;
		timeline_info.pSignalSemaphoreValues    = submit.signal_ticks.data();

		vk::SubmitInfo submit_info {};
		submit_info.pNext                = &timeline_info;
		submit_info.waitSemaphoreCount   = submit.num_wait_semaphores;
		submit_info.pWaitSemaphores      = submit.wait_semaphores.data();
		submit_info.pWaitDstStageMask    = submit.wait_stages.data();
		submit_info.commandBufferCount   = 1;
		submit_info.pCommandBuffers      = &buffer;
		submit_info.signalSemaphoreCount = submit.num_signal_semaphores;
		submit_info.pSignalSemaphores    = submit.signal_semaphores.data();

		result = graphics.queue.submit(1, &submit_info, nullptr);
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::QueueSubmits);

	if (result != vk::Result::eSuccess) {
		ReportStall(tick); // A lost device still reports its last checkpoints.
		ReportVulkanFatal("vkQueueSubmit", result, tick, m_command.m_debug_op,
		                  m_command.m_debug_submit_id, m_command.m_debug_arg0,
		                  m_command.m_debug_arg1, m_command.m_debug_arg2, m_command.m_debug_arg3,
		                  m_command.m_debug_arg4);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	EndGpuTimer(tick);
	m_recorded_since_submit = 0;

	m_command.m_buffer = nullptr;
	return tick;
}

void CommandScheduler::SubmitThread(std::stop_token stop) {
	for (;;) {
		QueuedSubmit job;
		{
			std::unique_lock lock(m_submit_mutex);
			m_submit_changed.wait(lock, stop, [this] { return !m_submit_queue.empty(); });
			if (m_submit_queue.empty()) {
				return; // Stop requested.
			}
			job = m_submit_queue.front();
			m_submit_queue.pop_front();
			m_submit_busy = true;
		}
		vk::TimelineSemaphoreSubmitInfo timeline_info {};
		timeline_info.signalSemaphoreValueCount = 1;
		timeline_info.pSignalSemaphoreValues    = &job.tick;
		const auto     semaphore                = m_master.Handle();
		vk::SubmitInfo submit_info {};
		submit_info.pNext                = &timeline_info;
		submit_info.commandBufferCount   = 1;
		submit_info.pCommandBuffers      = &job.buffer;
		submit_info.signalSemaphoreCount = 1;
		submit_info.pSignalSemaphores    = &semaphore;
		vk::Result result;
		{
			Common::LockGuard lock(m_graphics.queue_mutex);
			result = m_graphics.queue.submit(1, &submit_info, nullptr);
		}
		if (result != vk::Result::eSuccess) {
			ReportStall(job.tick); // A lost device still reports its last checkpoints.
			ReportVulkanFatal("vkQueueSubmit", result, job.tick, job.debug_op, job.debug_submit, 0,
			                  0, 0, 0, 0);
		}
		EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
		{
			std::lock_guard lock(m_submit_mutex);
			m_submit_busy = false;
		}
		m_submit_changed.notify_all();
	}
}

void CommandScheduler::DrainQueuedSubmits() {
	std::unique_lock lock(m_submit_mutex);
	m_submit_changed.wait(lock, [this] { return m_submit_queue.empty() && !m_submit_busy; });
}

void CommandScheduler::BeginNext() {
	CheckActive();
	BeginCommand();
}

} // namespace Libs::Graphics
