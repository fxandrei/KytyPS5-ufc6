#include "graphics/host_gpu/renderer/commandScheduler.h"

#include "common/abSwitch.h"
#include "common/assert.h"
#include "common/corePlacement.h"
#include "common/debugCounters.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep
#endif

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

bool GpuBufferWordProbeEnabled() noexcept {
	static const bool enabled = [] {
		const auto* path = std::getenv("KYTY_GPU_BUFFER_WORD_PROBE");
		return path != nullptr && *path != '\0';
	}();
	return enabled;
}

struct CommandScheduler::BufferWordProbe {
	static constexpr uint32_t CAPACITY = 4096;
	struct Sample {
		uint64_t tick = 0;
		uint64_t source_offset = 0;
		std::string label;
	};
	GraphicContext& graphics;
	std::FILE* file = nullptr;
	std::unique_ptr<Buffer> readback;
	vk::QueryPool queries;
	std::array<Sample, CAPACITY> samples;
	uint64_t stride = 4;
	uint32_t next = 0;
	uint32_t reported = 0;

	BufferWordProbe(GraphicContext& graphics_, CommandScheduler& scheduler): graphics(graphics_) {
		const auto* path = std::getenv("KYTY_GPU_BUFFER_WORD_PROBE");
		file = path != nullptr ? std::fopen(path, "w") : nullptr;
		if (file == nullptr) return;
		vk::QueryPoolCreateInfo create{};
		create.queryType = vk::QueryType::eTimestamp;
		create.queryCount = CAPACITY;
		if (graphics.device.createQueryPool(&create, nullptr, &queries) != vk::Result::eSuccess) {
			std::fclose(std::exchange(file, nullptr));
			return;
		}
		stride = std::max<uint64_t>(4, graphics.GetPhysicalDeviceProperties().limits.nonCoherentAtomSize);
		readback = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Download, 0,
		                                     vk::BufferUsageFlagBits::eTransferDst, CAPACITY * stride);
		SetVulkanObjectNameF(graphics.device, readback->Handle(), "GPU Buffer Word Probe");
		if (graphics.host_query_reset_enabled) graphics.device.resetQueryPool(queries, 0, CAPACITY);
		std::fprintf(file, "sample,tick,source_offset,gpu_timestamp,gpu_word_u32,gpu_word_i32,label\n");
		std::fflush(file);
	}

	~BufferWordProbe() {
		if (queries) graphics.device.destroyQueryPool(queries);
		if (file != nullptr) std::fclose(file);
	}

	void Report(bool stalled, uint64_t completed_tick) {
		if (file == nullptr) return;
		for (; reported < next; ++reported) {
			// Without host reset, even querying availability is invalid until the recorded
			// reset has executed. Such devices can report only completed submission ticks.
			if (!graphics.host_query_reset_enabled && samples[reported].tick > completed_tick) break;
			std::array<uint64_t, 2> result{};
			const auto status = graphics.device.getQueryPoolResults(
			    queries, reported, 1, sizeof(result), result.data(), sizeof(result),
			    vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability);
			if (status != vk::Result::eSuccess || result[1] == 0) break;
			const auto offset = reported * stride;
			readback->Invalidate(offset, 4);
			uint32_t word = 0;
			std::memcpy(&word, readback->Mapped().data() + offset, sizeof(word));
			const auto& sample = samples[reported];
			std::fprintf(file, "%u,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%u,%" PRId32 ",%s\n",
			             reported, sample.tick, sample.source_offset, result[0], word,
			             std::bit_cast<int32_t>(word), sample.label.c_str());
		}
		std::fflush(file);
		if (stalled) {
			std::printf("GPU buffer word probe: %u/%u copies available (capacity %u; no slot reuse)\n",
			            reported, next, CAPACITY);
			for (uint32_t i = reported; i < next; ++i) {
				std::printf("GPU buffer word probe: pending sample %u tick %" PRIu64 " %s\n",
				            i, samples[i].tick, samples[i].label.c_str());
			}
		}
	}
};

void CommandScheduler::ProbeBufferWord(vk::Buffer source, uint64_t offset, std::string_view label) {
	if (!GpuBufferWordProbeEnabled() || source == nullptr || (offset & 3u) != 0) return;
	if (!m_buffer_word_probe) m_buffer_word_probe = std::make_unique<BufferWordProbe>(m_graphics, *this);
	auto& probe = *m_buffer_word_probe;
	if (probe.file == nullptr) return;
	probe.Report(false, m_master.KnownGpuTick());
	// Every slot/query is written only once, so an old availability bit cannot publish a new copy.
	if (probe.next == BufferWordProbe::CAPACITY) return;
	EndRendering();
	const auto command = Current().Handle();
	const auto index = probe.next++;
	probe.samples[index] = {CurrentTick(), offset, std::string(label)};
	if (!m_graphics.host_query_reset_enabled) command.resetQueryPool(probe.queries, index, 1);
	vk::BufferMemoryBarrier2 before{};
	before.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands;
	before.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	before.dstStageMask = vk::PipelineStageFlagBits2::eTransfer;
	before.dstAccessMask = vk::AccessFlagBits2::eTransferRead;
	before.srcQueueFamilyIndex = before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer = source;
	before.offset = offset;
	before.size = 4;
	vk::DependencyInfo dependency{};
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers = &before;
	command.pipelineBarrier2(dependency);
	command.copyBuffer(source, probe.readback->Handle(), vk::BufferCopy{offset, index * probe.stride, 4});
	auto after = before;
	after.srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
	after.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
	after.dstStageMask = vk::PipelineStageFlagBits2::eHost;
	after.dstAccessMask = vk::AccessFlagBits2::eHostRead;
	after.buffer = probe.readback->Handle();
	after.offset = index * probe.stride;
	dependency.pBufferMemoryBarriers = &after;
	command.pipelineBarrier2(dependency);
	command.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, probe.queries, index);
}

void CommandScheduler::ReportBufferWordProbes() {
	if (m_buffer_word_probe) m_buffer_word_probe->Report(true, m_master.KnownGpuTick());
}

bool CommandScheduler::ReadbackTimingEnabled() noexcept {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_READBACK_TIMING");
		return value != nullptr && *value == '1';
	}();
	return enabled;
}

int64_t CommandScheduler::NowNs() noexcept {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(
	           std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

void CommandScheduler::NoteReadbackWait(uint64_t write_tick, bool writer_in_open_buffer,
                                        uint64_t in_flight, int64_t start_ns, int64_t flushed_ns,
                                        int64_t done_ns, int64_t end_ns) {
	struct Totals {
		uint64_t n = 0, open = 0, waited = 0, flight = 0;
		int64_t  total = 0, flush = 0, until_submit = 0, gpu = 0, priority = 0, max_total = 0;
	};
	static std::mutex mutex;
	static Totals     sums;
	static int64_t    window_start = 0;
	const int64_t     submit_ns =
	    m_submit_ns[write_tick % m_submit_ns.size()].load(std::memory_order_acquire);
	std::lock_guard lock(mutex);
	if (window_start == 0) window_start = start_ns;
	sums.n++;
	sums.open += writer_in_open_buffer ? 1 : 0;
	sums.flight += in_flight;
	sums.total += end_ns - start_ns;
	sums.max_total = std::max(sums.max_total, end_ns - start_ns);
	sums.flush += flushed_ns - start_ns;
	if (done_ns - flushed_ns > 2000) sums.waited++;
	// The writer's buffer may have been submitted before this readback started (submit_ns < start).
	const int64_t submitted = std::max(submit_ns, start_ns);
	sums.until_submit += std::clamp<int64_t>(submitted - start_ns, 0, done_ns - start_ns);
	sums.gpu += std::max<int64_t>(done_ns - submitted, 0);
	sums.priority += end_ns - done_ns;
	if (end_ns - window_start >= 5'000'000'000) {
		const double n       = static_cast<double>(sums.n);
		const double seconds = static_cast<double>(end_ns - window_start) / 1e9;
		std::printf("readback-timing: %.1fs n/s=%.1f avg_us total=%.1f flush=%.1f until_submit=%.1f "
		            "gpu=%.1f priority=%.1f max_total_us=%.0f writer_open=%.0f%% waited=%.0f%% "
		            "in_flight=%.2f\n",
		            seconds, n / seconds, static_cast<double>(sums.total) / n / 1e3,
		            static_cast<double>(sums.flush) / n / 1e3,
		            static_cast<double>(sums.until_submit) / n / 1e3,
		            static_cast<double>(sums.gpu) / n / 1e3,
		            static_cast<double>(sums.priority) / n / 1e3,
		            static_cast<double>(sums.max_total) / 1e3,
		            100.0 * static_cast<double>(sums.open) / n,
		            100.0 * static_cast<double>(sums.waited) / n,
		            static_cast<double>(sums.flight) / n);
		std::fflush(stdout);
		sums         = {};
		window_start = end_ns;
	}
}

bool CommandScheduler::IndirectNotesEnabled() noexcept {
	static const bool env_enabled = [] {
		const char* value = std::getenv("KYTY_INDIRECT_NOTES");
		return value != nullptr && *value == '1';
	}();
	return env_enabled || GpuCheckpointsEnabled() || HangDiagEnabled();
}

bool CommandScheduler::HangDiagEnabled() noexcept {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_HANG_DIAG");
		return value != nullptr && *value == '1';
	}();
	return enabled;
}

void CommandScheduler::NoteHangWork(char kind, uint64_t hash_a, uint64_t hash_b, uint32_t x,
                                    uint32_t y, uint32_t z, const Buffer* args,
                                    uint64_t args_offset, bool indexed, bool indirect) {
	if (!HangDiagEnabled()) return;
	std::lock_guard lock(m_hang_mutex);
	if (m_hang_work.empty()) m_hang_work.resize(HANG_WORK_NOTES);
	m_hang_work[m_hang_work_next++ % HANG_WORK_NOTES] = {
	    args, args_offset, CurrentTick(), hash_a, hash_b, x, y, z, kind, indexed,
	    indirect || args != nullptr};
}

void CommandScheduler::NoteHangBoundWord(const Buffer* buffer, uint64_t offset, uint64_t guest,
                                         uint64_t shader_hash, uint32_t cpu_word_at_record,
                                         bool gpu_dirty) {
	if (!HangDiagEnabled()) return;
	std::lock_guard lock(m_hang_mutex);
	m_hang_bounds[m_hang_bound_next++ % HANG_BOUND_NOTES] = {
	    buffer, offset, guest, CurrentTick(), shader_hash, cpu_word_at_record, gpu_dirty};
}

// Reads only host-coherent mappings (no Invalidate, no commands, no waits): safe while the GPU
// hangs. Buffers used by unfinished ticks are still alive (see ReportIndirectDraws).
void CommandScheduler::ReportHangNotes(uint64_t finished) {
	if (!HangDiagEnabled()) return;
	std::lock_guard lock(m_hang_mutex);
	const uint64_t stuck = finished + 1;
	const auto read_word = [](const Buffer* buffer, uint64_t offset, uint32_t* out,
	                          uint32_t words) -> const char* {
		if (buffer == nullptr) return "none";
		const auto mapped = buffer->Mapped();
		if (mapped.empty()) return "unmapped";
		if (!buffer->IsCoherent()) return "noncoherent";
		if (offset > mapped.size() || mapped.size() - offset < uint64_t {words} * 4u) {
			return "out-of-range";
		}
		std::memcpy(out, mapped.data() + offset, words * 4u);
		return "mapped";
	};
	// The work of the started, unfinished command buffer, in recording order.
	const uint64_t total = m_hang_work_next;
	const uint64_t first = total > HANG_WORK_NOTES ? total - HANG_WORK_NOTES : 0;
	uint64_t       in_stuck = 0;
	uint64_t       in_flight = 0;
	uint64_t       oldest_flight = 0;
	for (uint64_t i = first; i < total; ++i) {
		const auto& note = m_hang_work[i % HANG_WORK_NOTES];
		if (note.tick <= finished) continue;
		if (in_flight++ == 0) oldest_flight = note.tick;
		if (note.tick == stuck) ++in_stuck;
	}
	std::printf("hang-diag: finished %" PRIu64 ", stuck tick %" PRIu64 ": %" PRIu64
	            " draws/dispatches in it, %" PRIu64 " in unfinished ticks (oldest noted %" PRIu64
	            ", ring %s)\n",
	            finished, stuck, in_stuck, in_flight, oldest_flight,
	            first > 0 && oldest_flight > stuck ? "TOO SMALL" : "ok");
	uint64_t shown = 0;
	for (uint64_t i = first; i < total; ++i) {
		const auto& note = m_hang_work[i % HANG_WORK_NOTES];
		if (note.tick != stuck) continue;
		// The first 48 and the last 48 entries of the stuck command buffer.
		++shown;
		if (shown > 48 && in_stuck - shown >= 48) continue;
		std::array<uint32_t, 5> args {};
		const char*             state = "-";
		if (note.args != nullptr) {
			state = read_word(note.args, note.args_offset, args.data(), note.indexed ? 5u : 4u);
		}
		if (note.kind == 'D') {
			std::printf("hang-diag: #%" PRIu64 " draw vs-h%016" PRIx64 " ps-h%016" PRIx64
			            " rt%ux%u n%u%s args[%s] %u %u %u %u %u\n",
			            shown, note.hash_a, note.hash_b, note.x, note.y, note.z,
			            note.indirect ? " indirect" : "", state, args[0], args[1], args[2],
			            args[3], args[4]);
		} else {
			std::printf("hang-diag: #%" PRIu64 " dispatch cs-h%016" PRIx64 " g%ux%ux%u%s\n",
			            shown, note.hash_a, note.x, note.y, note.z,
			            note.indirect ? " indirect" : "");
		}
	}
	// The filter loop bound (signed compare: a word in [bound, INT32_MAX] loops that many times).
	for (uint64_t i = m_hang_bound_next; i > 0 && i + HANG_BOUND_NOTES > m_hang_bound_next; --i) {
		const auto& note = m_hang_bounds[(i - 1) % HANG_BOUND_NOTES];
		if (note.buffer == nullptr || note.tick <= finished) continue;
		uint32_t   gpu_word = 0;
		const auto state    = read_word(note.buffer, note.offset, &gpu_word, 1);
		uint32_t   cpu_now  = 0;
		const bool cpu_ok   = LibKernel::Memory::TryReadBacking(note.guest, &cpu_now, 4);
		std::printf("hang-diag: filter bound tick %" PRIu64 " ps-h%016" PRIx64 " guest 0x%" PRIx64
		            " gpu[%s] %d cpu@record %d cpu-now[%s] %d gpu-dirty@record %d\n",
		            note.tick, note.shader_hash, note.guest, state,
		            std::bit_cast<int32_t>(gpu_word), std::bit_cast<int32_t>(note.cpu_word),
		            cpu_ok ? "ok" : "unreadable", std::bit_cast<int32_t>(cpu_now),
		            note.gpu_dirty ? 1 : 0);
	}
	std::fflush(stdout);
}

void CommandScheduler::NoteIndirectDraw(const Buffer* args, uint64_t offset, uint32_t count,
                                        uint32_t stride, bool indexed, uint64_t vs_hash,
                                        uint64_t ps_hash) {
	if (!IndirectNotesEnabled() || args == nullptr) return;
	std::lock_guard lock(m_indirect_notes_mutex);
	auto& note = m_indirect_notes[m_indirect_notes_next++ % INDIRECT_NOTES];
	note       = {args, offset, args->CpuAddress() + offset, CurrentTick(), vs_hash, ps_hash,
	              count, stride, indexed};
}

// Buffers that draws of an unfinished tick use are destroyed only after that tick completes, so
// the notes of in-flight ticks still point at live buffers.
void CommandScheduler::ReportIndirectDraws(uint64_t finished) {
	if (!IndirectNotesEnabled()) return;
	std::lock_guard lock(m_indirect_notes_mutex);
	const uint64_t total = m_indirect_notes_next;
	const uint64_t first = total > INDIRECT_NOTES ? total - INDIRECT_NOTES : 0;
	int            shown = 0;
	for (uint64_t i = total; i > first && shown < 24; --i) {
		const auto& note = m_indirect_notes[(i - 1) % INDIRECT_NOTES];
		if (note.args == nullptr || note.tick <= finished) continue;
		++shown;
		const uint32_t words = note.indexed ? 5u : 4u;
		std::array<uint32_t, 5> gpu {};
		std::array<uint32_t, 5> cpu {};
		const auto mapped = note.args->Mapped();
		// Only host-coherent mappings (resizable BAR) are read; Invalidate() is for Download buffers.
		const bool gpu_ok = note.args->IsCoherent() && mapped.size() >= note.offset + words * 4u;
		if (gpu_ok) {
			std::memcpy(gpu.data(), mapped.data() + note.offset, words * 4u);
		}
		const bool cpu_ok = LibKernel::Memory::TryReadBacking(note.guest, cpu.data(), words * 4u);
		std::printf("GPU stall: indirect draw tick %" PRIu64 " vs-h%016" PRIx64 " ps-h%016" PRIx64
		            " guest 0x%" PRIx64 " count %u stride %u %s gpu[%s] %u %u %u %u %u cpu[%s] %u "
		            "%u %u %u %u\n",
		            note.tick, note.vs_hash, note.ps_hash, note.guest, note.count, note.stride,
		            note.indexed ? "indexed" : "plain", gpu_ok ? "mapped" : (mapped.empty() ? "unmapped" : "noncoherent"), gpu[0],
		            gpu[1], gpu[2], gpu[3], gpu[4], cpu_ok ? "ok" : "unreadable", cpu[0], cpu[1],
		            cpu[2], cpu[3], cpu[4]);
	}
	std::printf("GPU stall: %d indirect draws in flight shown (of %" PRIu64 " noted)\n", shown,
	            total);
}

CommandScheduler::CommandPool::CommandPool(GraphicContext& graphics, MasterSemaphore& master)
    : m_graphics(graphics), m_master(master) {
	EXIT_IF(graphics.queue_family == static_cast<uint32_t>(-1));
	vk::CommandPoolCreateInfo create {};
	create.queueFamilyIndex = graphics.queue_family;
	create.flags            = vk::CommandPoolCreateFlagBits::eTransient |
	               vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	const auto result = graphics.device.createCommandPool(&create, nullptr, &m_pool);
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
	ReportBufferWordProbes();
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
		        m_priority_active_tick != 0);
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

void CommandScheduler::Flush() {
	SubmitInfo submit;
	Flush(submit);
}

void CommandScheduler::DeferShaderAccessBarrier() {
	m_deferred_shader_access = true;
}

void CommandScheduler::FlushDeferredShaderAccessBarrier(bool full) {
	if (!m_deferred_shader_access) {
		return;
	}
	m_deferred_shader_access = false;
	if (m_command.IsInvalid()) {
		return;
	}
	EndRendering();
	if (full) {
		GuestFlushBarrier(m_command.Handle());
	} else {
		ShaderAccessBarrier(m_command.Handle(), vk::PipelineStageFlagBits::eComputeShader);
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::GuestFlushBarriers);
}

void CommandScheduler::FlushUrgent() {
	m_direct_submit = Common::AbSwitch::Enabled(Common::AbSwitch::DIRECT_URGENT_SUBMIT);
	Flush();
	m_direct_submit = false;
}

void CommandScheduler::RecordedWork() {
	const uint32_t interval =
	    Common::AbSwitch::Enabled(Common::AbSwitch::EAGER_FLUSH_COARSE) ? 64 : 16;
	const bool early = std::exchange(m_flush_after_record, false);
	if (m_command.IsInvalid()) {
		return;
	}
	++m_recorded_since_submit;
	if (early && m_recorded_since_submit >= 8) {
		Common::DebugCounters::Add(Common::DebugCounters::Counter::EagerFlushes);
		Flush();
		m_eager_submits++;
		return;
	}
	if (!Common::AbSwitch::Enabled(Common::AbSwitch::EAGER_FLUSH) ||
	    m_recorded_since_submit % interval != 0) {
		return;
	}
	if (!Common::AbSwitch::Enabled(Common::AbSwitch::LAZY_GPU_POLL)) {
		m_master.Refresh();
	}
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
	if (!Common::AbSwitch::Enabled(Common::AbSwitch::LAZY_GPU_POLL) || (++m_pop_calls & 63u) == 0) {
		m_master.Refresh();
	}
	ReadGpuTimers();
	for (;;) {
		PendingOperation operation;
		{
			std::lock_guard lock(m_operation_mutex);
			if (m_pending_operations.empty() || !IsFree(m_pending_operations.front().tick)) {
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

void CommandScheduler::DeferPriorityOperation(Common::UniqueFunction<void>&& operation,
                                              uint64_t guest_vaddr, uint64_t guest_size) {
	EXIT_IF(guest_size > UINT64_MAX - guest_vaddr);
	QueueOperation(std::move(operation), true, guest_vaddr, guest_vaddr + guest_size);
}

void CommandScheduler::QueueOperation(Common::UniqueFunction<void>&& operation, bool priority,
                                      uint64_t guest_begin, uint64_t guest_end) {
	CheckActive();
	EXIT_IF(!operation);
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		if (priority) {
			m_priority_operations.push_back(
			    {std::move(operation), CurrentTick(), guest_begin, guest_end});
		} else {
			m_pending_operations.push({std::move(operation), CurrentTick()});
		}
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
	return !m_priority_operations.empty() || m_priority_active_tick != 0;
}

uint64_t CommandScheduler::NewestGuestPublicationTick(uint64_t vaddr, uint64_t size) {
	const uint64_t   end = size > UINT64_MAX - vaddr ? UINT64_MAX : vaddr + size;
	uint64_t         tick = 0;
	std::lock_guard  lock(m_operation_mutex);
	const auto overlaps = [&](uint64_t begin, uint64_t finish) {
		return begin < finish && begin < end && vaddr < finish;
	};
	if (m_priority_active_tick != 0 && overlaps(m_priority_active_begin, m_priority_active_end)) {
		tick = m_priority_active_tick;
	}
	// Queue order is tick order, so the last overlapping entry is the newest; the queue holds a
	// few entries (completions of the last frames).
	for (const auto& operation: m_priority_operations) {
		if (overlaps(operation.guest_begin, operation.guest_end)) {
			tick = std::max(tick, operation.tick);
		}
	}
	return tick;
}

uint64_t CommandScheduler::PendingGuestPublications() {
	std::lock_guard lock(m_operation_mutex);
	uint64_t        count = m_priority_active_tick != 0 &&
	                         m_priority_active_begin < m_priority_active_end
	                            ? 1
	                            : 0;
	for (const auto& operation: m_priority_operations) {
		count += operation.guest_begin < operation.guest_end ? 1 : 0;
	}
	return count;
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
			m_priority_operations.pop_front();
			m_priority_active_tick  = operation.tick;
			m_priority_active_begin = operation.guest_begin;
			m_priority_active_end   = operation.guest_end;
		}
		m_master.Wait(operation.tick);
		if (!stop.stop_requested()) {
			RunOperation(std::move(operation.callback));
		}
		{
			std::lock_guard lock(m_operation_mutex);
			m_priority_active_tick  = 0;
			m_priority_active_begin = 0;
			m_priority_active_end   = 0;
		}
		m_operation_available.notify_all();
	}
}

void CommandScheduler::DrainPriorityOperations() {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(
	    lock, [this] { return m_priority_operations.empty() && m_priority_active_tick == 0; });
}

void CommandScheduler::WaitPriorityOperations(uint64_t tick) {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(lock, [this, tick] {
		const bool active_before_or_at =
		    m_priority_active_tick != 0 && m_priority_active_tick <= tick;
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
	// Only the GPU thread routes: BufferCache's constructor (BeginCommand + Fill on the main
	// thread) and the shutdown Finish stay direct. With RECORD_THREAD off the GPU thread stays
	// the stream's owner with a null buffer, so its direct calls drain what the stream holds.
	if (m_stream != nullptr && GuestGpu::IsGpuThread()) {
		const bool routed = Common::AbSwitch::Enabled(Common::AbSwitch::RECORD_THREAD);
		if (routed) {
			EXIT_IF(!RecordingHooksInstalled());
		}
		m_stream->Route(routed ? m_command.m_buffer : VK_NULL_HANDLE);
		MarkCommandHookThread();
	}
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
	// KYTY_GPU_MARKERS (diagnostic): a guest marker keeps its pair from PUSH to POP, possibly for
	// seconds, while the ring wraps. Handing that pair to a new timer let the marker's late end
	// stamp overwrite the new timer's end (inflating it) and vice versa. Skip pairs still owned.
	static const bool track_pairs = std::getenv("KYTY_GPU_MARKERS") != nullptr;
	if (track_pairs) {
		if (m_gpu_pair_busy.empty()) {
			m_gpu_pair_busy.assign(GPU_TIMER_QUERIES / 2, 0);
		}
		for (uint32_t tries = 0; m_gpu_pair_busy[m_gpu_timer_next / 2] != 0; tries++) {
			m_gpu_timer_next = (m_gpu_timer_next + 2) % GPU_TIMER_QUERIES;
			if (tries == 64) {
				return UINT32_MAX;
			}
		}
	}
	const auto query = m_gpu_timer_next;
	m_gpu_timer_next = (m_gpu_timer_next + 2) % GPU_TIMER_QUERIES;
	if (track_pairs) {
		m_gpu_pair_busy[query / 2] = 1;
	}
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

void CommandScheduler::ReleaseGpuTimerPair(uint32_t query) {
	if (query != UINT32_MAX && !m_gpu_pair_busy.empty()) {
		m_gpu_pair_busy[(query % GPU_TIMER_QUERIES) / 2] = 0;
	}
}

void CommandScheduler::EndGpuMarker(uint32_t scope, std::string_view name) {
	if (scope == UINT32_MAX) {
		return;
	}
	if (m_command.IsInvalid()) {
		ReleaseGpuTimerPair(scope);
		return;
	}
	m_command.Handle().writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, m_gpu_timer_pool,
	                                  scope + 1);
	m_gpu_timer_pending.push_back(
	    {CurrentTick(), scope, Common::DebugCounters::Counter::Count, MarkerId(name)});
}

uint32_t CommandScheduler::MarkerId(std::string_view name) {
	auto [it, inserted] =
	    m_marker_ids.try_emplace(std::string(name), static_cast<uint32_t>(m_marker_names.size()));
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
	// Runs after every draw: read the clock only when markers are being collected.
	if (path == nullptr || m_marker_names.empty()) {
		return;
	}
	const auto now = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                                           std::chrono::steady_clock::now().time_since_epoch())
	                                           .count());
	if (now - m_marker_dump_ms < 5000) {
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
	// 1500, not 300: names that drop out of a truncated list make per-window deltas unusable.
	for (size_t i = 0; i < order.size() && i < 1500; i++) {
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
		m_command.Handle().writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, m_gpu_timer_pool,
		                                  m_gpu_timer_query);
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
	if (scope == UINT32_MAX) {
		return;
	}
	if (m_command.IsInvalid()) {
		ReleaseGpuTimerPair(scope);
		return;
	}
	m_command.Handle().writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, m_gpu_timer_pool,
	                                  scope + 1);
	m_gpu_timer_pending.push_back({CurrentTick(), scope, counter});
}

// A wait of several seconds is a GPU hang or a deadlock. With KYTY_GPU_SHADERS the timer pairs
// around each draw and dispatch show which one the GPU started and has not finished.
void CommandScheduler::RecordRecentSubmit(uint64_t tick, uint32_t op, uint64_t submit_id) {
	std::lock_guard lock(m_recent_submit_mutex);
	m_recent_submits.push_back({tick, op, submit_id});
	while (m_recent_submits.size() > 64) {
		m_recent_submits.pop_front();
	}
}

std::optional<CommandScheduler::RecentSubmit>
CommandScheduler::FindRecentSubmit(uint64_t tick) const {
	std::lock_guard lock(m_recent_submit_mutex);
	for (auto it = m_recent_submits.rbegin(); it != m_recent_submits.rend(); ++it) {
		if (it->tick == tick) {
			return *it;
		}
	}
	return std::nullopt;
}

void CommandScheduler::ReportStall(uint64_t tick) {
	// Not Refresh(): it exits on a lost device, which is when this report matters most.
	uint64_t finished = 0;
	(void)m_graphics.device.getSemaphoreCounterValue(m_master.Handle(), &finished);
	std::printf("GPU stall: still waiting for tick %" PRIu64 "; the GPU finished tick %" PRIu64
	            ", recording tick %" PRIu64 "\n",
	            tick, finished, CurrentTick());
	if (m_stream != nullptr) {
		std::printf("GPU stall: record stream: %" PRIu64 " of %" PRIu64 " bytes recorded, "
		            "consumer %s\n",
		            m_stream->ReadBytes(), m_stream->WrittenBytes(),
		            m_stream->ConsumerSleeping() ? "sleeping" : "running");
	}
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
	ReportIndirectDraws(finished);
	ReportHangNotes(finished);
	if (GuestGpu::IsGpuThread()) ReportBufferWordProbes();
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
			std::printf("GPU stall: started, not finished: %s (tick %" PRIu64 ")", name,
			            timer.tick);
			if (timer.counter == Common::DebugCounters::Counter::GpuBusyNs) {
				if (const auto recent = FindRecentSubmit(timer.tick); recent.has_value()) {
					std::printf(" op=%u submit=%" PRIu64, recent->op, recent->submit_id);
				}
			}
			std::printf("\n");
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
		ReleaseGpuTimerPair(timer.query);
		std::array<uint64_t, 2> stamps {};
		if (m_graphics.device.getQueryPoolResults(
		        m_gpu_timer_pool, timer.query, 2, sizeof(stamps), stamps.data(), sizeof(uint64_t),
		        vk::QueryResultFlagBits::e64) == vk::Result::eSuccess &&
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
						Common::DebugCounters::Add(Common::DebugCounters::Counter::GpuIdleGapNs,
						                           gap);
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

	// Diagnostic only: with unmapped guest buffers, bit 11 otherwise has no readback effect.
	// Alternate the blanket host visibility operation without exposing direct BAR reads.
	if (m_host_read_barrier ||
	    (m_host_read_barrier_ab && Common::AbSwitch::Enabled(Common::AbSwitch::DIRECT_READBACK))) {
		m_command.EndRendering();
		vk::MemoryBarrier barrier {};
		barrier.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
		barrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
		m_command.Handle().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                                   vk::PipelineStageFlagBits::eHost, {}, 1, &barrier, 0,
		                                   nullptr, 0, nullptr);
	}
	// A chained dispatch's access barrier belongs before the end of this command buffer.
	FlushDeferredShaderAccessBarrier(!Common::AbSwitch::Enabled(Common::AbSwitch::BARRIER_LITE));
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
	// With the record stream, every submit without semaphores goes through it: its packets are
	// submitted in tick order by the submit thread. An urgent flush (DIRECT_URGENT_SUBMIT) also
	// pushes and wakes: the consumer is spinning. RECORD_URGENT_DIRECT selects the drain plus
	// direct submit below, for A/B.
	const bool stream_submit =
	    m_stream != nullptr && submit.num_wait_semaphores == 0 &&
	    submit.num_signal_semaphores == 0 &&
	    (!m_direct_submit || !Common::AbSwitch::Enabled(Common::AbSwitch::RECORD_URGENT_DIRECT));
	if (stream_submit) {
		// The tick is allocated at push time, so ticks follow the push order.
		tick = m_master.NextTick();
		const QueuedSubmit job {buffer, tick, m_command.m_debug_op, m_command.m_debug_submit_id};
		m_stream->Route(VK_NULL_HANDLE);
		m_stream->Push([this, job] { SubmitQueuedJob(job); });
		m_stream->Wake();
		Common::DebugCounters::Add(Common::DebugCounters::Counter::QueueSubmits);
		EndGpuTimer(tick);
		m_recorded_since_submit = 0;
		m_command.m_buffer      = nullptr;
		return tick;
	}
	if (submit.num_wait_semaphores == 0 && submit.num_signal_semaphores == 0 && !m_direct_submit &&
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
		RecordRecentSubmit(tick, m_command.m_debug_op, m_command.m_debug_submit_id);
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

		if (ReadbackTimingEnabled()) [[unlikely]] {
			m_submit_ns[tick % m_submit_ns.size()].store(NowNs(), std::memory_order_release);
		}
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

void CommandScheduler::EnableRecordThread() {
	// The hooks replace entries of the global dispatcher; install them once the device
	// dispatcher is initialized (RenderContext construction, after vulkanWindow.cpp's init).
	InstallCommandHooks();
	if (CommandRecordingDeferred()) {
		EXIT_IF(m_stream != nullptr);
		m_stream = std::make_unique<CommandStream>();
		// The submit thread records the stream's commands and submits in order; it starts now,
		// not lazily like the deque submitter. KYTY_VERIFY_RECORD=2 replays each packet inline on
		// the producer, so no consumer runs.
		if (!VerifyRecordInline()) {
			EXIT_IF(m_submit_thread.joinable());
			m_submit_thread = std::jthread([this](std::stop_token stop) { SubmitThread(stop); });
		}
	}
}

void CommandScheduler::DrainGpuCommands() {
	if (m_stream != nullptr) {
		m_stream->Drain();
	}
}

void CommandScheduler::SubmitThread(std::stop_token stop) {
	KYTY_PROFILER_THREAD("GpuQueueSubmit");
	if (m_stream != nullptr) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		m_record_thread_id.store(GetCurrentThreadId(), std::memory_order_relaxed);
#endif
		if (Common::AbSwitch::Enabled(Common::AbSwitch::GPU_THREAD_PRIORITY) &&
		    !Common::AbSwitch::Enabled(Common::AbSwitch::RESERVE_CORE)) {
			Common::CorePlacement::SetCurrentThreadAboveNormal(true);
		}
		m_stream->Consume(stop);
		m_record_thread_id.store(0, std::memory_order_relaxed);
		return;
	}
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
		SubmitQueuedJob(job);
		{
			std::lock_guard lock(m_submit_mutex);
			m_submit_busy = false;
		}
		m_submit_changed.notify_all();
	}
}

void CommandScheduler::SubmitQueuedJob(const QueuedSubmit& job) {
	RecordRecentSubmit(job.tick, job.debug_op, job.debug_submit);
	// KYTY_VERIFY_RECORD: the consumer submits in packet order, so its ticks increase. (The
	// direct branch allocates its ticks after draining, so a strict +1 does not always hold.)
	static const bool verify_ticks = [] {
		const char* value = std::getenv("KYTY_VERIFY_RECORD");
		return value != nullptr && (std::strcmp(value, "1") == 0 || std::strcmp(value, "2") == 0);
	}();
	if (verify_ticks) {
		if (m_last_verify_tick != UINT64_MAX) {
			EXIT_IF(job.tick <= m_last_verify_tick);
		}
		m_last_verify_tick = job.tick;
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
		if (ReadbackTimingEnabled()) [[unlikely]] {
			m_submit_ns[job.tick % m_submit_ns.size()].store(NowNs(), std::memory_order_release);
		}
		result = m_graphics.queue.submit(1, &submit_info, nullptr);
	}
	if (result != vk::Result::eSuccess) {
		ReportStall(job.tick); // A lost device still reports its last checkpoints.
		ReportVulkanFatal("vkQueueSubmit", result, job.tick, job.debug_op, job.debug_submit, 0, 0,
		                  0, 0, 0);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandScheduler::DrainQueuedSubmits() {
	// The stream's consumer needs the queue lock to submit, so never drain it under that lock.
	if (m_stream != nullptr) {
		m_stream->Drain();
	}
	std::unique_lock lock(m_submit_mutex);
	m_submit_changed.wait(lock, [this] { return m_submit_queue.empty() && !m_submit_busy; });
}

void CommandScheduler::BeginNext() {
	CheckActive();
	BeginCommand();
}

} // namespace Libs::Graphics
