#include "graphics/presentation/performanceOverlay.h"

#include "common/abSwitch.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/hostStats.h"
#include "graphics/host_gpu/graphicContext.h"
#include "imgui.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <stop_token>
#include <thread>

namespace Libs::Graphics {

namespace {

using Clock   = std::chrono::steady_clock;
using Counter = Common::DebugCounters::Counter;
using Peak    = Common::DebugCounters::Peak;

constexpr auto   REFRESH_INTERVAL  = std::chrono::seconds(1);
constexpr size_t FRAME_HISTORY     = 120;
constexpr float  FRAME_GRAPH_MS    = 50.0f;
constexpr float  FRAME_TIME_GOOD   = 17.5f;
constexpr float  FRAME_TIME_MEDIUM = 34.0f;
constexpr size_t COUNTER_COUNT     = static_cast<size_t>(Counter::Count);

// Tables as wide as their columns; the default fills the window, which an auto-resizing
// window then never shrinks.
constexpr ImGuiTableFlags TABLE_FLAGS =
    ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_NoHostExtendX;

constexpr ImU32 GOOD    = IM_COL32(96, 220, 120, 255);
constexpr ImU32 MEDIUM  = IM_COL32(230, 195, 60, 255);
constexpr ImU32 BAD     = IM_COL32(235, 80, 70, 255);
constexpr ImU32 NEUTRAL = IM_COL32(160, 160, 160, 255);

struct FrameWindow {
	std::array<float, FRAME_HISTORY> intervals_ms {};
	size_t                           next         = 0;
	size_t                           count        = 0;
	uint32_t                         flips        = 0;
	float                            max_ms       = 0.0f;
	uint32_t                         guest_width  = 0;
	uint32_t                         guest_height = 0;
	Clock::time_point                last_flip    = {};
	Clock::time_point                window_start = {};
};

// Shown values, recomputed once per refresh interval.
struct Snapshot {
	double                              fps          = 0.0;
	double                              frame_ms     = 0.0;
	double                              max_frame_ms = 0.0;
	double                              low_fps      = 0.0; // From the 99th percentile.
	double                              stalled_s    = 0.0;
	uint32_t                            guest_width  = 0;
	uint32_t                            guest_height = 0;
	std::array<double, COUNTER_COUNT>   rates {};
	std::array<uint64_t, COUNTER_COUNT> totals {};
	uint64_t                            shader_compile_peak_ns  = 0;
	uint64_t                            pipeline_create_peak_ns = 0;
	Common::HostStats                   host;
	std::optional<uint64_t>             device_memory;
	uint64_t                            device_budget = 0;
	std::array<GraphicContext::HeapUsage, VK_MAX_MEMORY_HEAPS> heaps {};
	uint32_t                                                   heap_count = 0;
	std::array<float, FRAME_HISTORY>                           history {};
	size_t                                                     history_count = 0;
	Clock::time_point                                          sample_time {};
	LibKernel::Memory::DebugStats                              guest_memory;
};

std::mutex            g_frame_mutex;
FrameWindow           g_frames;
std::atomic<uint64_t> g_flip_total {0}; // Never reset, unlike the panel's window.
std::atomic<uint64_t> g_generation {0};
std::atomic<bool>     g_details {false};
std::atomic<uint64_t> g_revision {0};

std::mutex                  g_host_mutex;
Common::HostStats           g_host_stats;
uint64_t                    g_host_generation = 0;
std::condition_variable_any g_sampler_wait;
std::jthread                g_sampler;

void SetOverlayEnabled(bool enabled) {
	{
		std::scoped_lock lock(g_host_mutex);
		if (enabled) {
			std::scoped_lock frame_lock(g_frame_mutex);
			g_frames = {};
			g_generation.fetch_add(1, std::memory_order_release);
		}
		Common::DebugCounters::SetEnabled(enabled);
	}
	g_sampler_wait.notify_all();
}

bool GuestStalled(Clock::time_point now) {
	std::scoped_lock lock(g_frame_mutex);
	return now - g_frames.last_flip >= REFRESH_INTERVAL;
}

// Host statistics can block for milliseconds (IOKit, thread enumeration), so sample them off
// the present path. While the guest stops flipping (loading screens), ask for one repaint per
// interval.
void SamplerThread(const std::stop_token& stop) {
	std::optional<Common::HostStatsSampler> sampler;
	uint64_t                                sampled_generation = 0;
	std::unique_lock                        lock(g_host_mutex);
	while (!stop.stop_requested()) {
		g_sampler_wait.wait(lock, stop, [] { return Common::DebugCounters::Enabled(); });
		if (stop.stop_requested()) {
			break;
		}
		const auto generation = g_generation.load(std::memory_order_acquire);
		lock.unlock();
		if (!sampler || generation != sampled_generation) {
			sampler.emplace();
			sampled_generation = generation;
		}
		auto stats = sampler->Sample();
		lock.lock();
		g_host_stats      = std::move(stats);
		g_host_generation = generation;
		if (Common::DebugCounters::Enabled() && GuestStalled(Clock::now())) {
			g_revision.fetch_add(1, std::memory_order_acq_rel);
		}
		g_sampler_wait.wait_for(lock, stop, REFRESH_INTERVAL, [generation] {
			return !Common::DebugCounters::Enabled() ||
			       generation != g_generation.load(std::memory_order_acquire);
		});
	}
}

void EnsureSampler() {
	static std::once_flag once;
	std::call_once(once, [] { g_sampler = std::jthread(SamplerThread); });
}

// KYTY_PERF_CSV=<path> writes one row per second for unattended A/B runs: guest flips, frame
// time percentiles, the GPU counters and process CPU. The counters it reads also drive the
// panel, so it turns the panel on.
std::jthread g_perf_csv;

// The largest wide depth target since the last CSV row, as width << 16 | height.
std::atomic<uint32_t> g_depth_target_max {0};

void PerfCsvThread(const std::stop_token& stop, std::FILE* file) {
	std::fputs(
	    "t_s,unix_s,flips,fps,frame_p50_ms,frame_p99_ms,frame_max_ms,draws,dispatches,"
	    "indirect_draws,submits,gpu_waits,gpu_wait_ms,gpu_faults,tex_up_mb,tex_down_mb,"
	    "buf_up_mb,buf_down_mb,shaders,shader_ms,pipelines,pipeline_ms,cmask_readback_ms,"
	    "late_vblanks,cpu_pct,gpu_thread_waits,gpu_thread_wait_ms,unmap_waits,unmap_wait_ms,"
	    "unmap_waits_unregistered,flip_waits,shreg_reads,shreg_clean,shreg_dirty,dargs_reads,"
	    "dargs_clean,dargs_dirty,cp_backing_reads,srt_faults,srt_backing,label_writes,"
	    "label_clean,label_dirty,label_in_order,written_bindings,written_mb,"
	    "dirty_reads_unchanged,dirty_reads_changed,tex_refresh_gpu_mb,tex_refresh_cpu_mb,"
	    "gpu_busy_ms,gpu_timed_buffers,tex_upload_gpu_ms,partial_image_skipped_mb,"
	    "gpu_busy_union_ms,gpu_idle_gap_ms,guest_flushes,eager_flushes,ab,depth_w,depth_h,"
	    "thread1,"
	    "thread1_pct,thread2,thread2_pct,"
	    "thread3,thread3_pct,record_packets,record_mb,record_drains,record_direct,record_sleeps,"
	    "draw_chain_taken,draw_chain_dyn_kept,draw_chain_pipe_kept\n",
	    file);
	const char*              ab_file = std::getenv("KYTY_AB_FILE");
	uint64_t                 ab      = Common::AbSwitch::Mask();
	Common::HostStatsSampler sampler;
	(void)sampler.Sample(); // CPU usage needs a previous sample.
	std::array<uint64_t, COUNTER_COUNT> last {};
	for (size_t i = 0; i < COUNTER_COUNT; i++) {
		last[i] = Common::DebugCounters::Get(static_cast<Counter>(i));
	}
	uint64_t                    last_flips = g_flip_total.load(std::memory_order_relaxed);
	const auto                  start      = Clock::now();
	auto                        last_time  = start;
	std::mutex                  mutex;
	std::condition_variable_any wait;
	std::unique_lock            lock(mutex);
	for (;;) {
		wait.wait_for(lock, stop, REFRESH_INTERVAL, [] { return false; });
		if (stop.stop_requested()) {
			break;
		}
		const auto   now     = Clock::now();
		const double seconds = std::chrono::duration<double>(now - last_time).count();
		last_time            = now;
		const auto flips     = g_flip_total.load(std::memory_order_relaxed);
		const auto new_flips = flips - last_flips;
		last_flips           = flips;

		std::array<float, FRAME_HISTORY> intervals {};
		size_t                           count = 0;
		{
			std::scoped_lock frame_lock(g_frame_mutex);
			count = std::min<size_t>(new_flips, g_frames.count);
			for (size_t i = 0; i < count; i++) {
				intervals[i] =
				    g_frames.intervals_ms[(g_frames.next + FRAME_HISTORY - count + i) % FRAME_HISTORY];
			}
		}
		std::sort(intervals.begin(), intervals.begin() + static_cast<ptrdiff_t>(count));
		const auto percentile = [&](size_t percent) {
			return count != 0 ? intervals[std::min(count - 1, count * percent / 100)] : 0.0f;
		};

		std::array<double, COUNTER_COUNT> rates {};
		for (size_t i = 0; i < COUNTER_COUNT; i++) {
			const auto value = Common::DebugCounters::Get(static_cast<Counter>(i));
			rates[i]         = static_cast<double>(value - last[i]) / seconds;
			last[i]          = value;
		}
		const auto rate = [&](Counter counter) { return rates[static_cast<size_t>(counter)]; };
		constexpr double MIB  = 1024.0 * 1024.0;
		const auto       host = sampler.Sample();
		std::fprintf(
		    file,
		    "%.1f,%.3f,%llu,%.2f,%.1f,%.1f,%.1f,%.0f,%.0f,%.0f,%.0f,%.0f,%.1f,%.0f,%.1f,%.1f,%.1f,"
		    "%.1f,%.0f,%.1f,%.0f,%.1f,%.1f,%.0f,%.1f,%.0f,%.1f,%.0f,%.1f,%.0f,%.0f,%.0f,%.0f,%.0f,"
		    "%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.1f,%.0f,%.0f,%.1f,%.1f,%.1f,"
		    "%.0f,%.1f,%.1f,%.1f,%.1f,%.0f,%.0f,%llu",
		    std::chrono::duration<double>(now - start).count(),
		    std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch())
		        .count(),
		    static_cast<unsigned long long>(new_flips), static_cast<double>(new_flips) / seconds,
		    percentile(50), percentile(99), count != 0 ? intervals[count - 1] : 0.0f,
		    rate(Counter::Draws), rate(Counter::Dispatches), rate(Counter::IndirectDraws),
		    rate(Counter::QueueSubmits), rate(Counter::GpuWaits), rate(Counter::GpuWaitNs) / 1e6,
		    rate(Counter::GpuFaults), rate(Counter::TextureUploadBytes) / MIB,
		    rate(Counter::TextureDownloadBytes) / MIB, rate(Counter::BufferUploadBytes) / MIB,
		    rate(Counter::BufferDownloadBytes) / MIB, rate(Counter::ShadersCompiled),
		    rate(Counter::ShaderCompileNs) / 1e6, rate(Counter::PipelinesCreated),
		    rate(Counter::PipelineCreateNs) / 1e6, rate(Counter::ColorMetadataReadbackNs) / 1e6,
		    rate(Counter::LateVblanks), host.process_cpu_percent.value_or(-1.0),
		    rate(Counter::GpuThreadWaits), rate(Counter::GpuThreadWaitNs) / 1e6,
		    rate(Counter::UnmapWaits), rate(Counter::UnmapWaitNs) / 1e6,
		    rate(Counter::UnmapWaitsUnregistered), rate(Counter::FlipWaits),
		    rate(Counter::ShRegTableReads), rate(Counter::ShRegTableReadsClean),
		    rate(Counter::ShRegTableReadsDirty), rate(Counter::DispatchArgReads),
		    rate(Counter::DispatchArgReadsClean), rate(Counter::DispatchArgReadsDirty),
		    rate(Counter::CpReadsFromBacking), rate(Counter::SrtWalkerFaults),
		    rate(Counter::SrtReadsFromBacking), rate(Counter::LabelWrites),
		    rate(Counter::LabelWritesClean), rate(Counter::LabelWritesDirty),
		    rate(Counter::LabelWritesInOrder), rate(Counter::WrittenBindings),
		    rate(Counter::WrittenBindingBytes) / MIB, rate(Counter::DirtyReadsUnchanged),
		    rate(Counter::DirtyReadsChanged), rate(Counter::TextureRefreshesGpu) / MIB,
		    rate(Counter::TextureRefreshesCpu) / MIB, rate(Counter::GpuBusyNs) / 1e6,
		    rate(Counter::GpuTimedBuffers), rate(Counter::TextureUploadGpuNs) / 1e6,
		    rate(Counter::PartialImageWritesSkipped) / MIB, rate(Counter::GpuBusyUnionNs) / 1e6,
		    rate(Counter::GpuIdleGapNs) / 1e6, rate(Counter::GuestFlushBarriers),
		    rate(Counter::EagerFlushes), static_cast<unsigned long long>(ab));
		const auto depth = g_depth_target_max.exchange(0, std::memory_order_relaxed);
		std::fprintf(file, ",%u,%u", depth >> 16u, depth & 0xffffu);
		// The three busiest threads, in percent of one core.
		for (size_t i = 0; i < 3; i++) {
			if (i < host.busiest_threads.size()) {
				auto name = host.busiest_threads[i].name;
				std::replace(name.begin(), name.end(), ',', ';');
				std::fprintf(file, ",%s,%.1f", name.c_str(), host.busiest_threads[i].cpu_percent);
			} else {
				std::fputs(",,", file);
			}
		}
		// The record stream (AbSwitch RECORD_THREAD): packets and bytes appended, drains,
		// calls that ran on the caller instead of the stream (a missing hook), consumer sleeps.
		std::fprintf(file, ",%.0f,%.1f,%.0f,%.0f,%.0f", rate(Counter::RecordStreamPackets),
		             rate(Counter::RecordStreamBytes) / 1e6, rate(Counter::RecordStreamDrains),
		             rate(Counter::RecordStreamDirectCalls), rate(Counter::RecordStreamSleeps));
		std::fprintf(file, ",%.0f,%.0f,%.0f", rate(Counter::DrawChainTaken),
		             rate(Counter::DrawChainDynKept), rate(Counter::DrawChainPipeKept));
		std::fputc('\n', file);
		std::fflush(file);
		// The variant applies from the next row on, so every row measures a single variant.
		if (ab_file != nullptr) {
			if (std::FILE* control = std::fopen(ab_file, "r"); control != nullptr) {
				unsigned long long mask = 0;
				if (std::fscanf(control, "%llu", &mask) == 1) {
					Common::AbSwitch::g_mask.store(mask & ~Common::AbSwitch::UNSET,
					                               std::memory_order_relaxed);
				}
				std::fclose(control);
			}
		}
		ab = Common::AbSwitch::Mask();
	}
	std::fclose(file);
}

void EnsurePerfCsv() {
	static std::once_flag once;
	std::call_once(once, [] {
		const char* path = std::getenv("KYTY_PERF_CSV");
		if (path == nullptr || *path == '\0') {
			return;
		}
		std::FILE* file = std::fopen(path, "w");
		if (file == nullptr) {
			std::printf("KYTY_PERF_CSV: cannot open %s\n", path);
			return;
		}
		if (!Common::DebugCounters::Enabled()) {
			SetOverlayEnabled(true);
			g_revision.fetch_add(1, std::memory_order_acq_rel);
		}
		g_perf_csv = std::jthread(PerfCsvThread, file);
	});
}

ImU32 FrameTimeColor(double ms) {
	return ms <= FRAME_TIME_GOOD ? GOOD : ms <= FRAME_TIME_MEDIUM ? MEDIUM : BAD;
}

// Green below `warn`, yellow below `bad`, red above.
ImU32 LevelColor(double value, double warn, double bad) {
	return value < warn ? GOOD : value < bad ? MEDIUM : BAD;
}

ImVec4 Color(ImU32 color) {
	return ImGui::ColorConvertU32ToFloat4(color);
}

std::string Bytes(uint64_t bytes) {
	constexpr double GIB = 1024.0 * 1024.0 * 1024.0;
	constexpr double MIB = 1024.0 * 1024.0;
	char             buffer[32];
	if (static_cast<double>(bytes) >= GIB) {
		std::snprintf(buffer, sizeof(buffer), "%.2f GB", static_cast<double>(bytes) / GIB);
	} else {
		std::snprintf(buffer, sizeof(buffer), "%.0f MB", static_cast<double>(bytes) / MIB);
	}
	return buffer;
}

std::string Bytes(std::optional<uint64_t> bytes) {
	return bytes ? Bytes(*bytes) : "n/a";
}

std::string Throughput(double bytes_per_second) {
	char buffer[32];
	std::snprintf(buffer, sizeof(buffer), "%.1f MB/s", bytes_per_second / (1024.0 * 1024.0));
	return buffer;
}

std::string Percent(std::optional<double> value) {
	if (!value) {
		return "n/a";
	}
	char buffer[16];
	std::snprintf(buffer, sizeof(buffer), "%.0f%%", *value);
	return buffer;
}

void Row(const char* label, const std::string& value, std::optional<ImU32> color = std::nullopt) {
	ImGui::TableNextRow();
	ImGui::TableNextColumn();
	ImGui::TextDisabled("%s", label);
	ImGui::TableNextColumn();
	if (color) {
		ImGui::TextColored(Color(*color), "%s", value.c_str());
	} else {
		ImGui::TextUnformatted(value.c_str());
	}
}

[[gnu::format(printf, 1, 2)]] std::string Fmt(const char* format, ...) {
	char    buffer[128];
	va_list args;
	va_start(args, format);
	std::vsnprintf(buffer, sizeof(buffer), format, args);
	va_end(args);
	return buffer;
}

bool BeginSection(const char* title) {
	ImGui::SeparatorText(title);
	return ImGui::BeginTable(title, 2, TABLE_FLAGS);
}

void DrawFrameGraph(const std::array<float, FRAME_HISTORY>& history, size_t count, float width,
                    float height) {
	const ImVec2 origin = ImGui::GetCursorScreenPos();
	auto*        draw   = ImGui::GetWindowDrawList();
	draw->AddRectFilled(origin, {origin.x + width, origin.y + height}, IM_COL32(0, 0, 0, 90));
	const auto y_of = [&](float ms) {
		return origin.y + height - std::min(ms, FRAME_GRAPH_MS) / FRAME_GRAPH_MS * height;
	};
	for (const float reference: {1000.0f / 60.0f, 1000.0f / 30.0f}) {
		draw->AddLine({origin.x, y_of(reference)}, {origin.x + width, y_of(reference)},
		              IM_COL32(255, 255, 255, 45));
	}
	const float bar = width / static_cast<float>(FRAME_HISTORY);
	for (size_t i = 0; i < count; i++) {
		const float ms = history[i];
		const float x  = origin.x + width - static_cast<float>(count - i) * bar;
		draw->AddRectFilled({x, y_of(ms)}, {x + std::max(bar - 1.0f, 1.0f), origin.y + height},
		                    FrameTimeColor(ms));
	}
	ImGui::Dummy({width, height});
}

const char* PressureName(Common::MemoryPressure pressure) {
	switch (pressure) {
		case Common::MemoryPressure::Normal: return "Normal";
		case Common::MemoryPressure::Warning: return "Warning";
		case Common::MemoryPressure::Critical: return "Critical";
		case Common::MemoryPressure::Unknown: break;
	}
	return "n/a";
}

ImU32 PressureColor(Common::MemoryPressure pressure) {
	switch (pressure) {
		case Common::MemoryPressure::Normal: return GOOD;
		case Common::MemoryPressure::Warning: return MEDIUM;
		case Common::MemoryPressure::Critical: return BAD;
		case Common::MemoryPressure::Unknown: break;
	}
	return NEUTRAL;
}

const char* ThermalName(Common::ThermalPressure thermal) {
	switch (thermal) {
		case Common::ThermalPressure::Nominal: return "Nominal";
		case Common::ThermalPressure::Moderate: return "Moderate";
		case Common::ThermalPressure::Heavy: return "Heavy (throttling)";
		case Common::ThermalPressure::Critical: return "Critical (throttling)";
		case Common::ThermalPressure::Unknown: break;
	}
	return "n/a";
}

ImU32 ThermalColor(Common::ThermalPressure thermal) {
	switch (thermal) {
		case Common::ThermalPressure::Nominal: return GOOD;
		case Common::ThermalPressure::Moderate: return MEDIUM;
		case Common::ThermalPressure::Heavy:
		case Common::ThermalPressure::Critical: return BAD;
		case Common::ThermalPressure::Unknown: break;
	}
	return NEUTRAL;
}

const char* PresentModeName(Config::PresentMode mode) {
	switch (mode) {
		case Config::PresentMode::Fifo: return "FIFO";
		case Config::PresentMode::Mailbox: return "Mailbox";
		case Config::PresentMode::Immediate: return "Immediate";
	}
	return "?";
}

double Rate(const Snapshot& shown, Counter counter) {
	return shown.rates[static_cast<size_t>(counter)];
}

uint64_t Total(const Snapshot& shown, Counter counter) {
	return shown.totals[static_cast<size_t>(counter)];
}

void TransferRow(const Snapshot& shown, const char* label, Counter upload, Counter download) {
	Row(label,
	    Throughput(Rate(shown, upload)) + " up / " + Throughput(Rate(shown, download)) + " down");
}

void GpuActivityRows(const Snapshot& shown) {
	const auto draws = Rate(shown, Counter::Draws);
	Row("Draw calls", Fmt("%.0f/frame  %.0f/s", draws / std::max(shown.fps, 1.0), draws));
	// Includes present pacing waits, so a high value alone is not a stall.
	Row("CPU waiting on GPU",
	    Fmt("%.0f ms/s  (%.0f waits/s)", Rate(shown, Counter::GpuWaitNs) / 1e6,
	        Rate(shown, Counter::GpuWaits)));
	Row("DCC/CMASK readbacks",
	    Fmt("%.0f ms/s  (%.0f/s)", Rate(shown, Counter::ColorMetadataReadbackNs) / 1e6,
	        Rate(shown, Counter::ColorMetadataReadbacks)));
	Row("DCC/CMASK GPU checks", Fmt("%.0f/s", Rate(shown, Counter::ColorMetadataChecks)));
}

void CompilationRows(const Snapshot& shown) {
	const auto row = [&](const char* label, Counter count_counter, Counter duration_counter,
	                     uint64_t peak_ns) {
		const auto count = Total(shown, count_counter);
		Row(label,
		    Fmt("%llu  (avg %.1f ms, peak %.1f ms)", static_cast<unsigned long long>(count),
		        count != 0 ? static_cast<double>(Total(shown, duration_counter)) / 1e6 /
		                         static_cast<double>(count)
		                   : 0.0,
		        static_cast<double>(peak_ns) / 1e6),
		    Rate(shown, count_counter) > 0.0 ? MEDIUM : GOOD);
	};
	row("Shader recompiler", Counter::ShadersCompiled, Counter::ShaderCompileNs,
	    shown.shader_compile_peak_ns);
	row("Pipeline compile", Counter::PipelinesCreated, Counter::PipelineCreateNs,
	    shown.pipeline_create_peak_ns);
}

void Refresh(Snapshot& shown, const GraphicContext& graphics, Clock::time_point now,
             uint64_t generation) {
	const double seconds = shown.sample_time == Clock::time_point {}
	                           ? 0.0
	                           : std::chrono::duration<double>(now - shown.sample_time).count();
	for (size_t i = 0; i < COUNTER_COUNT; i++) {
		const auto value = Common::DebugCounters::Get(static_cast<Counter>(i));
		shown.rates[i] =
		    seconds > 0.0 ? static_cast<double>(value - shown.totals[i]) / seconds : 0.0;
		shown.totals[i] = value;
	}
	shown.shader_compile_peak_ns  = Common::DebugCounters::Get(Peak::ShaderCompileNs);
	shown.pipeline_create_peak_ns = Common::DebugCounters::Get(Peak::PipelineCreateNs);
	shown.sample_time             = now;
	{
		std::scoped_lock lock(g_host_mutex);
		if (g_host_generation == generation) {
			shown.host = g_host_stats;
		}
	}
	shown.heap_count = graphics.GetHeapUsage(shown.heaps);
	shown.device_memory =
	    graphics.CanReportMemoryUsage() ? std::optional<uint64_t>(0) : std::nullopt;
	shown.device_budget = 0;
	const bool discrete =
	    graphics.GetPhysicalDeviceProperties().deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	for (uint32_t i = 0; i < shown.heap_count; i++) {
		const auto& heap = shown.heaps[i];
		if (shown.device_memory && (!discrete || heap.device_local)) {
			*shown.device_memory += heap.usage;
			shown.device_budget += heap.budget;
		}
	}
	if (g_details.load(std::memory_order_acquire)) {
		shown.guest_memory = LibKernel::Memory::GetDebugStats();
	}
}

void DrawCompact(const Snapshot& shown) {
	if (!ImGui::BeginTable("##performance", 2, TABLE_FLAGS)) {
		return;
	}
	const auto& host = shown.host;
	if (shown.max_frame_ms > 0.0) {
		Row("Worst frame", Fmt("%.1f ms", shown.max_frame_ms), FrameTimeColor(shown.max_frame_ms));
	}
	Row("CPU", host.process_cpu_percent
	               ? Fmt("%.0f%%  of %u threads", *host.process_cpu_percent, host.logical_cores)
	               : std::string("n/a"));
	Row("GPU", Percent(host.gpu_percent));
	GpuActivityRows(shown);
	CompilationRows(shown);
	if (shown.device_memory) {
		Row("GPU memory", Bytes(*shown.device_memory) + " / " + Bytes(shown.device_budget));
	}
	TransferRow(shown, "Texture transfers", Counter::TextureUploadBytes,
	            Counter::TextureDownloadBytes);
	TransferRow(shown, "Buffer transfers", Counter::BufferUploadBytes,
	            Counter::BufferDownloadBytes);
	Row("Emulator RAM", Bytes(host.process_memory));
	if (host.system_memory_used && host.system_memory_total) {
		Row("System RAM",
		    Bytes(*host.system_memory_used) + " / " + Bytes(*host.system_memory_total));
	}
	ImGui::EndTable();
}

void DrawFramesAndCpu(const Snapshot& shown, vk::Extent2D extent) {
	const auto& host = shown.host;
	if (BeginSection("Frames")) {
		if (shown.low_fps > 0.0) {
			Row("1% low", Fmt("%.1f FPS", shown.low_fps), FrameTimeColor(1000.0 / shown.low_fps));
		}
		Row("Worst frame", Fmt("%.1f ms", shown.max_frame_ms), FrameTimeColor(shown.max_frame_ms));
		Row("Guest output", shown.guest_width != 0
		                        ? Fmt("%ux%u", shown.guest_width, shown.guest_height)
		                        : std::string("n/a"));
		Row("Window", Fmt("%ux%u  %s  %u Hz", extent.width, extent.height,
		                  PresentModeName(Config::GetPresentMode()), Config::GetVblankFrequency()));
		Row("Flips", Fmt("%.0f/s requested", Rate(shown, Counter::FlipsRequested)));
		const auto rejected = Rate(shown, Counter::FlipsRejected);
		Row("Flip queue full", Fmt("%.0f/s", rejected), rejected > 0.0 ? BAD : GOOD);
		const auto late = Rate(shown, Counter::LateVblanks);
		Row("Late vblanks", Fmt("%.0f/s", late), LevelColor(late, 1.0, 10.0));
		Row("Swapchain rebuilds",
		    Fmt("%llu  (dropped %llu)",
		        static_cast<unsigned long long>(Total(shown, Counter::SwapchainRecreations)),
		        static_cast<unsigned long long>(Total(shown, Counter::DroppedPresents))));
		ImGui::EndTable();
	}
	if (BeginSection("CPU")) {
		Row("Emulator",
		    host.process_cpu_percent
		        ? Fmt("%.0f%% of %u threads  (%.1f cores)", *host.process_cpu_percent,
		              host.logical_cores, *host.process_cpu_percent * host.logical_cores / 100.0)
		        : std::string("n/a"));
		Row("Machine", Percent(host.system_cpu_percent),
		    host.system_cpu_percent ? std::optional(LevelColor(*host.system_cpu_percent, 70, 90))
		                            : std::nullopt);
		if (host.thread_count) {
			Row("Host threads", Fmt("%u", *host.thread_count));
		}
		for (const auto& thread: host.busiest_threads) {
			Row(Fmt("  %.24s", thread.name.c_str()).c_str(), Fmt("%.0f%%", thread.cpu_percent),
			    LevelColor(thread.cpu_percent, 70, 95));
		}
		if (host.context_switches_per_s) {
			Row("Context switches", Fmt("%.0f/s", *host.context_switches_per_s));
		}
		if (host.page_faults_per_s) {
			Row("Page faults", Fmt("%.0f/s  (%.0f from disk)", *host.page_faults_per_s,
			                       host.page_ins_per_s.value_or(0.0)));
		}
		Row("Thermal", ThermalName(host.thermal_pressure), ThermalColor(host.thermal_pressure));
		const auto emulated = Rate(shown, Counter::EmulatedInstructions);
		Row("Emulated instrs", Fmt("%.0f/s", emulated), LevelColor(emulated, 1000, 100000));
		const auto unresolved = Rate(shown, Counter::UnresolvedImportCalls);
		Row("Missing HLE calls",
		    Fmt("%.0f/s  (%llu total)", unresolved,
		        static_cast<unsigned long long>(Total(shown, Counter::UnresolvedImportCalls))),
		    unresolved > 0.0 ? MEDIUM : GOOD);
		ImGui::EndTable();
	}
}

void DrawGpu(const Snapshot& shown, const GraphicContext& graphics) {
	const auto& host = shown.host;
	if (BeginSection("GPU")) {
		Row("Device", host.gpu_model.empty()
		                  ? std::string(graphics.GetPhysicalDeviceProperties().deviceName.data())
		              : host.gpu_cores
		                  ? Fmt("%s (%u cores)", host.gpu_model.c_str(), *host.gpu_cores)
		                  : host.gpu_model);
		Row("Utilization", Percent(host.gpu_percent),
		    host.gpu_percent ? std::optional(LevelColor(*host.gpu_percent, 80, 95)) : std::nullopt);
		if (host.gpu_renderer_percent || host.gpu_tiler_percent) {
			Row("Renderer / tiler",
			    Percent(host.gpu_renderer_percent) + " / " + Percent(host.gpu_tiler_percent));
		}
		if (host.gpu_memory_in_use) {
			Row("Driver memory", Bytes(*host.gpu_memory_in_use) + " in use / " +
			                         Bytes(host.gpu_memory_allocated) + " allocated");
		}
		if (host.gpu_recoveries) {
			Row("GPU resets", Fmt("%llu", static_cast<unsigned long long>(*host.gpu_recoveries)),
			    *host.gpu_recoveries > 0 ? BAD : GOOD);
		}
		const double fps = std::max(shown.fps, 1.0);
		GpuActivityRows(shown);
		Row("Instances", Fmt("%.0f/frame", Rate(shown, Counter::DrawInstances) / fps));
		Row("Indirect draws", Fmt("%.0f/s", Rate(shown, Counter::IndirectDraws)));
		Row("Dispatches", Fmt("%.0f/frame  %.0f/s", Rate(shown, Counter::Dispatches) / fps,
		                      Rate(shown, Counter::Dispatches)));
		Row("Queue submits", Fmt("%.0f/s", Rate(shown, Counter::QueueSubmits)));
		const auto faults = Rate(shown, Counter::GpuFaults);
		Row("Write-tracking faults", Fmt("%.0f/s", faults), LevelColor(faults, 20000, 100000));
		CompilationRows(shown);
		ImGui::EndTable();
	}
	if (BeginSection("GPU memory")) {
		if (shown.device_memory) {
			const double share = shown.device_budget != 0
			                         ? static_cast<double>(*shown.device_memory) /
			                               static_cast<double>(shown.device_budget) * 100.0
			                         : 0.0;
			Row("Vulkan budget", Bytes(*shown.device_memory) + " / " + Bytes(shown.device_budget),
			    LevelColor(share, 80, 95));
		}
		for (size_t heap = 0; heap < shown.heap_count; heap++) {
			const auto& usage = shown.heaps[heap];
			Row(Fmt("  Heap %zu%s", heap, usage.device_local ? " (device)" : "").c_str(),
			    Bytes(usage.usage) + " / " + Bytes(usage.budget) +
			        Fmt("  %u allocs", usage.allocations));
		}
		TransferRow(shown, "Texture transfers", Counter::TextureUploadBytes,
		            Counter::TextureDownloadBytes);
		Row("Texture evictions", Fmt("%.0f/s", Rate(shown, Counter::TextureEvictions)));
		TransferRow(shown, "Buffer transfers", Counter::BufferUploadBytes,
		            Counter::BufferDownloadBytes);
		Row("Buffer evictions", Fmt("%.0f/s", Rate(shown, Counter::BufferEvictions)));
		ImGui::EndTable();
	}
}

void DrawMemoryAndIo(const Snapshot& shown) {
	const auto& host  = shown.host;
	const auto& guest = shown.guest_memory;
	if (BeginSection("PS5 memory")) {
		const auto direct_used = guest.direct_allocated + guest.pooled_allocated;
		Row("Direct", Bytes(direct_used) + " / " + Bytes(guest.direct_total),
		    LevelColor(guest.direct_total != 0 ? static_cast<double>(direct_used) /
		                                             static_cast<double>(guest.direct_total) * 100.0
		                                       : 0.0,
		               85, 97));
		Row("  Mapped", Bytes(guest.direct_mapped));
		Row("  Pool",
		    Bytes(guest.pooled_allocated) + " (" + Bytes(guest.pool_committed) + " committed)");
		Row("Flexible", Bytes(guest.flexible_used) + " / " + Bytes(guest.flexible_total));
		Row("Extended", Bytes(guest.automatic_allocated));
		// Past its total a side sees no page-table entries available.
		const auto percent = std::max(guest.cpu_page_entries * 100 / guest.cpu_page_entries_total,
		                              guest.gpu_page_entries * 100 / guest.gpu_page_entries_total);
		Row("Page table (2 MB)",
		    Fmt("CPU %llu / %llu  GPU %llu / %llu",
		        static_cast<unsigned long long>(guest.cpu_page_entries),
		        static_cast<unsigned long long>(guest.cpu_page_entries_total),
		        static_cast<unsigned long long>(guest.gpu_page_entries),
		        static_cast<unsigned long long>(guest.gpu_page_entries_total)),
		    LevelColor(static_cast<double>(percent), 90, 100));
		ImGui::EndTable();
	}
	if (BeginSection("Emulator process")) {
		Row("Footprint", Bytes(host.process_memory));
		Row("Resident", Bytes(host.process_resident));
		if (host.process_compressed) {
			Row("Compressed / swapped", Bytes(*host.process_compressed),
			    *host.process_compressed > (1ull << 30u) ? MEDIUM : GOOD);
		}
		Row("File-backed", Bytes(host.process_file_backed));
		Row("Address space", Bytes(host.process_virtual));
		ImGui::EndTable();
	}
	if (BeginSection("System memory")) {
		if (host.system_memory_used && host.system_memory_total) {
			Row("Used", Bytes(*host.system_memory_used) + " / " + Bytes(*host.system_memory_total));
		}
		if (host.system_wired) {
			Row("Wired", Bytes(*host.system_wired));
		}
		if (host.system_compressed) {
			Row("Compressed", Bytes(*host.system_compressed));
		}
		Row("Cached files", Bytes(host.system_cached));
		Row("Free", Bytes(host.system_free));
		if (host.swap_used) {
			Row("Swap", Bytes(*host.swap_used) + " / " + Bytes(host.swap_total),
			    *host.swap_used > (4ull << 30u) ? MEDIUM : GOOD);
		}
		Row("Pressure", PressureName(host.memory_pressure), PressureColor(host.memory_pressure));
		ImGui::EndTable();
	}
	if (BeginSection("I/O & audio")) {
		Row("Disk read / write", Throughput(host.disk_read_per_s.value_or(0.0)) + " / " +
		                             Throughput(host.disk_write_per_s.value_or(0.0)));
		Row("Guest file read / write", Throughput(Rate(shown, Counter::GuestFileReadBytes)) +
		                                   " / " +
		                                   Throughput(Rate(shown, Counter::GuestFileWriteBytes)));
		const auto underruns = Rate(shown, Counter::AudioUnderruns);
		Row("Audio underruns",
		    Fmt("%.0f/s  (%llu total)", underruns,
		        static_cast<unsigned long long>(Total(shown, Counter::AudioUnderruns))),
		    underruns > 0.0 ? BAD : GOOD);
		ImGui::EndTable();
	}
}

} // namespace

void PerformanceOverlayRecordDepthTarget(uint32_t width, uint32_t height) noexcept {
	// Only the perf CSV reads it, and the counters run only with the overlay or the CSV.
	if (!Common::DebugCounters::Enabled()) {
		return;
	}
	// Shadow maps and other square or narrow targets are not the scene's resolution.
	if (height == 0 || width * 10 < height * 15 || width * 10 > height * 20) {
		return;
	}
	const uint32_t packed   = (width << 16u) | (height & 0xffffu);
	uint32_t       previous = g_depth_target_max.load(std::memory_order_relaxed);
	while ((previous >> 16u) < width &&
	       !g_depth_target_max.compare_exchange_weak(previous, packed, std::memory_order_relaxed)) {
	}
}

void PerformanceOverlayRecordFlip(uint32_t guest_width, uint32_t guest_height) noexcept {
	EnsurePerfCsv();
	// KYTY_FPS_LOG=1: the guest's flip rate once a second, with no counters, no panel and no CSV
	// (the panel and the CSV both turn every counter and the per-buffer GPU timers on, so this is
	// how the release's real rate, and the overlay's own cost, are measured).
	static const bool fps_log = std::getenv("KYTY_FPS_LOG") != nullptr;
	if (fps_log) {
		static std::atomic<uint32_t> window_flips {0};
		static std::atomic<int64_t>  window_start {0};
		const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
		                        Clock::now().time_since_epoch())
		                        .count();
		if (window_start.load(std::memory_order_relaxed) == 0) {
			window_start.store(now_ns, std::memory_order_relaxed);
		}
		const auto flips = window_flips.fetch_add(1, std::memory_order_relaxed) + 1;
		const auto start = window_start.load(std::memory_order_relaxed);
		if (now_ns - start >= 1'000'000'000) {
			const auto seconds = static_cast<double>(now_ns - start) / 1e9;
			std::printf("Kyty FPS: %.2f (%u flips in %.0f ms)\n",
			            static_cast<double>(flips) / seconds, flips, seconds * 1000.0);
			std::fflush(stdout);
			window_flips.store(0, std::memory_order_relaxed);
			window_start.store(now_ns, std::memory_order_relaxed);
		}
	}
	if (!Common::DebugCounters::Enabled()) {
		return;
	}
	std::scoped_lock lock(g_frame_mutex);
	g_flip_total.fetch_add(1, std::memory_order_relaxed);
	const auto       now = Clock::now();
	if (g_frames.window_start == Clock::time_point {}) {
		g_frames.window_start = now;
	}
	auto& frames = g_frames;
	if (frames.last_flip != Clock::time_point {}) {
		const float ms = std::chrono::duration<float, std::milli>(now - frames.last_flip).count();
		frames.intervals_ms[frames.next] = ms;
		frames.next                      = (frames.next + 1) % FRAME_HISTORY;
		frames.count                     = std::min(frames.count + 1, FRAME_HISTORY);
		frames.max_ms                    = std::max(frames.max_ms, ms);
	}
	frames.last_flip    = now;
	frames.guest_width  = guest_width;
	frames.guest_height = guest_height;
	frames.flips++;
}

bool PerformanceOverlayEnabled() noexcept {
	return Common::DebugCounters::Enabled();
}

void TogglePerformanceOverlay() noexcept {
	SetOverlayEnabled(!Common::DebugCounters::Enabled());
	g_revision.fetch_add(1, std::memory_order_acq_rel);
}

void TogglePerformanceOverlayDetails() noexcept {
	if (Common::DebugCounters::Enabled()) {
		g_details.store(!g_details.load(std::memory_order_acquire), std::memory_order_release);
	} else {
		g_details.store(true, std::memory_order_release);
		SetOverlayEnabled(true);
	}
	g_revision.fetch_add(1, std::memory_order_acq_rel);
}

uint64_t PerformanceOverlayRevision() noexcept {
	return g_revision.load(std::memory_order_acquire);
}

float DrawPerformanceOverlay(const GraphicContext& graphics, vk::Extent2D extent) {
	if (!PerformanceOverlayEnabled()) {
		return 0.0f;
	}
	EnsureSampler();

	static Snapshot shown;
	static uint64_t shown_generation = 0;
	static bool     shown_details    = false;
	const auto      generation       = g_generation.load(std::memory_order_acquire);
	const bool      details          = g_details.load(std::memory_order_acquire);
	const auto      now              = Clock::now();
	if (generation != shown_generation) {
		shown            = {};
		shown_generation = generation;
	}
	if (shown.sample_time == Clock::time_point {} || now - shown.sample_time >= REFRESH_INTERVAL ||
	    details != shown_details) {
		{
			std::scoped_lock lock(g_frame_mutex);
			auto&            frames = g_frames;
			shown.history_count     = frames.count;
			for (size_t i = 0; i < frames.count; i++) {
				shown.history[i] =
				    frames.intervals_ms[(frames.next + FRAME_HISTORY - frames.count + i) %
				                        FRAME_HISTORY];
			}
			if (frames.window_start == Clock::time_point {}) {
				frames.window_start = now;
			}
			const double seconds = std::chrono::duration<double>(now - frames.window_start).count();
			shown.fps            = seconds > 0.0 ? frames.flips / seconds : 0.0;
			shown.frame_ms       = frames.flips > 0 ? seconds * 1000.0 / frames.flips : 0.0;
			shown.max_frame_ms   = frames.max_ms;
			shown.stalled_s   = frames.last_flip == Clock::time_point {}
			                        ? 0.0
			                        : std::chrono::duration<double>(now - frames.last_flip).count();
			shown.guest_width = frames.guest_width;
			shown.guest_height  = frames.guest_height;
			frames.flips        = 0;
			frames.max_ms       = 0.0f;
			frames.window_start = now;
		}
		shown_details = details;
		if (shown.history_count > 0) {
			auto       sorted     = shown.history;
			const auto end        = sorted.begin() + shown.history_count;
			const auto percentile = sorted.begin() + shown.history_count * 99 / 100;
			std::nth_element(sorted.begin(), percentile, end);
			shown.low_fps = *percentile > 0.0f ? 1000.0 / *percentile : 0.0;
		}
		Refresh(shown, graphics, now, generation);
	}

	const float scale = std::clamp(std::min(static_cast<float>(extent.width) / 1920.0f,
	                                        static_cast<float>(extent.height) / 1080.0f),
	                               0.75f, 1.5f);
	ImGui::SetNextWindowPos({static_cast<float>(extent.width) - 12.0f * scale, 12.0f * scale},
	                        ImGuiCond_Always, {1.0f, 0.0f});
	ImGui::SetNextWindowBgAlpha(details ? 0.82f : 0.72f);
	constexpr ImGuiWindowFlags FLAGS =
	    ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoNav |
	    ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
	    ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoBringToFrontOnFocus |
	    ImGuiWindowFlags_AlwaysAutoResize;
	ImGui::PushFont(nullptr, (details ? 13.0f : 15.0f) * scale);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f * scale);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {10.0f * scale, 8.0f * scale});
	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {8.0f * scale, 2.0f * scale});
	ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, {4.0f * scale, 1.0f * scale});
	if (ImGui::Begin("##kyty_performance_overlay", nullptr, FLAGS)) {
		if (shown.fps > 0.0) {
			ImGui::TextColored(Color(FrameTimeColor(shown.frame_ms)), "%.1f FPS   %.1f ms",
			                   shown.fps, shown.frame_ms);
		} else if (shown.stalled_s >= 1.0) {
			ImGui::TextColored(Color(BAD), "No frames for %.0f s", shown.stalled_s);
		} else {
			ImGui::TextDisabled("Waiting for frames");
		}
		DrawFrameGraph(shown.history, shown.history_count, (details ? 330.0f : 260.0f) * scale,
		               46.0f * scale);
		if (details) {
			// Three columns keep the detailed panel within a 720p window.
			if (ImGui::BeginTable("##details", 3, TABLE_FLAGS)) {
				ImGui::TableNextRow();
				ImGui::TableNextColumn();
				DrawFramesAndCpu(shown, extent);
				ImGui::TableNextColumn();
				DrawGpu(shown, graphics);
				ImGui::TableNextColumn();
				DrawMemoryAndIo(shown);
				ImGui::EndTable();
			}
		} else {
			DrawCompact(shown);
		}
#if defined(__APPLE__)
		ImGui::TextDisabled(details ? "Cmd+Shift+P compact   Cmd+P hide"
		                            : "Cmd+Shift+P details   Cmd+P hide");
#else
		ImGui::TextDisabled(details ? "Shift+F2 compact   F2 hide" : "Shift+F2 details   F2 hide");
#endif
	}
	const float bottom = ImGui::GetWindowPos().y + ImGui::GetWindowSize().y;
	ImGui::End();
	ImGui::PopStyleVar(4);
	ImGui::PopFont();
	return bottom;
}

} // namespace Libs::Graphics
