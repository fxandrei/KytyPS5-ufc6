#include "graphics/host_gpu/renderer/renderContext.h"
#include "common/profiler.h"

#include "common/abSwitch.h"
#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/presentation/videoOut.h"
#include "kernel/memory.h"
#include "libs/errno.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace Libs::Graphics {

namespace {
// KYTY_UNMAP_TRACE=1 (diagnostic, 2026-11-21): each guest unmap or fixed remap that drains the GPU
// (Finish on the GPU thread) because a cache holds the range: address, size, which cache, wait
// time; plus a summary every 5 s. 4K T-Mobile fights showed ~1.3 such drains per frame.
// With KYTY_UNMAP_TARGETED or its verify mode each line also says how the unmap waited: mode=F
// full drain, T targeted wait, S no wait needed; behind = current tick - waited tick after the
// wait was chosen (0: the open command buffer was submitted for it, -1: no tick; always 0 for F).
// dirty=1: the range held GPU-written buffer bytes, which the invalidation downloads after a wait
// for their own write tick (that wait is not in wait_us).
struct UnmapTrace {
	bool enabled = [] {
		const char* value = std::getenv("KYTY_UNMAP_TRACE");
		return value != nullptr && *value == '1';
	}();
	std::chrono::steady_clock::time_point window = std::chrono::steady_clock::now();
	uint64_t waits = 0, bytes = 0, wait_ns = 0, in_buffers = 0, in_textures = 0, lines = 0;
	uint64_t targeted = 0, skipped = 0, submitted = 0, dirty = 0;

	void Note(char mode, uint64_t vaddr, uint64_t size, bool buffers, bool textures,
	          bool gpu_dirty, int64_t behind, uint64_t ns) {
		waits++;
		bytes += size;
		wait_ns += ns;
		in_buffers += buffers ? 1 : 0;
		in_textures += textures ? 1 : 0;
		targeted += mode != 'F' ? 1 : 0;
		skipped += mode == 'S' ? 1 : 0;
		submitted += mode == 'T' && behind == 0 ? 1 : 0;
		dirty += gpu_dirty ? 1 : 0;
		if (lines < 400) {
			lines++;
			std::printf("unmap-wait: addr=0x%" PRIx64 " size=0x%" PRIx64
			            " buffers=%d textures=%d wait_us=%.0f mode=%c behind=%" PRId64
			            " dirty=%d\n",
			            vaddr, size, buffers ? 1 : 0, textures ? 1 : 0,
			            static_cast<double>(ns) / 1e3, mode, behind, gpu_dirty ? 1 : 0);
		}
		const auto now = std::chrono::steady_clock::now();
		if (now - window >= std::chrono::seconds(5)) {
			std::printf("unmap-wait-summary: waits=%" PRIu64 " MiB=%.1f wait_ms=%.1f buffers=%" PRIu64
			            " textures=%" PRIu64 " targeted=%" PRIu64 " skipped=%" PRIu64
			            " submitted=%" PRIu64 " dirty=%" PRIu64 "\n",
			            waits, static_cast<double>(bytes) / (1024.0 * 1024.0),
			            static_cast<double>(wait_ns) / 1e6, in_buffers, in_textures, targeted,
			            skipped, submitted, dirty);
			std::fflush(stdout);
			window = now;
			waits = bytes = wait_ns = in_buffers = in_textures = 0;
			targeted = skipped = submitted = dirty = 0;
		}
	}
};

UnmapTrace& GetUnmapTrace() {
	static UnmapTrace trace;
	return trace;
}

// KYTY_UNMAP_TARGETED (2026-11-21, experiment, default off): an unmap over a cached range waits
// only for what can still write the range's guest memory, instead of draining the GPU (Finish).
// Cached buffers and images are separate Vulkan allocations; the GPU never reads or writes guest
// pages. GPU results reach guest memory only through synchronous downloads, which wait for their
// own write ticks, and through priority operations (image downloads, the buffer collector's
// downloads), which name their guest range (DeferPriorityOperation). The unmap waits for the
// newest of those overlapping the range (submitting the open command buffer if it is that tick)
// and for the priority operations up to it. Images the unmap frees are destroyed by tick
// (DeferOperation), buffers stay allocated; in-flight GPU work only uses those allocations.
//   KYTY_UNMAP_TARGETED=1 on, =ab on while AbSwitch bit 39 is set (same-process A/B; bit 39 is
//   RECORD_BINDINGS_PAYLOAD, which no code reads, and KYTY_VBLANK_ALT, which must stay unset).
//   KYTY_UNMAP_TARGETED_RULE=use: also wait for the newest GPU use of the range's cache objects
//   (buffer bindings, uploads, copies, BDA programs; any image counts as the open command buffer).
//   KYTY_VERIFY_UNMAP_TARGETED=1: computes the targeted tick, still drains fully, counts the cases
//   the targeted rule would not have waited for although the GPU had not finished work on the
//   range's cache objects, and exits on a guest publication the rule cannot see.
struct UnmapTargetedConfig {
	int  mode     = 0; // 0 off, 1 on, 2 AbSwitch bit 39
	bool verify   = false;
	bool use_rule = false;
};

constexpr uint32_t UNMAP_TARGETED_AB_BIT = 39;

const UnmapTargetedConfig& GetUnmapTargetedConfig() {
	static const UnmapTargetedConfig config = [] {
		UnmapTargetedConfig result;
		if (const char* value = std::getenv("KYTY_UNMAP_TARGETED"); value != nullptr) {
			result.mode = std::strcmp(value, "ab") == 0 ? 2 : (std::atoi(value) != 0 ? 1 : 0);
		}
		const char* verify = std::getenv("KYTY_VERIFY_UNMAP_TARGETED");
		result.verify      = verify != nullptr && *verify == '1';
		const char* rule   = std::getenv("KYTY_UNMAP_TARGETED_RULE");
		result.use_rule    = rule != nullptr && std::strcmp(rule, "use") == 0;
		if (result.mode != 0 || result.verify) {
			std::printf("GPU: unmap wait: targeted=%s rule=%s verify=%d\n",
			            result.mode == 2   ? "ab (AbSwitch bit 39)"
			            : result.mode == 1 ? "on"
			                               : "off",
			            result.use_rule ? "use" : "publication", result.verify ? 1 : 0);
			std::fflush(stdout);
		}
		return result;
	}();
	return config;
}

// KYTY_VERIFY_UNMAP_TARGETED counters, printed every 5 s (GPU thread only).
// checked: unmaps that drained. publication: a publication into the range was pending
// (publication_busy: its tick had not completed; publication_open: it was the open command
// buffer). buffer_inflight: the GPU had not finished recorded work on a cache buffer of the range
// newer than the targeted tick (safe by the memory model; rule=use waits for it). images: the range
// held images, whose GPU uses are not tracked (images_busy: the GPU had not finished the
// submitted work). gpu_dirty: the range held GPU-written buffer bytes, which the invalidation
// downloads after waiting for their own write tick, in either mode. dirty_after: GPU-dirty buffer bytes left in the range after its invalidation
// (either mode; a later download would write the unmapped range).
struct UnmapVerify {
	std::chrono::steady_clock::time_point window = std::chrono::steady_clock::now();
	uint64_t checked = 0, publication = 0, publication_busy = 0, publication_open = 0;
	uint64_t buffer_inflight = 0, images = 0, images_busy = 0, gpu_dirty = 0, dirty_after = 0;

	void MaybePrint() {
		const auto now = std::chrono::steady_clock::now();
		if (now - window < std::chrono::seconds(5)) {
			return;
		}
		std::printf("unmap-verify: checked=%" PRIu64 " publication=%" PRIu64
		            " publication_busy=%" PRIu64 " publication_open=%" PRIu64
		            " buffer_inflight=%" PRIu64 " images=%" PRIu64 " images_busy=%" PRIu64
		            " gpu_dirty=%" PRIu64 " dirty_after=%" PRIu64 " unsafe=0\n",
		            checked, publication, publication_busy, publication_open, buffer_inflight,
		            images, images_busy, gpu_dirty, dirty_after);
		std::fflush(stdout);
		window = now;
		checked = publication = publication_busy = publication_open = 0;
		buffer_inflight = images = images_busy = gpu_dirty = dirty_after = 0;
	}
};

UnmapVerify& GetUnmapVerify() {
	static UnmapVerify verify;
	return verify;
}

// The newest tick of a priority operation that will write [vaddr, vaddr + size) of guest memory
// or change its protection, 0 when none is pending.
uint64_t GuestPublicationTick(CommandScheduler& scheduler, TextureCache& textures, uint64_t vaddr,
                              uint64_t size) {
	// Storage write-backs are image publications too; the fault path lists them separately.
	return std::max(scheduler.NewestGuestPublicationTick(vaddr, size),
	                textures.WriteBackTick(vaddr, size));
}

// KYTY_UNMAP_TARGETED_RULE=use: the newest tick of recorded GPU work that may use the range's
// cache objects. Image uses are not tracked: any image counts as the open command buffer.
uint64_t UseTick(CommandScheduler& scheduler, BufferCache& buffers, uint64_t vaddr, uint64_t size,
                 bool in_textures) {
	return std::max(buffers.NewestUseTick(vaddr, size),
	                in_textures ? scheduler.CurrentTick() : uint64_t {0});
}

// KYTY_VERIFY_UNMAP_TARGETED, before the full drain.
void VerifyTargetedUnmap(CommandScheduler& scheduler, BufferCache& buffers, TextureCache& textures,
                         uint64_t vaddr, uint64_t size, bool in_textures, bool gpu_dirty) {
	auto&      verify  = GetUnmapVerify();
	const auto current = scheduler.CurrentTick();
	verify.checked++;
	verify.gpu_dirty += gpu_dirty ? 1 : 0;
	// Read the tagged count first: a finishing image publication lowers the image counter before
	// it leaves the tagged list, never after, and only this thread queues new ones.
	const auto tagged     = scheduler.PendingGuestPublications();
	const auto publishes  = textures.PendingBackingPublishes();
	const auto tagged_pub = scheduler.NewestGuestPublicationTick(vaddr, size);
	const auto write_back = textures.WriteBackTick(vaddr, size);
	if (publishes > tagged) {
		EXIT("unmap-verify: UNSAFE: %u image downloads still publish into guest memory but only "
		     "%" PRIu64 " priority operations name a guest range (unmap addr=0x%016" PRIx64
		     " size=0x%" PRIx64 ")\n",
		     publishes, tagged, vaddr, size);
	}
	if (write_back > tagged_pub) {
		EXIT("unmap-verify: UNSAFE: a storage write-back over the range (tick %" PRIu64
		     ") has no tagged publication (newest tagged %" PRIu64 "), unmap addr=0x%016" PRIx64
		     " size=0x%" PRIx64 "\n",
		     write_back, tagged_pub, vaddr, size);
	}
	const auto& config      = GetUnmapTargetedConfig();
	const auto  publication = std::max(tagged_pub, write_back);
	const auto  buffer_use  = buffers.NewestUseTick(vaddr, size);
	const auto  target =
        config.use_rule
	         ? std::max(publication, UseTick(scheduler, buffers, vaddr, size, in_textures))
	         : publication;
	if (publication != 0) {
		verify.publication++;
		verify.publication_busy += scheduler.IsFree(publication) ? 0 : 1;
		verify.publication_open += publication >= current ? 1 : 0;
	}
	if (buffer_use > target && !scheduler.IsFree(buffer_use)) {
		verify.buffer_inflight++;
	}
	if (in_textures) {
		verify.images++;
		verify.images_busy += current > 1 && !scheduler.IsFree(current - 1) ? 1 : 0;
	}
	verify.MaybePrint();
}

int64_t TicksBehind(CommandScheduler& scheduler, uint64_t tick) {
	return tick == 0 ? -1 : static_cast<int64_t>(scheduler.CurrentTick() - tick);
}

uint64_t ElapsedNs(std::chrono::steady_clock::time_point start) {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now() - start)
	                                 .count());
}

// Before an unmap's caches drop the range (GPU thread, scheduler active).
void WaitForUnmap(CommandScheduler& scheduler, BufferCache& buffers, TextureCache& textures,
                  uint64_t vaddr, uint64_t size) {
	const auto& config      = GetUnmapTargetedConfig();
	auto&       trace       = GetUnmapTrace();
	const bool  in_buffers  = buffers.IsRegionRegistered(vaddr, size);
	const bool  in_textures = textures.IsRegionRegistered(vaddr, size);
	const bool  registered  = in_buffers || in_textures;
	// Before the invalidation downloads them (trace and verify only).
	const bool gpu_dirty =
	    (trace.enabled || config.verify) && in_buffers && buffers.HasGpuDirtyBytes(vaddr, size);
	const bool targeted =
        !config.verify && (config.mode == 1 || (config.mode == 2 && Common::AbSwitch::Enabled(
	                                                                    UNMAP_TARGETED_AB_BIT)));
	if (!targeted) {
		// Check cache ownership on the GPU thread. Guest-memory callbacks can still access a
		// range with no cached data, so they must finish before it is unmapped.
		if (!registered && !scheduler.HasPendingPriorityOperations()) {
			return;
		}
		if (!registered) {
			Common::DebugCounters::Add(Common::DebugCounters::Counter::UnmapWaitsUnregistered);
		}
		if (config.verify) [[unlikely]] {
			VerifyTargetedUnmap(scheduler, buffers, textures, vaddr, size, in_textures,
			                    gpu_dirty);
		}
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::UnmapWaits,
		                                         Common::DebugCounters::Counter::UnmapWaitNs);
		const auto tick  = scheduler.CurrentTick();
		const auto start = std::chrono::steady_clock::now();
		scheduler.Finish();
		scheduler.WaitPriorityOperations(tick);
		if (trace.enabled) {
			trace.Note('F', vaddr, size, in_buffers, in_textures, gpu_dirty, 0, ElapsedNs(start));
		}
		return;
	}
	auto target = GuestPublicationTick(scheduler, textures, vaddr, size);
	if (config.use_rule) {
		target = std::max(target, UseTick(scheduler, buffers, vaddr, size, in_textures));
	}
	if (target == 0) {
		if (trace.enabled && (registered || scheduler.HasPendingPriorityOperations())) {
			trace.Note('S', vaddr, size, in_buffers, in_textures, gpu_dirty, -1, 0);
		}
		return;
	}
	if (!registered) {
		Common::DebugCounters::Add(Common::DebugCounters::Counter::UnmapWaitsUnregistered);
	}
	const auto behind = TicksBehind(scheduler, target);
	const auto start  = std::chrono::steady_clock::now();
	{
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::UnmapWaits,
		                                         Common::DebugCounters::Counter::UnmapWaitNs);
		if (target >= scheduler.CurrentTick()) {
			// The open command buffer: waiting for it is the full drain.
			scheduler.Finish();
		} else {
			scheduler.Wait(target);
		}
		scheduler.WaitPriorityOperations(target);
	}
	// Every publication into the range was queued at the target tick or before, and priority
	// operations run in tick order. One queued later (a later tick) would race with any wait.
	if (const auto left = GuestPublicationTick(scheduler, textures, vaddr, size);
	    left != 0 && left <= target) {
		EXIT("unmap: a guest publication into addr=0x%016" PRIx64 " size=0x%" PRIx64
		     " (tick %" PRIu64 ") is still pending after the targeted wait for tick %" PRIu64 "\n",
		     vaddr, size, left, target);
	}
	if (trace.enabled) {
		trace.Note('T', vaddr, size, in_buffers, in_textures, gpu_dirty, behind,
		           ElapsedNs(start));
	}
}
} // namespace

RenderContext::RenderContext(GraphicContext& graphics, const std::filesystem::path& shader_store)
    : m_graphics(graphics), m_render_executor(*this), m_command_scheduler(*this, graphics),
      m_descriptor_heap(graphics, m_command_scheduler.GetMasterSemaphore()),
      m_sampler_cache(graphics),
      m_buffer_cache(graphics, m_command_scheduler, m_page_manager, m_texture_cache),
      m_resolve_cache(*this), m_texture_cache(graphics, m_command_scheduler, m_page_manager,
	                                          m_buffer_cache, m_resolve_cache),
      m_image_heap(*this, m_page_manager), m_pipeline_cache(graphics, &m_image_heap, shader_store), m_occlusion_queries(graphics, m_command_scheduler) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	// Runs after VULKAN_HPP_DEFAULT_DISPATCHER.init(device) (vulkanWindow.cpp), and only for the
	// render scheduler: with AbSwitch RECORD_THREAD, its commands are recorded by the submit
	// thread. The present scheduler stays plain.
	m_command_scheduler.EnableRecordThread();
	const auto& unmap_config = GetUnmapTargetedConfig();
	m_buffer_cache.SetUseTracking(unmap_config.verify || unmap_config.use_rule);
	if (graphics.ray_query_enabled) {
		m_ray_tracing = std::make_unique<RayTracing>(*this);
	}
}

RenderContext::~RenderContext() {
	ShutdownGpu();
	m_command_scheduler.Shutdown();
}

void RenderContext::InitializeGpu(VideoOut::VideoOutDriver* video_out) {
	EXIT_IF(m_gpu != nullptr);
	m_video_out = video_out;
	m_gpu       = std::make_unique<GuestGpu>(*this);
}

void RenderContext::ShutdownGpu() {
	if (m_gpu != nullptr) {
		m_gpu->Shutdown();
		m_gpu.reset();
	}
	if (m_video_out != nullptr) {
		if (m_command_scheduler.Active()) {
			m_command_scheduler.Finish();
		}
		m_command_scheduler.DrainPriorityOperations();
		m_video_out = nullptr;
	}
}

GuestGpu& RenderContext::GetGpu() const {
	EXIT_IF(m_gpu == nullptr);
	return *m_gpu;
}

VideoOut::VideoOutDriver& RenderContext::GetVideoOut() const {
	EXIT_IF(m_video_out == nullptr);
	return *m_video_out;
}

bool RenderContext::HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	// The host reports the faulting byte, not the instruction's access width. Both caches
	// resolve its page; guessing a width can cross the end of a valid guest mapping.
	constexpr uint64_t fault_size = 1;
	if (!IsMapped(fault_vaddr, fault_size)) {
		return false;
	}
	// The access retries once the image write-backs over its page land.
	if (WaitWriteBacks(fault_vaddr, fault_size)) {
		return true;
	}
	if (access == PageFaultAccess::Write) {
		m_buffer_cache.InvalidateMemory(fault_vaddr, fault_size);
		m_texture_cache.InvalidateMemory(fault_vaddr, fault_size);
		m_image_heap.InvalidateMemory(fault_vaddr, fault_size);
	} else {
		m_buffer_cache.ReadMemory(fault_vaddr, fault_size);
	}
	return true;
}

bool RenderContext::ReadCleanBytes(uint64_t vaddr, void* data, uint64_t size) noexcept {
	// The buffer cache belongs to the guest GPU thread.
	return GuestGpu::IsGpuThread() && IsMapped(vaddr, size) &&
	       m_buffer_cache.IsRegionGpuModified(vaddr, size) &&
	       !m_buffer_cache.HasGpuDirtyBytes(vaddr, size) &&
	       !m_texture_cache.HasPendingBackingPublish() &&
	       LibKernel::Memory::TryReadBacking(vaddr, data, size);
}

void RenderContext::TraceDirtyRead(const char* what, uint64_t vaddr) noexcept {
	static const int limit = [] {
		const char* value = std::getenv("KYTY_TRACE_WRITES");
		return value != nullptr ? std::atoi(value) : 0;
	}();
	static std::atomic_int printed = 0;
	if (limit <= 0 || !GuestGpu::IsGpuThread() ||
	    printed.fetch_add(1, std::memory_order_relaxed) >= limit) {
		return;
	}
	const auto [begin, end] = m_buffer_cache.GpuDirtyRangeAt(vaddr);
	std::printf("%s read 0x%" PRIx64 " waits for GPU-dirty range 0x%" PRIx64 "-0x%" PRIx64
	            " (%" PRIu64 " KiB)\n",
	            what, vaddr, begin, end, (end - begin) / 1024);
}

bool RenderContext::WaitWriteBacks(uint64_t vaddr, uint64_t size) {
	const auto tick = m_texture_cache.WriteBackTick(vaddr, size);
	if (tick == 0) {
		return false;
	}
	KYTY_PROFILER_BLOCK("Readback: pending image publication");
	// The GPU thread can fault inside a draw, so it submits without running deferred operations.
	if (GuestGpu::IsGpuThread()) {
		m_command_scheduler.Wait(tick);
	} else if (tick == m_command_scheduler.CurrentTick()) {
		GetGpu().SendCommandSync([this, tick] { m_command_scheduler.Wait(tick); });
	}
	m_command_scheduler.WaitPriorityOperations(tick);
	return true;
}

bool RenderContext::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	// Host I/O into the range, even one running past its mapping, would fail on its inaccessible
	// pages, and a later write-back would overwrite it.
	(void)WaitWriteBacks(vaddr, size);
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	m_buffer_cache.InvalidateMemory(vaddr, size);
	m_texture_cache.InvalidateMemory(vaddr, size);
	m_image_heap.InvalidateMemory(vaddr, size);
	return true;
}

bool RenderContext::IsMapped(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges.Contains(vaddr, size);
}

void RenderContext::MapMemory(uint64_t vaddr, uint64_t size) {
	std::lock_guard lock(m_mapped_ranges_mutex);
	m_mapped_ranges.Add(vaddr, size);
	m_buffer_cache.QueueBdaSync(vaddr, size);
}

void RenderContext::UnmapMemory(uint64_t vaddr, uint64_t size) {
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	const auto unmap = [this, vaddr, size] {
		if (m_command_scheduler.Active()) {
			WaitForUnmap(m_command_scheduler, m_buffer_cache, m_texture_cache, vaddr, size);
		}
		m_buffer_cache.InvalidateMemory(vaddr, size);
		if (GetUnmapTargetedConfig().verify && m_buffer_cache.HasGpuDirtyBytes(vaddr, size))
		    [[unlikely]] {
			GetUnmapVerify().dirty_after++;
		}
		m_texture_cache.UnmapMemory(vaddr, size);
		m_image_heap.UnmapMemory(vaddr, size);
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Subtract(vaddr, size);
	};
	// Shutdown still owns the GPU while queued rendering drains, but its command lane no
	// longer accepts external work. Use the guest GPU's state for the teardown route.
	if (m_gpu == nullptr || m_gpu->IsStopping()) {
		unmap();
		return;
	}
	m_gpu->SendCommandSync(unmap);
}

void RenderContext::PrepareBda() {
    m_buffer_cache.NoteBdaProgram(); // KYTY_UNMAP_TARGETED rule=use and its verify mode.
    if (!m_bda_logged) {
        Log::WriteToConsoleAndLog("GPU: using buffer device address (BDA) shader memory access.\n");
        m_bda_logged = true;
    }
    m_fault_process_pending = true;
    // AbSwitch BDA_SYNC_PER_SUBMIT (restored 2026-11-20; the 2026-11-12 upstream merge d42bf14fd had
    // dropped this gate, so every BDA draw synchronized): game threads write what a submission slice
    // reads before submitting it or before the label a wait resumes on, so synchronize once per slice
    // and after command-processor writes and passed waits (each calls RequestBdaSync).
    if (Common::AbSwitch::Enabled(Common::AbSwitch::BDA_SYNC_PER_SUBMIT) &&
        !std::exchange(m_bda_sync_requested, false)) {
        return;
    }
    m_bda_sync_requested = false;
    m_bda_written = {};
    std::shared_lock lock(m_mapped_ranges_mutex);
    m_buffer_cache.SynchronizeBda(m_mapped_ranges);
}

void RenderContext::RequestBdaSync(uint64_t vaddr, uint64_t size) {
    if (size != 0 && GuestRange {vaddr, size}.Valid()) {
        m_buffer_cache.QueueBdaSync(vaddr, size, true);
        m_bda_sync_requested = true;
    }
}

void RenderContext::RunGarbageCollector() {
	LibKernel::Memory::DrainDirectMapEvents();
	if (m_fault_process_pending) {
		m_fault_process_pending = false;
		m_buffer_cache.ProcessFaultBuffer();
	}
	m_image_heap.RunGarbageCollector();
	m_texture_cache.ProcessDownloadImages();
	m_texture_cache.RunGarbageCollector();
	m_buffer_cache.RunGarbageCollector();
	if (m_ray_tracing) {
		m_ray_tracing->RunGarbageCollector();
	}
}

void RenderContext::AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it != m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.push_back({eq, event_id});
}

void RenderContext::DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it == m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.erase(it);
}

void RenderContext::TriggerInterrupt(int event_id, uint32_t context_id) {
	std::vector<InterruptEqRegistration> registrations;
	{
		Common::LockGuard lock(m_interrupt_mutex);
		for (const auto& registration: m_interrupt_eqs) {
			if (registration.event_id == event_id) {
				registrations.push_back(registration);
			}
		}
	}

	const auto ident = static_cast<uintptr_t>(event_id);
	auto*      data  = reinterpret_cast<void*>((static_cast<uintptr_t>(context_id) << 16u) | ident);
	for (const auto& registration: registrations) {
		const auto result = LibKernel::EventQueue::KernelTriggerEvent(
		    registration.eq, ident, LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS, data);
		if (result == LibKernel::KERNEL_ERROR_EBADF || result == LibKernel::KERNEL_ERROR_ENOENT) {
			DeleteInterruptEq(registration.eq, registration.event_id);
			continue;
		}
		EXIT_NOT_IMPLEMENTED(result != OK);
	}
}

} // namespace Libs::Graphics
