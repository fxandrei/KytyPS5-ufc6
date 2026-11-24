#include "graphics/host_gpu/renderer/cache/bufferCache.h"

#include "common/abSwitch.h"
#include "common/alignment.h"
#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <smmintrin.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

namespace {

constexpr uint64_t MiB = 1024 * 1024;

// Reads device memory mapped through the BAR, which is write-combined: plain loads are uncached
// one at a time (about 30 MB/s), while streaming loads fetch a whole 64-byte line per request.
// Hands the bytes to `sink` in pieces of at most 4 KiB, as (bytes, offset, size).
template <typename Sink>
void ReadWriteCombined(const uint8_t* source, uint64_t size, Sink&& sink) {
	// Written by the line loop below before sink() sees it; zero-filling 4 KiB per call cost about
	// a third of this function.
	alignas(64) std::array<uint8_t, 4096 + 128> lines;
	uint64_t                                    done = 0;
	while (done < size) {
		const auto  piece = std::min<uint64_t>(size - done, 4096);
		const auto* from  = source + done;
		const auto  first = reinterpret_cast<uintptr_t>(from) & ~uintptr_t {63};
		const auto  last  = (reinterpret_cast<uintptr_t>(from) + piece + 63) & ~uintptr_t {63};
		auto*       to    = reinterpret_cast<__m128i*>(lines.data());
		for (auto line = first; line < last; line += 64) {
			const auto* in = reinterpret_cast<__m128i*>(line);
			_mm_store_si128(to++, _mm_stream_load_si128(in));
			_mm_store_si128(to++, _mm_stream_load_si128(in + 1));
			_mm_store_si128(to++, _mm_stream_load_si128(in + 2));
			_mm_store_si128(to++, _mm_stream_load_si128(in + 3));
		}
		sink(lines.data() + (reinterpret_cast<uintptr_t>(from) - first), done, piece);
		done += piece;
	}
}
constexpr uint64_t GdsBufferSize = 64 * 1024;
// GPU-modified ranges come as many small scattered islands, so a readback widens to a window.
constexpr uint64_t DownloadWindowSize = 512 * 1024;

// KYTY_VERIFY_READ_FAST_PATH (AbSwitch READ_BINDING_FAST_PATH), read once at startup so the hot path
// tests no static guard.
const bool g_verify_read_fast_path = std::getenv("KYTY_VERIFY_READ_FAST_PATH") != nullptr;

// KYTY_VERIFY_READ_FAST_PATH: checked fast-path decisions, and those a CPU write overtook before the
// full path ran (the race UPLOAD_PRECHECK accepts: the write is for later commands). GPU thread only.
void NoteReadFastPathVerify(bool overtaken) {
	static uint64_t checked = 0;
	static uint64_t late    = 0;
	++checked;
	late += overtaken ? 1u : 0u;
	if ((checked & 0xfffffu) == 0) {
		std::printf("read-fast-path verify: %" PRIu64 " checked, %" PRIu64
		            " overtaken by a CPU write, 0 mismatches\n",
		            checked, late);
		std::fflush(stdout);
	}
}

std::pair<Buffer*, uint64_t> UploadMemory(GraphicContext& graphics, CommandScheduler& scheduler,
                                          StreamBuffer& staging, uint64_t size, uint64_t alignment,
                                          vk::BufferUsageFlags usage, const auto& write) {
	const auto [mapped, offset] = staging.Map(size, alignment);
	if (mapped != nullptr) {
		write(mapped, offset);
		staging.Commit();
		return {&staging, offset};
	}

	// Large transfers use the same one-time host allocation as ordinary buffer uploads.
	auto temporary =
	    std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Upload, 0, usage, size);
	write(temporary->Mapped().data(), 0);
	temporary->Flush(0, size);
	auto* buffer = temporary.get();
	scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	return {buffer, 0};
}

// KYTY_TRACE_WRITES=<n>: print the first n distinct shader-written buffer bindings, which mark
// their whole range GPU-written.
void TraceWrittenBinding(uint64_t vaddr, uint64_t size) {
	static const int limit = [] {
		const char* value = std::getenv("KYTY_TRACE_WRITES");
		return value != nullptr ? std::atoi(value) : 0;
	}();
	if (limit <= 0) {
		return;
	}
	static std::mutex                              mutex;
	static std::set<std::pair<uint64_t, uint64_t>> seen;
	std::scoped_lock                               lock(mutex);
	if (static_cast<int>(seen.size()) >= limit || !seen.emplace(vaddr, size).second) {
		return;
	}
	std::printf("written binding 0x%" PRIx64 " size 0x%" PRIx64 " (%" PRIu64 " KiB)\n", vaddr, size,
	            size / 1024);
}

// KYTY_SRT_DIRTY_STATS (diagnostic, 2026-11-19): shader-written bindings by writer shader.
void NoteWrittenBindingStats(uint64_t writer, uint64_t size) {
	struct Entry {
		uint64_t n = 0, bytes = 0, max = 0;
	};
	static std::mutex                          mutex;
	static std::unordered_map<uint64_t, Entry> entries;
	static Entry                               total;
	static int64_t                             window_start = 0;
	const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
	                        std::chrono::steady_clock::now().time_since_epoch())
	                        .count();
	std::scoped_lock lock(mutex);
	if (window_start == 0) {
		window_start = now;
	}
	for (auto* e: {&entries[writer], &total}) {
		e->n++;
		e->bytes += size;
		e->max = std::max(e->max, size);
	}
	if (now - window_start < 5'000'000'000) {
		return;
	}
	const double s = static_cast<double>(now - window_start) / 1e9;
	std::printf("write-stats: %.1fs bindings/s=%.0f MB/s=%.0f max_kib=%" PRIu64 "\n", s,
	            static_cast<double>(total.n) / s, static_cast<double>(total.bytes) / 1e6 / s,
	            total.max / 1024);
	std::vector<std::pair<uint64_t, Entry>> sorted(entries.begin(), entries.end());
	std::sort(sorted.begin(), sorted.end(),
	          [](const auto& a, const auto& b) { return a.second.bytes > b.second.bytes; });
	for (size_t i = 0; i < sorted.size() && i < 12; i++) {
		const auto& [hash, e] = sorted[i];
		std::printf("write-stats:   writer %016" PRIx64 " n/s=%.1f MB/s=%.1f avg_kib=%.1f "
		            "max_kib=%" PRIu64 "\n",
		            hash, static_cast<double>(e.n) / s, static_cast<double>(e.bytes) / 1e6 / s,
		            static_cast<double>(e.bytes) / 1024.0 / static_cast<double>(e.n), e.max / 1024);
	}
	std::fflush(stdout);
	entries.clear();
	total        = {};
	window_start = now;
}

} // namespace

void BufferCache::WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source,
                                  uint64_t size) {
	auto* bytes = static_cast<const uint8_t*>(source);
	while (size != 0) {
		const auto chunk = std::min(size, m_staging_buffer.Size());
		// Copy waits for ring reuse and flushes the fresh host data before submission.
		const auto offset             = m_staging_buffer.Copy(bytes, chunk, 4);
		const auto destination_offset = buffer.Offset(address);
		EXIT_IF(destination_offset > buffer.Size() || chunk > buffer.Size() - destination_offset);
		m_scheduler.EndRendering();
		const auto               command = m_scheduler.Current().Handle();
		vk::BufferMemoryBarrier2 before {};
		before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		before.srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		before.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		before.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer              = buffer.Handle();
		before.offset              = destination_offset;
		before.size                = chunk;
		vk::DependencyInfo dependency {};
		dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
		dependency.bufferMemoryBarrierCount = 1;
		dependency.pBufferMemoryBarriers    = &before;
		command.pipelineBarrier2(dependency);
		const vk::BufferCopy copy {offset, destination_offset, chunk};
		command.copyBuffer(m_staging_buffer.Handle(), buffer.Handle(), 1, &copy);
		auto after          = before;
		after.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		after.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
		after.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		after.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		dependency.pBufferMemoryBarriers = &after;
		command.pipelineBarrier2(dependency);
		NoteRecordedUse(buffer);
		bytes += chunk;
		address += chunk;
		size -= chunk;
	}
}

void BufferCache::Register(BufferId id) {
	ChangeRegister<true>(id);
}

void BufferCache::Unregister(BufferId id) {
	ChangeRegister<false>(id);
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId id) {
	auto&                buffer = m_slot_buffers[id];
	PageTable::PageRange pages {};
	EXIT_IF(!(GuestRange {buffer.CpuAddress(), buffer.Size()}.Valid()) ||
	        !PageTable::TryGetPageRange(buffer.CpuAddress(), buffer.Size(), pages));
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		if constexpr (insert) {
			m_page_table[page] = id;
		} else {
			m_page_table[page] = {};
		}
	}
	const auto size_pages   = pages.last_exclusive - pages.first;
	const auto table_offset = PageIndex(buffer.CpuAddress()) * sizeof(vk::DeviceAddress);
	if constexpr (insert) {
		m_buffers.emplace(buffer.CpuAddress(), id);
		m_total_used_memory += buffer.Size();
		buffer.lru_id   = m_lru_cache.Insert(id, m_gc_tick);
		buffer.lru_tick = m_gc_tick;
		std::vector<vk::DeviceAddress> addresses;
		addresses.reserve(size_pages);
		for (uint64_t i = 0; i < size_pages; ++i) {
			addresses.push_back(buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS));
		}
		WriteDataBuffer(m_bda_pagetable_buffer, table_offset,
		                addresses.data(), addresses.size() * sizeof(vk::DeviceAddress));
		QueueBdaSync(buffer.CpuAddress(), buffer.Size());
	} else {
		m_buffers.erase(m_buffers.find(buffer.CpuAddress()));
		m_total_used_memory -= buffer.Size();
		m_lru_cache.Free(buffer.lru_id);
		m_bda_pagetable_buffer.Fill(table_offset, size_pages * sizeof(vk::DeviceAddress), 0);
		buffer.is_deleted = true;
	}
}

void BufferCache::RecordGpuWrite(uint64_t vaddr, uint64_t size) {
	auto& slot           = m_gpu_writes[m_next_gpu_write];
	m_evicted_write_tick = std::max(m_evicted_write_tick, slot.tick);
	slot                 = {vaddr, vaddr + size, m_scheduler.CurrentTick(), diag_writer};
	m_next_gpu_write     = (m_next_gpu_write + 1) % GPU_WRITE_RING;
	if (m_next_gpu_write == m_first_unretagged) {
		// More than a ring of writes without a draw: the oldest unretagged one was just evicted.
		m_first_unretagged = m_next_gpu_write;
	}
}

void BufferCache::RetagGpuWrites() {
	const auto tick = m_scheduler.CurrentTick();
	for (auto slot = m_first_unretagged; slot != m_next_gpu_write;
	     slot      = (slot + 1) % GPU_WRITE_RING) {
		m_gpu_writes[slot].tick = std::max(m_gpu_writes[slot].tick, tick);
	}
	m_first_unretagged = m_next_gpu_write;
	// The uses noted since the last call were recorded by now, at this tick or earlier. A reused
	// slot's new buffer only gets a later tick than needed.
	for (const auto id: m_pending_uses) {
		if (auto* buffer = m_slot_buffers.try_get(id);
		    buffer != nullptr && buffer->use_tick == PENDING_USE) {
			buffer->use_tick = tick;
		}
	}
	m_pending_uses.clear();
	if (std::exchange(m_bda_pending, false)) {
		m_bda_tick = tick;
	}
}

void BufferCache::NoteRecordedUse(Buffer& buffer) {
	if (m_track_uses && buffer.use_tick != PENDING_USE) {
		buffer.use_tick = std::max(buffer.use_tick, m_scheduler.CurrentTick());
	}
}

uint64_t BufferCache::NewestUseTick(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		return 0;
	}
	const auto current = m_scheduler.CurrentTick();
	const auto end     = vaddr + size;
	uint64_t   tick    = 0;
	bool       found   = false;
	auto       it      = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < end; ++it) {
		const auto& buffer = m_slot_buffers[it->second];
		if (buffer.CpuAddress() + buffer.Size() <= vaddr) {
			continue;
		}
		found = true;
		tick  = std::max({tick, buffer.use_tick == PENDING_USE ? current : buffer.use_tick,
		                  buffer.LastGpuWriteTick()});
	}
	if (!found) {
		return 0;
	}
	if (!m_track_uses) {
		return current;
	}
	// A BDA program can address every registered buffer.
	return std::min(current, std::max(tick, m_bda_pending ? current : m_bda_tick));
}

bool BufferCache::HasInFlightWriteAt(uint64_t vaddr) const {
	const auto known = m_scheduler.GetMasterSemaphore().KnownGpuTick();
	if (m_evicted_write_tick > known) {
		return true; // A write that left the ring may still be running.
	}
	// Newest first; ticks never decrease from oldest to newest, so the first completed write ends
	// the search: it and every older one are done.
	for (size_t i = 0; i < GPU_WRITE_RING; i++) {
		const auto& write =
		    m_gpu_writes[(m_next_gpu_write + GPU_WRITE_RING - 1 - i) % GPU_WRITE_RING];
		if (write.tick <= known) {
			return false;
		}
		if (write.begin <= vaddr && vaddr < write.end) {
			return true;
		}
	}
	return true;
}

BufferCache::InFlightTicks BufferCache::DiagInFlightTicks(uint64_t vaddr, uint64_t size,
                                                          uint64_t window_begin,
                                                          uint64_t window_end) const {
	InFlightTicks result;
	result.known = m_scheduler.GetMasterSemaphore().KnownGpuTick();
	for (size_t i = 0; i < GPU_WRITE_RING; i++) {
		const auto& write =
		    m_gpu_writes[(m_next_gpu_write + GPU_WRITE_RING - 1 - i) % GPU_WRITE_RING];
		if (write.tick <= result.known) {
			break;
		}
		if (write.begin < vaddr + size && vaddr < write.end) {
			if (result.exact == 0) {
				result.writer = write.writer;
			}
			result.exact = std::max(result.exact, write.tick);
		}
		if (write.begin < window_end && window_begin < write.end) {
			result.window = std::max(result.window, write.tick);
		}
	}
	return result;
}

BufferCache::CoveringWrite BufferCache::NewestCoveringWrite(uint64_t vaddr) const {
	for (size_t i = 0; i < GPU_WRITE_RING; i++) {
		const auto& write =
		    m_gpu_writes[(m_next_gpu_write + GPU_WRITE_RING - 1 - i) % GPU_WRITE_RING];
		if (write.end == 0) {
			break; // Never used: the ring has not wrapped yet.
		}
		if (write.begin <= vaddr && vaddr < write.end) {
			return {write.begin, write.end, write.tick, write.writer};
		}
	}
	return {};
}

uint64_t BufferCache::ReadbackWriteTick(const Buffer& buffer, uint64_t buffer_address,
                                        std::span<const vk::BufferCopy> copies) {
	const auto buffer_tick = buffer.LastGpuWriteTick();
	if (buffer_tick == 0 || !Common::AbSwitch::Enabled(Common::AbSwitch::RANGE_WRITE_TICKS)) {
		return buffer_tick;
	}
	const auto known = m_scheduler.GetMasterSemaphore().KnownGpuTick();
	uint64_t   tick  = m_evicted_write_tick > known ? m_evicted_write_tick : 0;
	// Newest first, down to the first write the GPU has completed: all older ones are done too.
	for (size_t i = 0; i < GPU_WRITE_RING; i++) {
		const auto& write =
		    m_gpu_writes[(m_next_gpu_write + GPU_WRITE_RING - 1 - i) % GPU_WRITE_RING];
		if (write.tick <= known) {
			break;
		}
		for (const auto& copy: copies) {
			const auto begin = buffer_address + copy.srcOffset;
			if (write.begin < begin + copy.size && begin < write.end) {
				tick = std::max(tick, write.tick);
				break;
			}
		}
	}
	if (tick != 0) {
		return tick;
	}
	// No write in flight covers the bytes: they are the GPU's as of the last completed tick.
	return known != 0 ? known : buffer_tick;
}

void BufferCache::TouchBuffer(Buffer& buffer) {
	if (buffer.is_deleted) {
		return;
	}
	if (Common::AbSwitch::Enabled(Common::AbSwitch::LRU_TOUCH_MEMO)) {
		if (buffer.lru_tick >= m_gc_tick) {
			return;
		}
		buffer.lru_tick = m_gc_tick;
	}
	m_lru_cache.Touch(buffer.lru_id, m_gc_tick);
}

void BufferCache::DeleteBuffer(BufferId id) {
	Unregister(id);
	if (m_scheduler.Active()) {
		m_scheduler.DeferOperation([this, id] { m_slot_buffers.erase(id); });
	} else {
		m_slot_buffers.erase(id);
	}
}

template <bool async>
bool BufferCache::DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	KYTY_PROFILER_BLOCK("Readback: buffer download");
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size     = 0;
	const auto                  buffer_address = buffer.CpuAddress();
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "buffer download");
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    copies.emplace_back(start - buffer_address, total_size, end - start);
			    // Start each packed range on its own 64-byte cache line.
			    total_size += Common::AlignUp(end - start, 64);
		    });
		    m_gpu_modified_ranges.Subtract(address, bytes);
		    ForgetColorMetadataChecks(address, bytes);
	    });
	if (copies.empty()) {
		return false;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferDownloadBytes, total_size);
	if constexpr (!async) {
		// The ticks up to the last writer carry the host-read barrier that Submit records, so the
		// mapped bytes are the GPU's once that tick has completed; later work is not waited for.
		const auto write_tick = ReadbackWriteTick(buffer, buffer_address, copies);
		// While the writes are still being recorded, the work has to be submitted anyway: a copy
		// to cached host memory recorded behind it is then cheaper than reading the resizable
		// BAR over PCIe afterwards (AbSwitch READBACK_COPY_WHEN_FLUSHING).
		const bool copy_instead =
		    write_tick >= m_scheduler.CurrentTick() &&
		    Common::AbSwitch::Enabled(Common::AbSwitch::READBACK_COPY_WHEN_FLUSHING);
		if (!buffer.Mapped().empty() && write_tick != 0 && !copy_instead &&
		    Common::AbSwitch::Enabled(Common::AbSwitch::DIRECT_READBACK)) {
			{
			KYTY_PROFILER_BLOCK("Readback: direct wait and priority completion");
			const bool     timing   = CommandScheduler::ReadbackTimingEnabled();
			const int64_t  start_ns = timing ? CommandScheduler::NowNs() : 0;
			const bool     open     = write_tick >= m_scheduler.CurrentTick();
			const uint64_t in_flight =
			    timing ? m_scheduler.CurrentTick() - 1 -
			                 m_scheduler.GetMasterSemaphore().KnownGpuTick()
			           : 0;
			if (open) {
				m_scheduler.FlushUrgent(); // The writes are still being recorded.
			}
			const int64_t flushed_ns = timing ? CommandScheduler::NowNs() : 0;
			if (!m_scheduler.IsFree(write_tick)) {
				buffer.MarkWaitedForReadback();
			}
			m_scheduler.Wait(write_tick);
			const int64_t done_ns = timing ? CommandScheduler::NowNs() : 0;
			m_scheduler.WaitPriorityOperations(write_tick);
			if (timing) [[unlikely]] {
				m_scheduler.NoteReadbackWait(write_tick, open, in_flight, start_ns, flushed_ns,
				                             done_ns, CommandScheduler::NowNs());
			}
			}
			KYTY_PROFILER_BLOCK("Readback: BAR copy and backing publication");
			const auto* mapped = buffer.Mapped().data();
			for (const auto& copy: copies) {
				ReadWriteCombined(mapped + copy.srcOffset, copy.size,
				                  [&](const uint8_t* bytes, uint64_t offset, uint64_t size) {
					                  Libs::LibKernel::Memory::WriteBacking(
					                      buffer_address + copy.srcOffset + offset, bytes, size);
				                  });
			}
			return true;
		}
	}

	auto download = m_download_buffer.MapDownload(total_size, 64);
	for (auto& copy: copies) {
		copy.dstOffset += download.offset;
	}

	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = 0;
	before.size                = buffer.Size();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                       nullptr);
	native.copyBuffer(buffer.Handle(), download.buffer->Handle(),
	                  static_cast<uint32_t>(copies.size()), copies.data());
	NoteRecordedUse(buffer);

	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = download.buffer->Handle();
	after.offset        = download.offset;
	after.size          = total_size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands |
	                           vk::PipelineStageFlagBits::eHost,
	                       {}, 0, nullptr, 1, &after, 0, nullptr);
	auto publish = [total_size, buffer_address, copies = std::move(copies),
	                download = std::move(download)] {
		download.buffer->Invalidate(download.offset, total_size);
		for (const auto& copy: copies) {
			Libs::LibKernel::Memory::WriteBacking(
			    buffer_address + copy.srcOffset,
			    download.mapped + (copy.dstOffset - download.offset), copy.size);
		}
	};
	if constexpr (async) {
		// Tagged with its guest range: an unmap of the range waits for it (KYTY_UNMAP_TARGETED).
		m_scheduler.DeferPriorityOperation(std::move(publish), vaddr, size);
	} else {
		const auto tick = m_scheduler.CurrentTick();
		{
		KYTY_PROFILER_BLOCK("Readback: staging wait and priority completion");
		if (Common::AbSwitch::Enabled(Common::AbSwitch::READBACK_COPY_WHEN_FLUSHING)) {
			m_scheduler.FlushUrgent(); // Waited for next: submit on this thread.
		}
		m_scheduler.Wait(tick);
		m_scheduler.WaitPriorityOperations(tick);
		}
		KYTY_PROFILER_BLOCK("Readback: staging publication");
		publish();
	}
	return true;
}

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_bda_pagetable_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                             BDA_PAGETABLE_SIZE),
      m_memory_tracker(page_manager),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, 512 * MiB),
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, 64 * MiB),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 64 * MiB),
      m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
      m_texture_cache(texture_cache) {
	std::memset(m_gds_buffer.Mapped().data(), 0, static_cast<size_t>(m_gds_buffer.Size()));
	m_gds_buffer.Flush(0, m_gds_buffer.Size());
	SetVulkanObjectNameF(m_graphics.device, m_bda_pagetable_buffer.Handle(),
	                     "BDA Page Table Buffer");
	m_scheduler.BeginCommand();
	for (auto* buffer: {&m_bda_pagetable_buffer, m_fault_manager.GetFaultBuffer()}) {
		buffer->Fill(0, buffer->Size(), 0);
	}
	// A host-visible type on a large device-local heap means a resizable BAR; a small (256 MiB) BAR
	// heap could not hold the guest buffers.
	const auto& memory = m_graphics.physical_device_memory_properties;
	const auto* device_only = std::getenv("KYTY_DEVICE_LOCAL_GUEST_BUFFERS");
	const bool force_device_only = device_only != nullptr && std::strcmp(device_only, "1") == 0;
	if (force_device_only) {
		Log::WriteToConsoleAndLog("Buffer cache diagnostic: unmapped device-local guest buffers (BAR path disabled)\n");
		const auto* host_ab = std::getenv("KYTY_HOST_BARRIER_AB");
		if (host_ab != nullptr && std::strcmp(host_ab, "1") == 0) {
			m_scheduler.SetHostReadBarrier(false, true);
			Log::WriteToConsoleAndLog("Host barrier diagnostic: bit 11 adds blanket host visibility; guest buffers stay unmapped\n");
		}
	}
	for (uint32_t i = 0; !force_device_only && i < memory.memoryTypeCount; i++) {
		const auto flags = memory.memoryTypes[i].propertyFlags;
		const auto heap  = memory.memoryHeaps[memory.memoryTypes[i].heapIndex];
		if ((flags & vk::MemoryPropertyFlagBits::eDeviceLocal) &&
		    (flags & vk::MemoryPropertyFlagBits::eHostVisible) &&
		    (flags & vk::MemoryPropertyFlagBits::eHostCoherent) && heap.size >= 4ull * 1024 * MiB) {
			m_guest_usage = MemoryUsage::Guest;
			m_scheduler.SetHostReadBarrier(true);
			break;
		}
	}
	const auto null_id =
	    m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 16);
	EXIT_IF(null_id != NULL_BUFFER_ID);
	SetVulkanObjectNameF(m_graphics.device, GetBuffer(null_id).Handle(), "Kyty.NullBuffer");
	if (!m_graphics.CanReportMemoryUsage()) {
		return;
	}
	constexpr int64_t GiB              = 1024ll * 1024 * 1024;
	constexpr int64_t target_threshold = 8 * GiB;
	const auto        budget =
	    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
	const auto threshold = std::min(budget, target_threshold);
	const auto expected  = std::min(budget - 6 * threshold / 10, budget - GiB);
	const auto critical  = std::min(budget - 2 * threshold / 10, budget - GiB / 2);
	m_trigger_gc_memory  = static_cast<uint64_t>(std::max<int64_t>(expected, GiB));
	m_critical_gc_memory = static_cast<uint64_t>(std::max<int64_t>(critical, 2 * GiB));
}

BufferCache::~BufferCache() {
	if (!m_gpu_modified_ranges.Empty()) {
		EXIT("BufferCache: destroyed with pending GPU-modified ranges\n");
	}
	for (const auto& [vaddr, id]: m_buffers) {
		(void)vaddr;
		const auto& buffer = m_slot_buffers[id];
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: destroyed with GPU-modified buffer\n");
		}
	}
}

void BufferCache::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	m_memory_tracker.InvalidateRegion(vaddr, size,
	                                  [this, vaddr, size] { ReadMemory(vaddr, size, true); });
	QueueBdaSync(vaddr, size);
}

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write) {
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	m_scheduler.Context().GetGpu().SendCommandSync([this, vaddr, size, is_write] {
		if (is_write && !IsRegionRegistered(vaddr, size)) {
			return;
		}
		auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];
		if (IsRegionGpuModified(vaddr, size)) {
			using namespace Common::AbSwitch;
			const uint64_t window = Enabled(DOWNLOAD_WINDOW_4K)      ? 4 * 1024
			                        : Enabled(DOWNLOAD_WINDOW_16K)   ? 16 * 1024
			                        : Enabled(SMALL_DOWNLOAD_WINDOW) ? 64 * 1024
			                                                         : DownloadWindowSize;
			const auto start = std::max(Common::AlignDown(vaddr, window), buffer.CpuAddress());
			const auto end = std::min(std::max(start + window, vaddr + size),
			                          buffer.CpuAddress() + buffer.Size());
			if (DownloadBufferMemory<false>(buffer, start, end - start)) {
				m_memory_tracker.UnmarkRegionAsGpuModified(start, end - start);
			}
		}
		if (is_write) {
			m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
			QueueBdaSync(vaddr, size);
		}
	});
}

BufferId BufferCache::FindBuffer(uint64_t vaddr, uint64_t size) {
	if (vaddr == 0) {
		return NULL_BUFFER_ID;
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid buffer discovery request\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			return *owner;
		}
	}
	return CreateBuffer(vaddr, size);
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(uint64_t vaddr, uint64_t size) {
	static constexpr int      StreamLeapThreshold = 16;
	static constexpr uint64_t StreamLeapSize      = CACHING_PAGESIZE * 128;

	auto       begin      = vaddr;
	auto       end        = vaddr + size;
	const auto find_first = [&](uint64_t address) {
		auto first = m_buffers.lower_bound(address);
		if (first != m_buffers.begin()) {
			const auto  previous = std::prev(first);
			const auto& buffer   = m_slot_buffers[previous->second];
			if (buffer.CpuAddress() + buffer.Size() > address) {
				first = previous;
			}
		}
		return first;
	};
	auto first           = find_first(begin);
	auto last            = first;
	int  stream_score    = 0;
	bool has_stream_leap = false;
	for (; last != m_buffers.end() && last->first < end; ++last) {
		const auto& buffer        = m_slot_buffers[last->second];
		const auto  buffer_begin  = buffer.CpuAddress();
		const auto  buffer_end    = buffer_begin + buffer.Size();
		const bool  expands_left  = buffer_begin < begin;
		const bool  expands_right = buffer_end > end;
		begin                     = std::min(begin, buffer_begin);
		end                       = std::max(end, buffer_end);
		if (!has_stream_leap && (stream_score += buffer.StreamScore()) > StreamLeapThreshold) {
			has_stream_leap = true;
			// Reserve space in the incoming stream's direction of growth.
			// The old buffer extending left of the request predicts growth to the right, and vice
			// versa.
			if (expands_left) {
				end += std::min(StreamLeapSize, (vaddr < LOWER_ADDRESS_SIZE
				                                     ? LOWER_ADDRESS_SIZE
				                                     : LibKernel::Memory::kExtendedMemoryBase +
				                                           LibKernel::Memory::kExtendedMemorySize) -
				                                    end);
			}
			if (expands_right) {
				const auto minimum = vaddr < LOWER_ADDRESS_SIZE
				                         ? CACHING_PAGESIZE * 2
				                         : LibKernel::Memory::kExtendedMemoryBase;
				if (begin > minimum) {
					begin -= std::min(StreamLeapSize, begin - minimum);
				}
				first = find_first(begin);
				begin = std::min(begin, first->first);
			}
		}
	}
	return {first, last, begin, end, has_stream_leap};
}

void BufferCache::JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score) {
	auto& new_buffer = m_slot_buffers[new_id];
	auto& overlap    = m_slot_buffers[overlap_id];
	if (accumulate_stream_score) {
		new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
	}
	new_buffer.CopyFrom(m_scheduler.Current(), overlap, 0,
	                    overlap.CpuAddress() - new_buffer.CpuAddress(), overlap.Size());
	new_buffer.MarkGpuWrite(m_scheduler.CurrentTick());
	NoteRecordedUse(new_buffer);
	RecordGpuWrite(overlap.CpuAddress(), overlap.Size());
	m_merge_count++;
	DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(uint64_t vaddr, uint64_t size) {
	const auto end     = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	vaddr              = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	size               = end - vaddr;
	const auto overlap = ResolveOverlaps(vaddr, size);

	const auto  id = m_slot_buffers.insert(m_graphics, m_scheduler, m_guest_usage, overlap.begin,
	                                       AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress,
	                                       overlap.end - overlap.begin);
	const auto& buffer = m_slot_buffers[id];
	SetVulkanObjectNameF(m_graphics.device, buffer.Handle(),
	                     "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", overlap.begin,
	                     overlap.end - overlap.begin);
	for (auto it = overlap.first; it != overlap.last;) {
		const auto old_id = (it++)->second;
		JoinOverlap(id, old_id, !overlap.has_stream_leap);
	}
	Register(id);
	return id;
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                    bool is_texel_buffer) {
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size = 0;
	m_memory_tracker.ForEachUploadRange(
	    vaddr, size, is_written,
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    copies.emplace_back(total_size, buffer.Offset(address), bytes);
		    total_size += bytes;
	    },
	    [&]() noexcept {
		    const BufferUpload upload {&buffer, copies.size()};
		    UploadCopies({&upload, 1}, copies, total_size);
	    });
	if (is_texel_buffer && !is_written) {
		return SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	return false;
}

void BufferCache::UploadCopies(std::span<const BufferUpload> uploads,
                               std::span<vk::BufferCopy> copies, uint64_t total_size) {
	if (copies.empty()) {
		return;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferUploadBytes, total_size);

	const auto source = UploadMemory(
	    m_graphics, m_scheduler, m_staging_buffer, total_size, 4,
	    vk::BufferUsageFlagBits::eTransferSrc, [&](uint8_t* mapped, uint64_t base_offset) {
		    size_t first = 0;
		    for (const auto& upload: uploads) {
			    for (auto& copy: copies.subspan(first, upload.end - first)) {
				    const auto address = upload.buffer->CpuAddress() + copy.dstOffset;
				    if (!LibKernel::Memory::CopySparseMemory(address, mapped + copy.srcOffset,
					                                         copy.size)) {
					    EXIT("BufferCache: failed to read sparse guest buffer memory\n");
				    }
				    copy.srcOffset += base_offset;
			    }
			    first = upload.end;
		    }
	    });
	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	const vk::MemoryBarrier before {
	    .srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
		                 vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite,
	    .dstAccessMask = vk::AccessFlagBits::eTransferWrite};
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits::eByRegion,
	                       1, &before, 0, nullptr, 0, nullptr);
	size_t first = 0;
	for (const auto& upload: uploads) {
		native.copyBuffer(source.first->Handle(), upload.buffer->Handle(),
		                  static_cast<uint32_t>(upload.end - first), copies.data() + first);
		NoteRecordedUse(*upload.buffer);
		first = upload.end;
	}
	const vk::MemoryBarrier after {.srcAccessMask = vk::AccessFlagBits::eTransferWrite,
	                               .dstAccessMask = vk::AccessFlagBits::eMemoryRead |
	                                                vk::AccessFlagBits::eMemoryWrite};
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands,
	                       vk::DependencyFlagBits::eByRegion, 1, &after, 0, nullptr, 0, nullptr);
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBuffer(uint64_t vaddr, uint64_t size,
                                                       bool is_written, bool is_texel_buffer,
                                                       BufferId id) {
	EXIT_IF(!(GuestRange {vaddr, size}.Valid()));

	// Whether the small-range query below found no CPU-written page (AbSwitch READ_BINDING_FAST_PATH);
	// it creates missing regions, which start CPU-written, so this is IsUploaded at that moment.
	bool known_uploaded = false;
	if (!is_written && size <= CACHING_PAGESIZE) {
		// One pass over the regions for the three questions the fast path used to ask separately.
		const auto region = m_memory_tracker.QueryRegionState(vaddr, size);
		known_uploaded    = !region.cpu_modified;
		if (region.cpu_modified && !region.gpu_modified) {
			// Within one draw window the guest has no signal to rewrite the range. A copy made in this
			// tick at most an eighth of a lap ago is nearly a lap ahead of the ring's writes, like a
			// fresh copy, even if the draw's preparation submits the tick.
			auto& copy = m_stream_copies[(vaddr * 0x9e3779b97f4a7c15ull) >> (64u - STREAM_COPY_BITS)];
			const auto tick = m_scheduler.CurrentTick();
			if (copy.vaddr == vaddr && copy.size == size && copy.window == m_draw_window &&
			    copy.tick == tick &&
			    m_stream_buffer.Position() - copy.position <= m_stream_buffer.Size() / 8) {
				return {&m_stream_buffer, copy.offset};
			}
			const auto alignment = std::max<uint64_t>(
			    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 1);
			auto [mapped, offset] = m_stream_buffer.Map(size, alignment, false);
			if (mapped != nullptr) {
				if (!LibKernel::Memory::CopySparseMemory(vaddr, mapped, size)) {
					EXIT("BufferCache: failed to read sparse guest buffer memory\n");
				}
				m_stream_buffer.Commit();
				copy = {vaddr, size, m_draw_window, tick, offset, m_stream_buffer.Position()};
				return {&m_stream_buffer, offset};
			}
		}
	}

	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	NoteUse(id, buffer);
	if (!is_written && Common::AbSwitch::Enabled(Common::AbSwitch::READ_BINDING_FAST_PATH) &&
	    Common::AbSwitch::Enabled(Common::AbSwitch::UPLOAD_PRECHECK)) {
		// Nothing to upload: SynchronizeBuffer would take UPLOAD_PRECHECK's lock-free exit and then
		// only search for an image starting at the address, which FindImageFromRange rules out on
		// the same start-bucket test.
		if (known_uploaded || m_memory_tracker.IsUploaded(vaddr, size)) {
			const bool image_search =
			    is_texel_buffer &&
			    (!Common::AbSwitch::Enabled(Common::AbSwitch::IMAGE_START_INDEX) ||
			     m_texture_cache.MayStartAt(vaddr));
			Common::DebugCounters::Add(Common::DebugCounters::Counter::ReadFastHits);
			if (g_verify_read_fast_path) [[unlikely]] {
				// The full path as well (so a verify run behaves as with the bit off).
				const bool overtaken    = !m_memory_tracker.IsUploaded(vaddr, size);
				const bool image_synced = SynchronizeBuffer(buffer, vaddr, size, false, is_texel_buffer);
				if (image_synced && !image_search) {
					EXIT("read-fast-path verify: an image at 0x%016" PRIx64 " size 0x%" PRIx64
					     " was synchronized although the fast path skipped the search\n",
					     vaddr, size);
				}
				NoteReadFastPathVerify(overtaken);
			} else if (image_search) {
				Common::DebugCounters::Add(Common::DebugCounters::Counter::ReadFastImageSyncs);
				(void)SynchronizeBufferFromImage(buffer, vaddr, size);
			}
			return {&buffer, buffer.Offset(vaddr)};
		}
		Common::DebugCounters::Add(Common::DebugCounters::Counter::ReadFastMisses);
	}
	(void)SynchronizeBuffer(buffer, vaddr, size, is_written, is_texel_buffer);
	if (is_written) {
		// The whole bound range counts as GPU-written, whatever the shader stores to.
		Common::DebugCounters::Add(Common::DebugCounters::Counter::WrittenBindings);
		Common::DebugCounters::Add(Common::DebugCounters::Counter::WrittenBindingBytes, size);
		TraceWrittenBinding(vaddr, size);
		if (LibKernel::Memory::SrtDirtyStatsEnabled()) [[unlikely]] {
			NoteWrittenBindingStats(diag_writer, size);
		}
		m_gpu_modified_ranges.Add(vaddr, size);
		buffer.MarkGpuWrite(m_scheduler.CurrentTick());
		RecordGpuWrite(vaddr, size);
		ForgetColorMetadataChecks(vaddr, size);
		if (buffer.WaitedForReadback()) {
			// Submit this work as soon as it is recorded: a later read of what it writes will
			// then find it done or running instead of waiting for it from submission.
			m_scheduler.RequestFlushAfterRecord();
		}
	}
	return {&buffer, buffer.Offset(vaddr)};
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size) {
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			TouchBuffer(buffer);
			NoteUse(*owner, buffer);
			(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
			return {&buffer, buffer.Offset(vaddr)};
		}
	}
	if (IsRegionGpuModified(vaddr, size)) {
		return ObtainBuffer(vaddr, size, false, false);
	}

	return UploadMemory(
	    m_graphics, m_scheduler, m_staging_buffer, size, 16,
	    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
	    [&](uint8_t* mapped, uint64_t) {
		    if (!Libs::LibKernel::Memory::TryReadSparseBacking(vaddr, mapped, size)) {
			    EXIT("BufferCache: failed to read mapped guest image backing\n");
		    }
	    });
}

void BufferCache::FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds) {
	if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 || size > UINT64_MAX - vaddr) {
		EXIT("BufferCache: fill range must be dword aligned\n");
	}
	if (is_gds) {
		if (vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - vaddr) {
			EXIT("BufferCache: GDS fill range is out of bounds\n");
		}
		m_gds_buffer.Fill(vaddr, size, value);
		return;
	}
	if (vaddr == 0) {
		EXIT("BufferCache: invalid fill memory address\n");
	}
	(void)m_texture_cache.ClearMeta(vaddr);
	if (!IsRegionGpuModified(vaddr, size)) {
		// Access the guest mapping so write faults invalidate cached buffers and images.
		auto* destination = reinterpret_cast<uint32_t*>(vaddr);
		std::fill(destination, destination + size / sizeof(uint32_t), value);
		return;
	}

	m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
	auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true);
	dst->Fill(dst_offset, size, value);
}

void BufferCache::CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
                             bool src_gds) {
	const bool dst_memory = !dst_gds;
	const bool src_memory = !src_gds;
	if ((dst_memory && dst_vaddr == 0) || (src_memory && src_vaddr == 0) || size == 0 ||
	    ((dst_gds || src_gds) && ((dst_vaddr | src_vaddr | size) & 3u) != 0) ||
	    size > UINT64_MAX - dst_vaddr || size > UINT64_MAX - src_vaddr || (dst_gds && src_gds) ||
	    (dst_gds && (dst_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - dst_vaddr)) ||
	    (src_gds && (src_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - src_vaddr))) {
		EXIT("BufferCache: invalid copy range, src=0x%016" PRIx64 " dst=0x%016" PRIx64
		     " size=0x%016" PRIx64 " src_gds=%d dst_gds=%d\n",
		     src_vaddr, dst_vaddr, size, static_cast<int>(src_gds), static_cast<int>(dst_gds));
	}
	// A copy into whole levels of a GPU-written image lands on its GPU contents, not stale guest
	// bytes, while no buffer holds the range.
	if (src_memory && dst_memory && !IsRegionRegistered(dst_vaddr, size) &&
	    m_texture_cache.CopyToImage(dst_vaddr, src_vaddr, size)) {
		return;
	}
	if (src_memory && dst_memory && !IsRegionGpuModified(dst_vaddr, size) &&
	    !IsRegionGpuModified(src_vaddr, size) &&
	    !m_texture_cache.FindImageFromRange(src_vaddr, size)) {
		std::memcpy(reinterpret_cast<void*>(dst_vaddr), reinterpret_cast<const void*>(src_vaddr),
		            size);
		return;
	}

	auto& command = m_scheduler.Current();
	if (dst_memory) {
		m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size);
	}
	const auto src_id      = src_memory ? FindBuffer(src_vaddr, size) : BufferId {};
	const auto dst_id      = dst_memory ? FindBuffer(dst_vaddr, size) : BufferId {};
	auto [src, src_offset] = src_memory ? ObtainBuffer(src_vaddr, size, false, true, src_id)
	                                    : std::pair {&m_gds_buffer, src_vaddr};
	auto [dst, dst_offset] = dst_memory ? ObtainBuffer(dst_vaddr, size, true, true, dst_id)
	                                    : std::pair {&m_gds_buffer, dst_vaddr};
	dst->CopyFrom(command, *src, src_offset, dst_offset, size);
}

bool BufferCache::IsRegionRegistered(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid registered-region query\n");
	}
	// Cached buffers are ordered and non-overlapping. The last buffer beginning before the query
	// end is therefore the only possible intersection.
	const auto candidate = m_buffers.lower_bound(vaddr + size);
	if (candidate == m_buffers.begin()) {
		return false;
	}
	const auto& [address, id] = *std::prev(candidate);
	return address + m_slot_buffers[id].Size() > vaddr;
}

void BufferCache::MarkColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) {
	for (uint32_t slot = 0; slot < m_checked_color_metadata.size(); slot++) {
		if ((keys & (1u << slot)) != 0) {
			m_checked_color_metadata[slot].Add(vaddr, size);
		}
	}
}

bool BufferCache::IsColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) const {
	for (uint32_t slot = 0; slot < m_checked_color_metadata.size(); slot++) {
		if ((keys & (1u << slot)) != 0 && !m_checked_color_metadata[slot].Contains(vaddr, size)) {
			return false;
		}
	}
	return true;
}

void BufferCache::ForgetColorMetadataChecks(uint64_t vaddr, uint64_t size) {
	for (auto& checked: m_checked_color_metadata) {
		checked.Subtract(vaddr, size);
	}
}

bool BufferCache::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionGpuModified(vaddr, size);
}

bool BufferCache::HasGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	return m_gpu_modified_ranges.Intersects(vaddr, size);
}

bool BufferCache::WriteLabelInOrder(uint64_t vaddr, const void* value, uint64_t size) {
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner == nullptr || !*owner || !m_slot_buffers[*owner].IsInBounds(vaddr, size) ||
	    m_texture_cache.HasPendingBackingPublish() ||
	    !LibKernel::Memory::TryWriteBacking(vaddr, value, size)) {
		return false;
	}
	WriteDataBuffer(m_slot_buffers[*owner], vaddr, value, size);
	m_gpu_modified_ranges.Subtract(vaddr, size);
	ForgetColorMetadataChecks(vaddr, size);
	// A GPU-modified tracker page must keep GPU-written bytes, or a read of it faults forever.
	const auto first = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto last  = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	for (auto page = first; page < last; page += TRACKER_PAGE_SIZE) {
		if (!m_gpu_modified_ranges.Intersects(page, TRACKER_PAGE_SIZE)) {
			m_memory_tracker.UnmarkRegionAsGpuModified(page, TRACKER_PAGE_SIZE);
		}
	}
	return true;
}

bool BufferCache::HasOnlyGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	return m_gpu_modified_ranges.Contains(vaddr, size);
}

bool BufferCache::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

void BufferCache::RunGarbageCollector() {
	const auto tick = m_gc_tick++;
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	// AbSwitch BUFFER_IDLE_GC: with an idle age the collector runs below the memory trigger too.
	// Under the trigger the sweep skips GPU-dirty buffers, so only clean, unused ones are freed;
	// one still needed is created again.
	static const uint64_t idle_frames = [] {
		const char* value = std::getenv("KYTY_BUFFER_IDLE_FRAMES");
		return value != nullptr ? std::strtoull(value, nullptr, 10) : uint64_t {60};
	}();
	const bool idle_collect =
	    Common::AbSwitch::Enabled(Common::AbSwitch::BUFFER_IDLE_GC) && idle_frames != 0;
	if (m_total_used_memory < m_trigger_gc_memory && !idle_collect) {
		return;
	}

	const bool     aggressive = m_total_used_memory >= m_critical_gc_memory;
	const uint64_t age =
	    idle_collect && !aggressive ? idle_frames : std::min<uint64_t>(aggressive ? 80 : 160, tick);
	const size_t limit = aggressive ? 64 : 32;

	std::vector<BufferId> dirty_buffers;
	size_t                retire_count = 0;
	m_lru_cache.ForEachItemBelow(tick - age, [&](BufferId id) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges, buffer.CpuAddress(),
		                                           buffer.Size(), "garbage collection");
		const bool dirty = m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size());
		if (dirty && !aggressive) {
			return false;
		}
		if (dirty) {
			EXIT_NOT_IMPLEMENTED(
			    !DownloadBufferMemory<true>(buffer, buffer.CpuAddress(), buffer.Size()));
			dirty_buffers.push_back(id);
		} else {
			m_memory_tracker.MarkRegionAsCpuModified(buffer.CpuAddress(), buffer.Size());
			DeleteBuffer(id);
			Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferEvictions);
		}
		return ++retire_count == limit;
	});
	if (dirty_buffers.empty()) {
		return;
	}

	// Publish all queued downloads before releasing their tracked pages and owners.
	const auto completion_tick = m_scheduler.CurrentTick();
	m_scheduler.Wait(completion_tick);
	m_scheduler.WaitPriorityOperations(completion_tick);
	for (const auto id: dirty_buffers) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
		m_memory_tracker.MarkRegionAsCpuModified(buffer.CpuAddress(), buffer.Size());
		Unregister(id);
		m_slot_buffers.erase(id);
	}
}

void BufferCache::ProcessFaultBuffer() {
	m_fault_manager.ProcessFaultBuffer();
}

void BufferCache::QueueBdaSync(uint64_t vaddr, uint64_t size, bool is_write) {
	const auto begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	std::lock_guard lock(m_bda_sync_mutex);
	(void)is_write; // All CPU writes must be synchronized, including first BDA access.
	m_bda_sync.Add(begin, end - begin);
}

void BufferCache::SynchronizeBda(const RangeSet& mapped) {
	const auto queued = [this] {
		std::lock_guard lock(m_bda_sync_mutex);
		return std::exchange(m_bda_sync, {});
	}();
	std::vector<vk::BufferCopy> copies;
	std::vector<BufferUpload>   uploads;
	uint64_t                    total_size = 0;
	queued.ForEach([&](uint64_t begin, uint64_t end) {
		mapped.ForEachInRange(begin, end - begin, [&](uint64_t start, uint64_t finish) {
			auto it = m_buffers.upper_bound(start);
			if (it != m_buffers.begin()) {
				--it;
			}
			for (; it != m_buffers.end() && it->first < finish; ++it) {
				auto&      buffer = m_slot_buffers[it->second];
				const auto first  = std::max(buffer.CpuAddress(), start);
				const auto last   = std::min(buffer.CpuAddress() + buffer.Size(), finish);
				if (first < last) {
					// Reads follow every range's protection; a write in between re-queues its page.
					m_memory_tracker.ForEachUploadRange(
					    first, last - first, false,
					    [&](uint64_t address, uint64_t bytes) noexcept {
						    if (uploads.empty() || uploads.back().buffer != &buffer) {
							    uploads.push_back({&buffer, 0});
						    }
						    copies.emplace_back(total_size, buffer.Offset(address), bytes);
						    uploads.back().end = copies.size();
						    total_size += bytes;
					    },
					    []() noexcept {});
				}
			}
		});
	});
	UploadCopies(uploads, copies, total_size);
}

} // namespace Libs::Graphics
