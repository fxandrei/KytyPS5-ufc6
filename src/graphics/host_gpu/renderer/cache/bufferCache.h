#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <array>
#include <map>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class TextureCache;

using BufferId = Common::SlotId;
inline constexpr BufferId NULL_BUFFER_ID {0};

class BufferCache {
public:
	static constexpr uint32_t CACHING_PAGEBITS  = 14;
	static constexpr uint64_t CACHING_PAGESIZE  = uint64_t {1} << CACHING_PAGEBITS;
	static constexpr uint64_t CACHING_NUMPAGES  = (LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemorySize) >> CACHING_PAGEBITS;
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    CACHING_NUMPAGES * sizeof(vk::DeviceAddress);
	// Tags the BDA page-table entries of pages a DMA shader accessed; an access to another page
	// records a fault.
	static constexpr vk::DeviceAddress BDA_ACCESSED = 1;

	static constexpr uint64_t PageIndex(uint64_t address) {
		return (address < LOWER_ADDRESS_SIZE
		            ? address
		            : address - LibKernel::Memory::kExtendedMemoryBase + LOWER_ADDRESS_SIZE) >>
		       CACHING_PAGEBITS;
	}
	static constexpr uint64_t GuestAddress(uint64_t offset) {
		return offset < LOWER_ADDRESS_SIZE
		           ? offset
		           : offset - LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemoryBase;
	}

	BufferCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	            TextureCache& texture_cache);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
	void                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false);
	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size);
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer = false,
	                                                        BufferId id              = {});
	[[nodiscard]] StreamBuffer&                GetUtilityBuffer(MemoryUsage usage) noexcept {
		switch (usage) {
			case MemoryUsage::Upload: return m_staging_buffer;
			case MemoryUsage::Stream: return m_stream_buffer;
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return m_device_buffer;
		}
		EXIT("BufferCache: invalid utility-buffer usage\n");
	}
	[[nodiscard]] const Buffer* GetGdsBuffer() const noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept { return &m_bda_pagetable_buffer; }
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// Cache-index and exact dirty-range queries require GPU-thread serialization.
	[[nodiscard]] bool IsRegionRegistered(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasOnlyGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	// A command-processor label over GPU-written bytes: writes it to guest memory and, after the
	// work recorded so far, to the cached copy, so the GPU bytes under it need no download.
	// False when it cannot; then store through the guest mapping.
	[[nodiscard]] bool WriteLabelInOrder(uint64_t vaddr, const void* value, uint64_t size);
	// The GPU-dirty range holding `vaddr` as {begin, end}, or {0, 0}. For diagnostics.
	[[nodiscard]] std::pair<uint64_t, uint64_t> GpuDirtyRangeAt(uint64_t vaddr) const {
		return m_gpu_modified_ranges.Containing(vaddr);
	}
	// KYTY_SRT_DIRTY_STATS diagnostics (2026-11-19): the newest recorded GPU write that covers
	// `vaddr` (end == 0 when none), tagged with the shader whose bindings were being prepared.
	struct CoveringWrite {
		uint64_t begin  = 0;
		uint64_t end    = 0;
		uint64_t tick   = 0;
		uint64_t writer = 0;
	};
	[[nodiscard]] CoveringWrite NewestCoveringWrite(uint64_t vaddr) const;
	// True when a recorded GPU write the GPU has not completed yet (as far as the CPU knows) may
	// cover `vaddr`, conservatively true when such a write left the ring (AbSwitch
	// SRT_OPTIONAL_DIRTY_UNKNOWN). GPU thread only.
	[[nodiscard]] bool HasInFlightWriteAt(uint64_t vaddr) const;
	// KYTY_SRT_DIRTY_STATS diagnostics: the newest in-flight write tick covering [vaddr, vaddr + size)
	// and the newest overlapping [window_begin, window_end), 0 when none; writer of the first.
	struct InFlightTicks {
		uint64_t exact = 0, window = 0, writer = 0, known = 0;
	};
	[[nodiscard]] InFlightTicks DiagInFlightTicks(uint64_t vaddr, uint64_t size,
	                                              uint64_t window_begin, uint64_t window_end) const;
	// The tag of later GPU writes: the shader hash RebindBuffers prepares, else 0.
	static inline uint64_t diag_writer = 0;
	// Color metadata the GPU found free of the clear keys in `keys` (bits of code >> 5) stays so
	// until a GPU write or download.
	void               MarkColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys);
	[[nodiscard]] bool IsColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) const;
	void               ProcessFaultBuffer();
	// Tags fault-scan pages as DMA-accessed and queues them; creates buffers for unbacked ones.
	void MarkDmaAccess(const RangeSet& pages);
	// Uploads the CPU-dirty bytes of the queued ranges `mapped` covers behind one barrier pair.
	void SynchronizeBda(const RangeSet& mapped);
	// Queue after the range's state change becomes visible, or a drain could consume it first.
	void                   QueueBdaSync(uint64_t vaddr, uint64_t size, bool is_write = false);
	// Every GPU write recorded since the last call now carries the current tick: a draw's or
	// dispatch's written bindings are marked during its preparation, and a flush there moves the
	// command into a later command buffer. Called right after each draw or dispatch is recorded.
	void RetagGpuWrites();
	// KYTY_UNMAP_TARGETED (2026-11-21): a program that may address guest buffers through BDA is
	// being prepared; it may use every registered buffer.
	void NoteBdaProgram() noexcept { m_bda_pending = m_track_uses; }
	// Use ticks are kept only for the unmap rule=use and the unmap verify mode.
	void SetUseTracking(bool enabled) noexcept { m_track_uses = enabled; }
	// An upper bound of the newest tick at which recorded GPU work may use a cached buffer
	// overlapping [vaddr, vaddr + size) (bindings, uploads, copies, BDA programs), or 0 when no
	// cached buffer overlaps it (the current tick when an overlapping buffer is cached and use
	// tracking is off). GPU thread only. Cached buffers are separate allocations, so this
	// work never touches guest memory; the unmap rule=use and its verify mode use it.
	[[nodiscard]] uint64_t NewestUseTick(uint64_t vaddr, uint64_t size);
	static constexpr uint64_t PENDING_USE = UINT64_MAX;
	// Bumped whenever JoinOverlap merges a buffer away (and deletes it).
	[[nodiscard]] uint64_t MergeCount() const { return m_merge_count; }
	void               RunGarbageCollector();

private:
	friend struct BufferCacheTestAccess;

	bool IsBufferInvalid(BufferId id) const { return !id || m_slot_buffers[id].is_deleted; }

	using BufferMap = std::map<uint64_t, BufferId>;
	struct OverlapResult {
		BufferMap::iterator first;
		BufferMap::iterator last;
		uint64_t            begin;
		uint64_t            end;
		bool                has_stream_leap;
	};

	// Copies [previous upload's end, end) go to `buffer`.
	struct BufferUpload {
		Buffer* buffer;
		size_t  end;
	};

	using PageTable = MultiLevelPageTable<BufferId, CACHING_PAGEBITS, 44, 20>;
	static_assert(CACHING_PAGESIZE == (uint64_t {1} << PageTable::kPageBits));
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size);
	void TouchBuffer(Buffer& buffer);
	// A use by the draw, dispatch or command being prepared: its tick is known once RetagGpuWrites
	// runs after it is recorded (a flush during the preparation moves it to a later tick).
	void NoteUse(BufferId id, Buffer& buffer) {
		if (m_track_uses && buffer.use_tick != PENDING_USE) {
			buffer.use_tick = PENDING_USE;
			m_pending_uses.push_back(id);
		}
	}
	// A command recorded right now into the current command buffer uses the buffer.
	void NoteRecordedUse(Buffer& buffer);
	// A GPU write into [vaddr, vaddr + size) at the current tick (AbSwitch RANGE_WRITE_TICKS).
	void RecordGpuWrite(uint64_t vaddr, uint64_t size);
	// The tick a readback of `copies` must wait for: the newest write in flight that covers them.
	[[nodiscard]] uint64_t      ReadbackWriteTick(const Buffer& buffer, uint64_t buffer_address,
	                                              std::span<const vk::BufferCopy> copies);
	[[nodiscard]] OverlapResult ResolveOverlaps(uint64_t vaddr, uint64_t size);
	void JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);
	void                   Register(BufferId id);
	void Unregister(BufferId id);
	template <bool insert>
	void ChangeRegister(BufferId id);
	void DeleteBuffer(BufferId id);
	[[nodiscard]] bool SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                     bool is_written, bool is_texel_buffer);
	void UploadCopies(std::span<const BufferUpload> uploads, std::span<vk::BufferCopy> copies,
	                  uint64_t total_size);
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// Synchronous downloads publish before returning; asynchronous callers wait before reuse.
	template <bool async>
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size);
	void               ForgetColorMetadataChecks(uint64_t vaddr, uint64_t size);

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	FaultManager                                      m_fault_manager;
	Buffer                                            m_gds_buffer;
	Buffer                                            m_bda_pagetable_buffer;
	Common::SlotVector<Buffer>                        m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                         m_buffers;
	PageTable                                         m_page_table;
	RangeSet                                          m_gpu_modified_ranges;
	// One per TileManager::ColorClearSlots key slot.
	std::array<RangeSet, 8>                           m_checked_color_metadata;
	MemoryTracker                                     m_memory_tracker;
	std::mutex                                        m_bda_sync_mutex;
	// Guest buffers are mapped through the resizable BAR when the device-local heap is host-visible
	// (AbSwitch DIRECT_READBACK reads them in place).
	MemoryUsage                                       m_guest_usage = MemoryUsage::DeviceLocal;
	RangeSet                                          m_bda_sync;
	// Pages DMA shaders accessed, guarded by m_bda_sync_mutex.
	RangeSet                                          m_dma_accessed;
	StreamBuffer                                      m_staging_buffer;
	StreamBuffer                                      m_stream_buffer;
	StreamBuffer                                      m_download_buffer;
	StreamBuffer                                      m_device_buffer;
	TextureCache&                                     m_texture_cache;
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	uint64_t m_gc_tick            = 0;
	// The latest GPU writes by guest range, oldest overwritten: a readback waits only for writes
	// that cover its bytes, not for the newest write anywhere in a merged buffer. Ticks never
	// decrease from oldest to newest: entries are recorded at the current tick, and RetagGpuWrites
	// raises every entry since the previous call to the same tick.
	struct GpuWrite {
		uint64_t begin  = 0;
		uint64_t end    = 0;
		uint64_t tick   = 0;
		uint64_t writer = 0; // diag_writer when recorded.
	};
	static constexpr size_t GPU_WRITE_RING   = 2048;
	std::vector<GpuWrite>   m_gpu_writes     = std::vector<GpuWrite>(GPU_WRITE_RING);
	size_t                  m_next_gpu_write = 0;
	// The newest tick of a write that left the ring.
	uint64_t m_evicted_write_tick = 0;
	// The first entry not re-tagged yet.
	size_t   m_first_unretagged = 0;
	uint64_t m_merge_count      = 0;
	// KYTY_UNMAP_TARGETED: buffers whose use_tick is PENDING_USE, retagged by RetagGpuWrites.
	std::vector<BufferId> m_pending_uses;
	// The tick of the newest recorded BDA program; m_bda_pending while one is being prepared.
	uint64_t m_bda_tick    = 0;
	bool     m_bda_pending = false;
	bool     m_track_uses  = false;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
