#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_

#include "common/common.h"
#include "graphics/host_gpu/regionDefinitions.h"

#include <memory>

namespace Libs::Graphics {

enum class PageFaultAccess { Read, Write, Execute, Unknown };

// Where generated code finds whether a page faults on reads (it has an access watcher): the byte
// at regions[address >> region_shift] + pages_offset + ((address >> page_shift) & page_mask),
// tested with access_mask, when the region pointer is not null and the index is in range.
// Code reads `regions` when it runs: it is null until a PageManager exists.
struct PageAccessLookup {
	const void* volatile regions = nullptr;
	uint64_t region_count        = 0;
	uint32_t region_shift        = 0;
	uint32_t page_shift          = 0;
	uint32_t page_mask           = 0;
	uint32_t pages_offset        = 0;
	uint8_t  access_mask         = 0;
};

class PageManager final {
public:
	PageManager();
	// The owner must stop all PageManager callers before destruction.
	~PageManager();

	KYTY_CLASS_NO_COPY(PageManager);

	// The page table layout, with the regions of the live PageManager.
	[[nodiscard]] static PageAccessLookup* AccessLookup() noexcept;

	[[nodiscard]] uint64_t GetPageSize() const;

	template <bool track>
	void UpdatePageWatchers(uint64_t vaddr, uint64_t size);
	template <bool track, bool is_read = false>
	void UpdatePageWatchersForRegion(uint64_t base_addr, RegionBits& mask);

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
