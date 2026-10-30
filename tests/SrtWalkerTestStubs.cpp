// The resource tracking and materialization tests link the SRT walker (SrtWalker.cpp) without the
// emulator's guest memory. The walker's page-table fast path reads a PageAccessLookup whose
// `regions` stays null here, so generated code always takes its ordinary load, and the slow path
// (SrtWalkerReadGuestDword) is never reached. These stubs satisfy the link; keep them in step with
// kernel/memory.h and graphics/host_gpu/pageManager.h.
#include "graphics/host_gpu/pageManager.h"
#include "kernel/memory.h"

namespace Libs::Graphics {

PageAccessLookup* PageManager::AccessLookup() noexcept {
	static PageAccessLookup empty;
	return &empty;
}

} // namespace Libs::Graphics

namespace Libs::LibKernel::Memory {

bool TryReadBacking(uint64_t /*vaddr*/, void* /*data*/, uint64_t /*size*/) {
	return false;
}

bool HandleGpuFault(Graphics::PageFaultAccess /*access*/, uint64_t /*fault_vaddr*/) noexcept {
	return false;
}

bool TryReadGpuCleanBytes(uint64_t /*vaddr*/, void* /*data*/, uint64_t /*size*/) noexcept {
	return false;
}

void TraceGpuDirtyRead(const char* /*what*/, uint64_t /*vaddr*/) noexcept {}

} // namespace Libs::LibKernel::Memory
