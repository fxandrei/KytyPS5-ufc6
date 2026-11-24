#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_STREAMLINE_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_STREAMLINE_H_

#include "graphics/host_gpu/vulkanCommon.h"

// NVIDIA Streamline (DLSS Frame Generation) integration, experimental (2026-11-20).
//
// KYTY_DLSSG=1 loads sl.interposer.dll from KYTY_STREAMLINE_PATH (default: the folder of the
// executable), calls slInit for DLSS-G, Reflex and PCL before SDL loads Vulkan, and makes SDL load
// Vulkan through the interposer (SDL_VULKAN_LIBRARY), so instance, device and swapchain creation go
// through Streamline's proxies. After device creation, Probe reports whether DLSS-G is supported.
// NGX needs a CUDA address window below 1 TiB that the guest arena normally fills: run with
// KYTY_CUDA_HOLE=1 (kernel/memoryAddressSpace.inc). Builds without KYTY_STREAMLINE_DIR (CMake)
// compile this to no-ops.
namespace Libs::Graphics::Streamline {

// Before SDL_CreateWindow. True when Streamline is initialized and SDL will load the interposer.
bool InitBeforeVulkan();
// After the Vulkan device exists: prints DLSS-G support and requirements.
void Probe(VkPhysicalDevice physical_device);
[[nodiscard]] bool Active();
void Shutdown();

// FG-3 (DLSS-G per frame). A native image to tag: Vulkan handles, format, size and the layout it is
// in at the point of the tag (in command order).
struct TagImage {
	VkImage     image  = VK_NULL_HANDLE;
	VkImageView view   = VK_NULL_HANDLE;
	uint32_t    format = 0;
	uint32_t    width  = 0;
	uint32_t    height = 0;
	uint32_t    layout = 0;
};
// The game's temporal AA pass (KYTY_DLSSG_TAA, hex; default UFC 6's c3af833d5e36e33c) and its
// motion-vector binding (KYTY_DLSSG_MVEC_BINDING, default 1).
[[nodiscard]] uint64_t TaaShaderHash();
[[nodiscard]] uint32_t MotionVectorBinding();
// GPU thread, in command order before the TAA dispatch: starts a frame and tags depth and motion
// vectors (copied by Streamline at that point). Runs on the record thread when recording is deferred.
// Per-frame camera data read from the game's TAA constant buffer (agents/2026-11-20/dlssg-constants.md):
// jitter in render pixels (cb[28..29]), history reset (cb[39] == 0), near/far from the depth
// linearization viewZ = cb[1] / (d - cb[0]), render size cb[4..5]. valid=false keeps the defaults.
struct FrameConstants {
	float    jitter_x = 0.0f;
	float    jitter_y = 0.0f;
	float    near_z   = 0.1f;
	float    far_z    = 1000.0f;
	uint32_t width    = 0;
	uint32_t height   = 0;
	bool     reset    = false;
	bool     valid    = false;
};
void TagFrame(VkCommandBuffer command, const TagImage& motion, const TagImage& depth,
              const FrameConstants& constants);
// Present thread, around vkQueuePresentKHR (which Streamline intercepts).
void BeforePresent(uint32_t width, uint32_t height, uint32_t format);
void AfterPresent();

} // namespace Libs::Graphics::Streamline

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_STREAMLINE_H_
