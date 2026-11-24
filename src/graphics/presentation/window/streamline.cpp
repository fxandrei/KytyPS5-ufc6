#include "graphics/presentation/window/streamline.h"

#include "common/common.h"

#include <cstdio>
#include <cstdlib>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS && defined(KYTY_STREAMLINE)

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep

#include <SDL3/SDL.h>
#include <sl.h>
#include <sl_dlss_g.h>
#include <sl_pcl.h>
#include <sl_reflex.h>

#include "graphics/host_gpu/renderer/commandHooks.h"

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>

namespace Libs::Graphics::Streamline {

namespace {

HMODULE                         g_interposer = nullptr;
PFun_slInit*                    g_init       = nullptr;
PFun_slShutdown*                g_shutdown   = nullptr;
PFun_slIsFeatureSupported*      g_supported  = nullptr;
PFun_slIsFeatureLoaded*         g_loaded     = nullptr;
PFun_slGetFeatureRequirements*  g_requirements = nullptr;
bool                            g_active     = false;
PFun_slGetFeatureFunction*      g_feature_function = nullptr;
PFun_slGetNewFrameToken*        g_new_token  = nullptr;
PFun_slSetTagForFrame*          g_set_tag    = nullptr;
PFun_slSetConstants*            g_set_constants = nullptr;
PFun_slPCLSetMarker*            g_pcl_marker = nullptr;
PFun_slReflexSetOptions*        g_reflex_options = nullptr;
PFun_slDLSSGSetOptions*         g_dlssg_options = nullptr;
PFun_slDLSSGGetState*           g_dlssg_state = nullptr;
std::atomic<sl::FrameToken*>    g_token {nullptr};
std::atomic<uint32_t>           g_frame_index {0};
// The game's render size from the TAA constants (dynamic resolution), for DLSSGOptions.
std::atomic<uint32_t>           g_render_width {0};
std::atomic<uint32_t>           g_render_height {0};
uint32_t                        g_options_dyn_width = 0;
bool                            g_options_set = false;
uint64_t                        g_presents    = 0;
std::wstring                    g_plugin_path;
std::wstring                    g_log_path;

bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DLSSG");
		return value != nullptr && *value == '1';
	}();
	return enabled;
}

std::wstring Folder() {
	if (const char* value = std::getenv("KYTY_STREAMLINE_PATH"); value != nullptr && *value != 0) {
		std::wstring path(value, value + std::strlen(value));
		while (!path.empty() && (path.back() == L'\\' || path.back() == L'/')) path.pop_back();
		return path;
	}
	wchar_t exe[MAX_PATH] {};
	GetModuleFileNameW(nullptr, exe, MAX_PATH);
	std::wstring path = exe;
	const auto   slash = path.find_last_of(L"\\/");
	return slash == std::wstring::npos ? std::wstring {L"."} : path.substr(0, slash);
}

void LogCallback(sl::LogType type, const char* message) {
	std::printf("streamline[%u]: %s", static_cast<uint32_t>(type), message != nullptr ? message : "");
	std::fflush(stdout);
}

const char* ResultName(sl::Result result) {
	switch (result) {
		case sl::Result::eOk: return "ok";
		case sl::Result::eErrorDriverOutOfDate: return "driver out of date";
		case sl::Result::eErrorOSOutOfDate: return "OS out of date";
		case sl::Result::eErrorOSDisabledHWS: return "hardware scheduling disabled";
		case sl::Result::eErrorAdapterNotSupported: return "adapter not supported";
		case sl::Result::eErrorNoSupportedAdapterFound: return "no supported adapter";
		case sl::Result::eErrorNoPlugins: return "no plugins";
		case sl::Result::eErrorVulkanAPI: return "Vulkan API error";
		case sl::Result::eErrorNGXFailed: return "NGX failed";
		case sl::Result::eErrorNotInitialized: return "not initialized";
		case sl::Result::eErrorFeatureMissing: return "feature missing";
		case sl::Result::eErrorFeatureNotSupported: return "feature not supported";
		case sl::Result::eErrorFeatureFailedToLoad: return "feature failed to load";
		case sl::Result::eErrorMissingProxy: return "missing proxy";
		default: return "other error";
	}
}

} // namespace

bool InitBeforeVulkan() {
	if (!Enabled() || g_active) return g_active;
	const auto folder      = Folder();
	const auto interposer  = folder + L"\\sl.interposer.dll";
	g_interposer           = LoadLibraryW(interposer.c_str());
	if (g_interposer == nullptr) {
		std::printf("streamline: cannot load sl.interposer.dll from the Streamline folder (error %lu)\n",
		            GetLastError());
		std::fflush(stdout);
		return false;
	}
	g_init         = reinterpret_cast<PFun_slInit*>(GetProcAddress(g_interposer, "slInit"));
	g_shutdown     = reinterpret_cast<PFun_slShutdown*>(GetProcAddress(g_interposer, "slShutdown"));
	g_supported    = reinterpret_cast<PFun_slIsFeatureSupported*>(
        GetProcAddress(g_interposer, "slIsFeatureSupported"));
	g_loaded       = reinterpret_cast<PFun_slIsFeatureLoaded*>(
        GetProcAddress(g_interposer, "slIsFeatureLoaded"));
	g_requirements = reinterpret_cast<PFun_slGetFeatureRequirements*>(
	    GetProcAddress(g_interposer, "slGetFeatureRequirements"));
	g_feature_function = reinterpret_cast<PFun_slGetFeatureFunction*>(
	    GetProcAddress(g_interposer, "slGetFeatureFunction"));
	g_new_token     = reinterpret_cast<PFun_slGetNewFrameToken*>(
        GetProcAddress(g_interposer, "slGetNewFrameToken"));
	g_set_tag       = reinterpret_cast<PFun_slSetTagForFrame*>(
        GetProcAddress(g_interposer, "slSetTagForFrame"));
	g_set_constants = reinterpret_cast<PFun_slSetConstants*>(
	    GetProcAddress(g_interposer, "slSetConstants"));
	if (g_init == nullptr || g_supported == nullptr || g_requirements == nullptr ||
	    g_feature_function == nullptr || g_new_token == nullptr || g_set_tag == nullptr ||
	    g_set_constants == nullptr) {
		std::printf("streamline: sl.interposer.dll lacks the expected exports\n");
		std::fflush(stdout);
		return false;
	}

	static const sl::Feature features[] = {sl::kFeatureDLSS_G, sl::kFeatureReflex, sl::kFeaturePCL};
	g_plugin_path                      = folder;
	g_log_path                         = folder;
	const wchar_t*  plugin_paths[]     = {g_plugin_path.c_str()};
	sl::Preferences preferences {};
	preferences.showConsole        = false;
	preferences.logLevel           = sl::LogLevel::eVerbose;
	preferences.pathsToPlugins     = plugin_paths;
	preferences.numPathsToPlugins  = 1;
	preferences.pathToLogsAndData  = g_log_path.c_str();
	preferences.logMessageCallback = LogCallback;
	// No over-the-air plugins: only the DLLs next to the interposer are used.
	preferences.flags              = sl::PreferenceFlags::eDisableCLStateTracking | sl::PreferenceFlags::eUseFrameBasedResourceTagging;
	preferences.featuresToLoad     = features;
	preferences.numFeaturesToLoad  = static_cast<uint32_t>(std::size(features));
	preferences.engine             = sl::EngineType::eCustom;
	preferences.engineVersion      = "KytyPS5";
	preferences.projectId          = "6f8c9a42-3c5e-4b1d-9a7e-2d4b8c1f0e53";
	preferences.renderAPI          = sl::RenderAPI::eVulkan;
	const auto result              = g_init(preferences, sl::kSDKVersion);
	std::printf("streamline: slInit -> %d (%s)\n", static_cast<int>(result), ResultName(result));
	std::fflush(stdout);
	if (result != sl::Result::eOk) return false;

	// SDL loads Vulkan while creating the window: give it the interposer instead of vulkan-1.dll.
	const std::string narrow(interposer.begin(), interposer.end());
	SDL_setenv_unsafe("SDL_VULKAN_LIBRARY", narrow.c_str(), 1);
	g_active = true;
	return true;
}

void Probe(VkPhysicalDevice physical_device) {
	if (!g_active) return;
	for (const auto feature: {sl::kFeatureDLSS_G, sl::kFeatureReflex, sl::kFeaturePCL}) {
		bool loaded = false;
		if (g_loaded != nullptr) (void)g_loaded(feature, loaded);
		sl::AdapterInfo adapter {};
		adapter.vkPhysicalDevice = physical_device;
		const auto supported     = g_supported(feature, adapter);
		sl::FeatureRequirements requirements {};
		const auto              required = g_requirements(feature, requirements);
		std::printf("streamline: feature %u loaded=%d supported=%d (%s) requirements=%d flags=0x%x "
		            "driver %u.%u.%u (needs %u.%u.%u) vk device ext %u instance ext %u "
		            "compute queues %u graphics queues %u optical flow queues %u\n",
		            static_cast<uint32_t>(feature), loaded ? 1 : 0, static_cast<int>(supported),
		            ResultName(supported), static_cast<int>(required),
		            static_cast<uint32_t>(requirements.flags), requirements.driverVersionDetected.major,
		            requirements.driverVersionDetected.minor, requirements.driverVersionDetected.build,
		            requirements.driverVersionRequired.major, requirements.driverVersionRequired.minor,
		            requirements.driverVersionRequired.build, requirements.vkNumDeviceExtensions,
		            requirements.vkNumInstanceExtensions, requirements.vkNumComputeQueuesRequired,
		            requirements.vkNumGraphicsQueuesRequired,
		            requirements.vkNumOpticalFlowQueuesRequired);
	}
	const auto function = [](sl::Feature feature, const char* name) -> void* {
		void* pointer = nullptr;
		return g_feature_function(feature, name, pointer) == sl::Result::eOk ? pointer : nullptr;
	};
	g_pcl_marker     = reinterpret_cast<PFun_slPCLSetMarker*>(function(sl::kFeaturePCL, "slPCLSetMarker"));
	g_reflex_options = reinterpret_cast<PFun_slReflexSetOptions*>(
	    function(sl::kFeatureReflex, "slReflexSetOptions"));
	g_dlssg_options  = reinterpret_cast<PFun_slDLSSGSetOptions*>(
        function(sl::kFeatureDLSS_G, "slDLSSGSetOptions"));
	g_dlssg_state    = reinterpret_cast<PFun_slDLSSGGetState*>(
        function(sl::kFeatureDLSS_G, "slDLSSGGetState"));
	if (g_reflex_options != nullptr) {
		sl::ReflexOptions reflex {};
		reflex.mode                 = sl::ReflexMode::eLowLatency;
		reflex.useMarkersToOptimize = false;
		std::printf("streamline: Reflex low latency -> %d\n", static_cast<int>(g_reflex_options(reflex)));
	}
	std::printf("streamline: functions pcl=%d reflex=%d dlssg=%d/%d\n", g_pcl_marker != nullptr ? 1 : 0,
	            g_reflex_options != nullptr ? 1 : 0, g_dlssg_options != nullptr ? 1 : 0,
	            g_dlssg_state != nullptr ? 1 : 0);
	std::fflush(stdout);
}

uint64_t TaaShaderHash() {
	static const uint64_t hash = [] {
		const char* value = std::getenv("KYTY_DLSSG_TAA");
		return value != nullptr ? std::strtoull(value, nullptr, 16) : 0xc3af833d5e36e33cull;
	}();
	return hash;
}

uint32_t MotionVectorBinding() {
	static const uint32_t binding = [] {
		const char* value = std::getenv("KYTY_DLSSG_MVEC_BINDING");
		return value != nullptr ? static_cast<uint32_t>(std::atoi(value)) : 1u;
	}();
	return binding;
}

namespace {

void Identity(sl::float4x4& m) {
	for (uint32_t r = 0; r < 4; ++r) m.setRow(r, {r == 0 ? 1.0f : 0.0f, r == 1 ? 1.0f : 0.0f, r == 2 ? 1.0f : 0.0f, r == 3 ? 1.0f : 0.0f});
}

float EnvFloat(const char* name, float fallback) {
	const char* value = std::getenv(name);
	return value != nullptr ? static_cast<float>(std::atof(value)) : fallback;
}

struct TagPayload {
	TagImage       motion;
	TagImage       depth;
	FrameConstants constants;
	uint32_t       frame;
};

sl::Resource MakeResource(const TagImage& image) {
	sl::Resource resource(sl::ResourceType::eTex2d, reinterpret_cast<void*>(image.image), nullptr,
	                      reinterpret_cast<void*>(image.view), image.layout);
	resource.width        = image.width;
	resource.height       = image.height;
	resource.nativeFormat = image.format;
	resource.mipLevels    = 1;
	resource.arrayLayers  = 1;
	return resource;
}

void RunTag(VkCommandBuffer command, const uint8_t* bytes) {
	TagPayload payload;
	std::memcpy(&payload, bytes, sizeof(payload));
	sl::FrameToken* token = nullptr;
	uint32_t        index = payload.frame;
	(void)index;
	if (g_new_token(token, nullptr) != sl::Result::eOk || token == nullptr) return;
	g_token.store(token, std::memory_order_release);
	if (g_pcl_marker != nullptr) {
		(void)g_pcl_marker(sl::PCLMarker::eSimulationStart, *token);
		(void)g_pcl_marker(sl::PCLMarker::eSimulationEnd, *token);
		(void)g_pcl_marker(sl::PCLMarker::eRenderSubmitStart, *token);
	}
	auto            motion = MakeResource(payload.motion);
	auto            depth  = MakeResource(payload.depth);
	const auto&     cb     = payload.constants;
	// The TAA's render size (cb[4..5]) bounds the valid part of the dynamic-resolution targets.
	const uint32_t  ext_w  = cb.valid && cb.width != 0 ? std::min(cb.width, payload.motion.width)
	                                                   : payload.motion.width;
	const uint32_t  ext_h  = cb.valid && cb.height != 0 ? std::min(cb.height, payload.motion.height)
	                                                    : payload.motion.height;
	sl::Extent      extent {0, 0, ext_w, ext_h};
	sl::ResourceTag tags[] = {
	    sl::ResourceTag(&depth, sl::kBufferTypeDepth, sl::ResourceLifecycle::eOnlyValidNow, &extent),
	    sl::ResourceTag(&motion, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eOnlyValidNow,
	                    &extent)};
	const sl::ViewportHandle viewport(0);
	const auto tag_result = g_set_tag(*token, viewport, tags, 2,
	                                  reinterpret_cast<sl::CommandBuffer*>(command));
	sl::Constants constants {};
	Identity(constants.cameraViewToClip);
	Identity(constants.clipToCameraView);
	Identity(constants.clipToLensClip);
	Identity(constants.clipToPrevClip);
	Identity(constants.prevClipToClip);
	const float jitter_sign        = EnvFloat("KYTY_DLSSG_JITTER_SIGN", 1.0f);
	constants.jitterOffset         = cb.valid ? sl::float2 {cb.jitter_x * jitter_sign, cb.jitter_y * jitter_sign}
	                                          : sl::float2 {0.0f, 0.0f};
	constants.mvecScale            = {EnvFloat("KYTY_DLSSG_MVEC_SCALE_X", 1.0f),
	                                  EnvFloat("KYTY_DLSSG_MVEC_SCALE_Y", 1.0f)};
	constants.cameraPinholeOffset  = {0.0f, 0.0f};
	constants.cameraPos            = {0.0f, 0.0f, 0.0f};
	constants.cameraUp             = {0.0f, 1.0f, 0.0f};
	constants.cameraRight          = {1.0f, 0.0f, 0.0f};
	constants.cameraFwd            = {0.0f, 0.0f, 1.0f};
	constants.cameraNear           = cb.valid ? cb.near_z : EnvFloat("KYTY_DLSSG_NEAR", 0.1f);
	constants.cameraFar            = cb.valid ? cb.far_z : EnvFloat("KYTY_DLSSG_FAR", 1000.0f);
	constants.cameraFOV            = 1.0f;
	constants.cameraAspectRatio    = static_cast<float>(ext_w) / static_cast<float>(ext_h == 0 ? 1 : ext_h);
	constants.motionVectorsInvalidValue = 0.0f;
	constants.depthInverted        = EnvFloat("KYTY_DLSSG_DEPTH_INVERTED", 1.0f) != 0.0f
	                                     ? sl::Boolean::eTrue : sl::Boolean::eFalse;
	constants.cameraMotionIncluded = sl::Boolean::eTrue;
	constants.motionVectors3D      = sl::Boolean::eFalse;
	constants.reset                = cb.valid && cb.reset ? sl::Boolean::eTrue : sl::Boolean::eFalse;
	// UFC 6's TAA motion vectors: UV units, current -> previous, dilated, not jittered.
	constants.motionVectorsDilated  = sl::Boolean::eTrue;
	constants.motionVectorsJittered = sl::Boolean::eFalse;
	const auto constants_result    = g_set_constants(constants, *token, viewport);
	static uint32_t reports = 0;
	if (reports < 5 || (tag_result != sl::Result::eOk && reports < 50)) {
		reports++;
		std::printf("streamline: frame %u tag -> %d (%s) constants -> %d; mvec %ux%u f%u layout %u, "
		            "depth %ux%u f%u layout %u; cb valid %d jitter %.4f,%.4f near %.4f far %.1f render %ux%u "
		            "reset %d\n",
		            payload.frame, static_cast<int>(tag_result), ResultName(tag_result),
		            static_cast<int>(constants_result), payload.motion.width, payload.motion.height,
		            payload.motion.format, payload.motion.layout, payload.depth.width,
		            payload.depth.height, payload.depth.format, payload.depth.layout, cb.valid ? 1 : 0,
		            cb.jitter_x, cb.jitter_y, cb.near_z, cb.far_z, cb.width, cb.height, cb.reset ? 1 : 0);
		std::fflush(stdout);
	}
}

} // namespace

void TagFrame(VkCommandBuffer command, const TagImage& motion, const TagImage& depth,
              const FrameConstants& constants) {
	if (!g_active) return;
	if (constants.valid) {
		g_render_width.store(constants.width, std::memory_order_relaxed);
		g_render_height.store(constants.height, std::memory_order_relaxed);
	}
	TagPayload payload {motion, depth, constants, g_frame_index.fetch_add(1, std::memory_order_relaxed) + 1};
	if (auto* room = ReserveRecordedCall(command, sizeof(payload)); room != nullptr) {
		std::memcpy(room, &payload, sizeof(payload));
		CommitRecordedCall(RunTag);
		return;
	}
	RunTag(command, reinterpret_cast<const uint8_t*>(&payload));
}

void BeforePresent(uint32_t width, uint32_t height, uint32_t format) {
	if (!g_active) return;
	auto* token = g_token.load(std::memory_order_acquire);
	// Dynamic resolution (KYTY_DLSSG_DYNRES, default on): once the TAA constants report the game's
	// render size, the options are set again with eDynamicResolutionEnabled and that size as the
	// optimal one (the tags already carry the valid extent per frame).
	static const bool dynres     = []() {
		const char* v = std::getenv("KYTY_DLSSG_DYNRES");
		return v == nullptr || std::atoi(v) != 0;
	}();
	const uint32_t    dyn_width  = dynres ? g_render_width.load(std::memory_order_relaxed) : 0;
	const uint32_t    dyn_height = dynres ? g_render_height.load(std::memory_order_relaxed) : 0;
	if ((!g_options_set || (g_options_dyn_width == 0 && dyn_width != 0)) && token != nullptr &&
	    g_dlssg_options != nullptr) {
		const char*      frames = std::getenv("KYTY_DLSSG_FRAMES");
		sl::DLSSGOptions options {};
		options.mode                = sl::DLSSGMode::eOn;
		options.numFramesToGenerate = frames != nullptr ? static_cast<uint32_t>(std::atoi(frames)) : 1u;
		options.colorWidth          = width;
		options.colorHeight         = height;
		options.colorBufferFormat   = format;
		if (dyn_width != 0 && dyn_height != 0) {
			options.flags            = sl::DLSSGFlags::eDynamicResolutionEnabled;
			options.dynamicResWidth  = dyn_width;
			options.dynamicResHeight = dyn_height;
		}
		const auto result           = g_dlssg_options(sl::ViewportHandle(0), options);
		std::printf("streamline: DLSS-G on (%u generated) for %ux%u, dynamic res %ux%u -> %d (%s)\n",
		            options.numFramesToGenerate, width, height, dyn_width, dyn_height,
		            static_cast<int>(result), ResultName(result));
		std::fflush(stdout);
		g_options_set       = true;
		g_options_dyn_width = dyn_width;
	}
	if (token != nullptr && g_pcl_marker != nullptr) {
		(void)g_pcl_marker(sl::PCLMarker::eRenderSubmitEnd, *token);
		(void)g_pcl_marker(sl::PCLMarker::ePresentStart, *token);
	}
}

void AfterPresent() {
	if (!g_active) return;
	auto* token = g_token.load(std::memory_order_acquire);
	if (token != nullptr && g_pcl_marker != nullptr) {
		(void)g_pcl_marker(sl::PCLMarker::ePresentEnd, *token);
	}
	if (g_dlssg_state != nullptr && g_options_set) {
		// Displayed frames: numFramesActuallyPresented is per application frame (2 with one
		// generated frame); summed over 2 s next to the application frames.
		static auto     window_start = std::chrono::steady_clock::now();
		static uint64_t app_frames = 0, shown_frames = 0;
		sl::DLSSGState state {};
		if (g_dlssg_state(sl::ViewportHandle(0), state, nullptr) == sl::Result::eOk) {
			app_frames++;
			shown_frames += state.numFramesActuallyPresented;
		}
		const auto now     = std::chrono::steady_clock::now();
		const auto seconds = std::chrono::duration<double>(now - window_start).count();
		if (seconds >= 2.0) {
			std::printf("streamline: DLSS-G app %.1f fps, displayed %.1f fps, status 0x%x, max %u\n",
			            static_cast<double>(app_frames) / seconds,
			            static_cast<double>(shown_frames) / seconds,
			            static_cast<uint32_t>(state.status), state.numFramesToGenerateMax);
			std::fflush(stdout);
			window_start = now;
			app_frames = shown_frames = 0;
		}
	}
}

bool Active() {
	return g_active;
}

void Shutdown() {
	if (g_active && g_shutdown != nullptr) (void)g_shutdown();
	g_active = false;
}

} // namespace Libs::Graphics::Streamline

#else

namespace Libs::Graphics::Streamline {

bool InitBeforeVulkan() {
	if (const char* value = std::getenv("KYTY_DLSSG"); value != nullptr && *value == '1') {
		std::printf("streamline: this build has no Streamline support (configure KYTY_STREAMLINE_DIR)\n");
		std::fflush(stdout);
	}
	return false;
}
void Probe(VkPhysicalDevice /*physical_device*/) {}
bool Active() {
	return false;
}
void Shutdown() {}
uint64_t TaaShaderHash() {
	return 0;
}
uint32_t MotionVectorBinding() {
	return 1;
}
void TagFrame(VkCommandBuffer /*command*/, const TagImage& /*motion*/, const TagImage& /*depth*/,
              const FrameConstants& /*constants*/) {}
void BeforePresent(uint32_t /*width*/, uint32_t /*height*/, uint32_t /*format*/) {}
void AfterPresent() {}

} // namespace Libs::Graphics::Streamline

#endif
