#include "graphics/host_gpu/renderer/renderDraw.h"

#include "common/abSwitch.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

std::pair<int32_t, uint32_t> ResolveDrawOffsets(uint32_t                     index_offset,
                                                const ShaderVertexInputInfo& vs_input_info) {
	auto     vertex_offset   = static_cast<int32_t>(index_offset);
	uint32_t instance_offset = 0;
	if (!vs_input_info.has_dynamic_vertex_fetch) {
		return {vertex_offset, instance_offset};
	}

	EXIT_IF(!vs_input_info.stage);
	const auto& program   = vs_input_info.stage.module->info;
	const auto& resources = *vs_input_info.stage.resources;
	if (index_offset == 0 &&
	    program.info.vertex_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.vertex_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			vertex_offset = static_cast<int32_t>(resources.user_data[index]);
		}
	}
	if (program.info.instance_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.instance_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			instance_offset = resources.user_data[index];
		}
	}

	return {vertex_offset, instance_offset};
}

static std::atomic<uint32_t> g_draw_state_log_count = 0;
static std::atomic<uint32_t> g_draw_input_log_count = 0;
static std::atomic<uint32_t> g_mrt_state_log_count  = 0;

static std::atomic<uint32_t> g_framebuffer_skip_log_count = 0;

static float ConvertPolygonOffsetConstantFactor(float guest_factor, const HW::PolyOffset& offset,
                                                vk::Format host_depth_format) {
	if (offset.db_is_float_fmt) {
		return guest_factor;
	}

	int host_depth_bits = 0;
	switch (host_depth_format) {
		case vk::Format::eD16Unorm:
		case vk::Format::eD16UnormS8Uint: host_depth_bits = 16; break;
		case vk::Format::eD24UnormS8Uint: host_depth_bits = 24; break;
		default:
			// A fixed-point guest bias cannot be represented exactly by a floating-point host
			// attachment without VK_EXT_depth_bias_control.
			return guest_factor;
	}
	return std::ldexp(guest_factor, host_depth_bits + offset.neg_num_db_bits);
}

static const char* RenderColorTypeName(const RenderColorInfo& color) {
	return color.image_id ? "RenderTexture" : "NoColorOutput";
}

static void LogFramebufferSkip(const char* draw_name, const RenderDepthInfo& depth,
                               const CommandBuffer& buffer, uint32_t index_count, uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	auto log_id = g_framebuffer_skip_log_count.fetch_add(1, std::memory_order_relaxed);
	if (log_id >= 128) {
		return;
	}

	LOGF("DrawFramebufferSkip[%u]: %s depth_format=%s depth_image=%s depth_vaddr_num=%d "
	     "target_mask=0x%08" PRIx32 " prim=%u index_count=%u flags=0x%08" PRIx32 "\n",
	     log_id, draw_name, vk::to_string(depth.desc.view_info.format).c_str(),
	     depth.image_id ? "yes" : "no",
	     static_cast<int>(!depth.desc.info.data.Empty()) +
	         static_cast<int>(depth.desc.info.HasStencil()),
	     ctx.GetRenderTargetMask(), static_cast<uint32_t>(ucfg.GetPrimType()), index_count, flags);
}

static void LogMrtState(const char* draw_name, const CommandBuffer& buffer,
                        const ShaderPixelInputInfo& ps_input_info) {
	const auto& ctx            = buffer.GetRegisters();
	const auto& sh_regs        = ctx.GetShaderRegisters();
	const auto  rt_mask        = ctx.GetRenderTargetMask();
	const auto  cb_shader_mask = sh_regs.m_cbShaderMask;
	const auto& bc0            = ctx.GetBlendControl(0);

	auto log_id = g_mrt_state_log_count.fetch_add(1);
	if (log_id >= 32) {
		return;
	}

	LOGF("MrtState[%u]: %s rt_mask=0x%08" PRIx32 " cb_shader_mask=0x%08" PRIx32
	     " blend0=%s src=%u dst=%u alpha_src=%u alpha_dst=%u sep_alpha=%s\n",
	     log_id, draw_name, rt_mask, cb_shader_mask, bc0.enable ? "true" : "false",
	     bc0.color_srcblend, bc0.color_destblend, bc0.alpha_srcblend, bc0.alpha_destblend,
	     bc0.separate_alpha_blend ? "true" : "false");

	for (uint32_t i = 0; i < 8; i++) {
		const auto& rt  = ctx.GetRenderTarget(i);
		const auto& bc  = ctx.GetBlendControl(i);
		const auto  ctm = (rt_mask >> (i * 4u)) & 0x0fu;
		const auto  csm = (cb_shader_mask >> (i * 4u)) & 0x0fu;

		if (rt.base.addr == 0 && ps_input_info.target_output_mode[i] == 0 && ctm == 0 && csm == 0 &&
		    !bc.enable) {
			continue;
		}

		LOGF("MrtState[%u]: slot=%u addr=0x%010" PRIx64
		     " target_mask=0x%x shader_mask=0x%x out_mode=%u"
		     " fmt=0x%08" PRIx32 " nfmt=0x%08" PRIx32 " order=0x%08" PRIx32
		     " width=%u height=%u tile=%u"
		     " blend=%s src=%u dst=%u alpha_src=%u alpha_dst=%u\n",
		     log_id, i, rt.base.addr, ctm, csm, ps_input_info.target_output_mode[i],
		     static_cast<uint32_t>(rt.info.format), static_cast<uint32_t>(rt.info.channel_type),
		     static_cast<uint32_t>(rt.info.channel_order), rt.attrib2.width + 1,
		     rt.attrib2.height + 1, static_cast<uint32_t>(rt.attrib3.tile_mode),
		     bc.enable ? "true" : "false", bc.color_srcblend, bc.color_destblend, bc.alpha_srcblend,
		     bc.alpha_destblend);
	}
}

static void LogDrawTargetState(const char* draw_name, const RenderColorInfo& color,
                               const RenderDepthInfo& depth, const CommandBuffer& buffer,
                               const ShaderPixelInputInfo& ps_input_info, uint32_t index_count,
                               uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (!color.image_id) {
		return;
	}

	auto log_id = g_draw_state_log_count.fetch_add(1);
	if (log_id >= 192) {
		return;
	}

	const auto& cc             = ctx.GetColorControl();
	const auto& bc             = ctx.GetBlendControl(color.target_slot);
	const auto& dc             = ctx.GetDepthControl();
	const auto& vp             = ctx.GetScreenViewport();
	const auto& vp0            = vp.viewports[0];
	const auto& ps_resources   = ps_input_info.stage.module->info.info;
	const auto  sampled_images = std::count_if(
        ps_resources.images.begin(), ps_resources.images.end(), [](const auto& image) {
            return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
        });

	const auto extent = color.Extent();
	const auto sc     = calc_final_scissor(vp, ctx.GetScanModeControl(), extent, 0);

	LOGF(
	    "DrawTargetState[%u]: frame=%d %s target=%s addr=0x%010" PRIx64
	    " extent=%ux%u prim=%u index_count=%u flags=0x%08" PRIx32 " color_mask=0x%08" PRIx32
	    " cc_mode=%u cc_op=0x%02x"
	    " blend=%s src=%u dst=%u comb=%u ps_tex=%d sampled=%d storage=%d ps_kill=%s target_mode0=%u"
	    " depth_test=%s depth_write=%s depth_func=%u depth_clear=%s viewport=(%.1f,%.1f %.1fx%.1f) "
	    "scissor=(%d,%d)-(%d,%d)\n",
	    log_id, buffer.GetContext().GetGpu().GetFrameNum(), draw_name, RenderColorTypeName(color),
	    color.desc.info.data.address, extent.width, extent.height,
	    static_cast<uint32_t>(ucfg.GetPrimType()), index_count, flags, ctx.GetRenderTargetMask(),
	    cc.mode, cc.op, bc.enable ? "true" : "false", bc.color_srcblend, bc.color_destblend,
	    bc.color_comb_fcn, static_cast<int>(ps_resources.images.size()),
	    static_cast<int>(sampled_images),
	    static_cast<int>(ps_resources.images.size() - sampled_images),
	    ps_input_info.ps_pixel_kill_enable ? "true" : "false", ps_input_info.target_output_mode[0],
	    dc.z_enable ? "true" : "false", dc.z_write_enable ? "true" : "false", dc.zfunc,
	    depth.depth_clear_enable ? "true" : "false", vp0.xoffset - vp0.xscale,
	    vp0.yoffset - vp0.yscale, vp0.xscale * 2.0f, vp0.yscale * 2.0f, sc.left, sc.top, sc.right,
	    sc.bottom);

	LogMrtState(draw_name, buffer, ps_input_info);
}

static void LogDrawInputState(const CommandBuffer& buffer, const RenderColorInfo& color,
                              const ShaderVertexInputInfo& vs_input_info,
                              uint32_t index_type_and_size, uint32_t index_count,
                              const void* index_addr) {
	auto log_id = g_draw_input_log_count.fetch_add(1);
	if (log_id >= 64) {
		return;
	}

	LOGF("DrawInputState[%u]: frame=%d target=%s addr=0x%010" PRIx64
	     " index_type=%u index_count=%u index_addr=0x%016" PRIx64
	     " vs_resources=%d vs_buffers=%d\n",
	     log_id, buffer.GetContext().GetGpu().GetFrameNum(), RenderColorTypeName(color),
	     color.desc.info.data.address, index_type_and_size, index_count,
	     reinterpret_cast<uint64_t>(index_addr), vs_input_info.resources_num,
	     vs_input_info.buffers_num);

	for (int bi = 0; bi < vs_input_info.buffers_num; bi++) {
		const auto& b = vs_input_info.buffers[bi];
		LOGF("DrawInputState[%u]: vb[%d] addr=0x%010" PRIx64
		     " stride=%u records=%u fetch_index=%u\n",
		     log_id, bi, b.addr, b.stride, b.num_records, b.fetch_index);

		const auto* bytes = reinterpret_cast<const uint8_t*>(b.addr);
		if (bytes != nullptr && b.stride != 0) {
			const uint32_t records = std::min<uint32_t>(b.num_records, 4u);
			for (uint32_t rec = 0; rec < records; rec++) {
				const auto* rec_bytes = bytes + static_cast<uint64_t>(rec) * b.stride;
				const auto  dword_num = std::min<uint32_t>(b.stride / 4u, 12u);
				uint32_t    raw[12]   = {};
				float       flt[12]   = {};
				for (uint32_t i = 0; i < dword_num; i++) {
					std::memcpy(&raw[i], rec_bytes + i * 4u, sizeof(raw[i]));
					std::memcpy(&flt[i], rec_bytes + i * 4u, sizeof(flt[i]));
				}
				LOGF("DrawInputState[%u]: vb[%d].rec[%u] stride=%u dwords=%u raw=%08" PRIx32
				     " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32
				     " %08" PRIx32 " %08" PRIx32 " %08" PRIx32
				     " f=(%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f)\n",
				     log_id, bi, rec, b.stride, dword_num, raw[0], raw[1], raw[2], raw[3], raw[4],
				     raw[5], raw[6], raw[7], raw[8], flt[0], flt[1], flt[2], flt[3], flt[4], flt[5],
				     flt[6], flt[7], flt[8]);

				for (int ai = 0; ai < vs_input_info.resources_num; ai++) {
					const auto& r  = vs_input_info.resources[ai];
					const auto& rd = vs_input_info.resources_dst[ai];
					if (rd.buffer_index != bi) {
						continue;
					}
					const auto offset = static_cast<uint32_t>(r.Base48() - b.addr);
					if (offset + 4u <= b.stride &&
					    r.Format() == Prospero::BufferFormat::k8_8_8_8UNorm) {
						uint32_t packed = 0;
						std::memcpy(&packed, rec_bytes + offset, sizeof(packed));
						const auto r8 = (packed >> 0u) & 0xffu;
						const auto g8 = (packed >> 8u) & 0xffu;
						const auto b8 = (packed >> 16u) & 0xffu;
						const auto a8 = (packed >> 24u) & 0xffu;
						LOGF("DrawInputState[%u]: vb[%d].rec[%u].attr[%d] dst=v%d fmt=56 "
						     "rgba8=%02" PRIx32 "%02" PRIx32 "%02" PRIx32 "%02" PRIx32
						     " rgba=(%.3f,%.3f,%.3f,%.3f)\n",
						     log_id, bi, rec, ai, rd.register_start, r8, g8, b8, a8,
						     static_cast<double>(r8) / 255.0, static_cast<double>(g8) / 255.0,
						     static_cast<double>(b8) / 255.0, static_cast<double>(a8) / 255.0);
					}
				}
			}
		}

		for (int ai = 0; ai < vs_input_info.resources_num; ai++) {
			const auto& r  = vs_input_info.resources[ai];
			const auto& rd = vs_input_info.resources_dst[ai];
			if (rd.buffer_index != bi) {
				continue;
			}
			LOGF("DrawInputState[%u]: attr[%d] offset=%u dst=v%d regs=%d fetch_index=%u "
			     "sharp=%08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 "\n",
			     log_id, ai, static_cast<uint32_t>(r.Base48() - b.addr), rd.register_start,
			     rd.registers_num, rd.fetch_index, r.fields[0], r.fields[1], r.fields[2],
			     r.fields[3]);
		}
	}
}

static void UpdateGraphicsDynamicState(const CommandBuffer& buffer, DynamicState& dynamic_state,
                                       const ShaderVertexInputInfo& vs_input_info,
                                       const RenderDepthInfo& depth, const RenderState& rendering) {
	KYTY_PROFILER_FUNCTION();

	const auto&        ctx = buffer.GetRegisters();
	const auto&        vp  = ctx.GetScreenViewport();
	const vk::Extent2D framebuffer_extent {rendering.width, rendering.height};
	const auto&        outputs = vs_input_info.stage.module->info.info.outputs;
	const bool         indexed_viewports =
	    std::any_of(outputs.begin(), outputs.end(), [](const auto& output) {
		    return output.kind == ShaderRecompiler::IR::StageOutputKind::ViewportIndex;
	    });
	constexpr uint32_t viewport_slots = std::size(HW::ScreenViewport {}.viewports);
	std::array<vk::Viewport, viewport_slots> viewports {};
	std::array<vk::Rect2D, viewport_slots>   scissors {};
	const uint32_t viewport_count = indexed_viewports ? viewport_slots : 1;
	for (uint32_t i = 0; i < viewport_count; i++) {
		const auto& guest    = vp.viewports[i];
		auto&       viewport = viewports[i];
		if (ctx.GetClipControl().clip_disable) {
			const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
			viewport.width  = static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u));
			viewport.height = static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u));
		} else {
			viewport.x      = guest.xoffset - guest.xscale;
			viewport.y      = guest.yoffset - guest.yscale;
			viewport.width  = guest.xscale * 2.0f;
			viewport.height = guest.yscale * 2.0f;
		}
		viewport.minDepth =
		    guest.zoffset - (ctx.GetClipControl().dx_clip_space ? 0.0f : guest.zscale);
		viewport.maxDepth = guest.zscale + guest.zoffset;

		const auto final_scissor =
		    calc_final_scissor(vp, ctx.GetScanModeControl(), framebuffer_extent, i);
		auto& scissor  = scissors[i];
		scissor.offset = {final_scissor.left, final_scissor.top};
		scissor.extent = {static_cast<uint32_t>(final_scissor.right - final_scissor.left),
		                  static_cast<uint32_t>(final_scissor.bottom - final_scissor.top)};
		if (viewport.width == 0.0f) {
			// Keep empty slots at their guest index; Vulkan requires a positive viewport width.
			viewport.width = 1.0f;
			scissor.extent = {0, 0};
		}
	}
	dynamic_state.SetViewports(std::span {viewports}.first(viewport_count));
	dynamic_state.SetScissors(std::span {scissors}.first(viewport_count));

	float line_width = ctx.GetLineWidth();
	if (line_width != 1.0f) {
		static bool logged = false;
		if (!logged) {
			LOGF("Render: temporary: clamping Vulkan line width %f to 1.0 because wideLines is "
			     "not enabled\n",
			     line_width);
			logged = true;
		}
		line_width = 1.0f;
	}
	dynamic_state.SetLineWidth(line_width);
	const auto&      blend = ctx.GetBlendColor();
	const std::array blend_constants {blend.red, blend.green, blend.blue, blend.alpha};
	dynamic_state.SetBlendConstants(blend_constants);
	dynamic_state.SetDepthTestEnabled(depth.depth_test_enable);
	dynamic_state.SetDepthWriteEnabled(depth.depth_write_enable);
	dynamic_state.SetDepthCompareOp(depth.depth_compare_op);
#if !defined(__APPLE__)
	dynamic_state.SetDepthBoundsTestEnabled(depth.depth_bounds_test_enable);
	if (depth.depth_bounds_test_enable) {
		dynamic_state.SetDepthBounds(ctx.GetDepthBoundsMin(), ctx.GetDepthBoundsMax());
	}
#endif

	const auto& mode              = ctx.GetModeControl();
	const auto& poly_offset       = ctx.GetPolyOffset();
	const bool  use_front         = mode.poly_offset_front_enable && !mode.cull_front;
	const bool  use_back          = mode.poly_offset_back_enable && !mode.cull_back;
	const bool  depth_bias_enable = use_front || use_back;
	dynamic_state.SetDepthBiasEnabled(depth_bias_enable);
	if (depth_bias_enable) {
		// Vulkan has one bias for both faces. Prefer a visible front face when both are enabled.
		const float guest_constant_factor =
		    use_front ? poly_offset.front_offset : poly_offset.back_offset;
		const float constant_factor = ConvertPolygonOffsetConstantFactor(
		    guest_constant_factor, poly_offset, depth.desc.view_info.format);
		const float slope_factor =
		    (use_front ? poly_offset.front_scale : poly_offset.back_scale) / 16.0f;
		dynamic_state.SetDepthBias(constant_factor, poly_offset.clamp, slope_factor);
	}

	// Rect lists are drawn without culling.
	vk::CullModeFlags cull_mode {};
	if (!Prospero::IsRectList(buffer.GetUserConfig().GetPrimType())) {
		if (mode.cull_front) {
			cull_mode |= vk::CullModeFlagBits::eFront;
		}
		if (mode.cull_back) {
			cull_mode |= vk::CullModeFlagBits::eBack;
		}
	}
	dynamic_state.SetCullMode(cull_mode);
	dynamic_state.SetFrontFace(mode.face ? vk::FrontFace::eClockwise
	                                     : vk::FrontFace::eCounterClockwise);

	dynamic_state.SetStencilTestEnabled(depth.stencil_test_enable);
	if (depth.stencil_test_enable) {
		const auto& front = depth.stencil_front;
		const auto& back  = depth.stencil_back;
		dynamic_state.SetStencilOps(
		    {front.failOp, front.passOp, front.depthFailOp, front.compareOp},
		    {back.failOp, back.passOp, back.depthFailOp, back.compareOp});
		dynamic_state.SetStencilCompareMasks(front.compareMask, back.compareMask);
		dynamic_state.SetStencilWriteMasks(front.writeMask, back.writeMask);
		dynamic_state.SetStencilReferences(front.reference, back.reference);
	}
}

// KYTY_VERIFY_DYNAMIC_KEEP=1: a draw that keeps the recorded dynamic state records it anyway, and
// exits if that would have changed anything.
static bool VerifyDynamicKeep() {
	static const bool enabled = std::getenv("KYTY_VERIFY_DYNAMIC_KEEP") != nullptr;
	return enabled;
}

static bool SameDynamicState(const DynamicState& a, const DynamicState& b) {
	return a.num_viewports == b.num_viewports && a.viewports == b.viewports &&
	       a.num_scissors == b.num_scissors && a.scissors == b.scissors &&
	       a.depth_test_enabled == b.depth_test_enabled &&
	       a.depth_write_enabled == b.depth_write_enabled &&
	       a.depth_compare_op == b.depth_compare_op &&
	       a.depth_bounds_test_enabled == b.depth_bounds_test_enabled &&
	       a.depth_bounds_min == b.depth_bounds_min && a.depth_bounds_max == b.depth_bounds_max &&
	       a.depth_bias_enabled == b.depth_bias_enabled &&
	       a.depth_bias_constant == b.depth_bias_constant &&
	       a.depth_bias_clamp == b.depth_bias_clamp && a.depth_bias_slope == b.depth_bias_slope &&
	       a.stencil_test_enabled == b.stencil_test_enabled &&
	       a.stencil_front_ops == b.stencil_front_ops && a.stencil_back_ops == b.stencil_back_ops &&
	       a.stencil_front_reference == b.stencil_front_reference &&
	       a.stencil_back_reference == b.stencil_back_reference &&
	       a.stencil_front_write_mask == b.stencil_front_write_mask &&
	       a.stencil_back_write_mask == b.stencil_back_write_mask &&
	       a.stencil_front_compare_mask == b.stencil_front_compare_mask &&
	       a.stencil_back_compare_mask == b.stencil_back_compare_mask &&
	       a.cull_mode == b.cull_mode && a.front_face == b.front_face &&
	       a.blend_constants == b.blend_constants && a.line_width == b.line_width &&
	       a.feedback_loop_aspects == b.feedback_loop_aspects;
}

static bool DrawHasValidVertexShader(const HW::Shader& sh_ctx) {

	const auto& vs = sh_ctx.GetVs();
	return vs.es_regs.data_addr != 0;
}

struct DrawCallInfo {
	bool     indexed        = true;
	uint32_t index_count    = 0;
	uint32_t instance_count = 0;
	uint32_t first_instance = 0;

	[[nodiscard]] bool        IsIndexed() const { return indexed; }
	[[nodiscard]] const char* Name() const { return indexed ? "DrawIndex" : "DrawIndexAuto"; }
};

RenderState RenderExecutor::AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
                                                 uint32_t color_count, RenderDepthInfo& depth,
                                                 vk::ImageAspectFlags& feedback_aspects,
                                                 std::span<PreparedBindings* const> stages) {
	EXIT_IF(colors == nullptr || color_count > RENDER_COLOR_ATTACHMENTS_MAX);
	feedback_aspects  = {};
	auto&       cache = m_context.GetTextureCache();
	RenderState state {};
	state.width                 = std::numeric_limits<uint32_t>::max();
	state.height                = std::numeric_limits<uint32_t>::max();
	state.num_layers            = std::numeric_limits<uint32_t>::max();
	state.num_color_attachments = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		auto& target = colors[i];
		EXIT_IF(!target.image_id);
		const auto image_view = cache.FindRenderTarget(target.image_id, target.desc);
		auto&      image      = cache.GetImage(target.image_id);
		EXIT_IF(image.backing.samples != target.desc.info.samples || image_view == nullptr);
		const auto& view                = target.desc.view_info;
		const auto  layout              = image.binding.is_bound ? vk::ImageLayout::eGeneral
		                                                         : vk::ImageLayout::eColorAttachmentOptimal;
		image.binding.attachment_layout = layout;
		image.binding.attachment_access =
		    vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite;
		image.Transit(layout, image.binding.attachment_access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Handle());
		const auto extent           = target.Extent();
		state.width                 = std::min(state.width, extent.width);
		state.height                = std::min(state.height, extent.height);
		state.num_layers            = std::min(state.num_layers, view.layer_count);
		state.num_color_attachments = std::max(state.num_color_attachments, target.target_slot + 1);
		auto& attachment            = state.color_attachments[target.target_slot];
		attachment.image_view       = image_view;
		attachment.image_layout     = layout;
	}
	if (depth.image_id) {
		const auto  image_view = cache.FindDepthTarget(depth.image_id, depth.desc);
		const auto& metadata   = depth.desc.info.metadata;
		const bool  meta_clear =
		    metadata.kind == ImageMetadataKind::Htile &&
		    cache.IsMetaCleared(metadata.range.address, depth.desc.view_info.base_layer);
		depth.load_clear_aspects = {};
		if (depth.depth_clear_enable || meta_clear)
			depth.load_clear_aspects |= vk::ImageAspectFlagBits::eDepth;
		// Apply pending clears to compressed stencil as well as depth.
		if (depth.stencil_clear_enable || (meta_clear && metadata.stencil_compressed))
			depth.load_clear_aspects |= vk::ImageAspectFlagBits::eStencil;
		if (meta_clear &&
		    !cache.TouchMeta(metadata.range.address, depth.desc.view_info.base_layer, false)) {
			EXIT("failed to consume HTile clear state\n");
		}
		auto& image = cache.GetImage(depth.image_id);
		EXIT_IF(image_view == nullptr || image.backing.samples != depth.desc.info.samples);
		const auto           draw_writes = depth.AttachmentWriteAspects();
		vk::ImageAspectFlags sampled_aspects;
		for (const auto* stage: stages) {
			for (const auto& binding: stage->images) {
				if (binding.image_id != depth.image_id ||
				    binding.desc.type != TextureCache::BindingType::Texture)
					continue;
				const auto native =
				    std::ranges::find(image.views, binding.image_view, &CachedImageView::view);
				EXIT_IF(native == image.views.end());
				sampled_aspects |= native->info.aspect;
				feedback_aspects |=
				    DepthFeedbackAspects(draw_writes, depth.desc.view_info, native->info);
			}
		}
		if (feedback_aspects && !m_context.GetGraphics().attachment_feedback_loop_enabled) {
			static std::atomic_bool logged = false;
			if (!logged.exchange(true, std::memory_order_relaxed)) {
				std::printf("depth attachment feedback loop is not supported by the host\n");
			}
		}
		auto layout = depth_attachment_layout(depth);
		if (sampled_aspects & ~DepthReadableAspects(layout)) {
			layout = m_context.GetGraphics().attachment_feedback_loop_enabled
			             ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
			             : vk::ImageLayout::eGeneral;
		}
		// The attachment store writes even when guest depth/stencil tests do not.
		const auto access = vk::AccessFlagBits2::eDepthStencilAttachmentRead |
		                    vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
		image.binding.attachment_layout = layout;
		image.binding.attachment_access = access;
		const auto& view                = depth.desc.view_info;
		image.Transit(layout, access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Handle());
		state.width               = std::min(state.width, depth.desc.info.extent.width);
		state.height              = std::min(state.height, depth.desc.info.extent.height);
		state.num_layers          = std::min(state.num_layers, view.layer_count);
		const auto aspects        = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		auto&      attachment     = state.depth_stencil_attachment;
		attachment.image_view     = image_view;
		attachment.image_layout   = layout;
		attachment.clear_value[0] = std::bit_cast<uint32_t>(depth.depth_clear_value);
		attachment.clear_value[1] = depth.stencil_clear_value;
		attachment.has_depth      = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eDepth);
		attachment.depth_clear =
		    static_cast<bool>(depth.load_clear_aspects & vk::ImageAspectFlagBits::eDepth);
		attachment.has_stencil = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eStencil);
		attachment.stencil_clear =
		    static_cast<bool>(depth.load_clear_aspects & vk::ImageAspectFlagBits::eStencil);
	}
	if (color_count == 0 && !depth.image_id) {
		const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
		state.width        = limits.maxFramebufferWidth;
		state.height       = limits.maxFramebufferHeight;
	}
	if (state.num_layers == std::numeric_limits<uint32_t>::max()) {
		state.num_layers = 1;
	}
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.width == std::numeric_limits<uint32_t>::max() ||
	        state.height == std::numeric_limits<uint32_t>::max());
	// The guest extents above can exceed the host image an attachment was placed in. Rendering past
	// an attachment faults NVIDIA GPUs ("3D WIDTH ZT Violation") and loses the device, so keep the
	// render area, and with it the scissors, inside every attachment.
	const auto clamp_to_attachment = [&](const char* kind, ImageId id, uint32_t level) {
		const auto&    host   = cache.GetImage(id);
		const uint32_t width  = std::max(host.info.extent.width >> level, 1u);
		const uint32_t height = std::max(host.info.extent.height >> level, 1u);
		if (state.width <= width && state.height <= height) {
			return;
		}
		static std::atomic_int reported = 0;
		if (reported.fetch_add(1, std::memory_order_relaxed) < 64) {
			std::printf("render area %ux%u exceeds the %s attachment %ux%u (level %u, %s, guest "
			            "0x%" PRIx64 ")\n",
			            state.width, state.height, kind, width, height, level,
			            vk::to_string(host.info.pixel_format).c_str(), host.info.data.address);
		}
		state.width  = std::min(state.width, width);
		state.height = std::min(state.height, height);
	};
	for (uint32_t i = 0; i < color_count; i++) {
		clamp_to_attachment("color", colors[i].image_id, colors[i].desc.view_info.base_level);
	}
	if (depth.image_id) {
		clamp_to_attachment("depth", depth.image_id, depth.desc.view_info.base_level);
	}
	return state;
}

enum class CbColorMode : uint8_t {
	Disable            = 0,
	Normal             = 1,
	EliminateFastClear = 2,
	Resolve            = 3,
	FmaskDecompress    = 5,
	DccDecompress      = 6,
};

bool RenderExecutor::ConsumeMetadataColorOperation(CommandBuffer& buffer) {
	const auto& ctx  = buffer.GetRegisters();
	const auto  mode = ctx.GetColorControl().mode;
	if (mode == static_cast<uint8_t>(CbColorMode::EliminateFastClear)) {
		// FCE can precede any ordinary draw. Discover the target to materialize its native
		// clear metadata before the expanded image is sampled.
		RenderColorInfo target {};
		ResolveRenderColorTarget(buffer, target, 0, true);
		return true;
	}
	// Metadata/decompression shaders must not export colors into the expanded image.
	return mode == static_cast<uint8_t>(CbColorMode::FmaskDecompress) ||
	       mode == static_cast<uint8_t>(CbColorMode::DccDecompress);
}

// vertex_offset is the first vertex of non-indexed draws.
struct DrawEmitInfo {
	int32_t          vertex_offset  = 0;
	uint32_t         first_instance = 0;
	DrawIndirectArgs indirect;
};

struct DrawIndexBufferSource {
	uint64_t    address            = 0;
	const void* host_data          = nullptr;
	uint64_t    size               = 0;
	uint32_t    guest_element_size = 0;
};

// The task shader's inputs for an indirect mesh draw: Params in gpu_mesh_indirect.task.
struct MeshTaskParams {
	vk::DeviceAddress records        = 0;
	vk::DeviceAddress index_address  = 0;
	uint32_t          stride         = 0;
	uint32_t          index_size     = 0;
	uint32_t          index_limit    = 0;
	uint32_t          primitive_size = 0;
	uint32_t          group_step     = 0;
	uint32_t          max_groups     = 0;
	uint32_t          max_instances  = 0;
	uint32_t          max_total      = 0;
};

struct PreparedIndexBuffer {
	vk::Buffer     buffer = nullptr;
	vk::DeviceSize offset = 0;
	vk::IndexType  type   = vk::IndexType::eUint16;
};

static uint64_t VertexBufferDescriptorSize(int binding, const ShaderVertexInputInfo& info) {
	const auto& buffer = info.buffers[binding];
	if (buffer.stride != 0 || buffer.num_records == 0) {
		return static_cast<uint64_t>(buffer.stride) * buffer.num_records;
	}

	uint64_t size = 0;
	for (int i = 0; i < info.resources_num; i++) {
		if (info.resources_dst[i].buffer_index != binding) {
			continue;
		}
		const auto& resource = info.resources[i];
		// RDNA2 OOB_SELECT=2 only checks NumRecords != 0. A constant attribute still
		// fetches its entire format; NumRecords is not a byte count in this mode.
		const uint64_t extent =
		    resource.OutOfBounds() == 2
		        ? resource.Base48() - buffer.addr +
		              ShaderRecompiler::Format::GetFormatInfo(resource.Format()).byte_size
		        : buffer.num_records;
		size = std::max(size, extent);
	}
	return size;
}

struct VertexBufferRange {
	uint64_t                     base_address  = 0;
	uint64_t                     requested_end = 0;
	uint64_t                     acquired_end  = 0;
	std::pair<Buffer*, uint64_t> binding;

	[[nodiscard]] uint64_t RequestedSize() const { return requested_end - base_address; }
};

struct PreparedVertexBuffers {
	static constexpr uint32_t MaxBuffers = ShaderVertexInputInfo::RES_MAX;

	std::array<vk::Buffer, MaxBuffers>     buffers {};
	std::array<vk::DeviceSize, MaxBuffers> offsets {};
	std::array<vk::DeviceSize, MaxBuffers> sizes {};
	uint32_t                               count = 0;
};

// Opt-in observation only: no prepared binding or synchronization is ever reused. Keys describe
// effective guest requests, not SRT hits. Timings sample roughly 1/64 draws; exact key counters
// include every non-mesh draw. No extra dirty-state query is made: that can allocate a region and
// warm the memory tracker before ObtainBuffer. plan_ns includes descriptor-size calculation, so
// repeat_plan_ns is an optimistic bound before memo-key overhead. Vertex calls include empty
// plans; active_calls/active_*_ns select calls with a non-empty guest range. Clock overhead and
// the probe's own key/counter work still affect timings; verify FPS with probes off.
struct VertexIndexProbe {
	using Clock = std::chrono::steady_clock;
	struct VertexKey {
		std::array<uint64_t, PreparedVertexBuffers::MaxBuffers> addresses {};
		std::array<uint64_t, PreparedVertexBuffers::MaxBuffers> sizes {};
		uint32_t                                                count = 0;
		bool operator==(const VertexKey&) const                       = default;
	};
	struct IndexKey {
		uint64_t      address                           = 0;
		uint64_t      size                              = 0;
		vk::IndexType type                              = vk::IndexType::eUint16;
		bool          operator==(const IndexKey&) const = default;
	};
	struct ProgramEntry {
		uint64_t  module_id = 0;
		uint64_t  last_draw = 0;
		VertexKey vertex;
		IndexKey  index;
		bool      vertex_valid = false;
		bool      index_valid  = false;
	};
	struct Timing {
		uint64_t calls                       = 0;
		uint64_t active_calls                = 0;
		uint64_t plan_ns                     = 0;
		uint64_t active_plan_ns              = 0;
		uint64_t obtain_ns                   = 0;
		uint64_t slots_ns                    = 0;
		uint64_t active_slots_ns             = 0;
		uint64_t repeat_plan_ns              = 0;
		uint64_t repeat_calls                = 0;
		uint64_t repeat_obtain_ns            = 0;
		uint64_t ranges                      = 0;
		uint64_t stream                      = 0;
		uint64_t repeat_stream               = 0;
		uint64_t repeat_same_program_calls   = 0;
		uint64_t repeat_same_program_plan_ns = 0;
	};
	struct Window {
		uint64_t              draws                        = 0;
		uint64_t              sampled_draws                = 0;
		uint64_t              vertex_draws                 = 0;
		uint64_t              vertex_repeat                = 0;
		uint64_t              index_draws                  = 0;
		uint64_t              index_repeat                 = 0;
		uint64_t              host_index_draws             = 0;
		uint64_t              retry_passes                 = 0;
		uint64_t              program_cache_hits           = 0;
		uint64_t              program_cache_misses         = 0;
		uint64_t              program_cache_evictions      = 0;
		uint64_t              vertex_same_program_draws    = 0;
		uint64_t              vertex_same_program_compares = 0;
		uint64_t              vertex_repeat_same_program   = 0;
		uint64_t              index_same_program_draws     = 0;
		uint64_t              index_same_program_compares  = 0;
		uint64_t              index_repeat_same_program    = 0;
		std::array<Timing, 4> timings {}; // Vertex/index, then their merge-retry calls.
	} window;
	// Stable compiled-module IDs distinguish shader specializations. This is a bounded LRU of
	// the last key seen per module, so interleaved shaders do not look like compulsory misses.
	static constexpr size_t                   ProgramCapacity = 64;
	std::array<ProgramEntry, ProgramCapacity> programs {};
	ProgramEntry*                             current_program = nullptr;
	size_t                                    program_count   = 0;
	uint64_t                                  draw_serial     = 0;
	VertexKey                                 previous_vertex;
	IndexKey                                  previous_index;
	bool                                      vertex_valid               = false;
	bool                                      index_valid                = false;
	bool                                      vertex_repeat              = false;
	bool                                      index_repeat               = false;
	bool                                      vertex_repeat_same_program = false;
	bool                                      index_repeat_same_program  = false;
	bool                                      sampled                    = false;
	bool                                      retry                      = false;
	uint32_t                                  sequence                   = 0;
	Clock::time_point                         window_start               = Clock::now();

	static bool Enabled() {
		static const bool enabled = [] {
			const char* value = std::getenv("KYTY_PROBE_VERTEX_INDEX");
			return value != nullptr && std::strcmp(value, "1") == 0;
		}();
		return enabled;
	}
	static VertexIndexProbe& Get() {
		thread_local VertexIndexProbe probe;
		return probe;
	}
	static uint64_t Ns(Clock::time_point begin, Clock::time_point end) {
		return static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
	}
	void BeginDraw(uint64_t module_id) {
		++window.draws;
		// High bits of a multiplicative sequence avoid sampling the same draw modulo 64.
		sampled = ((++sequence * uint32_t {0x9e3779b9}) >> 26u) == 0;
		window.sampled_draws += sampled;
		retry      = false;
		auto end   = programs.begin() + program_count;
		auto entry = std::find_if(programs.begin(), end, [module_id](const ProgramEntry& value) {
			return value.module_id == module_id;
		});
		if (entry != end) {
			++window.program_cache_hits;
		} else {
			++window.program_cache_misses;
			if (program_count < ProgramCapacity) {
				entry = programs.begin() + program_count++;
			} else {
				++window.program_cache_evictions;
				entry = std::min_element(programs.begin(), end,
				                         [](const ProgramEntry& a, const ProgramEntry& b) {
					                         return a.last_draw < b.last_draw;
				                         });
			}
			*entry           = {};
			entry->module_id = module_id;
		}
		entry->last_draw = ++draw_serial;
		current_program  = &*entry;
	}
	void ObserveVertex(const ShaderVertexInputInfo&                                   info,
	                   const std::array<uint64_t, PreparedVertexBuffers::MaxBuffers>& sizes) {
		if (retry) return;
		VertexKey key;
		key.count = static_cast<uint32_t>(info.buffers_num);
		for (uint32_t i = 0; i < key.count; ++i) {
			// Empty slots always name the null buffer, regardless of their descriptor address.
			key.addresses[i] = sizes[i] != 0 ? info.buffers[i].addr : 0;
			key.sizes[i]     = sizes[i];
		}
		vertex_repeat = vertex_valid && key == previous_vertex;
		window.vertex_draws += key.count != 0;
		window.vertex_repeat += key.count != 0 && vertex_repeat;
		vertex_valid               = key.count != 0;
		previous_vertex            = key;
		const bool comparable      = key.count != 0 && current_program->vertex_valid;
		vertex_repeat_same_program = comparable && key == current_program->vertex;
		window.vertex_same_program_draws += key.count != 0;
		window.vertex_same_program_compares += comparable;
		window.vertex_repeat_same_program += vertex_repeat_same_program;
		current_program->vertex_valid = key.count != 0;
		current_program->vertex       = key;
	}
	void ObserveIndex(const DrawIndexBufferSource& source) {
		if (retry) return;
		// Converted 8-bit indices live in a temporary vector/stream allocation. Pointer identity
		// would say nothing about their bytes or allocation lifetime, so exclude them from hits.
		const bool     eligible = source.size != 0 && source.host_data == nullptr;
		const IndexKey key {source.address, source.size,
		                    source.guest_element_size == 4 ? vk::IndexType::eUint32
		                                                   : vk::IndexType::eUint16};
		window.index_draws += eligible;
		index_repeat = eligible && index_valid && key == previous_index;
		window.index_repeat += index_repeat;
		window.host_index_draws += source.size != 0 && source.host_data != nullptr;
		index_valid               = eligible;
		previous_index            = key;
		const bool comparable     = eligible && current_program->index_valid;
		index_repeat_same_program = comparable && key == current_program->index;
		window.index_same_program_draws += eligible;
		window.index_same_program_compares += comparable;
		window.index_repeat_same_program += index_repeat_same_program;
		current_program->index_valid = eligible;
		current_program->index       = key;
	}
	void Report() {
		// At most one clock query per 1024 draws, outside the timed work.
		if ((sequence & 1023u) != 0) return;
		const auto now = Clock::now();
		if (now - window_start < std::chrono::seconds {5}) return;
		const auto  seconds = std::chrono::duration<double>(now - window_start).count();
		std::string output  = fmt::format(
            "vi-probe seconds={:.3f} draws={} sampled={} vertex_draws={} vertex_repeat={} "
		     "index_draws={} index_repeat={} host_index_draws={} merge_retry_passes={} "
		     "version=2 timing_unit=ns dirty_observation=unavailable\n",
            seconds, window.draws, window.sampled_draws, window.vertex_draws, window.vertex_repeat,
            window.index_draws, window.index_repeat, window.host_index_draws, window.retry_passes);
		output += fmt::format(
		    "vi-probe program_identity=module_id program_capacity={} program_entries={} "
		    "program_cache_hits={} program_cache_misses={} program_cache_evictions={} "
		    "vertex_same_program_draws={} vertex_same_program_compares={} "
		    "vertex_repeat_same_program={} index_same_program_draws={} "
		    "index_same_program_compares={} index_repeat_same_program={}\n",
		    ProgramCapacity, program_count, window.program_cache_hits, window.program_cache_misses,
		    window.program_cache_evictions, window.vertex_same_program_draws,
		    window.vertex_same_program_compares, window.vertex_repeat_same_program,
		    window.index_same_program_draws, window.index_same_program_compares,
		    window.index_repeat_same_program);
		constexpr std::array names {"vertex", "index", "vertex_retry", "index_retry"};
		for (size_t i = 0; i < names.size(); ++i) {
			const auto& t = window.timings[i];
			output +=
			    fmt::format("vi-probe kind={} sampled_calls={} active_calls={} plan_ns={} "
			                "active_plan_ns={} obtain_ns={} slots_ns={} active_slots_ns={} "
			                "repeat_plan_ns={} repeat_calls={} repeat_obtain_ns={} ranges={} "
			                "stream_ranges={} "
			                "repeat_stream_ranges={} repeat_same_program_calls={} "
			                "repeat_same_program_plan_ns={}\n",
			                names[i], t.calls, t.active_calls, t.plan_ns, t.active_plan_ns,
			                t.obtain_ns, t.slots_ns, t.active_slots_ns, t.repeat_plan_ns,
			                t.repeat_calls, t.repeat_obtain_ns, t.ranges, t.stream, t.repeat_stream,
			                t.repeat_same_program_calls, t.repeat_same_program_plan_ns);
		}
		Log::WriteToConsoleAndLog(output);
		window       = {};
		window_start = Clock::now(); // Logging time is outside the next observation window.
	}
};

template <bool Probe = false>
static PreparedVertexBuffers AcquireVertexBuffers(CommandBuffer&               buffer,
                                                  const ShaderVertexInputInfo& vs_input_info) {
	EXIT_IF(vs_input_info.buffers_num < 0 ||
	        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX);
	if constexpr (!Probe) {
		if (vs_input_info.buffers_num == 0 &&
		    Common::AbSwitch::Enabled(Common::AbSwitch::EMPTY_VERTEX_FAST_PATH)) [[unlikely]] {
			return {};
		}
	}
	VertexIndexProbe::Clock::time_point phase_start;
	if constexpr (Probe) {
		if (VertexIndexProbe::Get().sampled) phase_start = VertexIndexProbe::Clock::now();
	}

	// Collect the non-empty guest vertex ranges.
	std::array<uint64_t, ShaderVertexInputInfo::RES_MAX>          sizes {};
	std::array<VertexBufferRange, ShaderVertexInputInfo::RES_MAX> ranges {};
	uint32_t                                                      range_count = 0;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = VertexBufferDescriptorSize(i, vs_input_info);
		sizes[i]           = size;
		if (size == 0) {
			continue;
		}
		if (vertex.addr == 0 || size > UINT64_MAX - vertex.addr) {
			EXIT("invalid vertex buffer range: addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
			     vertex.addr, size);
		}
		ranges[range_count++] = {vertex.addr, vertex.addr + size};
	}

	std::sort(ranges.begin(), ranges.begin() + range_count,
	          [](const VertexBufferRange& left, const VertexBufferRange& right) {
		          return left.base_address < right.base_address;
	          });

	// Merge overlapping or touching ranges before acquiring host buffers.
	std::array<VertexBufferRange, ShaderVertexInputInfo::RES_MAX> merged_ranges {};
	uint32_t                                                      merged_count = 0;
	for (uint32_t i = 0; i < range_count; i++) {
		const auto& range = ranges[i];
		if (merged_count != 0 &&
		    merged_ranges[merged_count - 1].requested_end >= range.base_address) {
			merged_ranges[merged_count - 1].requested_end =
			    std::max(merged_ranges[merged_count - 1].requested_end, range.requested_end);
			continue;
		}
		merged_ranges[merged_count++] = {range.base_address, range.requested_end};
	}
	if constexpr (Probe) {
		auto& probe = VertexIndexProbe::Get();
		if (probe.sampled) {
			const auto elapsed = VertexIndexProbe::Ns(phase_start, VertexIndexProbe::Clock::now());
			auto&      t       = probe.window.timings[probe.retry ? 2 : 0];
			++t.calls;
			t.plan_ns += elapsed;
			if (range_count != 0) {
				++t.active_calls;
				t.active_plan_ns += elapsed;
			}
			probe.ObserveVertex(vs_input_info, sizes);
			if (!probe.retry && probe.vertex_repeat) {
				++t.repeat_calls;
				t.repeat_plan_ns += elapsed;
			}
			if (!probe.retry && probe.vertex_repeat_same_program) {
				++t.repeat_same_program_calls;
				t.repeat_same_program_plan_ns += elapsed;
			}
		} else {
			probe.ObserveVertex(vs_input_info, sizes);
		}
	}

	auto& cache = buffer.GetContext().GetBufferCache();
	for (uint32_t i = 0; i < merged_count; i++) {
		auto& range = merged_ranges[i];
		// PPSA20298
		const auto size =
		    Libs::LibKernel::Memory::ClampRangeSize(range.base_address, range.RequestedSize());
		range.acquired_end = range.base_address + size;
		if constexpr (Probe) {
			auto& probe = VertexIndexProbe::Get();
			if (probe.sampled) {
				auto& t = probe.window.timings[probe.retry ? 2 : 0];
				++t.ranges;
				phase_start = VertexIndexProbe::Clock::now();
			}
		}
		range.binding = cache.ObtainBuffer(range.base_address, size, false);
		if constexpr (Probe) {
			auto& probe = VertexIndexProbe::Get();
			if (probe.sampled) {
				auto&      t = probe.window.timings[probe.retry ? 2 : 0];
				const auto elapsed =
				    VertexIndexProbe::Ns(phase_start, VertexIndexProbe::Clock::now());
				const bool stream =
				    range.binding.first == &cache.GetUtilityBuffer(MemoryUsage::Stream);
				t.obtain_ns += elapsed;
				t.stream += stream;
				if (!probe.retry && probe.vertex_repeat) {
					t.repeat_obtain_ns += elapsed;
					t.repeat_stream += stream;
				}
			}
		}
	}

	// Rebuild slot bindings, offsetting non-empty slots into their acquired merged range.
	if constexpr (Probe) {
		if (VertexIndexProbe::Get().sampled) phase_start = VertexIndexProbe::Clock::now();
	}
	PreparedVertexBuffers prepared;
	prepared.count         = static_cast<uint32_t>(vs_input_info.buffers_num);
	vk::Buffer null_buffer = nullptr;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = sizes[i];
		if (size == 0) {
			if (null_buffer == nullptr) {
				null_buffer = cache.GetBuffer(NULL_BUFFER_ID).Handle();
			}
			prepared.buffers[i] = null_buffer;
			prepared.offsets[i] = 0;
			continue;
		}

		const auto range = std::find_if(merged_ranges.begin(), merged_ranges.begin() + merged_count,
		                                [&](const VertexBufferRange& value) {
			                                return vertex.addr >= value.base_address &&
			                                       vertex.addr < value.acquired_end;
		                                });
		if (range == merged_ranges.begin() + merged_count) {
			EXIT("vertex buffer address is outside the acquired range: addr=0x%016" PRIx64 "\n",
			     vertex.addr);
		}

		prepared.buffers[i] = range->binding.first->Handle();
		prepared.offsets[i] = range->binding.second + vertex.addr - range->base_address;
		prepared.sizes[i]   = std::min(size, range->acquired_end - vertex.addr);
	}
	if constexpr (Probe) {
		auto& probe = VertexIndexProbe::Get();
		if (probe.sampled) {
			const auto elapsed = VertexIndexProbe::Ns(phase_start, VertexIndexProbe::Clock::now());
			auto&      t       = probe.window.timings[probe.retry ? 2 : 0];
			t.slots_ns += elapsed;
			if (range_count != 0) t.active_slots_ns += elapsed;
		}
	}

	return prepared;
}

static void SetDrawDebugPhase(CommandBuffer& buffer, uint64_t submit_id, const DrawCallInfo& draw,
                              uint32_t phase) {
	const auto op =
	    draw.IsIndexed() ? CommandBufferDebugOp::DrawIndex : CommandBufferDebugOp::DrawIndexAuto;
	buffer.SetDebugInfo(static_cast<uint32_t>(op), submit_id, phase, draw.index_count, 0,
	                    draw.instance_count, draw.first_instance);
}

static bool GetDrawTopology(const HW::UserConfig& ucfg, vk::PrimitiveTopology& topology) {

	topology = vk::PrimitiveTopology::ePointList;

	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kNone: return false;
		case Prospero::PrimitiveType::kPointList:
			topology = vk::PrimitiveTopology::ePointList;
			break;
		case Prospero::PrimitiveType::kLineList: topology = vk::PrimitiveTopology::eLineList; break;
		case Prospero::PrimitiveType::kLineStrip:
			topology = vk::PrimitiveTopology::eLineStrip;
			break;
		case Prospero::PrimitiveType::kTriList:
			topology = vk::PrimitiveTopology::eTriangleList;
			break;
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kPolygon:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		case Prospero::PrimitiveType::kTriStrip:
			topology = vk::PrimitiveTopology::eTriangleStrip;
			break;
		case Prospero::PrimitiveType::kPatch:
			if (!Config::TessellationEnabled()) return false;
			[[fallthrough]];
		case Prospero::PrimitiveType::kRectList:
		case Prospero::PrimitiveType::kRectListLegacy:
			topology = vk::PrimitiveTopology::ePatchList;
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		default: {
			static std::atomic_bool logged = false;
			if (!logged.exchange(true, std::memory_order_relaxed)) {
				std::printf("Skipping draw with unknown primitive type: %u\n",
				            static_cast<uint32_t>(ucfg.GetPrimType()));
			}
			return false;
		}
	}

	return true;
}

static bool ResolvePrimitiveRestart(const CommandBuffer&         buffer,
                                    const DrawIndexBufferSource& source) {
	const auto control = buffer.GetUserConfig().GetPrimitiveResetControl();
	EXIT_NOT_IMPLEMENTED((control & ~0x3u) != 0);
	if ((control & 0x1u) == 0) {
		return false;
	}
	switch (buffer.GetUserConfig().GetPrimType()) {
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip: break;
		default: return false;
	}

	const auto element_size = source.guest_element_size;
	const auto index_mask   = UINT32_MAX >> ((4 - element_size) * 8);
	const auto reset_index  = buffer.GetRegisters().GetPrimitiveResetIndex();
	if ((control & 0x2u) != 0 && (reset_index & ~index_mask) != 0) {
		return false;
	}
	const auto restart_index = reset_index & index_mask;
	if (restart_index == index_mask) {
		// Use native restart; the 8-bit path widens its marker to 0xffff.
		return true;
	}

	// A game can set a custom reset value without using it in the index buffer.
	// Keep restart off in that case; fail if we actually find the value.
	// Scan before preparing draw resources: readback can restart the command buffer.
	EXIT_NOT_IMPLEMENTED(source.address == 0);
	const auto* indices = reinterpret_cast<const uint8_t*>(source.address);
	for (uint64_t offset = 0; offset < source.size; offset += element_size) {
		uint32_t index = 0;
		std::memcpy(&index, indices + offset, element_size);
		EXIT_NOT_IMPLEMENTED(index == restart_index);
	}
	return false;
}

DrawRenderState& RenderExecutor::BeginDrawState(std::optional<DrawRenderState>& local,
                                                const DrawIndirectArgs&         indirect) {
	if (Common::AbSwitch::Enabled(Common::AbSwitch::EXACT_DRAW_STATE)) {
		m_draw_state.Reset(indirect.offsets, indirect.address != 0);
		return m_draw_state;
	}
	auto& fresh            = local.emplace();
	fresh.indirect_offsets = indirect.offsets;
	fresh.indirect_draw    = indirect.address != 0;
	return fresh;
}

uint64_t g_target_state_serial = 0;

// Consecutive draws mostly go to the same render targets: a pass sets them once and then draws
// hundreds of meshes, changing only shader registers between them. Each draw resolved the targets
// from the registers and acquired them again. A draw keeps the whole result (the targets'
// descriptions, their attachments and the rendering state) and the next one takes it when all of
// this holds (AbSwitch DRAW_CHAIN; Bruno Cardenas Torres 28073f9ed):
//  - no command since could have changed the registers the targets resolve from
//    (g_target_state_serial), and the draw writes the same slots at the same slice offset;
//  - the rendering instance the kept draw began is still open on the same command buffer: no
//    barrier, copy or clear was recorded in between, which each end it;
//  - no image was registered or unregistered since (the texture cache's structure generation),
//    and every kept target is still registered and needs no rebind;
//  - the kept draw cleared nothing and read none of its targets, and neither does this one
//    (TargetMemoBound, once its textures are bound: such a draw acquires its targets itself).
// KYTY_VERIFY_TARGET_MEMO=1 acquires again for every taken draw and exits on a difference.
struct TargetMemo {
	bool                                                      valid                = false;
	const void*                                               owner                = nullptr;
	const CommandBuffer*                                      buffer               = nullptr;
	uint64_t                                                  rendering_serial     = 0;
	uint64_t                                                  state_serial         = 0;
	uint64_t                                                  structure_generation = 0;
	uint32_t                                                  mrt_mask             = 0;
	uint32_t                                                  slice_offset         = 0;
	bool                                                      ps_active            = false;
	uint32_t                                                  color_count          = 0;
	std::array<RenderColorInfo, RENDER_COLOR_ATTACHMENTS_MAX> colors;
	std::array<vk::ImageLayout, RENDER_COLOR_ATTACHMENTS_MAX> color_layouts {};
	RenderDepthInfo                                           depth;
	vk::ImageLayout depth_layout = vk::ImageLayout::eUndefined;
	RenderState     rendering;
};
static TargetMemo g_target_memo;

static bool VerifyTargetMemo() {
	static const bool enabled = std::getenv("KYTY_VERIFY_TARGET_MEMO") != nullptr;
	return enabled;
}

bool RenderExecutor::TakeTargetMemo(CommandBuffer& buffer, DrawRenderState& state) {
	const auto& memo = g_target_memo;
	if (!Common::AbSwitch::Enabled(Common::AbSwitch::DRAW_CHAIN)) {
		return false;
	}
	if (!memo.valid || memo.owner != this || memo.buffer != &buffer) {
		Common::DebugCounters::Add(Common::DebugCounters::Counter::TargetMemoInvalidMiss);
		Common::DebugCounters::Add(Common::DebugCounters::Counter::TargetMemoStateMiss);
		return false;
	}
	if (memo.state_serial != g_target_state_serial) {
		Common::DebugCounters::Add(Common::DebugCounters::Counter::TargetMemoSerialMiss);
		Common::DebugCounters::Add(Common::DebugCounters::Counter::TargetMemoStateMiss);
		return false;
	}
	if (!buffer.IsRendering() || memo.rendering_serial != buffer.RenderingSerial()) {
		Common::DebugCounters::Add(Common::DebugCounters::Counter::TargetMemoRenderingMiss);
		Common::DebugCounters::Add(Common::DebugCounters::Counter::TargetMemoStateMiss);
		return false;
	}
	if (memo.mrt_mask != state.mrt_mask || memo.slice_offset != state.slice_offset ||
	    memo.ps_active != state.ps_active) {
		Common::DebugCounters::Add(Common::DebugCounters::Counter::TargetMemoMaskMiss);
		Common::DebugCounters::Add(Common::DebugCounters::Counter::TargetMemoStateMiss);
		return false;
	}
	auto& cache = m_context.GetTextureCache();
	if (memo.structure_generation != cache.StructureGeneration()) {
		Common::DebugCounters::Add(Common::DebugCounters::Counter::TargetMemoStructureMiss);
		return false;
	}
	for (uint32_t i = 0; i < memo.color_count; i++) {
		if (!memo.colors[i].image_id) {
			Common::DebugCounters::Add(Common::DebugCounters::Counter::TargetMemoTargetMiss);
			return false;
		}
		const auto& image = cache.m_slot_images[memo.colors[i].image_id];
		if (!image.registered || image.binding.needs_rebind) {
			Common::DebugCounters::Add(Common::DebugCounters::Counter::TargetMemoTargetMiss);
			return false;
		}
	}
	if (memo.depth.image_id) {
		const auto& image = cache.m_slot_images[memo.depth.image_id];
		if (!image.registered || image.binding.needs_rebind) {
			Common::DebugCounters::Add(Common::DebugCounters::Counter::TargetMemoTargetMiss);
			return false;
		}
	}
	// As resolving and acquiring leave the targets for the rest of the draw.
	for (uint32_t i = 0; i < memo.color_count; i++) {
		m_targets.colors[i]             = memo.colors[i];
		auto& image                     = cache.GetImage(memo.colors[i].image_id);
		image.binding.is_target         = true;
		image.binding.attachment_layout = memo.color_layouts[i];
		image.binding.attachment_access =
		    vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite;
		BindRenderTarget(memo.colors[i].image_id);
	}
	m_targets.color_count = memo.color_count;
	if (memo.depth.image_id) {
		m_targets.depth                = memo.depth;
		auto& image                     = cache.GetImage(memo.depth.image_id);
		image.binding.is_target         = true;
		image.binding.attachment_layout = memo.depth_layout;
		image.binding.attachment_access = vk::AccessFlagBits2::eDepthStencilAttachmentRead |
		                                  vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
		BindRenderTarget(memo.depth.image_id);
	}
	state.target_memo = true;
	Common::DebugCounters::Add(Common::DebugCounters::Counter::DrawChainTaken);
	return true;
}

// Whether the draw that took the memo binds one of its targets as a texture or storage image.
bool RenderExecutor::TargetMemoBound(const DrawRenderState& state) const {
	auto& cache = m_context.GetTextureCache();
	for (uint32_t i = 0; i < m_targets.color_count; i++) {
		if (cache.GetImage(m_targets.colors[i].image_id).binding.is_bound) {
			return true;
		}
	}
	return m_targets.depth.image_id && cache.GetImage(m_targets.depth.image_id).binding.is_bound;
}

void RenderExecutor::KeepTargetMemo(CommandBuffer& buffer, const DrawRenderState& state,
                                    const RenderState&   rendering,
                                    vk::ImageAspectFlags feedback_aspects) {
	auto& memo        = g_target_memo;
	memo.valid        = false;
	const auto& depth = m_targets.depth;
	if (feedback_aspects || !buffer.IsRendering() || TargetMemoBound(state) ||
	    (m_targets.color_count == 0 && !depth.image_id) || depth.depth_clear_enable ||
	    depth.stencil_clear_enable || rendering.depth_stencil_attachment.depth_clear ||
	    rendering.depth_stencil_attachment.stencil_clear) {
		return;
	}
	auto& cache = m_context.GetTextureCache();
	for (uint32_t i = 0; i < m_targets.color_count; i++) {
		const auto& attachment = rendering.color_attachments[m_targets.colors[i].target_slot];
		memo.colors[i]         = m_targets.colors[i];
		memo.color_layouts[i]  = attachment.image_layout;
	}
	memo.color_count = m_targets.color_count;
	memo.depth       = depth;
	if (depth.image_id) {
		memo.depth_layout = cache.GetImage(depth.image_id).binding.attachment_layout;
	}
	memo.rendering            = rendering;
	memo.owner                = this;
	memo.buffer               = &buffer;
	memo.rendering_serial     = buffer.RenderingSerial();
	memo.state_serial         = g_target_state_serial;
	memo.structure_generation = cache.StructureGeneration();
	memo.mrt_mask             = state.mrt_mask;
	memo.slice_offset         = state.slice_offset;
	memo.ps_active            = state.ps_active;
	memo.valid                = true;
}

bool RenderExecutor::PrepareDrawRenderState(CommandBuffer& buffer, const DrawCallInfo& draw,
                                            DrawRenderState& state) {
	m_context.GetImageHeap().BeginDraw();
	m_context.GetPipelineCache().GetGraphicsPrograms(buffer, state);
	// Vertex programs beyond the host mesh limits or with unsupported call targets are not
	// compiled.
	if (state.vertex_info[0].stage.module == nullptr) {
		return false;
	}
	// An empty screen scissor rasterizes nothing in any viewport, so only vertex-stage buffer
	// writes would remain (PPSA02110 ends mip reductions with a 0x0 pass past the last level).
	const auto& vp = buffer.GetRegisters().GetScreenViewport();
	if ((vp.screen_scissor_right <= vp.screen_scissor_left ||
	     vp.screen_scissor_bottom <= vp.screen_scissor_top) &&
	    std::ranges::none_of(state.VertexStages(), HasShaderBufferWrites,
	                         &ShaderVertexInputInfo::stage)) {
		return false;
	}
	auto& targets = m_targets;
	targets.color_count = 0;
	targets.depth = {};
	// Raw render-target view registers carry the complete slice selection.
	state.slice_offset = 0;
	if (TakeTargetMemo(buffer, state)) {
		return true;
	}
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		if ((state.mrt_mask & (1u << slot)) != 0) {
			ResolveRenderColorTarget(buffer, targets.colors[targets.color_count], slot);
			if (targets.colors[targets.color_count].image_id) {
				targets.color_count++;
			}
		}
	}
	ResolveRenderDepthTarget(buffer, targets.depth);
	// TODO(PPSA09243): Vulkan passes take one sample count. Where PS5 attachments mix counts,
	// keep those matching the raster count instead of resolving multisampled shadows.
	const auto colors       = std::span {targets.colors, targets.color_count};
	const auto pass_samples = targets.depth.image_id ? targets.depth.desc.info.samples
	                                                 : targets.colors[0].desc.info.samples;
	if (std::ranges::any_of(colors, [&](const RenderColorInfo& color) {
		    return color.desc.info.samples != pass_samples;
	    })) {
		const auto raster =
		    render_sample_count(buffer.GetRegisters().GetAaConfig().msaa_num_samples);
		const auto dropped = std::ranges::remove_if(colors, [&](const RenderColorInfo& color) {
			return color.desc.info.samples != raster;
		});
		targets.color_count -= static_cast<uint32_t>(dropped.size());
		if (targets.depth.image_id && targets.depth.desc.info.samples != raster) {
			targets.depth = {};
		}
		static std::atomic_bool logged = false;
		if (!logged.exchange(true, std::memory_order_relaxed)) {
			Log::WriteToConsoleAndLog(
			    "Warning: dropping attachments whose sample count differs from the raster.\n");
		}
	}

	if (targets.color_count == 0 && !targets.depth.image_id && !state.ps_active) {
		LogFramebufferSkip(draw.Name(), targets.depth, buffer, draw.index_count, 0);
		return false;
	}

	return true;
}

template <bool Probe = false>
static PreparedIndexBuffer PrepareIndexBuffer(CommandBuffer&               buffer,
                                              const DrawIndexBufferSource& source) {
	PreparedIndexBuffer prepared;
	if constexpr (Probe) VertexIndexProbe::Get().ObserveIndex(source);
	if (source.size == 0) {
		return prepared;
	}
	VertexIndexProbe::Clock::time_point phase_start;
	if constexpr (Probe) {
		auto& probe = VertexIndexProbe::Get();
		if (probe.sampled) {
			auto& t = probe.window.timings[probe.retry ? 3 : 1];
			++t.calls;
			++t.active_calls;
			t.repeat_calls += !probe.retry && probe.index_repeat;
			t.repeat_same_program_calls += !probe.retry && probe.index_repeat_same_program;
			++t.ranges;
			phase_start = VertexIndexProbe::Clock::now();
		}
	}
	prepared.type =
	    source.guest_element_size == 4 ? vk::IndexType::eUint32 : vk::IndexType::eUint16;
	if (source.host_data != nullptr) {
		auto& stream = buffer.GetContext().GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
		prepared.offset = stream.Copy(source.host_data, source.size, 16);
		prepared.buffer = stream.Handle();
	} else {
		auto [buffer_ptr, offset] =
		    buffer.GetContext().GetBufferCache().ObtainBuffer(source.address, source.size, false);
		prepared.buffer = buffer_ptr->Handle();
		prepared.offset = offset;
	}
	if constexpr (Probe) {
		auto& probe = VertexIndexProbe::Get();
		if (probe.sampled) {
			auto&      t       = probe.window.timings[probe.retry ? 3 : 1];
			const auto elapsed = VertexIndexProbe::Ns(phase_start, VertexIndexProbe::Clock::now());
			const bool stream =
			    prepared.buffer ==
			    buffer.GetContext().GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream).Handle();
			t.obtain_ns += elapsed;
			t.stream += stream;
			if (!probe.retry && probe.index_repeat) {
				t.repeat_obtain_ns += elapsed;
				t.repeat_stream += stream;
			}
		}
	}
	return prepared;
}

static void CommitVertexBuffers(vk::CommandBuffer            vk_buffer,
                                const PreparedVertexBuffers& prepared) {
	if (prepared.count != 0) {
		// Guest descriptor bounds must survive allocation merging in the cache.
		vk_buffer.bindVertexBuffers2(0, prepared.count, prepared.buffers.data(),
		                             prepared.offsets.data(), prepared.sizes.data(), nullptr);
	}
}

static void CommitIndexBuffer(vk::CommandBuffer vk_buffer, const PreparedIndexBuffer& prepared) {
	if (prepared.buffer == nullptr) {
		return;
	}
	vk_buffer.bindIndexBuffer(prepared.buffer, prepared.offset, prepared.type);
}

static void LogDrawStateIfNeeded(const CommandBuffer& buffer, const DrawCallInfo& draw,
                                 const DrawRenderState& state, const DrawTargets& targets,
                                 uint32_t index_type_and_size, const void* index_addr) {
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	if (!draw.indexed && !Prospero::IsRectList(buffer.GetUserConfig().GetPrimType())) {
		return;
	}

	static const RenderColorInfo none {};

	const auto& color = targets.color_count != 0 ? targets.colors[0] : none;
	if (state.ps_active) {
		LogDrawTargetState(draw.Name(), color, targets.depth, buffer, state.ps_input_info,
		                   draw.index_count, 0);
	}
	LogDrawInputState(buffer, color, state.vertex_info[0], index_type_and_size, draw.index_count,
	                  index_addr);
}

static void EmitDrawPrimitives(const HW::UserConfig& ucfg, vk::CommandBuffer vk_buffer,
                               const DrawCallInfo& draw, const DrawEmitInfo& emit) {
	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kPointList:
		case Prospero::PrimitiveType::kLineList:
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriList:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip:
		case Prospero::PrimitiveType::kRectList:
		case Prospero::PrimitiveType::kPolygon:
		case Prospero::PrimitiveType::kRectListLegacy:
		case Prospero::PrimitiveType::kPatch:
			if (draw.indexed) {
				vk_buffer.drawIndexed(draw.index_count, draw.instance_count, 0, emit.vertex_offset,
				                      emit.first_instance);
			} else {
				vk_buffer.draw(draw.index_count, draw.instance_count,
				               static_cast<uint32_t>(emit.vertex_offset), emit.first_instance);
			}
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			EXIT_NOT_IMPLEMENTED((draw.index_count & 0x3u) != 0);
			for (uint32_t i = 0; i < draw.index_count; i += 4) {
				if (draw.indexed) {
					vk_buffer.drawIndexed(4, draw.instance_count, i, emit.vertex_offset,
					                      emit.first_instance);
				} else {
					vk_buffer.draw(4, draw.instance_count,
					               i + static_cast<uint32_t>(emit.vertex_offset),
					               emit.first_instance);
				}
			}
			break;
		default: EXIT("unknown primitive type: %u\n", static_cast<uint32_t>(ucfg.GetPrimType()));
	}
}

bool RenderExecutor::CopyDepthToColor(CommandBuffer& buffer, const DrawCallInfo& draw,
                                      const DrawEmitInfo& emit, const DrawRenderState& state) {
	if (!buffer.GetRegisters().GetRenderControl().copy_depth_to_color) {
		return false;
	}
	auto& ucfg = buffer.GetUserConfig();
	const auto& hw = buffer.GetRegisters();
	auto&       depth = m_targets.depth;
	auto&       color = m_targets.colors[0];
	EXIT_NOT_IMPLEMENTED(
	    !m_context.GetGraphics().maintenance8_enabled ||
	    !Prospero::IsRectList(ucfg.GetPrimType()) || draw.indexed || draw.index_count != 3 ||
	    draw.instance_count != 1 || emit.vertex_offset != 0 || emit.first_instance != 0 ||
	    m_targets.color_count != 1 || color.target_slot != 0 || !depth.image_id ||
	    color.desc.info.samples != 1 || depth.desc.info.extent.width != color.Extent().width ||
	    depth.desc.info.extent.height != color.Extent().height ||
	    depth.desc.view_info.layer_count != 1 || color.desc.view_info.layer_count != 1 ||
	    hw.GetBlendControl(0).enable || state.ps_input_info.ps_pixel_kill_enable ||
	    state.ps_input_info.ps_depth_export_enable ||
	    state.ps_input_info.ps_sample_mask_export_enable);
	vk::ImageAspectFlags feedback;
	const auto           rendering = AcquireRenderTargets(buffer, &color, 1, depth, feedback);
	if (depth.load_clear_aspects) {
		buffer.BeginRendering(rendering);
	}
	auto& cache       = m_context.GetTextureCache();
	auto& source      = cache.GetImage(depth.image_id);
	auto& destination = cache.GetImage(color.image_id);
	EXIT_NOT_IMPLEMENTED((source.backing.format != vk::Format::eD32Sfloat &&
	                      source.backing.format != vk::Format::eD32SfloatS8Uint) ||
	                     destination.backing.format != vk::Format::eR32Sfloat);
	const auto rect = calc_final_scissor(hw.GetScreenViewport(), hw.GetScanModeControl(),
	                                     {rendering.width, rendering.height}, 0);
	if (rect.right > rect.left && rect.bottom > rect.top) {
		const vk::ImageCopy region {
		    {vk::ImageAspectFlagBits::eDepth, depth.desc.view_info.base_level,
		     depth.desc.view_info.base_layer, 1},
		    {rect.left, rect.top, 0},
		    {vk::ImageAspectFlagBits::eColor, color.desc.view_info.base_level,
		     color.desc.view_info.base_layer, 1},
		    {rect.left, rect.top, 0},
		    {static_cast<uint32_t>(rect.right - rect.left),
		     static_cast<uint32_t>(rect.bottom - rect.top), 1}};
		buffer.EndRendering();
		const auto command = buffer.Handle();
		source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {},
		               command);
		destination.Transit(vk::ImageLayout::eTransferDstOptimal,
		                    vk::AccessFlagBits2::eTransferWrite, {}, command);
		command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal,
		                  destination.backing.image, vk::ImageLayout::eTransferDstOptimal, 1,
		                  &region);
	}
	return true;
}

void RenderExecutor::ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer,
                                         const DrawCallInfo& draw, DrawRenderState& state,
                                         const DrawEmitInfo&          emit,
                                         const DrawIndexBufferSource& index_source) {
	auto&      ucfg                     = buffer.GetUserConfig();
	const bool primitive_restart_enable = state.primitive_restart_enable;
	if (CopyDepthToColor(buffer, draw, emit, state)) {
		return;
	}
	const auto  vertex_stages = state.VertexStages();
	const bool  mesh_active   = state.vertex_info[0].stage.module->info.stage == ShaderType::Mesh;
	const bool  mesh_task     = state.vertex_info[0].stage.module->mesh_task;
	const auto& mesh          = state.vertex_info[0].mesh;
	const auto& limits        = m_context.GetGraphics().mesh_shader_properties;
	uint32_t    mesh_groups   = 0;
	if (mesh_active) {
		EXIT_NOT_IMPLEMENTED(mesh.fast_launch && (draw.indexed || primitive_restart_enable));
		static std::atomic_bool restart_warned = false;
		if (primitive_restart_enable && !restart_warned.exchange(true, std::memory_order_relaxed)) {
			std::printf("Warning: primitive restart is not implemented for mesh shaders; "
			            "continuing draw (primitive=%u indexed=%u)\n",
			            static_cast<uint32_t>(ucfg.GetPrimType()), draw.indexed);
		}
		if (mesh.primitives_per_group == 0) {
			EXIT("unsupported mesh draw: primitive=%u indexed=%u restart=%u\n",
			     static_cast<uint32_t>(ucfg.GetPrimType()), draw.indexed, primitive_restart_enable);
		}
	}
	if (mesh_task) {
		EXIT_NOT_IMPLEMENTED(emit.indirect.count > std::min(limits.maxTaskWorkGroupCount[0],
		                                                    limits.maxTaskWorkGroupTotalCount));
	} else if (mesh_active) {
		const auto primitives = mesh.InputPrimitiveCount(draw.index_count);
		if (primitives == 0 || draw.instance_count == 0) {
			return;
		}
		mesh_groups = (primitives - 1u) / mesh.primitives_per_group + 1u;
		if (mesh_groups > limits.maxMeshWorkGroupCount[0] ||
		    draw.instance_count > limits.maxMeshWorkGroupCount[1] ||
		    static_cast<uint64_t>(mesh_groups) * draw.instance_count >
		        limits.maxMeshWorkGroupTotalCount) {
			EXIT("mesh draw exceeds host workgroup limits: %ux%u\n", mesh_groups,
			     draw.instance_count);
		}
	}

	// Mesh shaders read the original guest indices in whole DWORDs by device address. Their buffer
	// exists before the descriptors bind, so creating it merges no buffer this draw binds.
	uint64_t index_begin = 0;
	uint64_t index_end   = 0;
	if (mesh_active && draw.indexed) {
		const auto size = static_cast<uint64_t>(draw.index_count) * index_source.guest_element_size;
		index_begin     = index_source.address & ~uint64_t {3};
		index_end       = (index_source.address + size + 3u) & ~uint64_t {3};
		(void)m_context.GetBufferCache().FindBuffer(index_begin, index_end - index_begin);
	}
	auto&                            bindings = m_graphics_bindings;
	std::array<PreparedBindings*, 4> descriptor_stages {};
	uint32_t                         stage_count = 0;
	for (uint32_t i = 0; i < vertex_stages.size(); i++) {
		PrepareBindings(state.vertex_info[i].stage, bindings.vertex[i]);
		descriptor_stages[stage_count++] = &bindings.vertex[i];
	}
	if (state.ps_active) {
		PrepareBindings(state.ps_input_info.stage, bindings.pixel);
		descriptor_stages[stage_count++] = &bindings.pixel;
	}
	const auto stages       = std::span {descriptor_stages.data(), stage_count};
	auto&      buffer_cache = m_context.GetBufferCache();
	auto       merges       = buffer_cache.MergeCount();
	PrepareGraphicsBindings(stages, std::span {m_targets.colors, m_targets.color_count});
	PreparedVertexBuffers vertex_bindings;
	PreparedIndexBuffer   index_binding;
	vk::DeviceAddress     mesh_indices = 0;
	const bool            probe_vertex_index = !mesh_active && VertexIndexProbe::Enabled();
	if (!mesh_active) {
		if (probe_vertex_index) [[unlikely]] {
			VertexIndexProbe::Get().BeginDraw(state.vertex_info[0].stage.module->id);
			vertex_bindings = AcquireVertexBuffers<true>(buffer, state.vertex_info[0]);
			index_binding   = PrepareIndexBuffer<true>(buffer, index_source);
		} else {
			vertex_bindings = AcquireVertexBuffers(buffer, state.vertex_info[0]);
			index_binding   = PrepareIndexBuffer(buffer, index_source);
		}
	} else if (draw.indexed) {
		const auto [indices, offset] =
		    m_context.GetBufferCache().ObtainBuffer(index_begin, index_end - index_begin, false);
		mesh_indices = indices->BufferDeviceAddress() + offset + (index_source.address & 3u);
	}
	std::pair<Buffer*, uint64_t> args_buffer {};
	if (emit.indirect.address != 0) {
		const auto size = static_cast<uint64_t>(emit.indirect.count - 1) * emit.indirect.stride +
		                  (draw.indexed ? sizeof(vk::DrawIndexedIndirectCommand)
		                                : sizeof(vk::DrawIndirectCommand));
		args_buffer = m_context.GetBufferCache().ObtainBuffer(emit.indirect.address, size, false);
		EXIT_IF((args_buffer.second & 3u) != 0);
	}
	auto&                pipeline = m_context.GetPipelineCache().GetGraphicsPipeline(buffer, state, m_targets);
	vk::ImageAspectFlags feedback_aspects {};
	const bool           target_memo_taken = state.target_memo && !TargetMemoBound(state);
	auto                 rendering = target_memo_taken ? g_target_memo.rendering : RenderState {};
	if (!target_memo_taken) {
		rendering = AcquireRenderTargets(buffer, m_targets.colors, m_targets.color_count,
		                                 m_targets.depth, feedback_aspects, stages);
	} else if (VerifyTargetMemo()) [[unlikely]] {
		vk::ImageAspectFlags check_aspects;
		const auto check = AcquireRenderTargets(buffer, m_targets.colors, m_targets.color_count,
		                                        m_targets.depth, check_aspects, stages);
		if (!(check == rendering) || check_aspects) {
			EXIT("target memo: the kept rendering state differs from the one acquired now\n");
		}
	}
	// A buffer obtained after the shader bindings (vertex, index, arguments, target refreshes) can
	// merge away and delete a buffer they already name: bind them all again until none merges.
	for (uint32_t pass = 0; pass < 4 && buffer_cache.MergeCount() != merges; pass++) {
		merges = buffer_cache.MergeCount();
		for (auto* stage: stages) {
			RebindBuffers(*stage);
		}
		if (!mesh_active) {
			if (probe_vertex_index) [[unlikely]] {
				auto& probe = VertexIndexProbe::Get();
				probe.retry = true;
				++probe.window.retry_passes;
				vertex_bindings = AcquireVertexBuffers<true>(buffer, state.vertex_info[0]);
				index_binding   = PrepareIndexBuffer<true>(buffer, index_source);
			} else {
				vertex_bindings = AcquireVertexBuffers(buffer, state.vertex_info[0]);
				index_binding   = PrepareIndexBuffer(buffer, index_source);
			}
		} else if (draw.indexed) {
			const auto [indices, offset] =
			    buffer_cache.ObtainBuffer(index_begin, index_end - index_begin, false);
			mesh_indices = indices->BufferDeviceAddress() + offset + (index_source.address & 3u);
		}
		if (emit.indirect.address != 0) {
			const auto size =
			    static_cast<uint64_t>(emit.indirect.count - 1) * emit.indirect.stride +
			    (draw.IsIndexed() ? sizeof(vk::DrawIndexedIndirectCommand)
			                      : sizeof(vk::DrawIndirectCommand));
			args_buffer = buffer_cache.ObtainBuffer(emit.indirect.address, size, false);
		}
	}
	if (probe_vertex_index) [[unlikely]]
		VertexIndexProbe::Get().Report();

	vk::DeviceAddress task_params = 0;
	if (mesh_task) {
		const MeshTaskParams params {
		    .records        = args_buffer.first->BufferDeviceAddress() + args_buffer.second,
		    .index_address  = mesh_indices,
		    .stride         = emit.indirect.stride,
		    .index_size     = index_source.guest_element_size,
		    .index_limit    = draw.index_count,
		    .primitive_size = mesh.InputPrimitiveSize(),
		    .group_step     = mesh.primitives_per_group * mesh.InputPrimitiveStep(),
		    .max_groups     = limits.maxMeshWorkGroupCount[0],
		    .max_instances  = limits.maxMeshWorkGroupCount[1],
		    .max_total      = limits.maxMeshWorkGroupTotalCount,
		};
		auto& stream = m_context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
		task_params  = stream.BufferDeviceAddress() + stream.Copy(&params, sizeof(params), 8);
	}

	// Resource preparation above may synchronously finish and restart the scheduler, and so may
	// CommitBindings (the image heap's stream copy). After it, every operation targets the current
	// command buffer and cannot touch guest memory.
	SetDrawDebugPhase(buffer, submit_id, draw, draw.IsIndexed() ? 0x100u : 0x200u);
	if (state.ps_active && !draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x300u);
	}
	CommitBindings(buffer, vk::PipelineBindPoint::eGraphics, pipeline, stages);
	auto vk_buffer = buffer.Handle();
	if (!mesh_active) {
		CommitVertexBuffers(vk_buffer, vertex_bindings);
		CommitIndexBuffer(vk_buffer, index_binding);
	}
	using ShaderRecompiler::IR::PushData;
	const auto draw_dwords =
	    PushData::GraphicsDrawDwords(mesh_active, state.ps_input_info.object_primitive_id);
	if (draw_dwords != 0) {
		std::array<uint32_t, PushData::MeshDrawDwordCount> draw_data {};
		if (mesh_task) {
			draw_data[0] = static_cast<uint32_t>(task_params);
			draw_data[1] = static_cast<uint32_t>(task_params >> 32u);
		} else if (mesh_active) {
			draw_data = {draw.index_count, static_cast<uint32_t>(emit.vertex_offset),
			             index_source.guest_element_size,
			             static_cast<uint32_t>(mesh_indices),
			             static_cast<uint32_t>(mesh_indices >> 32u)};
		} else {
			draw_data[0] = buffer.GetUserConfig().GetObjectId();
		}
		vk_buffer.pushConstants(pipeline.pipeline_layout, pipeline.push_stages, 0,
		                        draw_dwords * sizeof(uint32_t), draw_data.data());
	}
	if (const auto planes = vertex_stages.back().user_clip_planes; planes != 0) {
		vk_buffer.pushConstants(pipeline.pipeline_layout, pipeline.push_stages,
		                        PushData::UserClipPlaneDword * sizeof(uint32_t),
		                        std::bit_width(planes) * 4u * sizeof(uint32_t),
		                        buffer.GetRegisters().GetUserClipPlanes().data());
	}

	auto& dynamic_state = m_context.GetCommandScheduler().GetDynamicState();
	// A draw that took the previous draw's targets, with the same last pre-rasterization stage and
	// no command since that could have changed a context register, would record the same values:
	// working them out (sixteen viewports and scissors, a dozen other values) was 3% of the GPU
	// thread in the fork's GPU Jungle (Bruno Cardenas Torres dc7524760).
	// KYTY_VERIFY_DYNAMIC_KEEP=1 records anyway and exits on a difference.
	const bool dynamic_kept = state.target_memo &&
	                          dynamic_state.recorded_module == vertex_stages.back().stage.module &&
	                          dynamic_state.recorded_serial == g_target_state_serial &&
	                          dynamic_state.recorded_rendering == buffer.RenderingSerial() &&
	                          Common::AbSwitch::Enabled(Common::AbSwitch::DRAW_CHAIN_DYNAMIC);
	if (!dynamic_kept) {
		UpdateGraphicsDynamicState(buffer, dynamic_state, vertex_stages.back(), m_targets.depth,
		                           rendering);
	} else {
		Common::DebugCounters::Add(Common::DebugCounters::Counter::DrawChainDynKept);
		if (VerifyDynamicKeep()) [[unlikely]] {
			const auto before = dynamic_state;
			UpdateGraphicsDynamicState(buffer, dynamic_state, vertex_stages.back(),
			                           m_targets.depth, rendering);
			if (!SameDynamicState(before, dynamic_state)) {
				EXIT("dynamic keep: the kept dynamic state differs from the one recorded now\n");
			}
		}
	}
	dynamic_state.SetAttachmentFeedbackLoopAspects(feedback_aspects);
	dynamic_state.Commit(m_context.GetGraphics(), vk_buffer);

	vk::PipelineStageFlags shader_write_stages = {};
	for (const auto& stage: vertex_stages) {
		if (HasShaderBufferWrites(stage.stage)) {
			shader_write_stages |= ShaderPipelineStages(NativeShaderStage(stage.logical_stage));
		}
	}
	if (state.ps_active && HasShaderBufferWrites(state.ps_input_info.stage)) {
		shader_write_stages |= vk::PipelineStageFlagBits::eFragmentShader;
	}
	if (shader_write_stages) {
		// As for dispatches, buffer writes wait for earlier accesses to the memory.
		buffer.EndRendering();
		ShaderWriteHazardBarrier(vk_buffer, shader_write_stages);
	}
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x400u);
	}
	buffer.BeginRendering(rendering);
	if (!target_memo_taken) {
		KeepTargetMemo(buffer, state, rendering, feedback_aspects);
	}
	// The values above are what this command buffer now has for this program, these registers and
	// this rendering instance. Recorded after BeginRendering because the pass serial only names the
	// open instance afterwards: recording before it would cost one recompute per pass.
	dynamic_state.recorded_module    = vertex_stages.back().stage.module;
	dynamic_state.recorded_serial    = g_target_state_serial;
	dynamic_state.recorded_rendering = buffer.RenderingSerial();
	auto& bound = m_context.GetCommandScheduler().GetDynamicState().bound_graphics_pipeline;
	if (bound != pipeline.pipeline ||
	    !Common::AbSwitch::Enabled(Common::AbSwitch::PIPELINE_BIND_DEDUPE)) {
		vk_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.pipeline);
		bound = pipeline.pipeline;
	}
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x500u);
	}
	auto&         occlusion_queries = m_context.GetOcclusionQueries();
	const int32_t occlusion_query   = occlusion_queries.Begin(vk_buffer);
	auto&         scheduler         = m_context.GetCommandScheduler();
	const auto    gpu_timer         = scheduler.BeginGpuShaderTimer();
	const auto    label             = [&] {
        const auto& shaders = buffer.GetShaders();
        const auto& vs      = shaders.GetVs();
        return fmt::format("draw vs {:#x} ps {:#x}",
                           vs.gs_regs.data_addr != 0 ? vs.gs_regs.data_addr : vs.es_regs.data_addr,
		                                  shaders.GetPs().ps_regs.data_addr);
	};
	// The checkpoint also names the programs and the draw size, to find a draw that hangs the GPU.
	scheduler.GpuCheckpoint([&] {
		const auto* ps = state.ps_active ? state.ps_input_info.stage.module : nullptr;
		return fmt::format("{} vs-h{:016x} ps-h{:016x} n{} i{}{}{} rt{}x{}", label(),
		                   state.vertex_info[0].stage.module->info.shader_hash,
		                   ps != nullptr ? ps->info.shader_hash : 0, draw.index_count,
		                   draw.instance_count, args_buffer.first != nullptr ? " indirect" : "",
		                   mesh_active ? " mesh" : "", rendering.width, rendering.height);
	});
	if (mesh_task) {
		vk_buffer.drawMeshTasksEXT(emit.indirect.count, 1, 1);
	} else if (mesh_active) {
		vk_buffer.drawMeshTasksEXT(mesh_groups, draw.instance_count, 1);
	} else if (args_buffer.first == nullptr) {
		EmitDrawPrimitives(ucfg, vk_buffer, draw, emit);
	} else if (draw.indexed) {
		vk_buffer.drawIndexedIndirect(args_buffer.first->Handle(), args_buffer.second,
		                              emit.indirect.count, emit.indirect.stride);
	} else {
		vk_buffer.drawIndirect(args_buffer.first->Handle(), args_buffer.second, emit.indirect.count,
		                       emit.indirect.stride);
	}
	if (gpu_timer != UINT32_MAX) {
		scheduler.EndGpuMarker(gpu_timer, label());
	}
	m_context.GetBufferCache().RetagGpuWrites();
	occlusion_queries.End(vk_buffer, occlusion_query);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::Draws);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::DrawInstances, draw.instance_count);

	if (shader_write_stages) {
		buffer.EndRendering();
		ShaderWriteBarrier(vk_buffer, shader_write_stages);
	}
}

// Vulkan offsets for a prepared draw, or nullopt when its indirect records need the CPU:
// quad-list draws for counts, tessellation draws for the index/SGPR prolog and mesh draws without
// room for the task payload. Indirect records carry their own offsets.
static std::optional<DrawEmitInfo> ResolveDrawEmit(const HW::UserConfig&        ucfg,
                                                   const ShaderVertexInputInfo& vs,
                                                   DrawOffsetSource source, int32_t first_vertex,
                                                   uint32_t                first_instance,
                                                   const DrawIndirectArgs& indirect) {
	if ((indirect.address != 0 || source == DrawOffsetSource::IndirectArgs) &&
	    ucfg.GetIndexOffset() != 0) {
		static std::once_flag warning_once;
		std::call_once(warning_once, [&] {
			Log::WriteToConsoleAndLog(fmt::format(
			    "GPU: indirect draw ignores GE_INDX_OFFSET=0x{:08x}; geometry may be incorrect.\n",
			    ucfg.GetIndexOffset()));
		});
	}
	if (indirect.address != 0) {
		if (!vs.indirect_draw || ucfg.GetPrimType() == Prospero::PrimitiveType::kQuadListLegacy ||
		    vs.indirect_offsets != indirect.offsets) {
			return std::nullopt;
		}
		return DrawEmitInfo {.indirect = indirect};
	}
	if (source == DrawOffsetSource::IndirectArgs) {
		// Native fetches consume the patched SGPRs; rewritten fetches need Vulkan offsets.
		return vs.has_dynamic_vertex_fetch ? DrawEmitInfo {first_vertex, first_instance}
		                                   : DrawEmitInfo {};
	}
	const auto [vertex_offset, instance_offset] = ResolveDrawOffsets(ucfg.GetIndexOffset(), vs);
	return DrawEmitInfo {vertex_offset + first_vertex, instance_offset};
}

bool RenderExecutor::DrawIndex(uint64_t submit_id, CommandBuffer& buffer,
                               const DrawIndexArgs& args) {
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndex), submit_id,
	                    args.index_count, 0, 1, args.instance_count,
	                    reinterpret_cast<uint64_t>(args.index_addr));

	Common::LockGuard lock(m_context.GetMutex());
	Common::AbSwitch::DelayProbe();
	// A chained compute dispatch's barrier belongs before anything this draw records (AbSwitch
	// COMPUTE_CHAIN; the draw reads what the dispatch wrote).
	m_context.GetCommandScheduler().FlushDeferredShaderAccessBarrier(
	    !Common::AbSwitch::Enabled(Common::AbSwitch::BARRIER_LITE));
	if (args.index_count == 0 || (args.instance_count == 0 && args.indirect.address == 0)) {
		return true;
	}

	if (ConsumeMetadataColorOperation(buffer) || DepthStencilCopy(buffer) ||
	    ResolveColorTargets(buffer)) {
		ResetBindings();
		return true;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		return true;
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndex():Shader:\n");
		uc_print("GraphicsRenderDrawIndex():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndex():Parameters:\n"
		     "\t index_type_and_size = 0x%08" PRIx32 "\n"
		     "\t index_count         = 0x%08" PRIx32 "\n"
		     "\t index_addr          = 0x%016" PRIx64 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t base_vertex         = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     args.index_type_and_size, args.index_count,
		     reinterpret_cast<uint64_t>(args.index_addr), args.instance_count,
		     static_cast<uint32_t>(args.base_vertex), args.first_instance);
	}

	uc_check(ucfg);

	hw_check(buffer);

	std::optional<DrawRenderState> local_state;
	DrawRenderState&               state = BeginDrawState(local_state, args.indirect);
	if (!GetDrawTopology(ucfg, state.topology)) {
		return true;
	}

	DrawIndexBufferSource index_source {};
	index_source.address = reinterpret_cast<uint64_t>(args.index_addr);
	switch (static_cast<Prospero::IndexType>(args.index_type_and_size)) {
		case Prospero::IndexType::kIndex16: index_source.guest_element_size = 2; break;
		case Prospero::IndexType::kIndex32: index_source.guest_element_size = 4; break;
		case Prospero::IndexType::kIndex8: index_source.guest_element_size = 1; break;
		default: EXIT("unknown index_type_and_size: %u\n", args.index_type_and_size);
	}
	index_source.size = static_cast<uint64_t>(args.index_count) * index_source.guest_element_size;
	state.primitive_restart_enable = ResolvePrimitiveRestart(buffer, index_source);

	std::vector<uint16_t> expanded_indices;
	if (index_source.guest_element_size == 1) {
		EXIT_NOT_IMPLEMENTED(args.index_addr == nullptr);
		const auto* src = static_cast<const uint8_t*>(args.index_addr);
		expanded_indices.resize(args.index_count);
		for (uint32_t i = 0; i < args.index_count; i++) {
			expanded_indices[i] =
			    state.primitive_restart_enable && src[i] == 0xffu ? 0xffffu : src[i];
		}
		index_source.host_data = expanded_indices.data();
		index_source.size      = expanded_indices.size() * sizeof(uint16_t);
	}

	const DrawCallInfo draw {true, args.index_count, args.instance_count, args.first_instance};
	if (!PrepareDrawRenderState(buffer, draw, state)) {
		ResetBindings();
		return true;
	}

	LogDrawStateIfNeeded(buffer, draw, state, m_targets, args.index_type_and_size, args.index_addr);
	const auto emit = ResolveDrawEmit(ucfg, state.vertex_info[0], args.offset_source,
	                                  args.base_vertex, args.first_instance, args.indirect);
	if (emit) {
		ExecutePreparedDraw(submit_id, buffer, draw, state, *emit, index_source);
	}
	ResetBindings();
	return emit.has_value();
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool RenderExecutor::DrawAuto(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args) {
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndexAuto), submit_id,
	                    args.vertex_count, 0, args.first_vertex, args.instance_count,
	                    args.first_instance);

	Common::LockGuard lock(m_context.GetMutex());
	Common::AbSwitch::DelayProbe();
	// A chained compute dispatch's barrier belongs before anything this draw records.
	m_context.GetCommandScheduler().FlushDeferredShaderAccessBarrier(
	    !Common::AbSwitch::Enabled(Common::AbSwitch::BARRIER_LITE));
	if (args.indirect.address == 0 && (args.vertex_count == 0 || args.instance_count == 0)) {
		return true;
	}

	if (ConsumeMetadataColorOperation(buffer) || DepthStencilCopy(buffer) ||
	    ResolveColorTargets(buffer)) {
		ResetBindings();
		return true;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		return true;
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndexAuto():Shader:\n");
		uc_print("GraphicsRenderDrawIndexAuto():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndexAuto():Parameters:\n"
		     "\t vertex_count        = 0x%08" PRIx32 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t first_vertex        = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     args.vertex_count, args.instance_count, args.first_vertex, args.first_instance);
	}

	uc_check(ucfg);

	hw_check(buffer);

	const DrawCallInfo draw {false, args.vertex_count, args.instance_count, args.first_instance};

	std::optional<DrawRenderState> local_state;
	DrawRenderState&               state = BeginDrawState(local_state, args.indirect);
	if (!GetDrawTopology(ucfg, state.topology)) {
		ResetBindings();
		return true;
	}
	if (!PrepareDrawRenderState(buffer, draw, state)) {
		ResetBindings();
		return true;
	}

	const bool rect_list = Prospero::IsRectList(ucfg.GetPrimType());
	if (rect_list && state.vertex_info[0].buffers_num == 0 &&
	    state.vertex_info[0].stage.module->info.param_export_mask == 0 &&
	    state.ps_input_info.input_num != 0) {
		if (graphics_debug_dump_enabled()) {
			LOGF("DrawIndexAuto: skipping rect-list draw with no VS param exports and PS inputs: "
			     "ps_inputs=%u ps=0x%016" PRIx64 " es=0x%016" PRIx64 " gs=0x%016" PRIx64 "\n",
			     state.ps_input_info.input_num, sh_ctx.GetPs().ps_regs.data_addr,
			     sh_ctx.GetVs().es_regs.data_addr, sh_ctx.GetVs().gs_regs.data_addr);
		}
		ResetBindings();
		return true;
	}

	LogDrawStateIfNeeded(buffer, draw, state, m_targets, 0, nullptr);
	const auto emit = ResolveDrawEmit(ucfg, state.vertex_info[0], args.offset_source,
	                                  static_cast<int32_t>(args.first_vertex), args.first_instance,
	                                  args.indirect);
	if (emit) {
		ExecutePreparedDraw(submit_id, buffer, draw, state, *emit, {});
	}
	ResetBindings();
	return emit.has_value();
}

bool RenderExecutor::ResolveColorTargets(CommandBuffer& buffer) {
	const auto& hw = buffer.GetRegisters();
	if (hw.GetColorControl().mode != 3) {
		return false;
	}

	const auto& src_rt = hw.GetRenderTarget(0);
	const auto& dst_rt = hw.GetRenderTarget(1);
	if (src_rt.base.addr == 0 || dst_rt.base.addr == 0) {
		return false;
	}

	RenderColorInfo src {};
	RenderColorInfo dst {};
	ResolveRenderColorTarget(buffer, src, 0, true, true);
	ResolveRenderColorTarget(buffer, dst, 1, true, true);
	if (src.desc.info.data.address == dst.desc.info.data.address &&
	    src.guest_mip_level == dst.guest_mip_level &&
	    src.guest_array_layer == dst.guest_array_layer) {
		return true;
	}

	auto& cache = m_context.GetTextureCache();
	cache.MarkGpuWritten(dst.image_id);
	auto& source      = cache.GetImage(src.image_id);
	auto& destination = cache.GetImage(dst.image_id);
	destination.Resolve(source,
	                    {src.desc.view_info.base_level, 1, src.desc.view_info.base_layer, 1},
	                    {dst.desc.view_info.base_level, 1, dst.desc.view_info.base_layer, 1});
	return true;
}

} // namespace Libs::Graphics
