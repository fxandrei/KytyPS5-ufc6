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
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/drawPassStats.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/presentation/window/streamline.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <algorithm>
#include <bit>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fmt/format.h>
#include <string>
#include <limits>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {
static bool FillSourcesDisjoint(std::span<const ShaderRecompiler::IR::DescriptorValue> sources,
                                GuestRange destination, uint32_t output_buffer = UINT32_MAX) {
	for (uint32_t i = 0; i < sources.size(); ++i) {
		if (i == output_buffer) continue;
		const auto source = DecodeNativeDescriptor<ShaderBufferResource>(sources[i]);
		const auto bytes  = source.GetSize();
		if (source.Base48() < destination.End() && destination.address < source.Base48() + bytes)
			return false;
	}
	return true;
}

bool RenderExecutor::TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
                                                const CommandBuffer&          buffer) {
	const auto& program   = input.stage.module->info;
	const auto& resources = *input.stage.resources;
	if (resources.buffers.size() != program.info.buffers.size()) {
		EXIT("compute runtime buffer count does not match shader metadata\n");
	}
	auto& cache = buffer.GetContext().GetTextureCache();
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		const auto& resource   = program.info.buffers[i];
		const auto  descriptor = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
		// A metadata resource that is also read is not proven to be a full overwrite. Execute it
		// conservatively instead of replacing the dispatch with a coarse full-surface clear.
		if ((!resource.written || resource.read) && cache.IsMeta(descriptor.Base48())) {
			return false;
		}
	}

	if (!program.info.has_bitwise_xor) {
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& resource = program.info.buffers[i];
			if (resource.written) {
				const auto descriptor =
				    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
				if (cache.ClearMeta(descriptor.Base48())) {
					return true;
				}
			}
		}
	}
	return false;
}

bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                              uint32_t group_y, uint32_t group_z, uint32_t mode,
                              ShaderBufferResource& resolved_descriptor, uint32_t& resolved_clear,
                              uint64_t& resolved_size) {
	const auto& resources = *input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	if (fill.kind != ShaderRecompiler::IR::UniformFillKind::Buffer) {
		return false;
	}
	const auto element_size = fill.words * sizeof(uint32_t);
	const auto descriptor =
	    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[fill.resource]);
	constexpr std::array formats {
	    Prospero::BufferFormat::k32UInt, Prospero::BufferFormat::k32_32UInt,
	    Prospero::BufferFormat::k32_32_32UInt, Prospero::BufferFormat::k32_32_32_32UInt};
	// Offset-or-swizzle bounds do not describe the element range proved by the shader.
	if (descriptor.Stride() != element_size || descriptor.Format() != formats[fill.words - 1] ||
	    descriptor.SwizzleEnabled() || descriptor.IndexStride() != 0 || descriptor.AddTid() ||
	    descriptor.OutOfBounds() == 3 || descriptor.Base48() == 0) {
		return false;
	}
	if (input.threads_num[0] == 0 || input.threads_num[0] != fill.group_stride[0] ||
	    input.threads_num[1] != 1 || input.threads_num[2] != 1 || group_x == 0 || group_y != 1 ||
	    group_z != 1 || mode != (input.dispatch_thread_dimensions ? 0x61u : 0x41u)) {
		return false;
	}
	const uint64_t invocations = input.dispatch_thread_dimensions
	                                 ? group_x
	                                 : static_cast<uint64_t>(group_x) * input.threads_num[0];
	const auto     size        = descriptor.GetSize();
	const auto     elements    = descriptor.NumRecords();
	const bool     covered     = fill.bounded_elements != 0
	                                 ? fill.bounded_elements == elements && invocations >= elements
	                                 : invocations == elements;
	if (!covered || invocations > uint64_t {UINT32_MAX} + 1 || size == 0 || size > UINT32_MAX ||
	    (input.dispatch_thread_dimensions &&
	     (group_x % input.threads_num[0] != 0 || input.dispatch_threads_num[0] != group_x ||
	      input.dispatch_threads_num[1] != 1 || input.dispatch_threads_num[2] != 1))) {
		return false;
	}
	if (!FillSourcesDisjoint(resources.buffers, {descriptor.Base48(), size}, fill.resource))
		return false;
	resolved_descriptor = descriptor;
	resolved_clear      = fill.value;
	resolved_size       = size;
	return true;
}

// KYTY_TRACE_PASS_IMAGES=<seconds> (2026-11-20, frame-generation track): for that long, starting
// KYTY_TRACE_PASS_AT seconds after the first traced dispatch (default 150), prints every compute
// dispatch with its shader hash, grid and bound images (guest address, Vulkan format, extent,
// sampled/storage). Used to find the game's temporal AA / upscale pass and its depth and motion
// vector inputs.
static void TracePassImages(const char* path, const ShaderComputeInputInfo& input,
                            const PreparedBindings& bindings, uint32_t x, uint32_t y,
                            uint32_t z) {
	static const double window = [] {
		const char* value = std::getenv("KYTY_TRACE_PASS_IMAGES");
		return value != nullptr ? std::atof(value) : 0.0;
	}();
	if (window <= 0.0) return;
	static const double start_after = [] {
		const char* value = std::getenv("KYTY_TRACE_PASS_AT");
		return value != nullptr ? std::atof(value) : 150.0;
	}();
	static const auto first = std::chrono::steady_clock::now();
	const double      since =
	    std::chrono::duration<double>(std::chrono::steady_clock::now() - first).count();
	if (since < start_after || since > start_after + window) return;
	std::string line = fmt::format("pass-images: t={:.4f} {} cs={:016x} grid={}x{}x{}", since, path,
	                               input.stage.module->info.shader_hash, x, y, z);
	for (size_t i = 0; i < bindings.images.size(); ++i) {
		const auto& image = bindings.images[i];
		if (!image.image_id) continue;
		const auto& info = image.desc.info;
		line += fmt::format(" [{} {:x} f{} {}x{}x{} {}]", i, info.data.address,
		                    static_cast<uint32_t>(info.pixel_format), info.extent.width,
		                    info.extent.height, info.resources.layers,
		                    image.desc.type == TextureCache::BindingType::Storage ? "st" : "tx");
	}
	line += '\n';
	std::fputs(line.c_str(), stdout);
}

// KYTY_DLSSG=1 (frame generation, 2026-11-20): per frame, counts the R32_SFLOAT sampled images the
// compute passes bind (UFC 6 samples its depth that way in ~40 passes) and, at the game's temporal AA
// dispatch, tags its motion vectors and the most used R32F image of the same size as depth.
struct DlssgDepthCandidate {
	ImageId                       id;
	Streamline::TagImage          image;
	uint32_t                      uses = 0;
};
static std::vector<DlssgDepthCandidate> g_dlssg_depth;

static Streamline::TagImage DlssgTagImage(TextureCache& cache, const TextureBinding& binding) {
	Streamline::TagImage tag {};
	auto& image = cache.GetImage(binding.image_id);
	tag.image   = static_cast<VkImage>(image.backing.image);
	tag.view    = static_cast<VkImageView>(binding.image_view);
	tag.format  = static_cast<uint32_t>(image.backing.format);
	tag.width   = image.backing.extent.width;
	tag.height  = image.backing.extent.height;
	tag.layout  = static_cast<uint32_t>(image.backing.subresource_states.empty() ? image.backing.state.layout : image.backing.subresource_states[0].layout);
	return tag;
}

// Measurement only (2026-11-21): KYTY_SKIP_CS=<hash>[;<hash>] drops the dispatches of those compute
// shaders (shader_hash in hex, as the GPU timer labels print it). Rendering changes; never an FPS claim.
static bool SkipComputeShader(uint64_t hash) {
	static const std::vector<uint64_t> list = [] {
		std::vector<uint64_t> out;
		const char* v = std::getenv("KYTY_SKIP_CS");
		if (v == nullptr) return out;
		std::string text(v);
		size_t      begin = 0;
		while (begin < text.size()) {
			const auto end = text.find_first_of(";|+ ", begin);
			const auto tok = text.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
			if (!tok.empty()) out.push_back(std::strtoull(tok.c_str(), nullptr, 16));
			begin = end == std::string::npos ? text.size() : end + 1;
		}
		std::printf("skip-cs: %zu shader hashes\n", out.size());
		return out;
	}();
	return !list.empty() && std::find(list.begin(), list.end(), hash) != list.end();
}

static void DlssgNoteDispatch(RenderContext& context, CommandBuffer& buffer,
                              const ShaderComputeInputInfo& input, const PreparedBindings& bindings) {
	if (!Streamline::Active()) return;
	auto& cache = context.GetTextureCache();
	for (const auto& binding: bindings.images) {
		if (!binding.image_id || binding.desc.type == TextureCache::BindingType::Storage ||
		    binding.desc.info.pixel_format != vk::Format::eR32Sfloat || !binding.image_view)
			continue;
		auto it = std::ranges::find(g_dlssg_depth, binding.image_id, &DlssgDepthCandidate::id);
		if (it == g_dlssg_depth.end()) {
			g_dlssg_depth.push_back({binding.image_id, {}, 0});
			it = std::prev(g_dlssg_depth.end());
		}
		it->image = DlssgTagImage(cache, binding);
		it->uses++;
	}
	if (input.stage.module->info.shader_hash != Streamline::TaaShaderHash()) return;
	const auto mvec_index = Streamline::MotionVectorBinding();
	if (mvec_index < bindings.images.size() && bindings.images[mvec_index].image_id &&
	    bindings.images[mvec_index].image_view) {
		const auto motion = DlssgTagImage(cache, bindings.images[mvec_index]);
		const DlssgDepthCandidate* best = nullptr;
		for (const auto& candidate: g_dlssg_depth) {
			if (candidate.image.width == motion.width && candidate.image.height == motion.height &&
			    (best == nullptr || candidate.uses > best->uses))
				best = &candidate;
		}
		if (best != nullptr) {
			// The TAA constant buffer (buffer 0): read at dispatch time - it is ring-allocated per frame.
			Streamline::FrameConstants constants {};
			const auto& snapshot = *input.stage.resources;
			if (!snapshot.buffers.empty()) {
				const auto vsharp = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[0]);
				const auto base   = vsharp.Base48();
				uint32_t   cb[40] {};
				if (base != 0 && vsharp.GetSize() >= sizeof(cb) &&
				    !context.GetBufferCache().HasGpuDirtyBytes(base, sizeof(cb)) &&
				    LibKernel::Memory::TryReadBacking(base, cb, sizeof(cb))) {
					const auto f        = [&](uint32_t i) { return std::bit_cast<float>(cb[i]); };
					const float a       = f(0);
					const float b       = f(1);
					constants.jitter_x  = f(28);
					constants.jitter_y  = f(29);
					constants.width     = cb[4];
					constants.height    = cb[5];
					constants.reset     = cb[39] == 0;
					constants.near_z    = (1.0f - a) != 0.0f ? b / (1.0f - a) : 0.1f;
					constants.far_z     = a < 0.0f ? -b / a : 100000.0f;
					constants.valid     = constants.near_z > 0.0f && constants.width != 0 &&
					                      constants.height != 0;
				}
			}
			Streamline::TagFrame(static_cast<VkCommandBuffer>(buffer.Handle()), motion, best->image,
			                     constants);
		}
	}
	g_dlssg_depth.clear();
}

// Opt-in attribution only. This logs the physical buffer actually bound (which can be a stream
// copy), and can surround selected dispatches with full dependencies. It is not an FPS mode.
static bool ComputeDiagnosticEnabled() {
	static const bool enabled = [] {
		const auto* path = std::getenv("KYTY_COMPUTE_DIAG");
		return path != nullptr && *path != '\0';
	}();
	return enabled;
}

static bool TraceComputeDiagnostic(RenderContext& context, const ShaderComputeInputInfo& input,
                                   const PreparedBindings& bindings, bool indirect, bool& isolate) {
	struct Trace {
		std::FILE*                             file = nullptr;
		std::vector<uint64_t>                  hashes;
		std::unordered_map<uint64_t, uint64_t> last;
		bool                                   isolate = false;
		uint32_t                               rows    = 0;
		Trace() {
			const auto* path     = std::getenv("KYTY_COMPUTE_DIAG");
			file                 = path != nullptr ? std::fopen(path, "w") : nullptr;
			const auto* selected = std::getenv("KYTY_COMPUTE_DIAG_HASHES");
			if (selected != nullptr) {
				std::string list(selected);
				for (size_t begin = 0; begin < list.size();) {
					const auto end    = list.find(';', begin);
					const auto token  = list.substr(begin, end - begin);
					char*      parsed = nullptr;
					const auto hash   = std::strtoull(token.c_str(), &parsed, 16);
					if (parsed != token.c_str() && *parsed == '\0') hashes.push_back(hash);
					if (end == std::string::npos) break;
					begin = end + 1;
				}
			}
			const auto* serial = std::getenv("KYTY_COMPUTE_DIAG_ISOLATE");
			isolate            = serial != nullptr && std::strcmp(serial, "1") == 0;
			if (file != nullptr) {
				std::fprintf(
				    file, "monotonic_us,tick,hash,module,indirect,role,resource,descriptor,guest,"
				          "guest_size,stride,records,written,native,offset,range,allocation_size,"
				          "usage,memory_flags,mapped,coherent\n");
				std::fflush(file);
				std::printf(
				    "Compute diagnostic: enabled; isolate=%s selected=%zu timing_gates=2 (not an FPS run)\n",
				    isolate ? "true" : "false", hashes.size());
			}
		}
		~Trace() {
			if (file != nullptr) std::fclose(file);
		}
	};
	static Trace trace;
	const auto&  program = input.stage.module->info;
	if (trace.file == nullptr ||
	    (!trace.hashes.empty() &&
	     std::ranges::find(trace.hashes, program.shader_hash) == trace.hashes.end()))
		return false;
	const auto now  = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                               std::chrono::steady_clock::now().time_since_epoch())
	                                            .count());
	auto&      last = trace.last[program.shader_hash];
	if (trace.rows < 20000 && (last == 0 || now - last >= 1000000)) {
		last             = now;
		auto&      cache = context.GetBufferCache();
		const auto emit  = [&](const char* role, int resource, int descriptor, uint64_t guest,
                              uint64_t size, uint32_t stride, uint32_t records, bool written,
                              const vk::DescriptorBufferInfo& native, const Buffer* backing) {
            const auto raw    = static_cast<VkBuffer>(native.buffer);
            uint64_t   handle = 0;
            static_assert(sizeof(raw) == sizeof(handle));
            std::memcpy(&handle, &raw, sizeof(handle));
            const auto flags =
                backing != nullptr ? static_cast<uint32_t>(backing->MemoryProperties()) : 0u;
            std::fprintf(trace.file,
			              "%" PRIu64 ",%" PRIu64 ",%016" PRIx64 ",%" PRIu64
			              ",%u,%s,%d,%d,%016" PRIx64 ",%" PRIu64 ",%u,%u,%u,%016" PRIx64 ",%" PRIu64
			              ",%" PRIu64 ",%" PRIu64 ",%d,%u,%u,%u\n",
			              now, context.GetCommandScheduler().CurrentTick(), program.shader_hash,
			              input.stage.module->id, indirect ? 1u : 0u, role, resource, descriptor,
			              guest, size, stride, records, written ? 1u : 0u, handle,
			              uint64_t {native.offset}, uint64_t {native.range},
                         backing != nullptr ? backing->Size() : 0,
                         backing != nullptr ? static_cast<int>(backing->Usage()) : -1, flags,
			              backing != nullptr && !backing->Mapped().empty(),
			              backing != nullptr && backing->IsCoherent());
            ++trace.rows;
		};
		for (uint32_t i = 0; i < program.info.buffers.size(); ++i) {
			const auto& metadata = program.info.buffers[i];
			const auto  index    = metadata.descriptor_index;
			if (index == UINT32_MAX || index >= bindings.buffers.size() ||
			    index >= bindings.buffer_sources.size())
				continue;
			const auto descriptor =
			    DecodeNativeDescriptor<ShaderBufferResource>(input.stage.resources->buffers[i]);
			const auto&   native  = bindings.buffers[index];
			const Buffer* backing = nullptr;
			if (const auto id = bindings.buffer_sources[index].id; id) {
				const auto& candidate = cache.GetBuffer(id);
				if (candidate.Handle() == native.buffer) backing = &candidate;
			}
			if (backing == nullptr) {
				for (const auto usage: {MemoryUsage::Stream, MemoryUsage::Upload,
				                        MemoryUsage::Download, MemoryUsage::DeviceLocal}) {
					const auto& candidate = cache.GetUtilityBuffer(usage);
					if (candidate.Handle() == native.buffer) backing = &candidate;
				}
			}
			emit("buffer", static_cast<int>(i), static_cast<int>(index), descriptor.Base48(),
			     descriptor.GetSize(), descriptor.Stride(), descriptor.NumRecords(),
			     metadata.written, native, backing);
		}
		if (program.info.uses_gds) {
			const auto* gds = cache.GetGdsBuffer();
			emit("gds", -1, -1, 0, gds->Size(), 4, static_cast<uint32_t>(gds->Size() / 4), true,
			     vk::DescriptorBufferInfo {gds->Handle(), 0, gds->Size()}, gds);
		}
		std::fflush(trace.file);
	}
	isolate = trace.isolate;
	return true;
}

static void TraceByteStorageFallback(const char* path, const ShaderComputeInputInfo& input,
                                    uint32_t x, uint32_t y, uint32_t z, uint32_t mode) {
	static const bool enabled = std::getenv("KYTY_TRACE_STENCIL_STORAGE") != nullptr;
	if (!enabled) return;
	const auto& program = input.stage.module->info;
	const auto& resources = *input.stage.resources;
	for (size_t i = 0; i < std::min(program.info.images.size(), resources.images.size()); ++i) {
		const auto& resource = program.info.images[i];
		const auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
		if (!resource.written ||
		    resource.resource_class != ShaderRecompiler::IR::ImageResourceClass::Storage ||
		    descriptor.Format() != Prospero::BufferFormat::k8UInt) continue;
		const auto& fill = resources.uniform_fill;
		Log::WriteToConsoleAndLog(fmt::format(
		    "Byte storage fallback: path={} shader=0x{:016x} module={} resource={} "
		    "address=0x{:016x} dispatch={}x{}x{} mode=0x{:x} threads={}x{}x{} "
		    "thread_dimensions={} fill_kind={} fill_value={} stride={}x{}x{} "
		    "specialization_reads={} type={} extent={}x{} layers={}+{} mip={}..{} "
		    "meta={} write_compress={} swizzle_x={}\n",
		    path, program.shader_hash, input.stage.module->id, i, descriptor.Base40(),
		    x, y, z, mode, input.threads_num[0], input.threads_num[1], input.threads_num[2],
		    input.dispatch_thread_dimensions, static_cast<uint32_t>(fill.kind), fill.value,
		    fill.group_stride[0], fill.group_stride[1], fill.group_stride[2],
		    resources.specialization_reads.size(), static_cast<uint32_t>(descriptor.Type()),
		    descriptor.Width5() + 1u, descriptor.Height5() + 1u, descriptor.BaseArray5(),
		    descriptor.Depth() + 1u, descriptor.BaseLevel(), descriptor.LastLevel(),
		    descriptor.MetaCompress(), descriptor.WriteCompress(), descriptor.DstSelX()));
	}
}

bool RenderExecutor::TryConsumeComputeFill(const ShaderComputeInputInfo& input,
                                           CommandBuffer& command, uint32_t group_x,
                                           uint32_t group_y, uint32_t group_z, uint32_t mode) {
	const auto& program   = input.stage.module->info;
	const auto& resources = *input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	auto&       cache     = command.GetContext().GetTextureCache();
	if (fill.kind == ShaderRecompiler::IR::UniformFillKind::Image) {
		if (mode != 0x41u || input.dispatch_thread_dimensions || fill.value > 255 ||
		    input.threads_num[2] != 1)
			return false;
		const auto  descriptor = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[0]);
		const auto& resource   = program.info.images[0];
		if (descriptor.IsNull() || descriptor.Format() != Prospero::BufferFormat::k8UInt ||
		    descriptor.Type() != Prospero::ImageType::kColor2DArray || descriptor.MetaCompress() ||
		    descriptor.WriteCompress() || descriptor.BaseLevel() > descriptor.LastLevel() ||
		    descriptor.BaseLevel() > descriptor.MaxMip() ||
		    descriptor.BaseArray5() > descriptor.Depth() || descriptor.DstSelX() != 4)
			return false;
		const std::array extents {
		    std::max(1u, (descriptor.Width5() + 1u) >> descriptor.BaseLevel()),
		    std::max(1u, (descriptor.Height5() + 1u) >> descriptor.BaseLevel()),
		    descriptor.Depth() - descriptor.BaseArray5() + 1u};
		const std::array groups {group_x, group_y, group_z};
		for (uint32_t axis = 0; axis < 3; ++axis) {
			const uint64_t threads = input.threads_num[axis];
			// Guest image writes outside the descriptor dimensions are discarded. Only the
			// final workgroup may extend beyond the selected image view.
			if (threads == 0 || threads != fill.group_stride[axis] ||
			    groups[axis] != (extents[axis] + threads - 1) / threads ||
			    groups[axis] * threads > UINT32_MAX)
				return false;
		}
		// Proven native clears use the canonical stencil attachment, not an R8 alias.
		const auto  binding     = ResolveTexture(resource, resources.images[0], 0, UINT32_MAX, false);
		const auto& destination = binding.desc.info.data;
		if (!FillSourcesDisjoint(resources.buffers, destination)) return false;
		std::scoped_lock lock {cache.m_lock};
		const auto&      image = cache.GetImage(binding.image_id);
		const auto&      view  = binding.desc.view_info;
		if (image.backing.format != vk::Format::eD32SfloatS8Uint || image.info.samples != 1 ||
		    image.info.stencil != destination || view.base_level >= image.backing.mip_levels ||
		    view.base_layer >= image.backing.layers || view.layer_count != extents[2] ||
		    view.layer_count > image.backing.layers - view.base_layer ||
		    std::max(1u, image.info.extent.width >> view.base_level) != extents[0] ||
		    std::max(1u, image.info.extent.height >> view.base_level) != extents[1])
			return false;
		const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eStencil, view.base_level,
		                                       1, view.base_layer, view.layer_count};
		vk::ClearValue                  clear {};
		clear.depthStencil = vk::ClearDepthStencilValue {0.0f, fill.value};
		cache.ClearImage(command, binding.image_id, image.backing.format, range, clear);
		return true;
	}
	ShaderBufferResource descriptor;
	uint32_t             packed_clear = 0;
	uint64_t             size         = 0;
	if (!ResolveComputeBufferFill(input, group_x, group_y, group_z, mode, descriptor, packed_clear,
	                              size)) {
		return false;
	}
	const bool image_clear =
	    cache.ClearImageFromBuffer(command, descriptor.Base48(), size, packed_clear);
	if (!image_clear) {
		// CPU fills invalidate overlapping cached images. Preserve the native dispatch for
		// partial or ambiguous image backings, including pooled metadata aliases.
		if ((descriptor.Base48() & 3u) != 0 || !m_context.IsMapped(descriptor.Base48(), size) ||
		    cache.IsRegionRegistered(descriptor.Base48(), size)) {
			return false;
		}
		// Metadata need not have an image backing yet. Keep proven fills in guest memory
		// when CPU-owned, so color-clear discovery can read them without a GPU download.
		m_context.GetBufferCache().FillBuffer(descriptor.Base48(), size, packed_clear, false);
	}
	static std::atomic<uint32_t> logged_clears {0};
	if (logged_clears.fetch_add(1, std::memory_order_relaxed) < 32) {
		LOGF("GraphicsRenderDispatchDirect: compute %s shader=0x%016" PRIx64 " addr=0x%016" PRIx64
		     " size=0x%016" PRIx64 " value=0x%08" PRIx32 "\n",
		     image_clear ? "image clear" : "buffer fill", input.stage.module->info.shader_hash,
		     descriptor.Base48(), size, packed_clear);
	}
	return true;
}

// Global loads read pages without a buffer as zero. Back the 32 KiB after each base the shader
// loads from scalar data before the dispatch reads it.
static void BackAddressBases(RenderContext& context, const ShaderStageRuntime& stage) {
	constexpr uint64_t Page = BufferCache::CACHING_PAGESIZE;
	const auto&        flat = stage.resources->flattened_srt;
	for (const auto [low, high]: stage.module->info.info.address_bases) {
		const auto base = flat[low] | uint64_t {flat[high]} << 32u;
		for (uint64_t index = 0; index <= 2; ++index) {
			const auto page = base + index * Page;
			if (context.IsMapped(page, 1)) (void)context.GetBufferCache().FindBuffer(page, 1);
		}
	}
}

static void BindSharedMemory(RenderContext& context, ShaderComputeInputInfo& input,
                             PreparedBindings& bindings, uint64_t indirect_args = 0) {
	if (input.stage.module->info.bindings.descriptor_counts[static_cast<size_t>(
	        ShaderRecompiler::IR::DescriptorBindingKind::SharedMemory)] == 0) {
		return;
	}
	auto& cache = context.GetBufferCache();
	if (indirect_args != 0) {
		cache.ReadMemory(indirect_args, sizeof(vk::DispatchIndirectCommand));
		std::memcpy(input.workgroup_counts, reinterpret_cast<const void*>(indirect_args),
		            sizeof(input.workgroup_counts));
	}
	// LDS has no contents to preserve between dispatches. The existing shader hazard
	// barriers also order other users of this GPU-only utility buffer.
	auto&      storage = cache.GetUtilityBuffer(MemoryUsage::DeviceLocal);
	const auto limit   = std::min<uint64_t>(
        storage.Size(),
        context.GetGraphics().GetPhysicalDeviceProperties().limits.maxStorageBufferRange);
	uint64_t size = sizeof(uint32_t);
	if (std::ranges::find(input.workgroup_counts, 0u) == std::end(input.workgroup_counts)) {
		size = uint64_t {input.lds_size_dwords} * sizeof(uint32_t);
		EXIT_IF(size == 0 || size > limit);
		for (const auto count: input.workgroup_counts) {
			EXIT_IF(size > limit / count);
			size *= count;
		}
	}
	bindings.shared_memory = {storage.Handle(), 0, size};
}

static bool HasStorageWrites(const ShaderComputeInputInfo& input_info,
                             const PreparedBindings&       bindings) {
	const auto& images = input_info.stage.module->info.info.images;
	return bindings.shared_memory.buffer != nullptr || HasShaderBufferWrites(input_info.stage) ||
	       std::ranges::any_of(images, [](const auto& image) {
		       return image.written &&
		              image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Storage;
	       });
}

void RenderExecutor::BindComputePipeline(CommandBuffer& buffer, vk::Pipeline pipeline) {
	// The tracker belongs to the scheduler's current command buffer: bind into that one.
	auto& bound = m_context.GetCommandScheduler().GetDynamicState().bound_compute_pipeline;
	if (bound != pipeline || !Common::AbSwitch::Enabled(Common::AbSwitch::COMPUTE_BIND_DEDUPE)) {
		buffer.Handle().bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
		bound = pipeline;
	}
}

// KYTY_DISPATCH_TIMING=1 (measurement only, 2026-11-21): splits the GPU thread's time per dispatch
// into program lookup, binding preparation and commit/record (perf.csv disp_*_ms). Off by default:
// it reads the clock four times per dispatch (~0.1 us). DispatchNs also counts dispatches that end
// early (consumed as fills, skipped programs); the phases count only recorded ones.
static const bool g_dispatch_timing = [] {
	const char* value = std::getenv("KYTY_DISPATCH_TIMING");
	return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0;
}();

class DispatchPhaseTimer final {
public:
	using Counter = Common::DebugCounters::Counter;
	DispatchPhaseTimer() noexcept {
		if (g_dispatch_timing && Common::DebugCounters::Enabled()) {
			m_start = m_last = Now();
		}
	}
	~DispatchPhaseTimer() {
		if (m_start != 0) {
			Common::DebugCounters::Add(Counter::DispatchNs, Now() - m_start);
		}
	}
	void Phase(Counter counter) noexcept {
		if (m_start == 0) return;
		const auto now = Now();
		Common::DebugCounters::Add(counter, now - m_last);
		m_last = now;
	}
	DispatchPhaseTimer(const DispatchPhaseTimer&)            = delete;
	DispatchPhaseTimer& operator=(const DispatchPhaseTimer&) = delete;

private:
	static uint64_t Now() noexcept {
		return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                                 std::chrono::steady_clock::now().time_since_epoch())
		                                 .count());
	}
	uint64_t m_start = 0;
	uint64_t m_last  = 0;
};

void RenderExecutor::DispatchDirect(uint64_t submit_id, CommandBuffer& buffer,
                                    uint32_t thread_group_x, uint32_t thread_group_y,
                                    uint32_t thread_group_z, uint32_t mode, bool async_compute) {
	EXIT_IF(buffer.IsInvalid());
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchDirect), submit_id,
	                    thread_group_x, thread_group_y, thread_group_z, mode,
	                    sh_ctx.GetCs().cs_regs.data_addr);

	if (thread_group_x == 0 || thread_group_y == 0 || thread_group_z == 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: skipping zero-sized dispatch groups=%ux%ux%u "
			     "mode=0x%08" PRIx32 " shader=0x%016" PRIx64 "\n",
			     thread_group_x, thread_group_y, thread_group_z, mode,
			     sh_ctx.GetCs().cs_regs.data_addr);
		}
		return;
	}

	Common::LockGuard lock(m_context.GetMutex());
	DispatchPhaseTimer timer;
	Common::AbSwitch::DelayProbe();
	// A chained dispatch's barrier belongs before anything this one records.
	m_context.GetCommandScheduler().FlushDeferredShaderAccessBarrier(
	    !Common::AbSwitch::Enabled(Common::AbSwitch::BARRIER_LITE));
	if (sh_ctx.GetCs().cs_regs.data_addr == 0) {
		LOGF("GraphicsRenderDispatchDirect: temporary: ignoring dispatch with null CS shader, "
		     "groups=%ux%ux%u mode=%u\n",
		     thread_group_x, thread_group_y, thread_group_z, mode);
		return;
	}

	constexpr uint32_t DISPATCH_INITIATOR_BASE_BITS     = 0x41u;
	constexpr uint32_t DISPATCH_INITIATOR_MODIFIER_BITS = 0xa038u;
	constexpr uint32_t DISPATCH_INITIATOR_KNOWN_MASK =
	    DISPATCH_INITIATOR_BASE_BITS | DISPATCH_INITIATOR_MODIFIER_BITS;

	const uint32_t unknown_mode_bits = mode & ~DISPATCH_INITIATOR_KNOWN_MASK;
	if (unknown_mode_bits != 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: unknown dispatch initiator bits "
			     "mode=0x%08" PRIx32 " unknown=0x%08" PRIx32 " shader=0x%016" PRIx64
			     " groups=%ux%ux%u\n",
			     mode, unknown_mode_bits, sh_ctx.GetCs().cs_regs.data_addr, thread_group_x,
			     thread_group_y, thread_group_z);
		}
	}

	const auto& cs_regs = sh_ctx.GetCs();

	ShaderComputeInputInfo input_info {};
	input_info.async_compute = async_compute;
	const bool use_thread_dimensions =
	    (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	input_info.dispatch_thread_dimensions = use_thread_dimensions;
	input_info.workgroup_counts[0]        = thread_group_x;
	input_info.workgroup_counts[1]        = thread_group_y;
	input_info.workgroup_counts[2]        = thread_group_z;
	if (use_thread_dimensions) {
		const uint32_t group_sizes[] = {cs_regs.cs_regs.num_thread_x, cs_regs.cs_regs.num_thread_y,
		                                cs_regs.cs_regs.num_thread_z};
		for (uint32_t axis = 0; axis < 3u; ++axis) {
			const auto size                   = std::max(group_sizes[axis], 1u);
			const auto threads                = input_info.workgroup_counts[axis];
			input_info.workgroup_counts[axis] = threads / size + (threads % size != 0u);
		}
	}
	m_context.GetImageHeap().BeginDraw();
	if (!m_context.GetPipelineCache().GetComputeProgram(cs_regs, input_info)) return;
	timer.Phase(Common::DebugCounters::Counter::DispatchProgramNs);
	if (use_thread_dimensions) {
		input_info.dispatch_threads_num[0] = thread_group_x;
		input_info.dispatch_threads_num[1] = thread_group_y;
		input_info.dispatch_threads_num[2] = thread_group_z;
	}

	const auto& program   = input_info.stage.module->info;
	const auto& resources = *input_info.stage.resources;
	if (resources.specialization_reads.empty() &&
	    (TryConsumeComputeMetaClear(input_info, buffer) ||
	     TryConsumeComputeFill(input_info, buffer, thread_group_x, thread_group_y, thread_group_z,
	                           mode))) {
		ResetBindings();
		return;
	}

	if (use_thread_dimensions) {
		thread_group_x = input_info.workgroup_counts[0];
		thread_group_y = input_info.workgroup_counts[1];
		thread_group_z = input_info.workgroup_counts[2];
	}
	TraceByteStorageFallback("direct", input_info, thread_group_x, thread_group_y,
	                         thread_group_z, mode);

	buffer.EndRendering();
	auto& pipeline = m_context.GetPipelineCache().GetComputePipeline(input_info);
	auto& bindings = m_compute_bindings;
	PrepareBindings(input_info.stage, bindings);
	if (program.bindings.dispatch_thread_dword != ShaderRecompiler::IR::PushData::NoStart) {
		std::copy(std::begin(input_info.dispatch_threads_num),
		          std::end(input_info.dispatch_threads_num),
		          bindings.shader_data.begin() + program.bindings.dispatch_thread_dword);
	}
	PreparedBindings* descriptor_stage = &bindings;
	BackAddressBases(m_context, input_info.stage);
	FindBuffers(std::span {&descriptor_stage, 1u});
	if (program.info.uses_dma) {
		m_context.PrepareBda();
	}
	if (program.info.uses_image_heap) {
		m_context.GetImageHeap().Rebind(*this);
	}
	RebindImages(bindings);
	TracePassImages("direct", input_info, bindings, thread_group_x, thread_group_y, thread_group_z);
	DlssgNoteDispatch(m_context, buffer, input_info, bindings);
	BindSharedMemory(m_context, input_info, bindings);
	RebindBuffers(bindings);
	timer.Phase(Common::DebugCounters::Counter::DispatchBindNs);

	CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
	               std::span {&descriptor_stage, 1u});
	// After CommitBindings, which can begin the next command buffer.
	const auto vk_buffer = buffer.Handle();
	// Emulated LDS and GDS live in buffers every such dispatch shares, so those stay serialized.
	const bool lds_shared = bindings.shared_memory.buffer != nullptr || program.info.uses_gds;
	const bool chain =
	    Common::AbSwitch::Enabled(Common::AbSwitch::COMPUTE_CHAIN) && !lds_shared && !async_compute;
	const bool guest_ordered = Common::AbSwitch::Enabled(Common::AbSwitch::GUEST_BARRIERS) &&
	                           !Common::AbSwitch::Enabled(Common::AbSwitch::COMPUTE_CHAIN) &&
	                           !lds_shared;
	if (HasStorageWrites(input_info, bindings) && !guest_ordered && !chain) {
		// A host fence used to serialize every dispatch. Preserve its read-before-write ordering
		// while allowing the queue to execute asynchronously.
		ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	}
	if (Config::GraphicsDebugDumpEnabled()) {
		const auto sampled_images = std::count_if(
		    program.info.images.begin(), program.info.images.end(), [](const auto& image) {
			    return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
		    });
		const uint32_t frame_num = static_cast<uint32_t>(m_context.GetGpu().GetFrameNum());
		LOGF("GraphicsRenderDispatchDirect: frame=%u shader=0x%016" PRIx64 " hash=0x%016" PRIx64
		     " tick=%" PRIu64 " completed_tick=%" PRIu64 " groups=%ux%ux%u mode=0x%08" PRIx32
		     " local=%ux%ux%u "
		     "buffers=%zu textures=%zu sampled=%zu storage=%zu samplers=%zu push=%u\n",
		     frame_num, sh_ctx.GetCs().cs_regs.data_addr, program.shader_hash,
		     m_context.GetCommandScheduler().CurrentTick(),
		     m_context.GetCommandScheduler().GetMasterSemaphore().KnownGpuTick(), thread_group_x,
		     thread_group_y, thread_group_z, mode, input_info.threads_num[0],
		     input_info.threads_num[1], input_info.threads_num[2], program.info.buffers.size(),
		     program.info.images.size(), sampled_images,
		     program.info.images.size() - sampled_images, program.info.samplers.size(),
		     program.bindings.UsesPushData()
		         ? static_cast<uint32_t>(sizeof(ShaderRecompiler::IR::PushData))
		         : 0u);
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& buffer = program.info.buffers[i];
			const auto  r      = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
			LOGF("  CS buffer[%u]: source=%u usage=%s addr=0x%012" PRIx64
			     " stride=%u records=%u format=%u\n",
			     i, buffer.source, buffer.written ? "read-write" : "read-only", r.Base48(),
			     r.Stride(), r.NumRecords(), r.RawFormat());
		}
		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			const auto& image = program.info.images[i];
			const auto  r     = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
			const auto  max_mip = image.r128 ? r.LastLevel() : r.MaxMip();
			LOGF("  CS texture[%u]: source=%u usage=%s sampled=%s addr=0x%010" PRIx64
			     " type=%u fmt=%u extent=%ux%u depth=%u levels=%u tile=%u\n",
			     i, image.source, image.written ? "read-write" : "read-only",
			     image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled
			         ? "true"
			         : "false",
			     r.Base40(), static_cast<uint32_t>(r.Type()), static_cast<uint32_t>(r.Format()),
			     static_cast<uint32_t>(r.Width5()) + 1u, static_cast<uint32_t>(r.Height5()) + 1u,
			     static_cast<uint32_t>(r.Depth()) + 1u,
			     r.Type() == Prospero::ImageType::kColor2DMsaa ||
			             r.Type() == Prospero::ImageType::kColor2DMsaaArray
			         ? 1u
			         : static_cast<uint32_t>(max_mip) + 1u,
			     static_cast<uint32_t>(r.TileMode()));
		}
		for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
			const auto r = DecodeNativeDescriptor<ShaderSamplerResource>(
			    resources.samplers[program.info.samplers[i].snapshot_index]);
			LOGF("  CS sampler[%u]: source=%u clamp=%u/%u/%u filter=%u/%u/%u mip=%u "
			     "lod=%u-%u bias=%d\n",
			     i, program.info.samplers[i].source, static_cast<uint32_t>(r.ClampX()),
			     static_cast<uint32_t>(r.ClampY()), static_cast<uint32_t>(r.ClampZ()),
			     static_cast<uint32_t>(r.XyMagFilter()), static_cast<uint32_t>(r.XyMinFilter()),
			     static_cast<uint32_t>(r.ZFilter()), static_cast<uint32_t>(r.MipFilter()),
			     static_cast<uint32_t>(r.MinLod()), static_cast<uint32_t>(r.MaxLod()),
			     static_cast<int32_t>(r.LodBias()));
		}
	}

	BindComputePipeline(buffer, pipeline.pipeline);
	auto&      scheduler = m_context.GetCommandScheduler();
	bool isolate = false;
	const bool diagnostic = ComputeDiagnosticEnabled() &&
	                        TraceComputeDiagnostic(m_context, input_info, bindings, false, isolate);
	if (isolate) GuestFlushBarrier(vk_buffer);
	const auto gpu_timer = diagnostic ? scheduler.BeginGpuMarker() : scheduler.BeginGpuShaderTimer();
	// A timestamp orders earlier work but does not gate later work. Keep the
	// diagnostic dispatch after its start stamp, and its successor after the end stamp.
	if (isolate && gpu_timer != UINT32_MAX) GuestFlushBarrier(vk_buffer);
	const auto label     = [&] {
        return fmt::format("cs {:#x} h{:016x} g{}x{}x{} t{}x{}x{} w{}{}{}",
		                       cs_regs.cs_regs.data_addr, program.shader_hash, thread_group_x,
		                       thread_group_y, thread_group_z, input_info.threads_num[0],
		                       input_info.threads_num[1], input_info.threads_num[2],
		                       input_info.wave_size, input_info.lds_storage ? " lds-storage" : "",
                           program.info.uses_gds ? " gds" : "");
	};
	scheduler.GpuCheckpoint(label);
	if (CommandScheduler::HangDiagEnabled()) [[unlikely]] {
		scheduler.NoteHangWork('C', program.shader_hash, cs_regs.cs_regs.data_addr, thread_group_x,
		                       thread_group_y, thread_group_z);
	}
	constexpr uint32_t DISPATCH_INITIATOR_ORDERED_APPEND_ENABLE = 1u << 3u;
	const uint64_t     group_threads = uint64_t {std::max(input_info.threads_num[0], 1u)} *
	                               std::max(input_info.threads_num[1], 1u) *
	                               std::max(input_info.threads_num[2], 1u);
	// KYTY_ORDERED_SERIAL=1 (2026-11-21, experiment): also run multi-wave ordered-append GDS groups one
	// after another. UFC 6's strand-hair rasterizer (b37644325c6f10c6, 16x16 groups) otherwise spins on
	// GPUs that do not launch groups in order (4K fighter-select screen at ~4.5 FPS).
	// KYTY_ORDERED_SERIAL=ab: on only while AbSwitch bit 40 (no other code) is set, for a same-process A/B.
	static const int ordered_serial_mode = [] {
		const char* v = std::getenv("KYTY_ORDERED_SERIAL");
		if (v == nullptr) return 0; // default OFF again (23:30): one group then takes every hair tile -> TDR risk
		if (std::strcmp(v, "ab") == 0) return 2;
		return std::atoi(v) != 0 ? 1 : 0;
	}();
	const bool ordered_serial = ordered_serial_mode == 1 ||
	                            (ordered_serial_mode == 2 && Common::AbSwitch::Enabled(40));
	if ((mode & DISPATCH_INITIATOR_ORDERED_APPEND_ENABLE) != 0 && program.info.uses_gds &&
	    (ordered_serial || (group_threads <= input_info.wave_size && !input_info.lds_storage))) {
		// A single guest wave per group can run in launch order without a device-wide
		// spinlock. Keep the guest workgroup IDs when splitting the dispatch.
		for (uint32_t z = 0; z < thread_group_z; ++z) {
			for (uint32_t y = 0; y < thread_group_y; ++y) {
				for (uint32_t x = 0; x < thread_group_x; ++x) {
					if ((x | y | z) != 0u) {
						ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
					}
					vk_buffer.dispatchBase(x, y, z, 1, 1, 1);
				}
			}
		}
	} else if (!SkipComputeShader(input_info.stage.module->info.shader_hash)) {
		vk_buffer.dispatch(thread_group_x, thread_group_y, thread_group_z);
	}
	if (isolate) GuestFlushBarrier(vk_buffer);
	m_context.GetBufferCache().RetagGpuWrites();
	if (gpu_timer != UINT32_MAX) {
		scheduler.EndGpuMarker(gpu_timer, diagnostic ? (isolate ? "isolated " : "selected ") + label() : label());
		if (isolate) GuestFlushBarrier(vk_buffer);
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::Dispatches);
	// Groups are non-zero here, so an OR of 1 means 1x1x1.
	if ((thread_group_x | thread_group_y | thread_group_z) == 1u) {
		Common::DebugCounters::Add(Common::DebugCounters::Counter::DispatchesTiny);
	}
	if (DrawPassStats::Active()) [[unlikely]] {
		DrawPassStats::NoteDispatch(cs_regs.cs_regs.data_addr, program.shader_hash, thread_group_x,
		                            thread_group_y, thread_group_z, false);
	}

	// The removed host fence also ordered read-only dispatches before later writers.
	if (chain) {
		scheduler.DeferShaderAccessBarrier();
	} else if (!guest_ordered) {
		ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	}
	ResetBindings();
	timer.Phase(Common::DebugCounters::Counter::DispatchCommitNs);
}

void RenderExecutor::GuestFlush() {
	auto& scheduler = m_context.GetCommandScheduler();
	if (!scheduler.Active()) {
		return;
	}
	if (Common::AbSwitch::Enabled(Common::AbSwitch::COMPUTE_CHAIN)) {
		// The chained dispatches' policy owns the guest sync points: flush the one deferred
		// barrier (full unless BARRIER_LITE) only if a dispatch left one.
		scheduler.FlushDeferredShaderAccessBarrier(
		    !Common::AbSwitch::Enabled(Common::AbSwitch::BARRIER_LITE));
		return;
	}
	if (!Common::AbSwitch::Enabled(Common::AbSwitch::GUEST_BARRIERS)) {
		return;
	}
	scheduler.EndRendering();
	GuestFlushBarrier(scheduler.Current().Handle());
	Common::DebugCounters::Add(Common::DebugCounters::Counter::GuestFlushBarriers);
}

void RenderExecutor::DispatchIndirect(uint64_t submit_id, CommandBuffer& buffer, uint64_t args_addr,
                                      uint32_t mode, bool async_compute) {
	EXIT_IF(buffer.IsInvalid() || args_addr == 0 || (args_addr & 3u) != 0 ||
	        (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchIndirect), submit_id,
	                    static_cast<uint32_t>(args_addr), static_cast<uint32_t>(args_addr >> 32u),
	                    0, mode, buffer.GetShaders().GetCs().cs_regs.data_addr);
	Common::LockGuard lock(m_context.GetMutex());
	DispatchPhaseTimer timer;
	Common::AbSwitch::DelayProbe();
	// A chained dispatch's barrier belongs before anything this one records.
	m_context.GetCommandScheduler().FlushDeferredShaderAccessBarrier(
	    !Common::AbSwitch::Enabled(Common::AbSwitch::BARRIER_LITE));
	vk::DispatchIndirectCommand args {};
	if (!m_context.GetBufferCache().HasGpuDirtyBytes(args_addr, sizeof(args)) &&
	    LibKernel::Memory::TryReadBacking(args_addr, &args, sizeof(args)) &&
	    (args.x == 0 || args.y == 0 || args.z == 0)) {
		return;
	}
	const auto& cs_regs = buffer.GetShaders().GetCs();
	if (cs_regs.cs_regs.data_addr == 0) {
		return;
	}
	ShaderComputeInputInfo input_info {};
	input_info.async_compute         = async_compute;
	input_info.indirect_args_address = args_addr;
	m_context.GetImageHeap().BeginDraw();
	{
		KYTY_PROFILER_BLOCK("DispatchIndirect: program");
		if (!m_context.GetPipelineCache().GetComputeProgram(cs_regs, input_info)) return;
	}
	timer.Phase(Common::DebugCounters::Counter::DispatchProgramNs);
	TraceByteStorageFallback("indirect-unresolved", input_info, args.x, args.y, args.z, mode);
	buffer.EndRendering();
	auto& pipeline = m_context.GetPipelineCache().GetComputePipeline(input_info);
	auto& bindings = m_compute_bindings;
	PrepareBindings(input_info.stage, bindings);
	PreparedBindings* descriptor_stage = &bindings;
	BackAddressBases(m_context, input_info.stage);
	FindBuffers(std::span {&descriptor_stage, 1u});
	const auto& program = input_info.stage.module->info;
	if (program.info.uses_dma) {
		KYTY_PROFILER_BLOCK("DispatchIndirect: BDA upload");
		m_context.PrepareBda();
	}
	BindSharedMemory(m_context, input_info, bindings, args_addr);
	if (program.info.uses_image_heap) {
		m_context.GetImageHeap().Rebind(*this);
	}
	RebindImages(bindings);
	TracePassImages("indirect", input_info, bindings, args.x, args.y, args.z);
	DlssgNoteDispatch(m_context, buffer, input_info, bindings);
	// Acquiring arguments can merge cache buffers; finalize shader bindings afterward.
	const auto [args_buffer, args_offset] = [&] {
		KYTY_PROFILER_BLOCK("DispatchIndirect: argument buffer");
		return m_context.GetBufferCache().ObtainBuffer(
		    args_addr, sizeof(vk::DispatchIndirectCommand), false);
	}();
	EXIT_IF((args_offset & 3u) != 0);
	RebindBuffers(bindings);
	timer.Phase(Common::DebugCounters::Counter::DispatchBindNs);
	CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
	               std::span {&descriptor_stage, 1u});
	const auto vk_buffer  = buffer.Handle();
	const bool lds_shared = bindings.shared_memory.buffer != nullptr || program.info.uses_gds;
	const bool chain =
	    Common::AbSwitch::Enabled(Common::AbSwitch::COMPUTE_CHAIN) && !lds_shared && !async_compute;
	const bool guest_ordered = Common::AbSwitch::Enabled(Common::AbSwitch::GUEST_BARRIERS) &&
	                           !Common::AbSwitch::Enabled(Common::AbSwitch::COMPUTE_CHAIN) &&
	                           !lds_shared;
	if (HasStorageWrites(input_info, bindings) && !guest_ordered && !chain) {
		ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	}
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
	vk_buffer.pipelineBarrier(
	    vk::PipelineStageFlagBits::eAllGraphics | vk::PipelineStageFlagBits::eComputeShader |
	        vk::PipelineStageFlagBits::eTransfer,
	    vk::PipelineStageFlagBits::eDrawIndirect, {}, 1, &barrier, 0, nullptr, 0, nullptr);
	BindComputePipeline(buffer, pipeline.pipeline);
	auto&      scheduler = m_context.GetCommandScheduler();
	bool isolate = false;
	const bool diagnostic = ComputeDiagnosticEnabled() &&
	                        TraceComputeDiagnostic(m_context, input_info, bindings, true, isolate);
	if (isolate) GuestFlushBarrier(vk_buffer);
	const auto gpu_timer = diagnostic ? scheduler.BeginGpuMarker() : scheduler.BeginGpuShaderTimer();
	if (isolate && gpu_timer != UINT32_MAX) GuestFlushBarrier(vk_buffer);
	const auto label = [&] {
		return fmt::format("cs {:#x} h{:016x} indirect", cs_regs.cs_regs.data_addr, program.shader_hash);
	};
	scheduler.GpuCheckpoint(label);
	if (CommandScheduler::HangDiagEnabled()) [[unlikely]] {
		scheduler.NoteHangWork('C', program.shader_hash, cs_regs.cs_regs.data_addr, 0, 0, 0,
		                       nullptr, 0, false, true);
	}
	if (!SkipComputeShader(program.shader_hash)) {
		vk_buffer.dispatchIndirect(args_buffer->Handle(), args_offset);
	}
	if (isolate) GuestFlushBarrier(vk_buffer);
	m_context.GetBufferCache().RetagGpuWrites();
	if (gpu_timer != UINT32_MAX) {
		scheduler.EndGpuMarker(gpu_timer, diagnostic ? (isolate ? "isolated " : "selected ") + label() : label());
		if (isolate) GuestFlushBarrier(vk_buffer);
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::Dispatches);
	if (DrawPassStats::Active()) [[unlikely]] {
		DrawPassStats::NoteDispatch(cs_regs.cs_regs.data_addr, program.shader_hash, 0, 0, 0, true);
	}
	if (!guest_ordered) {
		ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	}
	ResetBindings();
	timer.Phase(Common::DebugCounters::Counter::DispatchCommitNs);
}

} // namespace Libs::Graphics
