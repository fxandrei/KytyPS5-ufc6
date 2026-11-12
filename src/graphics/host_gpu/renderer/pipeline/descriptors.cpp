#include "graphics/host_gpu/renderer/pipeline/descriptors.h"

#include "common/abSwitch.h"
#include "common/alignment.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <limits>
#include <memory>
#include <set>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

namespace Libs::Graphics {

namespace {

using BindingKind = ShaderRecompiler::IR::DescriptorBindingKind;

} // namespace

vk::DescriptorType NativeDescriptorType(BindingKind kind) {
	const auto image_class = ShaderRecompiler::IR::ImageBindingResourceClass(kind);
	if (image_class == ShaderRecompiler::IR::ImageResourceClass::Sampled) {
		return vk::DescriptorType::eSampledImage;
	}
	if (image_class == ShaderRecompiler::IR::ImageResourceClass::Storage) {
		return vk::DescriptorType::eStorageImage;
	}
	switch (kind) {
		case BindingKind::Samplers: return vk::DescriptorType::eSampler;
		case BindingKind::Buffers:
		case BindingKind::Gds:
		case BindingKind::BdaPagetable:
		case BindingKind::FaultBuffer:
		case BindingKind::FlattenedSrt:
		case BindingKind::ShaderData:
		case BindingKind::SharedMemory: return vk::DescriptorType::eStorageBuffer;
		case BindingKind::Count: EXIT("invalid native descriptor binding kind");
	}
	EXIT("invalid native descriptor binding kind");
}

vk::DescriptorImageInfo MakeImageInfo(const TextureBinding& texture, uint32_t element) {
	if (!texture.image_id) {
		EXIT_IF(texture.image_view != nullptr || !texture.mip_views.empty());
		return {nullptr, nullptr, vk::ImageLayout::eGeneral};
	}
	vk::ImageView view = nullptr;
	if (texture.mip_views.empty()) {
		if (element == 0u) {
			view = texture.image_view;
		}
	} else if (element < texture.mip_views.size()) {
		view = texture.mip_views[element];
	}
	EXIT_IF(view == nullptr || texture.layout == vk::ImageLayout::eUndefined);
	return {nullptr, view, texture.layout};
}

static Prospero::ImageType TextureType(const ShaderTextureResource& descriptor) {
	const auto type = descriptor.Type();
	return type == Prospero::ImageType::kCube ? Prospero::ImageType::kColor2DArray : type;
}

// 1D textures are 2D images one texel high, so they share 2D surfaces at their address.
static Prospero::ImageType TextureBaseType(Prospero::ImageType type) {
	switch (type) {
		case Prospero::ImageType::kColor1D:
		case Prospero::ImageType::kColor1DArray:
		case Prospero::ImageType::kColor2DArray:
		case Prospero::ImageType::kColor2DMsaa:
		case Prospero::ImageType::kColor2DMsaaArray: return Prospero::ImageType::kColor2D;
		default: return type;
	}
}

static bool IsMultisampledTexture(Prospero::ImageType type) {
	return type == Prospero::ImageType::kColor2DMsaa ||
	       type == Prospero::ImageType::kColor2DMsaaArray;
}

// KYTY_TRACE_WRITES=<n>: the first n distinct shader-written bindings that start at an image,
// which then refreshes from the buffer copy.
static void TraceImageWriteBinding(uint64_t address, uint64_t size,
                                   const ShaderRecompiler::IR::BufferResource& resource) {
	static const int limit = [] {
		const char* value = std::getenv("KYTY_TRACE_WRITES");
		return value != nullptr ? std::atoi(value) : 0;
	}();
	static std::set<std::pair<uint64_t, uint64_t>> seen;
	if (limit <= 0 || static_cast<int>(seen.size()) >= limit ||
	    !seen.emplace(address, size).second) {
		return;
	}
	std::printf("image write binding 0x%" PRIx64 " size 0x%" PRIx64 " max_byte_extent 0x%x "
	            "formatted %d stride 0x%x\n",
	            address, size, resource.max_byte_extent, resource.formatted ? 1 : 0,
	            resource.packed_stride);
}

static vk::DescriptorBufferInfo
NativeStorageBuffer(RenderContext& context, const PreparedBindings::BufferSource& source,
                    const ShaderRecompiler::IR::BufferResource& resource, uint32_t& buffer_offset) {
	buffer_offset = 0;

	const auto& [address, size, id] = source;
	if (address == 0 || size == 0) {
		// Null descriptors read zero and drop writes, with or without shader bounds checks.
		return {nullptr, 0, vk::WholeSize};
	}
	const auto& graphics  = context.GetGraphics();
	const auto  alignment = graphics.StorageMinAlignment();
	if (size > graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange) {
		EXIT("storage buffer range is unsupported\n");
	}
	auto [buffer, offset] = context.GetBufferCache().ObtainBuffer(
	    address, size, resource.written,
	    resource.formatted || (resource.read && Config::SyncRawImageBuffersEnabled()), id);
	const auto aligned_offset = Common::AlignDown(offset, alignment);
	const auto adjustment     = offset - aligned_offset;
	const auto max_range      = graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange;
	if (adjustment >= 256 || size > max_range - adjustment) {
		EXIT("storage buffer offset adjustment is unsupported\n");
	}
	buffer_offset = static_cast<uint32_t>(adjustment);
	const vk::DescriptorBufferInfo result {buffer->Handle(), aligned_offset, size + adjustment};
	if (resource.written && context.GetTextureCache().InvalidateMemoryFromGPU(address, size) != 0) {
		TraceImageWriteBinding(address, size, resource);
	}
	return result;
}

// Reserved and unsupported T# fields must be zero; perf_mod only scales sampler state and the
// mip-stats counter only counts sampling.
static bool IsCanonicalTextureEncoding(const ShaderTextureResource& descriptor, bool r128) {
	constexpr uint32_t field1_reserved_mask = 1u << 29u;
	constexpr uint32_t field2_reserved_mask = 0xf0003000u;
	const uint32_t     field3_expected = descriptor.DstSelXYZW() |
	                                     (static_cast<uint32_t>(descriptor.BaseLevel()) << 12u) |
	                                     (static_cast<uint32_t>(descriptor.LastLevel()) << 16u) |
	                                     (static_cast<uint32_t>(descriptor.TileMode()) << 20u) |
	                                     (static_cast<uint32_t>(descriptor.Type()) << 28u);
	const uint32_t     field4_expected = descriptor.Depth() | (descriptor.BaseArray5() << 16u);
	const uint32_t     field5_expected = (static_cast<uint32_t>(descriptor.PerfMod5()) << 20u) |
	                                     (uint32_t {descriptor.MipStatsCntEn()} << 25u) |
	                                     (static_cast<uint32_t>(descriptor.MaxMip()) << 4u);
	if ((descriptor.fields[1] & field1_reserved_mask) != 0 ||
	    (descriptor.fields[2] & field2_reserved_mask) != 0 ||
	    descriptor.fields[3] != field3_expected) {
		return false;
	}
	if (r128) {
		return descriptor.fields[4] == 0 && descriptor.fields[5] == 0 &&
		       descriptor.fields[6] == 0 && descriptor.fields[7] == 0;
	}
	return descriptor.fields[4] == field4_expected && descriptor.fields[5] == field5_expected;
}

bool IsSupportedDepthTextureEncoding(const ShaderTextureResource& descriptor, bool r128) {
	if (!IsCanonicalTextureEncoding(descriptor, r128)) {
		return false;
	}
	if (r128) {
		return true;
	}
	if (descriptor.MsaaDepth() && !IsMultisampledTexture(descriptor.Type())) {
		return false;
	}
	// Bits 0-7 name the mip-stats counter. A clear COMPRESSION_EN disables the metadata and its
	// settings.
	const auto         metadata_control = descriptor.fields[6] & 0x00ffff00u;
	constexpr uint32_t pipe_aligned     = 1u << 19u;
	if (!descriptor.MetaCompress()) {
		return true;
	}
	constexpr uint32_t htile_control = pipe_aligned | 1u << 21u;
	const uint32_t expected_control  = htile_control | (descriptor.MsaaDepth() ? (1u << 10u) : 0u);
	const auto     metadata_addr     = descriptor.MetaAddr() << 8u;
	return metadata_control == expected_control && GuestRange {metadata_addr, 1}.Valid() &&
	       (metadata_addr & 0x7fffu) == 0 && descriptor.TileMode() == Prospero::TileMode::kDepth;
}

static void ValidateSampledDepthBinding(const ShaderRecompiler::IR::ImageResource& resource,
                                        const ShaderTextureResource& descriptor, const Image& image,
                                        vk::Format view_format, uint64_t size) {
	const bool resource_ok = IsSupportedSampledDepthResource(resource);
	// Comparison sampling changes the host format, not the guest metadata encoding.
	const bool encoding_ok = descriptor.TileMode() != Prospero::TileMode::kDepth ||
	                         IsSupportedDepthTextureEncoding(descriptor, resource.r128);
	const bool view_ok =
	    IsSupportedSampledDepthView(image.info.pixel_format, view_format, descriptor.DstSelXYZW());
	if (resource_ok && encoding_ok && view_ok) {
		return;
	}
	const auto descriptor_pitch =
	    TileGetTexturePitch(descriptor.Format(), static_cast<uint32_t>(descriptor.Width5()) + 1u,
	                        descriptor.TileMode());
	EXIT("unsupported sampled depth image: resource=%d encoding=%d view=%d "
	     "class=%u numeric=%u dimension=%u mip_mode=%u read=%d written=%d atomic=%d compare=%d "
	     "guest_format=%u swizzle=0x%03x image_format=%d view_format=%d image_layers=%u "
	     "descriptor_type=%u base_array=%u depth=%u descriptor_pitch=%u target_pitch=%u "
	     "addr=0x%016" PRIx64 " size=0x%016" PRIx64
	     " dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
	     resource_ok, encoding_ok, view_ok, static_cast<uint32_t>(resource.resource_class),
	     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(resource.dimension),
	     static_cast<uint32_t>(resource.mip_mode), resource.read, resource.written, resource.atomic,
	     resource.depth_compare, static_cast<uint32_t>(descriptor.Format()),
	     descriptor.DstSelXYZW(), static_cast<int>(image.info.pixel_format),
	     static_cast<int>(view_format), image.info.resources.layers,
	     static_cast<uint32_t>(descriptor.Type()), descriptor.BaseArray5(), descriptor.Depth(),
	     descriptor_pitch, image.info.pitch, descriptor.Base40(), size, descriptor.fields[0],
	     descriptor.fields[1], descriptor.fields[2], descriptor.fields[3], descriptor.fields[4],
	     descriptor.fields[5], descriptor.fields[6], descriptor.fields[7]);
}

static bool IsSupportedStorageTextureDescriptor(const ShaderRecompiler::IR::ImageResource& resource,
                                                const ShaderTextureResource& descriptor) {
	const auto tile              = descriptor.TileMode();
	const bool is_color_1d       = descriptor.Type() == Prospero::ImageType::kColor1D;
	const bool is_color_1d_array = descriptor.Type() == Prospero::ImageType::kColor1DArray;
	const bool valid_1d_slice =
	    (is_color_1d && descriptor.Depth() == 0 && descriptor.BaseArray5() == 0) ||
	    (is_color_1d_array && descriptor.BaseArray5() <= descriptor.Depth());
	const bool is_1d = resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim1D &&
	                   descriptor.Height5() == 0 && valid_1d_slice;
	const bool is_1d_array =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim1DArray &&
	    is_color_1d_array && descriptor.Height5() == 0 &&
	    descriptor.BaseArray5() <= descriptor.Depth();
	const bool is_color_2d       = descriptor.Type() == Prospero::ImageType::kColor2D;
	const bool is_color_2d_array = descriptor.Type() == Prospero::ImageType::kColor2DArray;
	const bool valid_2d_slice =
	    (is_color_2d && descriptor.Depth() == 0 && descriptor.BaseArray5() == 0) ||
	    (is_color_2d_array && descriptor.BaseArray5() <= descriptor.Depth());
	const bool is_2d =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2D && valid_2d_slice;
	// Storage cube coordinates address individual faces, including partial cube views.
	const bool is_cube = resource.cube && descriptor.Type() == Prospero::ImageType::kCube &&
	                     descriptor.Width5() == descriptor.Height5() &&
	                     descriptor.BaseArray5() <= descriptor.Depth();
	const bool is_2d_array =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2DArray &&
	    ((!resource.cube && is_color_2d_array && descriptor.BaseArray5() <= descriptor.Depth()) ||
	     is_cube);
	const bool is_3d = resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim3D &&
	                   descriptor.Type() == Prospero::ImageType::kColor3D &&
	                   descriptor.BaseArray5() == 0;
	const bool is_2d_msaa =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaa &&
	    descriptor.Type() == Prospero::ImageType::kColor2DMsaa && descriptor.Depth() == 0 &&
	    descriptor.BaseArray5() == 0;
	TileTextureBlockLayout tile_layout {};
	bool                   supported_tile = false;
	switch (tile) {
		case Prospero::TileMode::kLinear: supported_tile = true; break;
		case Prospero::TileMode::kDepth:
			supported_tile =
			    !resource.read && !Prospero::IsFmaskTextureFormat(descriptor.Format()) &&
			    (is_2d || is_2d_array) &&
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		case Prospero::TileMode::kStandard256B:
			supported_tile =
			    (is_2d || is_2d_array) &&
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		case Prospero::TileMode::kStandard4KB:
		case Prospero::TileMode::kStandard64KB:
			supported_tile =
			    TileGetTextureBlockLayout(descriptor.Format(), tile, is_3d, tile_layout);
			break;
		case Prospero::TileMode::kRenderTarget:
			supported_tile =
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		default: break;
	}
	const auto swizzle = descriptor.DstSelXYZW();
	const bool supported_swizzle =
	    IsValidImageSwizzle(swizzle) &&
	    (swizzle == DstSel(4, 5, 6, 7) || !resource.read || resource.atomic);
	// A 4.8 fixed-point MIN_LOD at or below BASE_LEVEL clamps no level of the view.
	return (is_1d || is_1d_array || is_2d || is_2d_array || is_3d || is_2d_msaa) &&
	       supported_tile && descriptor.BaseLevel() <= descriptor.LastLevel() &&
	       descriptor.MinLod() <= descriptor.BaseLevel() * 256u && supported_swizzle &&
	       descriptor.BCSwizzle() == 0 && !descriptor.MsaaDepth();
}

void ValidateStorageTexture(const ShaderRecompiler::IR::ImageResource& resource,
                            const ShaderTextureResource& descriptor, uint64_t size) {
	const auto format        = descriptor.Format();
	const bool resource_ok   = IsSupportedStorageImageResource(resource);
	const bool descriptor_ok = IsSupportedStorageTextureDescriptor(resource, descriptor);
	const bool encoding_ok   = IsCanonicalTextureEncoding(descriptor, resource.r128);
	const bool uint_resource = resource.numeric_class == Prospero::TextureNumericClass::Uint;
	const bool raw_sint_storage =
	    uint_resource && ShaderRecompiler::IR::RawSignedStorageFormat(resource, format) !=
	                         Prospero::BufferFormat::kInvalid;
	const auto numeric_class =
	    Prospero::SampledTextureNumericClass(Prospero::RemapTextureFormat(format));
	const bool raw_float_atomic = format == Prospero::BufferFormat::k32Float && uint_resource &&
	                              resource.atomic && !resource.atomic64;
	const bool format_ok =
	    raw_sint_storage || raw_float_atomic ||
	    (numeric_class != Prospero::TextureNumericClass::Unsupported &&
	     numeric_class != Prospero::TextureNumericClass::Sint &&
	     uint_resource == (numeric_class == Prospero::TextureNumericClass::Uint) &&
	     (!resource.atomic || format == (resource.atomic64 ? Prospero::BufferFormat::k32_32UInt
	                                                       : Prospero::BufferFormat::k32UInt)));
	if (resource_ok && descriptor_ok && encoding_ok && format_ok && size != 0) {
		return;
	}
	EXIT("unsupported storage texture: resource=%d descriptor=%d encoding=%d format=%d "
	     "class=%u numeric=%u dimension=%u mip_mode=%u atomic=%d compare=%d "
	     "base_level=%u last_level=%u max_mip=%u min_lod=%u base_array=%u bc=%u msaa=%d "
	     "depth_tile_bpe=%u swizzle_ok=%d "
	     "addr=0x%016" PRIx64 " size=0x%016" PRIx64
	     " extent=%ux%ux%u type=%u format=%u tile=%u swizzle=0x%03x read=%d written=%d "
	     "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
	     resource_ok, descriptor_ok, encoding_ok, format_ok,
	     static_cast<uint32_t>(resource.resource_class),
	     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(resource.dimension),
	     static_cast<uint32_t>(resource.mip_mode), resource.atomic, resource.depth_compare,
	     descriptor.BaseLevel(), descriptor.LastLevel(), descriptor.MaxMip(), descriptor.MinLod(),
	     descriptor.BaseArray5(), descriptor.BCSwizzle(), descriptor.MsaaDepth(),
	     Prospero::RenderTargetBytesPerElement(format),
	     IsValidImageSwizzle(descriptor.DstSelXYZW()), descriptor.Base40(), size,
	     static_cast<uint32_t>(descriptor.Width5()) + 1u,
	     static_cast<uint32_t>(descriptor.Height5()) + 1u,
	     static_cast<uint32_t>(descriptor.Depth()) + 1u, static_cast<uint32_t>(descriptor.Type()),
	     static_cast<uint32_t>(format), static_cast<uint32_t>(descriptor.TileMode()),
	     descriptor.DstSelXYZW(), resource.read, resource.written, descriptor.fields[0],
	     descriptor.fields[1], descriptor.fields[2], descriptor.fields[3], descriptor.fields[4],
	     descriptor.fields[5], descriptor.fields[6], descriptor.fields[7]);
}

// sRGB views decode before filtering; a sampler that skips degamma reads through a UNorm view.
static vk::Format SampledViewFormat(const GraphicContext& graphics, Prospero::BufferFormat format,
                                    vk::Format pixel_format, bool skip_degamma) {
	if (skip_degamma) return SrgbUnormViewFormat(pixel_format);
	switch (format) {
		case Prospero::BufferFormat::k8Srgb:
			return graphics.r8_srgb_views ? vk::Format::eR8Srgb : pixel_format;
		case Prospero::BufferFormat::k8_8Srgb:
			return graphics.r8g8_srgb_views ? vk::Format::eR8G8Srgb : pixel_format;
		default: return pixel_format;
	}
}

// Atomics and raw signed stores use unsigned integer views; sRGB images store through UNorm.
static vk::Format StorageViewFormat(const ShaderRecompiler::IR::ImageResource& resource,
                                    Prospero::BufferFormat format, vk::Format pixel_format) {
	if (resource.atomic64) return vk::Format::eR64Uint;
	if (resource.atomic) return vk::Format::eR32Uint;
	const auto raw_sint_format = ShaderRecompiler::IR::RawSignedStorageFormat(resource, format);
	if (raw_sint_format != Prospero::BufferFormat::kInvalid) return VulkanFormat(raw_sint_format);
	return SrgbUnormViewFormat(pixel_format);
}

static ImageViewInfo TextureViewInfo(const ShaderRecompiler::IR::ImageResource& resource,
                                     const ShaderTextureResource& descriptor, vk::Format format,
                                     const SurfaceFormatInfo& surface_format, bool storage,
                                     uint32_t view_levels, uint32_t image_layers) {
	ImageViewInfo view {};
	view.format      = format;
	view.aspect      = vk::ImageAspectFlagBits::eColor;
	view.base_level  = descriptor.BaseLevel();
	view.level_count = view_levels;
	if (descriptor.MinLod() > descriptor.LastLevel() * 256u) {
		EXIT("texture minimum LOD exceeds last mip level: min_lod=%u last_level=%u\n",
		     descriptor.MinLod(), descriptor.LastLevel());
	}
	const auto base_lod = view.base_level * 256u;
	if (descriptor.MinLod() > base_lod) {
		view.min_lod = descriptor.MinLod() - base_lod;
	}
	view.usage = storage ? vk::ImageUsageFlagBits::eStorage : vk::ImageUsageFlagBits::eSampled;
	view.mapping =
	    storage || resource.conversion_format != Prospero::BufferFormat::kInvalid
	        ? vk::ComponentMapping {}
	        : TextureGetComponentMapping(descriptor.DstSelXYZW(), surface_format.host_to_storage);
	switch (resource.dimension) {
		case ShaderRecompiler::Decoder::ImageDimension::Dim3D:
			view.type        = vk::ImageViewType::e3D;
			view.base_layer  = 0;
			view.layer_count = 1;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim1DArray:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DArray:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaaArray:
			view.type       = vk::ImageViewType::e2DArray;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture array base layer is out of bounds\n");
			}
			view.layer_count = image_layers - view.base_layer;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim1D:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2D:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaa:
			view.type       = vk::ImageViewType::e2D;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture base layer is out of bounds\n");
			}
			view.layer_count = 1;
			break;
		default: EXIT("unsupported texture view dimension\n");
	}
	return view;
}

static bool ResolveTextureMipView(ImageInfo& info, uint32_t physical_levels, bool metadata,
                                  uint32_t view_levels, uint32_t& base_level) {
	if (!info.IsTiled()) {
		return false;
	}
	ImageInfo physical = info;
	// A view entirely past MAX_MIP addresses its own chain starting at its base mip (PPSA02110
	// compresses mip 1 through a T# that declares one level).
	if (base_level >= physical_levels) {
		const auto mip  = info.MipExtent(base_level);
		physical.extent = {mip.width, mip.height, info.IsVolume() ? mip.depth : 1u};
		physical_levels = view_levels;
	}
	physical.resources.levels = physical_levels;
	physical.UpdateSize();
	info.UpdateSize();
	const auto same_mip = [&](uint32_t target, uint32_t source) {
		const auto& left  = physical.mip_layout[target];
		const auto& right = info.mip_layout[source];
		return left.offset == right.offset && left.size == right.size &&
		       left.pitch == right.pitch && left.height == right.height &&
		       left.tail_x == right.tail_x && left.tail_y == right.tail_y &&
		       left.surface_z == right.surface_z &&
		       physical.MipExtent(target) == info.MipExtent(source);
	};
	if (physical.first_tail_level == info.first_tail_level &&
	    physical.tiled_slice_stride == info.tiled_slice_stride &&
	    physical.data.size == info.data.size) {
		bool matches = true;
		for (uint32_t level = 0; level < physical_levels; ++level) {
			matches &= same_mip(level, level);
		}
		if (matches) {
			return true;
		}
	}
	if (metadata ||
	    (info.TransferLayers() > 1 && physical.tiled_slice_stride != info.tiled_slice_stride)) {
		return false;
	}
	// T# addresses the last mip. A view can select the same stored subresources
	// with different mip indices; inaccessible mips need no host representation.
	for (uint32_t base = 0; base + view_levels <= physical_levels; ++base) {
		bool matches = true;
		for (uint32_t i = 0; i < view_levels; ++i) {
			const auto source = base_level + i;
			const auto target = base + i;
			if (!same_mip(target, source) ||
			    (target >= physical.first_tail_level) != (source >= info.first_tail_level)) {
				matches = false;
				break;
			}
		}
		if (matches) {
			info       = std::move(physical);
			base_level = base;
			return true;
		}
	}
	return false;
}

namespace {

// The image layout a texture descriptor decoded to (AbSwitch TEXTURE_INFO_MEMO): UpdateSize lays
// out every mip level, and the same descriptors are bound draw after draw.
struct TextureInfoMemoEntry {
	uint32_t  fields[8] = {};
	uint32_t  flags     = 0;
	uint32_t  view_base = 0;
	uint32_t  view_levels = 0;
	bool      valid     = false;
	ImageInfo info;
};

TextureInfoMemoEntry& TextureInfoMemoSlot(const ShaderTextureResource& descriptor, uint32_t flags) {
	static constexpr size_t                              SLOTS = 512;
	thread_local std::unique_ptr<TextureInfoMemoEntry[]> slots;
	if (slots == nullptr) {
		slots = std::make_unique<TextureInfoMemoEntry[]>(SLOTS);
	}
	uint64_t hash = flags;
	for (const auto field: descriptor.fields) {
		hash = (hash ^ field) * 0x9E3779B97F4A7C15ull;
	}
	return slots[(hash >> 40u) & (SLOTS - 1)];
}

} // namespace

RenderExecutor::TextureResolvePlan
RenderExecutor::BuildTextureResolvePlan(const ShaderRecompiler::IR::ImageResource& resource,
                                        const ShaderTextureResource&               descriptor,
                                        bool                                       use_info_memo) {
	const bool storage         = resource.written;
	const auto address         = descriptor.Base40();
	const auto width           = static_cast<uint32_t>(descriptor.Width5()) + 1u;
	const auto height          = static_cast<uint32_t>(descriptor.Height5()) + 1u;
	const auto base_level      = descriptor.BaseLevel();
	const auto last_level      = descriptor.LastLevel();
	const auto type            = TextureType(descriptor);
	const bool multisampled    = IsMultisampledTexture(type);
	const auto max_mip         = resource.r128 ? last_level : descriptor.MaxMip();
	const auto physical_levels = multisampled ? 1u : static_cast<uint32_t>(max_mip) + 1u;
	// IMAGE_STORE addresses BASE_LEVEL; only IMAGE_STORE_MIP selects other view mips.
	const bool single_storage_mip =
	    storage && resource.mip_mode != ShaderRecompiler::IR::ImageMipMode::Dynamic;
	auto view_levels = multisampled || single_storage_mip
	                       ? 1u
	                       : static_cast<uint32_t>(last_level - base_level) + 1u;
	auto levels =
	    multisampled ? 1u : std::max(physical_levels, base_level + view_levels);
	const auto tile       = descriptor.TileMode();
	const bool depth_tile = tile == Prospero::TileMode::kDepth;
	const bool msaa_tile  = depth_tile || tile == Prospero::TileMode::kRenderTarget;
	const bool msaa_array = type == Prospero::ImageType::kColor2DMsaaArray;
	if ((!multisampled && base_level > last_level) ||
	    (multisampled &&
	     (base_level != 0 || last_level == 0 || last_level > 3 || max_mip != last_level ||
	      !msaa_tile || (descriptor.MsaaDepth() && !depth_tile) ||
	      (!msaa_array && (descriptor.Depth() != 0 || descriptor.BaseArray5() != 0))))) {
		EXIT("unsupported texture mip view: base=%u last=%u levels=%u max=%u type=%u tile=%u "
		     "class=%u numeric=%u dimension=%u mip_mode=%u read=%d written=%d "
		     "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
		     base_level, last_level, levels, descriptor.MaxMip(),
		     static_cast<uint32_t>(descriptor.Type()), static_cast<uint32_t>(tile),
		     static_cast<uint32_t>(resource.resource_class),
		     static_cast<uint32_t>(resource.numeric_class),
		     static_cast<uint32_t>(resource.dimension), static_cast<uint32_t>(resource.mip_mode),
		     resource.read, resource.written, descriptor.fields[0], descriptor.fields[1],
		     descriptor.fields[2], descriptor.fields[3], descriptor.fields[4], descriptor.fields[5],
		     descriptor.fields[6], descriptor.fields[7]);
	}
	const auto samples     = multisampled ? 1u << last_level : 1u;
	const auto depth       = static_cast<uint32_t>(descriptor.Depth()) + 1u;
	const bool reads_depth = ShaderRecompiler::IR::ReadsDepthBits(resource);
	const auto format      = reads_depth ? Prospero::BufferFormat::k16UNorm : descriptor.Format();
	const auto surface_format = TextureGetSurfaceFormatInfo(format);
	const bool shader_conversion =
	    surface_format.conversion_format != Prospero::BufferFormat::kInvalid;
	const bool sampled_numeric_class =
	    storage || resource.numeric_class ==
	                   Prospero::SampledTextureNumericClass(Prospero::RemapTextureFormat(format));
	if (!storage && resource.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled &&
	    !sampled_numeric_class) {
		EXIT("sampled image numeric class mismatch: numeric=%u format=%u addr=0x%016" PRIx64 "\n",
		     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(format), address);
	}

	const bool    volume       = type == Prospero::ImageType::kColor3D;
	const bool    layered      = type == Prospero::ImageType::kColor1DArray ||
	                             type == Prospero::ImageType::kColor2DArray ||
	                             type == Prospero::ImageType::kColor2DMsaaArray;
	auto          image_layers = layered ? depth : 1u;
	// A cube holds six faces even when DEPTH, its last face slice, ends before them (Teardown's
	// sky cube has DEPTH 0).
	if (descriptor.Type() == Prospero::ImageType::kCube) {
		image_layers = std::max(image_layers, 6u);
	}
	auto          view_base    = static_cast<uint32_t>(base_level);
	auto pixel_format = surface_format.vk_format;
	if (resource.depth_compare) {
		if (const auto* depth_format = FindGuestDepthFormatPolicy(format)) {
			pixel_format = depth_format->depth_attachment_format;
		}
	}
	const auto view_format =
	    storage ? StorageViewFormat(resource, format, pixel_format) : pixel_format;
	const auto              block_bytes = Prospero::BlockCompressedBytesPerBlock(format);
	TextureCache::ImageDesc desc {};
	// Everything below that decides desc.info besides the descriptor's own bits.
	const uint32_t memo_flags = (resource.r128 ? 1u : 0u) | (reads_depth ? 2u : 0u) |
	                            (resource.depth_compare ? 4u : 0u) | (storage ? 8u : 0u) |
	                            (single_storage_mip ? 16u : 0u) |
	                            (resource.mip_mode == ShaderRecompiler::IR::ImageMipMode::Dynamic ? 32u : 0u);
	TextureInfoMemoEntry* memo =
	    use_info_memo && Common::AbSwitch::Enabled(Common::AbSwitch::TEXTURE_INFO_MEMO)
	        ? &TextureInfoMemoSlot(descriptor, memo_flags)
	        : nullptr;
	if (memo != nullptr && memo->valid && memo->flags == memo_flags &&
	    std::memcmp(memo->fields, descriptor.fields, sizeof(memo->fields)) == 0) {
		desc.info = memo->info;
		view_base = memo->view_base;
		view_levels = memo->view_levels;
	} else {
		desc.info.data         = {address, 0};
		desc.info.pixel_format = pixel_format;
		desc.info.guest_format = format;
		desc.info.type         = TextureBaseType(type);
		desc.info.extent       = {width, height, volume ? depth : 1u};
		desc.info.resources    = {levels, image_layers};
		desc.info.bytes_per_block =
		    block_bytes != 0 ? block_bytes : Prospero::NumBytesPerElement(format);
		desc.info.samples   = samples;
		desc.info.tile_mode = tile;
		if (!resource.r128 && descriptor.MetaCompress() && tile != Prospero::TileMode::kDepth) {
			desc.info.metadata.kind = ImageMetadataKind::Dcc;
		}
		if (levels > physical_levels) {
			if (!ResolveTextureMipView(desc.info, physical_levels,
			                           !resource.r128 && descriptor.MetaCompress(), view_levels,
			                           view_base)) {
				// The layout stores no mip past MAX_MIP here, so a view starting in the chain ends at
				// MAX_MIP; dynamic-mip shaders still count mips up to LAST_LEVEL.
				if (base_level >= physical_levels ||
				    resource.mip_mode == ShaderRecompiler::IR::ImageMipMode::Dynamic) {
					EXIT("unsupported texture mip view changes physical layout: base=%u last=%u "
					     "max=%u extent=%ux%ux%u tile=%u\n",
					     base_level, last_level, max_mip, width, height, depth,
					     static_cast<uint32_t>(tile));
				}
				view_levels                = physical_levels - base_level;
				desc.info.resources.levels = physical_levels;
				desc.info.data.size        = 0;
				desc.info.UpdateSize();
			}
		} else {
			desc.info.UpdateSize();
		}
		if (memo != nullptr) {
			std::memcpy(memo->fields, descriptor.fields, sizeof(memo->fields));
			memo->flags     = memo_flags;
			memo->view_base = view_base;
			memo->view_levels = view_levels;
			memo->info      = desc.info;
			memo->valid     = true;
		}
	}
	const auto alignment = desc.info.DataAlignment();
	EXIT_NOT_IMPLEMENTED(desc.info.data.size == 0 || alignment == 0 ||
	                     (address & (static_cast<uint64_t>(alignment) - 1u)) != 0);
	// Image heaps keep the descriptors of unmapped textures, which PS5 never samples.
	if (resource.heap && !m_context.IsMapped(address, desc.info.data.size)) {
		return {.mapped = false};
	}
	if (storage) {
		ValidateStorageTexture(resource, descriptor, desc.info.data.size);
	}
	if (!resource.r128 && descriptor.MetaCompress() && tile != Prospero::TileMode::kDepth) {
		TileSizeAlign metadata_size {};
		(void)TileGetDccSize(width, height, volume ? depth : image_layers,
		                     desc.info.bytes_per_block, physical_levels, tile, metadata_size,
		                     std::countr_zero(samples));
		desc.info.metadata.range         = {descriptor.MetaAddr() << 8u, metadata_size.size};
		desc.info.metadata.dcc_alpha_msb = descriptor.DccAlphaPos();
	}
	desc.view_info = TextureViewInfo(resource, descriptor, view_format, surface_format, storage,
	                                 view_levels, desc.info.resources.layers);
	desc.view_info.base_level = view_base;
	desc.type = storage ? TextureCache::BindingType::Storage : TextureCache::BindingType::Texture;
	desc.heap = resource.heap;

	return {.desc              = std::move(desc),
	        .pixel_format      = pixel_format,
	        .view_format       = view_format,
	        .shader_conversion = shader_conversion};
}

bool RenderExecutor::SameTextureResolvePlan(const TextureResolvePlan& left,
                                            const TextureResolvePlan& right) {
	// Compare semantic fields, including all derived tiling, rather than padding bytes.
	const auto image_key = [](const ImageInfo& info) {
		return std::tie(
		    info.data, info.stencil, info.metadata.range, info.metadata.kind,
		    info.metadata.clear_word, info.metadata.compression, info.metadata.stencil_compressed,
		    info.metadata.clear_register_valid, info.metadata.dcc_alpha_msb,
		    info.htile_initial_clear, info.pixel_format, info.guest_format, info.type, info.extent,
		    info.resources, info.pitch, info.bytes_per_block, info.samples, info.tile_mode,
		    info.bgra16, info.mip_layout, info.tiling.block.family,
		    info.tiling.block.bytes_per_element, info.tiling.block.block_size,
		    info.tiling.block.block_width, info.tiling.block.block_height,
		    info.tiling.block.block_depth, info.tiling.texel_width, info.tiling.texel_height,
		    info.tiled_slice_stride, info.linear_slice_stride, info.first_tail_level);
	};
	return image_key(left.desc.info) == image_key(right.desc.info) &&
	       left.desc.view_info == right.desc.view_info && left.desc.type == right.desc.type &&
	       left.desc.heap == right.desc.heap && left.pixel_format == right.pixel_format &&
	       left.view_format == right.view_format &&
	       left.shader_conversion == right.shader_conversion && left.mapped == right.mapped;
}

TextureBinding RenderExecutor::ResolveTexture(const ShaderRecompiler::IR::ImageResource&   resource,
                                              const ShaderRecompiler::IR::DescriptorValue& value,
                                              uint64_t module_id, uint32_t resource_index) {
	if (resource.atomic64 && !m_context.GetGraphics().shader_image_int64_atomics_enabled) {
		EXIT("64-bit image atomics require shaderImageInt64Atomics\n");
	}
	auto       descriptor = DecodeNativeDescriptor<ShaderTextureResource>(value);
	const bool storage    = resource.written;
	if (storage) {
		ValidateStorageImageResource(resource);
	}

	auto& texture_cache = m_context.GetTextureCache();
	if (descriptor.IsNull()) {
		return {.layout = vk::ImageLayout::eGeneral};
	}
	// Count every sampled binding, including texture-plan memo hits. Heap roots and storage
	// accesses report no samples.
	if (m_lod_stats && descriptor.MipStatsCntEn() && !resource.heap && !storage) {
		auto&         counter = m_mip_stats[descriptor.MipStatsCntId()];
		const uint8_t base    = descriptor.BaseLevel();
		counter.finest_level  = std::min(counter.finest_level, base);
		if (descriptor.MinLod() > base << 8u) {
			counter.clamped = true;
		} else {
			counter.unclamped = true;
		}
	}

	auto plan = [&]() -> TextureResolvePlan {
		// Heap mapping and storage validation retain their original per-call path. Module id/slot
		// select a bounded entry; exact resource inputs guard collisions and module-id reuse.
		if (!resource.heap && !storage && resource_index != UINT32_MAX &&
		    Common::AbSwitch::Enabled(Common::AbSwitch::TEXTURE_PLAN_MEMO)) {
			constexpr size_t Slots = 128;
			if (m_texture_plan_memo.empty()) m_texture_plan_memo.resize(Slots);
			const uint32_t flags =
			    (resource.read ? 1u : 0u) | (resource.written ? 2u : 0u) |
			    (resource.atomic ? 4u : 0u) | (resource.atomic64 ? 8u : 0u) |
			    (resource.depth_compare ? 16u : 0u) | (resource.r128 ? 32u : 0u) |
			    (resource.heap ? 64u : 0u) | (resource.cube ? 128u : 0u) |
			    (resource.unfiltered_access_only ? 256u : 0u) | (resource.skip_degamma ? 512u : 0u);
			const std::array<uint32_t, 7> key {flags,
			                                   static_cast<uint32_t>(resource.resource_class),
			                                   static_cast<uint32_t>(resource.numeric_class),
			                                   static_cast<uint32_t>(resource.dimension),
			                                   static_cast<uint32_t>(resource.mip_mode),
			                                   static_cast<uint32_t>(resource.conversion_format),
			                                   static_cast<uint32_t>(resource.srgb_decode)};
			const auto hash = (module_id ^ (uint64_t {resource_index} * 0x9E3779B97F4A7C15ull));
			auto&      memo = m_texture_plan_memo[(hash ^ (hash >> 32u)) & (Slots - 1)];
			const bool hit =
			    memo.valid && memo.module_id == module_id &&
			    memo.resource_index == resource_index && memo.resource_key == key &&
			    std::memcmp(memo.fields.data(), descriptor.fields, sizeof(descriptor.fields)) == 0;
			static const bool probe = std::getenv("KYTY_PROBE_TEXTURE_PLAN") != nullptr;
			if (probe) {
				++m_texture_plan_probe_calls;
				m_texture_plan_probe_hits += hit;
				if ((m_texture_plan_probe_calls & 0xffffu) == 0u) {
					std::printf("texture-plan-probe calls=%" PRIu64 " hits=%" PRIu64
					            " misses=%" PRIu64 "\n",
					            m_texture_plan_probe_calls, m_texture_plan_probe_hits,
					            m_texture_plan_probe_calls - m_texture_plan_probe_hits);
				}
			}
			if (hit) {
				static const bool verify = std::getenv("KYTY_VERIFY_TEXTURE_PLAN") != nullptr;
				if (verify &&
				    !SameTextureResolvePlan(memo.plan,
				                            BuildTextureResolvePlan(resource, descriptor, false))) {
					EXIT("texture plan verify: mismatch module=0x%016" PRIx64 " resource=%u\n",
					     module_id, resource_index);
				}
				return memo.plan;
			}
			auto fresh          = BuildTextureResolvePlan(resource, descriptor, true);
			memo.module_id      = module_id;
			memo.resource_index = resource_index;
			memo.resource_key   = key;
			std::copy_n(descriptor.fields, memo.fields.size(), memo.fields.begin());
			memo.plan  = fresh;
			memo.valid = true;
			return fresh;
		}
		return BuildTextureResolvePlan(resource, descriptor, true);
	}();
	if (!plan.mapped) return {.layout = vk::ImageLayout::eGeneral};
	// FindImage may remap this copy. Never overwrite the memo's original requested view.
	auto       desc              = std::move(plan.desc);
	const auto shader_conversion = plan.shader_conversion;
	const auto pixel_format      = plan.pixel_format;
	const auto view_format       = plan.view_format;
	const auto format            = desc.info.guest_format;

	auto id = texture_cache.FindImage(desc, shader_conversion);
	if (!id) {
		// Another extent at the address holds it; the heap retries once an image is created.
		return {.desc = std::move(desc), .layout = vk::ImageLayout::eGeneral};
	}
	auto*      image               = &texture_cache.GetImage(id);
	const bool stencil_association = static_cast<bool>(image->depth_id);
	if (stencil_association) {
		id    = image->depth_id;
		image = &texture_cache.GetImage(id);
	} else if (image->info.IsDepth()) {
		if (storage) {
			EXIT("depth target cannot be bound as a storage image\n");
		}
		ValidateSampledDepthBinding(resource, descriptor, *image, pixel_format,
		                            desc.info.data.size);
	} else if (storage) {
		ValidateStorageColorView(image->info.pixel_format, view_format, descriptor.DstSelXYZW());
	} else {
		// Set after FindImage, whose metadata clears need the attachment-capable format.
		desc.view_info.format =
		    SampledViewFormat(m_context.GetGraphics(), format, pixel_format, resource.skip_degamma);
		(void)SelectSampledColorView(image->info.pixel_format, desc.view_info.format,
		                             descriptor.DstSelXYZW());
	}
	return {id, nullptr, std::move(desc)};
}

static vk::Sampler NativeSampler(RenderContext&                                  context,
                                 const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                 uint32_t                                        index,
                                 const ShaderRecompiler::IR::DescriptorValue&    value) {
	auto        descriptor = DecodeNativeDescriptor<ShaderSamplerResource>(value);
	const auto& sampler    = program.info.samplers[index];
	if (!sampler.depth_compare) {
		descriptor.fields[0] &= ~(0x7u << 12u);
	}
	if (sampler.force_point_filtering) {
		descriptor.SetPointFiltering();
	}
	return context.GetSamplerCache().GetSampler(descriptor, sampler.integer_border);
}

static vk::DescriptorBufferInfo NativeUpload(RenderContext&            context,
                                             std::span<const uint32_t> data) {
	EXIT_IF(data.empty());
	auto&      buffer = context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
	const auto offset = buffer.Copy(data.data(), data.size_bytes(), 256);
	return {buffer.Handle(), offset, data.size_bytes()};
}

void RenderExecutor::BindImage(ImageId id, bool storage) {
	if (!id) {
		return;
	}
	auto& image = m_context.GetTextureCache().GetImage(id);
	if (image.binding.is_bound) {
		image.binding.force_general |= image.binding.shader_write != storage;
	}
	image.binding.is_bound = true;
	image.binding.shader_write |= storage;
	m_bound_images.push_back(id);
}

void RenderExecutor::BindRenderTarget(ImageId id) {
	auto& image             = m_context.GetTextureCache().GetImage(id);
	image.binding.is_target = true;
	m_bound_images.push_back(id);
}

void RenderExecutor::ResetBindings() {
	for (const auto id: m_bound_images) {
		m_context.GetTextureCache().m_slot_images[id].binding = {};
	}
	m_bound_images.clear();
}

void RenderExecutor::PrepareBindings(const ShaderStageRuntime& runtime,
                                     PreparedBindings&         prepared) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(!runtime);
	const auto& program         = runtime.module->info;
	const auto& snapshot        = *runtime.resources;
	prepared.runtime            = &runtime;
	prepared.gds                = {nullptr, 0, VK_WHOLE_SIZE};
	prepared.flattened_srt      = {};
	prepared.shader_data_buffer = {};
	prepared.shared_memory      = {};
	prepared.images.resize(program.info.images.size());
	prepared.samplers.clear();
	prepared.shader_data.clear();
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		if (program.info.images[i].heap) {
			prepared.images[i] = {};
			continue;
		}
		auto binding =
		    ResolveTexture(program.info.images[i], snapshot.images[i], runtime.module->id, i);
		BindImage(binding.image_id, binding.desc.type == TextureCache::BindingType::Storage);
		binding.mip_views.swap(prepared.images[i].mip_views);
		binding.mip_views.clear();
		prepared.images[i] = std::move(binding);
	}
	prepared.samplers.reserve(program.info.samplers.size());
	for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
		prepared.samplers.push_back(NativeSampler(
		    m_context, program, i, snapshot.samplers[program.info.samplers[i].snapshot_index]));
	}
	prepared.shader_data.reserve(program.bindings.ShaderDataDwords());
	for (const auto reg: program.info.user_data_registers) {
		prepared.shader_data.push_back(snapshot.user_data[reg - program.user_data_base]);
	}
	prepared.shader_data.resize(program.bindings.ShaderDataDwords());
	if (program.bindings.descriptor_counts[static_cast<size_t>(BindingKind::Gds)] != 0) {
		prepared.gds.buffer = m_context.GetBufferCache().GetGdsBuffer()->Handle();
	}
}

void RenderExecutor::FindBuffers(std::span<PreparedBindings* const> stages) {
	KYTY_PROFILER_FUNCTION();
	auto& cache = m_context.GetBufferCache();
	for (auto* stage: stages) {
		auto&       prepared = *stage;
		const auto& program  = prepared.runtime->module->info;
		const auto& snapshot = *prepared.runtime->resources;
		const auto& layout   = program.bindings;
		prepared.buffer_sources.resize(
		    layout.descriptor_counts[static_cast<size_t>(BindingKind::Buffers)]);
		if (layout.descriptor_counts[static_cast<size_t>(BindingKind::Buffers)] == 0) {
			continue;
		}
		for (uint32_t resource = 0; resource < program.info.buffers.size(); ++resource) {
			const auto& metadata = program.info.buffers[resource];
			if (metadata.descriptor_index == UINT32_MAX) continue;
			auto&      source = prepared.buffer_sources[metadata.descriptor_index];
			const auto descriptor =
			    DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[resource]);
			const auto address = descriptor.Base48();
			auto       size    = descriptor.GetSize();
			if (size != 0 && descriptor.Stride() == 0 && descriptor.OutOfBounds() == 2 &&
			    !metadata.scalar) {
				// OOB_SELECT=2 only tests NumRecords != 0; a zero-stride fetch
				// still needs its full payload when NumRecords is smaller than it.
				size = std::max(size, uint64_t {metadata.max_byte_extent});
			}
			if (address == 0 || size == 0) {
				source = {};
				continue;
			}
			if (descriptor.NumRecords() == UINT32_MAX) {
				if (program.info.buffers[resource].written) {
					// Sentinel write ranges end before tables captured by any bound stage.
					for (const auto* reader: stages) {
						for (const auto [read_address, read_size]:
						     reader->runtime->resources->specialization_reads) {
							if (read_size == 0) continue;
							if (read_address > address) {
								size = std::min(size, read_address - address);
							} else if (address - read_address < read_size) {
								EXIT("scalar resource reads overlap a shader buffer write\n");
							}
						}
					}
				}
				//  PPSA15246
				const auto& graphics = m_context.GetGraphics();
				const auto  limit =
				    uint64_t {graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange} -
				    (graphics.StorageMinAlignment() - 1);
				size = std::min({size, uint64_t {256} * 1024 * 1024, limit});
			}
			// Finite GPU descriptors can span sparse PRT holes; clamp only oversized/sentinel
			// ranges.
			if (size >= (uint64_t {1} << 30u) || descriptor.NumRecords() == UINT32_MAX) {
				size = Libs::LibKernel::Memory::ClampRangeSize(address, size);
			}
			source = {address, size, cache.FindBuffer(address, size)};
		}
	}
}

void RenderExecutor::RebindBuffers(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	const auto& program  = prepared.runtime->module->info;
	const auto& snapshot = *prepared.runtime->resources;
	const auto& layout   = program.bindings;

	prepared.buffers.resize(layout.descriptor_counts[static_cast<size_t>(BindingKind::Buffers)]);
	std::fill(prepared.shader_data.begin() + layout.memory_offset_dword, prepared.shader_data.end(),
	          0);
	auto pack_memory_offset = [&](uint32_t index, uint32_t offset) {
		const auto dword = layout.memory_offset_dword + index / 4u;
		const auto shift = (index % 4u) * 8u;
		prepared.shader_data[dword] |= offset << shift;
	};
	for (const auto& resource: program.info.buffers) {
		const auto index = resource.descriptor_index;
		if (index == UINT32_MAX) continue;
		uint32_t buffer_offset = 0;
		prepared.buffers[index] =
		    NativeStorageBuffer(m_context, prepared.buffer_sources[index], resource, buffer_offset);
		pack_memory_offset(index, buffer_offset);
		// Null bindings allow 0 bytes, not the shared null buffer's 16, so no store reaches it.
		prepared.shader_data[layout.MemorySizeDword() + index] =
		    static_cast<uint32_t>(prepared.buffer_sources[index].size + buffer_offset);
	}
	if (layout.descriptor_counts[static_cast<size_t>(BindingKind::FlattenedSrt)] != 0) {
		prepared.flattened_srt = NativeUpload(m_context, snapshot.flattened_srt);
	}
	if (layout.descriptor_counts[static_cast<size_t>(BindingKind::ShaderData)] != 0) {
		prepared.shader_data_buffer = NativeUpload(m_context, prepared.shader_data);
	}
}

void RenderExecutor::RebindImages(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	const auto& program       = prepared.runtime->module->info;
	const auto& snapshot      = *prepared.runtime->resources;
	auto&       images        = prepared.images;
	auto&       texture_cache = m_context.GetTextureCache();
	// Acquire each view before a later overlapping descriptor can replace its image.
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		if (!images[i].image_id) continue;
		auto& old_image = texture_cache.m_slot_images[images[i].image_id];
		if (!old_image.registered || old_image.binding.needs_rebind) {
			old_image.binding = {};
			images[i]         = ResolveTexture(program.info.images[i], snapshot.images[i],
			                                   prepared.runtime->module->id, i);
			BindImage(images[i].image_id,
			          images[i].desc.type == TextureCache::BindingType::Storage);
		}
		auto& binding = images[i];
		binding.mip_views.clear();
		const auto& resource = program.info.images[i];
		if (resource.mip_mode == ShaderRecompiler::IR::ImageMipMode::Dynamic) {
			EXIT_IF(resource.mip_count == 0u ||
			        resource.mip_count != binding.desc.view_info.level_count);
			binding.mip_views.reserve(resource.mip_count);
			for (uint32_t mip = 0; mip < resource.mip_count; mip++) {
				auto desc = binding.desc;
				desc.view_info.base_level += mip;
				desc.view_info.level_count = 1;
				// The shader selects the mip after applying the guest minimum LOD.
				desc.view_info.min_lod = 0;
				binding.mip_views.push_back(texture_cache.FindTexture(binding.image_id, desc));
			}
			binding.image_view = binding.mip_views.front();
		} else {
			binding.image_view = texture_cache.FindTexture(binding.image_id, binding.desc);
		}
	}
}

void RenderExecutor::PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
                                             std::span<RenderColorInfo>         colors) {
	bool uses_dma        = false;
	bool uses_image_heap = false;
	FindBuffers(stages);
	for (auto* stage: stages) {
		uses_dma |= stage->runtime->module->info.info.uses_dma;
		uses_image_heap |= stage->runtime->module->info.info.uses_image_heap;
	}
	if (uses_dma) {
		m_context.PrepareBda();
	}
	if (uses_image_heap) {
		m_context.GetImageHeap().Rebind(*this);
	}
	for (auto* stage: stages) {
		RebindImages(*stage);
	}
	auto& cache = m_context.GetTextureCache();
	for (auto& target: colors) {
		EXIT_IF(!target.image_id);
		auto& old_image = cache.m_slot_images[target.image_id];
		if (!old_image.registered || old_image.binding.needs_rebind) {
			old_image.binding                = {};
			target.desc.view_info.base_level = target.guest_mip_level;
			target.desc.view_info.base_layer = target.guest_array_layer;
			target.image_id                  = cache.FindImage(target.desc);
			BindRenderTarget(target.image_id);
		}
	}
	// Discovery can read back PS5 metadata and submit the scheduler. Reserve draw buffers only
	// after image identities are final; attachment layout transitions follow buffer alias copies.
	for (auto* stage: stages) {
		RebindBuffers(*stage);
	}
}

// A deliberately backing-only observation of the two screen-space filters seen near a stall.
// TryReadBacking bypasses guest faults: this must never cause a download, submission or wait.
// Recorded upload copies may not have run yet, so these bytes are NOT a GPU-verified loop bound.
static bool TraceFilterBoundEnabled() {
	static const bool enabled = [] {
		const auto* path = std::getenv("KYTY_TRACE_FILTER_BOUND");
		return path != nullptr && *path != '\0';
	}();
	return enabled;
}

static void TraceFilterBound(RenderContext& context, const PreparedBindings& prepared) {
	const auto& program = prepared.runtime->module->info;
	if (program.stage != ShaderType::Pixel ||
	    (program.shader_hash != UINT64_C(0x26ec5a6f42265b1f) &&
	     program.shader_hash != UINT64_C(0x6869ba5373a429b9))) {
		return;
	}
	struct Trace {
		std::FILE* file = nullptr;
		uint32_t   rows = 0;
		Trace() {
			const auto* path = std::getenv("KYTY_TRACE_FILTER_BOUND");
			if (path == nullptr || *path == '\0') return;
			file = std::fopen(path, "w");
			if (file == nullptr) {
				std::fprintf(stderr, "filter-bound trace: cannot open diagnostic output\n");
				return;
			}
			std::fprintf(
			    file, "state,unix_ms,monotonic_us,tick,shader_hash,module_id,resource_index,"
			          "descriptor_words,descriptor0,descriptor1,descriptor2,descriptor3,"
			          "source_id,guest_base,source_size,native_buffer,native_offset,native_range,"
			          "adjustment,shader_size,shader_index,shader_size_guard,misaligned,"
			          "guest_word_address,source_word_in_range,cpu_backing_read_ok,"
			          "cpu_backing_bound_u32,cpu_backing_bound_i32,gpu_dirty,gpu_dirty_begin,gpu_"
			          "dirty_end\n");
		}
		~Trace() {
			if (file != nullptr) std::fclose(file);
		}
	};
	static Trace trace;
	if (trace.file == nullptr || trace.rows >= 100000) return;
	const auto& layout   = program.bindings;
	const auto  resource = std::ranges::find_if(
        program.info.buffers, [](const auto& value) { return value.descriptor_index == 0; });
	if (resource == program.info.buffers.end() || prepared.buffer_sources.empty() ||
	    prepared.buffers.empty() || layout.memory_offset_dword >= prepared.shader_data.size() ||
	    layout.MemorySizeDword() >= prepared.shader_data.size()) {
		return; // Diagnostics never turn an unsupported observation into an emulator failure.
	}
	const auto  resource_index = static_cast<size_t>(resource - program.info.buffers.begin());
	const auto& snapshot       = *prepared.runtime->resources;
	if (resource_index >= snapshot.buffers.size()) return;
	const auto& descriptor   = snapshot.buffers[resource_index];
	const auto& source       = prepared.buffer_sources[0];
	const auto& native       = prepared.buffers[0];
	const auto  adjustment   = prepared.shader_data[layout.memory_offset_dword] & 255u;
	const auto  shader_size  = prepared.shader_data[layout.MemorySizeDword()];
	const auto  shader_index = (adjustment >> 2u) + 6u;
	const bool  guard        = shader_index < (shader_size >> 2u);
	// The shader rounds the packed byte adjustment down to a dword before adding six.
	const uint64_t relative_word   = 24u - (adjustment & 3u);
	const bool     source_in_range = source.address != 0 && relative_word <= source.size &&
	                             sizeof(uint32_t) <= source.size - relative_word &&
	                             source.address <= UINT64_MAX - relative_word - sizeof(uint32_t);
	const auto address = source_in_range ? source.address + relative_word : 0;
	uint32_t   value   = 0;
	const bool read_ok = guard && source_in_range &&
	                     LibKernel::Memory::TryReadBacking(address, &value, sizeof(value));
	auto&      cache     = context.GetBufferCache();
	const bool gpu_dirty = source_in_range && cache.HasGpuDirtyBytes(address, sizeof(value));
	const auto dirty =
	    source_in_range ? cache.GpuDirtyRangeAt(address) : std::pair<uint64_t, uint64_t> {};
	const auto raw_handle = static_cast<VkBuffer>(native.buffer);
	uint64_t   handle     = 0;
	static_assert(sizeof(handle) == sizeof(raw_handle));
	std::memcpy(&handle, &raw_handle, sizeof(handle));
	using namespace std::chrono;
	const auto unix_ms =
	    duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	const auto monotonic_us =
	    duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
	const auto signed_value = std::bit_cast<int32_t>(value);
	const int  written      = std::fprintf(
        trace.file,
        "cpu_backing_bound_not_gpu_verified,%lld,%lld,%" PRIu64 ",%016" PRIx64 ",%016" PRIx64
        ",%zu,%u,%08x,%08x,%08x,%08x,%u,%016" PRIx64 ",%" PRIu64 ",%016" PRIx64 ",%" PRIu64
        ",%" PRIu64 ",%u,%u,%u,%u,%u,%016" PRIx64 ",%u,%u,%u,%" PRId32 ",%u,%016" PRIx64
        ",%016" PRIx64 "\n",
        static_cast<long long>(unix_ms), static_cast<long long>(monotonic_us),
        context.GetCommandScheduler().CurrentTick(), program.shader_hash,
        prepared.runtime->module->id, resource_index, descriptor.dword_count, descriptor.dwords[0],
        descriptor.dwords[1], descriptor.dwords[2], descriptor.dwords[3], source.id.index,
        source.address, source.size, handle, uint64_t {native.offset}, uint64_t {native.range},
        adjustment, shader_size, shader_index, guard ? 1u : 0u, adjustment & 3u, address,
        source_in_range ? 1u : 0u, read_ok ? 1u : 0u, value, signed_value, gpu_dirty ? 1u : 0u,
        dirty.first, dirty.second);
	++trace.rows;
	// Preserve the last observation if the next draw stalls and the test kills the process.
	if (written < 0 || std::fflush(trace.file) != 0) {
		std::fclose(trace.file);
		trace.file = nullptr; // Output failures disable the diagnostic, never the game.
	}
}

void RenderExecutor::CommitBindings(CommandBuffer&                     buffer,
                                    vk::PipelineBindPoint              pipeline_bind_point,
                                    const PipelineCache::Pipeline&     pipeline,
                                    std::span<PreparedBindings* const> prepared_bindings) {
	KYTY_PROFILER_FUNCTION();
	auto                           vk_buffer              = buffer.Handle();
	size_t                         image_descriptor_count = 0;
	ShaderRecompiler::IR::PushData push_data;
	bool                           has_push_data = false;
	for (const auto* prepared: prepared_bindings) {
		const auto& program = prepared->runtime->module->info;
		for (auto mask = program.bindings.descriptor_mask; mask != 0; mask &= mask - 1) {
			const auto index = std::countr_zero(mask);
			const auto count = program.bindings.descriptor_counts[index];
			if (index >= ShaderRecompiler::IR::FirstImageBinding &&
			    index <= static_cast<size_t>(BindingKind::Samplers)) {
				image_descriptor_count += count;
			}
		}
		EXIT_IF(!(pipeline.push_stages & NativeShaderStage(program.stage)));
	}
	if (pipeline.uses_image_heap) {
		m_context.GetImageHeap().Commit(buffer, pipeline_bind_point, pipeline.pipeline_layout);
		// Its stream copy can wait for the GPU, which submits and begins the next command buffer.
		vk_buffer = buffer.Handle();
	}
	m_descriptor_images.resize(image_descriptor_count);
	m_descriptor_writes.clear();
	size_t                                  image_cursor = 0;
	std::array<vk::DescriptorBufferInfo, 2> address_buffers;

	for (auto* prepared: prepared_bindings) {
		const auto& program     = prepared->runtime->module->info;
		auto&       descriptors = *prepared;
		if (descriptors.gds.buffer != nullptr) {
			buffer.EndRendering();
			const auto barrier       = MakeGdsDependency(descriptors.gds.buffer);
			const auto shader_stages = ShaderPipelineStages(NativeShaderStage(program.stage));
			vk_buffer.pipelineBarrier(
			    vk::PipelineStageFlagBits::eHost | vk::PipelineStageFlagBits::eTransfer |
			        vk::PipelineStageFlagBits::eAllGraphics |
			        vk::PipelineStageFlagBits::eComputeShader,
			    shader_stages, vk::DependencyFlags {}, 0, nullptr, 1, &barrier, 0, nullptr);
		}

		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			auto& binding = descriptors.images[i];
			if (!binding.image_id) continue;
			auto&       image = m_context.GetTextureCache().GetImage(binding.image_id);
			const auto& view  = binding.desc.view_info;
			const ImageSubresourceRange range {view.base_level, view.level_count, view.base_layer,
			                                   view.layer_count};
			const bool storage = binding.desc.type == TextureCache::BindingType::Storage;
			if (image.binding.is_target) {
				const auto layout = image.binding.attachment_layout;
				EXIT_IF(layout == vk::ImageLayout::eUndefined);
				if (image.info.IsDepth()) {
					const auto host_view =
					    std::ranges::find(image.views, binding.image_view, &CachedImageView::view);
					EXIT_IF(storage || host_view == image.views.end());
					const auto aspect = host_view->info.aspect;
					if (aspect & ~DepthReadableAspects(layout)) {
						EXIT("sampling a writable depth/stencil attachment aspect\n");
					}
				}
				image.Transit(layout,
				              image.binding.attachment_access | vk::AccessFlagBits2::eShaderRead |
				                  (image.binding.shader_write ? vk::AccessFlagBits2::eShaderWrite
				                                              : vk::AccessFlags2 {}),
				              {}, vk_buffer);
			} else if (image.binding.force_general && !image.info.IsDepth()) {
				const vk::AccessFlags2 storage_access = image.binding.shader_write
				                                            ? vk::AccessFlagBits2::eShaderWrite
				                                            : vk::AccessFlags2 {};
				image.Transit(vk::ImageLayout::eGeneral,
				              vk::AccessFlagBits2::eShaderRead | storage_access, {}, vk_buffer);
			} else if (storage) {
				image.Transit(vk::ImageLayout::eGeneral,
				              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
				              range, vk_buffer);
			} else {
				image.Transit(image.ReadLayout(), vk::AccessFlagBits2::eShaderRead, range,
				              vk_buffer);
			}
			binding.layout = image.backing.state.layout;
		}

		std::array<size_t, ShaderRecompiler::IR::ImageBindingCount> image_starts {};
		for (auto mask = program.bindings.descriptor_mask; mask != 0; mask &= mask - 1) {
			const auto             index = std::countr_zero(mask);
			const auto             kind  = static_cast<BindingKind>(index);
			vk::WriteDescriptorSet write {};
			write.dstBinding       = ShaderRecompiler::IR::NativeBinding(program.stage, kind);
			write.descriptorType   = NativeDescriptorType(kind);
			write.descriptorCount  = program.bindings.descriptor_counts[index];
			const auto image_start = image_cursor;
			if (ShaderRecompiler::IR::ImageBindingResourceClass(kind) !=
			    ShaderRecompiler::IR::ImageResourceClass::None) {
				image_starts[ShaderRecompiler::IR::ImageBindingIndex(kind)] = image_start;
				image_cursor += write.descriptorCount;
			} else {
				switch (kind) {
					case BindingKind::Buffers:
						// Every source was checked by RebindBuffers/FindBuffers already.
						EXIT_IF(descriptors.buffers.size() != write.descriptorCount);
						write.pBufferInfo = descriptors.buffers.data();
						break;
					case BindingKind::BdaPagetable:
					case BindingKind::FaultBuffer: {
						auto&       cache      = m_context.GetBufferCache();
						const auto* bda_buffer = kind == BindingKind::BdaPagetable
						                             ? cache.GetBdaPageTableBuffer()
						                             : cache.GetFaultBuffer();
						auto&       view       = address_buffers[kind == BindingKind::FaultBuffer];
						view                   = {bda_buffer->Handle(), 0, bda_buffer->Size()};
						write.pBufferInfo      = &view;
						break;
					}
					case BindingKind::FlattenedSrt:
					case BindingKind::ShaderData:
					case BindingKind::SharedMemory:
					case BindingKind::Gds: {
						const vk::DescriptorBufferInfo* view = &descriptors.gds;
						if (kind == BindingKind::FlattenedSrt) {
							view = &descriptors.flattened_srt;
						} else if (kind == BindingKind::ShaderData) {
							view = &descriptors.shader_data_buffer;
						} else if (kind == BindingKind::SharedMemory) {
							view = &descriptors.shared_memory;
						}
						EXIT_IF(view->buffer == nullptr);
						write.pBufferInfo = view;
						break;
					}
					case BindingKind::Samplers:
						for (const auto sampler: descriptors.samplers) {
							m_descriptor_images[image_cursor++] = {sampler, nullptr,
							                                       vk::ImageLayout::eUndefined};
						}
						break;
					case BindingKind::Count: EXIT("invalid descriptor binding kind");
				}
			}
			if (image_cursor != image_start) {
				write.pImageInfo = m_descriptor_images.data() + image_start;
			}
			m_descriptor_writes.push_back(write);
		}
		for (uint32_t i = 0; i < program.info.images.size(); ++i) {
			const auto& resource = program.info.images[i];
			const auto& texture  = descriptors.images[i];
			if (resource.heap) continue;
			const auto kind  = *ShaderRecompiler::IR::DescriptorBindingForImage(resource);
			const auto start = image_starts[ShaderRecompiler::IR::ImageBindingIndex(kind)] +
			                   resource.descriptor_index;
			for (uint32_t mip = 0; mip < resource.mip_count; ++mip) {
				m_descriptor_images[start + mip] = MakeImageInfo(texture, mip);
			}
		}

		const auto shader_data_dwords = program.bindings.ShaderDataDwords();
		EXIT_IF(prepared->shader_data.size() != shader_data_dwords);
		if (TraceFilterBoundEnabled()) [[unlikely]] {
			TraceFilterBound(m_context, *prepared);
		}
		if (program.bindings.UsesPushData()) {
			std::ranges::copy(prepared->shader_data,
			                  push_data.dwords.begin() + program.bindings.push_data_start_dword);
			has_push_data = true;
		}
	}

	if (has_push_data) {
		vk_buffer.pushConstants(pipeline.pipeline_layout, pipeline.push_stages, 0,
		                        sizeof(push_data), push_data.dwords.data());
	}

	if (!m_descriptor_writes.empty()) {
		if (pipeline.uses_push_descriptors) {
			vk_buffer.pushDescriptorSetKHR(pipeline_bind_point, pipeline.pipeline_layout, 0,
			                               static_cast<uint32_t>(m_descriptor_writes.size()),
			                               m_descriptor_writes.data());
		} else {
			const auto set = m_context.GetDescriptorHeap().Commit(pipeline.descriptor_set_layout);
			for (auto& write: m_descriptor_writes) {
				write.dstSet = set;
			}
			m_context.GetGraphics().device.updateDescriptorSets(
			    static_cast<uint32_t>(m_descriptor_writes.size()), m_descriptor_writes.data(), 0,
			    nullptr);
			vk_buffer.bindDescriptorSets(pipeline_bind_point, pipeline.pipeline_layout, 0, 1, &set,
			                             0, nullptr);
		}
	}
}

} // namespace Libs::Graphics
