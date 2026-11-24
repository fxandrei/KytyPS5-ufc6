#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_

#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

// Runtime-only memo of one Program's last successful post-walk materialization (AbSwitch
// SRT_REFRESH_MEMO, bit 52). The key is the walker scratch bytes [0, OutputOffset(plan)); equal
// bytes mean an identical resource snapshot and specialization, so the descriptor loops and
// BuildResourceSpecialization are skipped. Never serialized and never shared across programs.
struct MaterializationMemo {
	std::vector<uint8_t> key;          // previous scratch [0, OutputOffset(plan))
	uint32_t             miss_streak = 0;  // consecutive eligible whole-key misses
	uint32_t             cooldown    = 0;  // refreshes to skip after a miss streak
	bool                 valid       = false; // a successful store set the key
	bool                 reused      = false; // the last refresh was an L1 hit
	int8_t               eligible    = -1;    // -1 unknown, 0 no, 1 yes
	// AbSwitch SRT_PREWALK_MEMO (2026-11-19): the inputs of this Program's last walk - the user data
	// it read, the shader base, the guest map epoch and every DWORD it loaded with its value. Equal
	// inputs mean the walker, scratch, flat SRT and snapshot are all unchanged, so the walk itself is
	// skipped. Walks with unknown optional loads or loads completed by the fault handler are not
	// recorded; a load from a page with GPU-written bytes makes the check fail.
	std::vector<std::pair<uint64_t, uint64_t>> prewalk_captures; // walker capture list (scratch)
	std::vector<uint32_t>                      prewalk_user_data;  // the walker's DWORDs (mask order)
	uint64_t                                   prewalk_user_mask = 0; // 0: the srt_user_data prefix
	std::vector<uint32_t>                      prewalk_workgroups;
	std::vector<uint64_t>                      prewalk_addresses;
	std::vector<uint32_t>                      prewalk_values;
	uint64_t                                   prewalk_shader_base = 0;
	uint64_t                                   prewalk_epoch       = 0;
	bool                                       prewalk_valid       = false;
};

// Canonical module-affecting resource state. Runtime addresses and descriptor payloads remain in
// ResourceSnapshot and therefore do not create shader permutations.
struct ResourceSpecialization {
	struct Buffer {
		uint32_t               packed_stride                   = 0;
		Prospero::BufferFormat descriptor_format               = Prospero::BufferFormat::kInvalid;
		uint32_t               descriptor_swizzle              = DstSel(4, 5, 6, 7);
		bool                   zero_stride_oob                 = false;
		bool                   scalar_empty                    = false;
		uint32_t               indirect_root                   = BufferResource::NoIndirectBuffer;
		uint32_t               indirect_mapping_offset         = 0;
		uint32_t               indirect_search_iterations      = 0;
		bool                   operator==(const Buffer&) const = default;
	};

	struct Image {
		Prospero::TextureNumericClass numeric_class = Prospero::TextureNumericClass::Unsupported;
		Decoder::ImageDimension       dimension     = Decoder::ImageDimension::Unknown;
		uint32_t                      mip_count     = 1;
		Prospero::BufferFormat        conversion_format          = Prospero::BufferFormat::kInvalid;
		uint32_t                      shader_swizzle             = ShaderImageIdentitySwizzle;
		Prospero::BufferFormat        srgb_decode                = Prospero::BufferFormat::kInvalid;
		uint32_t                      indirect_root              = ImageResource::NoIndirectImage;
		uint32_t                      indirect_mapping_offset    = 0;
		uint32_t                      indirect_search_iterations = 0;
		bool                          cube                       = false;
		bool                          skip_degamma               = false;
		bool                          fmask                      = false;
		bool                          heap                       = false;
		bool                          operator==(const Image&) const = default;
	};

	struct Sampler {
		uint32_t indirect_root = SamplerResource::NoIndirectSampler;
		uint32_t indirect_mapping_offset = 0;
		uint32_t indirect_search_iterations = 0;
		bool operator==(const Sampler&) const = default;
	};

	std::vector<Buffer> buffers;
	std::vector<Image>  images;
	std::vector<Sampler> samplers;

	bool operator==(const ResourceSpecialization&) const = default;
};

// Extracts the descriptor/SRT value graph before resource specialization. The returned plan owns
// its values and is independent of the translated shader CFG.
ResourcePlan ExtractResourcePlan(const Program& program);

// Shader cache record of a plan; loading it into a default-constructed plan rebuilds the walker
// and scratch state like extraction. A failed load leaves a plan that must not be used.
std::vector<uint8_t> SaveResourcePlan(const ResourcePlan& plan);
bool                 LoadResourcePlan(std::span<const uint8_t> bytes, ResourcePlan& plan);

// Refreshes cached resources and specialization in place. A failed refresh must not be used.
// `memo`, when non-null, skips the post-walk descriptor loops and BuildResourceSpecialization when
// the walk produced the same scratch bytes as the previous successful refresh (see
// MaterializationMemo).
bool MaterializeResources(const ResourcePlan& program, const SrtRuntime& runtime,
                          ResourceSnapshot& snapshot, ResourceSpecialization& specialization,
                          MaterializationMemo* memo = nullptr);

// Applies an already-derived specialization to native IR before layout and emission.
void ApplyResourceSpecialization(Program& program, const ResourceSpecialization& specialization);

// Kind of an image heap entry as root sees it, or 0 (and a zeroed value) when root reads it as
// null. Entries of one kind specialize identically.
uint32_t ImageHeapKind(const ImageResource& root, DescriptorValue& value);
// The sampled resource that root resolves entries of kind as; kind 0 gives the identity of
// root's view of a heap.
ImageResource ImageHeapResource(const ImageResource& root, uint32_t kind);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_ */
