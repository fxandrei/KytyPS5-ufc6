#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/file.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/rectListShader.h"
#include "graphics/shader/shader.h"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <mutex>
#include <queue>
#include <span>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
struct ShaderParams;
class CommandBuffer;

namespace HW {
class Context;
class UserConfig;
class Shader;
} // namespace HW

namespace ShaderRecompiler {
struct CompileOptions;
struct ShaderSource;
struct TranslateResult;
} // namespace ShaderRecompiler

namespace ShaderRecompiler::IR {
class ImageHeapSource;
} // namespace ShaderRecompiler::IR

namespace ShaderRecompiler::Spirv {
struct Profile;
} // namespace ShaderRecompiler::Spirv

#pragma pack(push, 1)

struct PipelineStaticParameters {
	bool                       negative_one_to_one      = false;
	bool                       depth_clip_enable        = true;
	vk::PrimitiveTopology      topology                 = vk::PrimitiveTopology::ePointList;
	// 0 without tessellation, 3 for a rectangle list, else the input control points.
	uint32_t                   patch_control_points     = 0;
	bool                       primitive_restart_enable = false;
	uint32_t                   samples                  = 1;
	bool                       sample_shading_enable    = false;
	bool                       alpha_to_coverage_enable = false;
	uint32_t                   color_mask[RENDER_COLOR_ATTACHMENTS_MAX]           = {};
	bool                       provoking_vtx_last                                 = false;
	vk::PolygonMode            polygon_mode                                       = vk::PolygonMode::eFill;
	uint8_t                    color_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    color_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    color_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	uint8_t                    alpha_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    alpha_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    alpha_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	bool                       separate_alpha_blend[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	bool                       blend_enable[RENDER_COLOR_ATTACHMENTS_MAX]         = {};
};

struct PipelineRenderingState {
	std::array<vk::Format, RENDER_COLOR_ATTACHMENTS_MAX> color_formats {};
	vk::Format                                           depth_format   = vk::Format::eUndefined;
	vk::Format                                           stencil_format = vk::Format::eUndefined;
	uint32_t                                             color_count    = 0;
};

struct PipelineVertexInputState {
	struct Binding {
		uint32_t stride   = 0;
		uint8_t  instance = 0;
	};
	struct Attribute {
		uint32_t offset  = 0;
		uint8_t  binding = 0;
		uint8_t  format  = 0; // Guest buffer format.
	};

	std::array<Binding, ShaderVertexInputInfo::RES_MAX>   bindings {};
	std::array<Attribute, ShaderVertexInputInfo::RES_MAX> attributes {};
	uint8_t                                               binding_count   = 0;
	uint8_t                                               attribute_count = 0;
};

// Everything a graphics pipeline is made of besides the stage modules its ids name. Aligned so
// that rendering.color_formats can be passed to Vulkan.
struct alignas(4) GraphicsPipelineKey {
	// VS, LS or mesh; HS; TES; PS. 0 marks an absent stage.
	std::array<uint64_t, 4>  stage_ids {};
	PipelineRenderingState   rendering;
	PipelineVertexInputState vertex_input;
	PipelineStaticParameters static_params;

	bool operator==(const GraphicsPipelineKey& other) const;
};

#pragma pack(pop)

// Keys are compared and hashed as bytes.
static_assert(std::has_unique_object_representations_v<GraphicsPipelineKey>);

// A compiled stage with what pipeline creation needs from it besides the key.
struct ShaderModule {
	uint64_t                                     id     = 0;
	vk::ShaderModule                             module = nullptr;
	ShaderRecompiler::IR::CompiledShaderInfo     info;
	ShaderRecompiler::IR::ResourceSpecialization specialization;
	std::vector<uint32_t>                        function_code;
	// Rectangle-list PS: the parameters it reads.
	std::vector<RectListParameter>               rect_parameters;
	// Indirect mesh draw: a task shader reads the records and launches the groups.
	bool                                         mesh_task = false;
};

// A draw's depth target and its first color_count color targets.
struct DrawTargets {
	RenderDepthInfo depth;
	RenderColorInfo colors[RENDER_COLOR_ATTACHMENTS_MAX];
	uint32_t        color_count = 0;
};

// A draw's topology and shader stages (from PipelineCache::GetGraphicsPrograms).
struct DrawRenderState {
	// Color targets the pixel program writes.
	uint32_t                             mrt_mask = 0;
	vk::PrimitiveTopology                topology = vk::PrimitiveTopology::ePointList;
	bool                                 primitive_restart_enable = false;
	bool                                 ps_active                = false;
	std::array<ShaderVertexInputInfo, 3> vertex_info;
	ShaderPixelInputInfo                 ps_input_info;
	// Input: the user-data slots a GPU-read indirect draw patches.
	ShaderIndirectOffsets                indirect_offsets;
	bool                                 indirect_draw = false;
	// The slice offset the targets resolved with, and whether this draw took the previous draw's
	// targets (AbSwitch DRAW_CHAIN; see TargetMemo in renderDraw.cpp).
	uint32_t slice_offset = 0;
	bool     target_memo  = false;

	[[nodiscard]] std::span<const ShaderVertexInputInfo> VertexStages() const {
		return {vertex_info.data(), vertex_info[1].stage ? 3u : 1u};
	}

	// Readies reused shader/draw state for the next draw (AbSwitch EXACT_DRAW_STATE).
	// Targets are held separately by RenderExecutor and resolved or restored for each draw.
	void Reset(ShaderIndirectOffsets offsets, bool indirect, bool tessellation) {
		mrt_mask                 = 0;
		topology                 = vk::PrimitiveTopology::ePointList;
		primitive_restart_enable = false;
		ps_active                = false;
		// PrepareProgram now relies on caller initialization, including indirect-draw and mesh
		// metadata. Reset every active stage, not just its resolved module pointer.
		vertex_info[0] = {};
		for (uint32_t i = 1; i < vertex_info.size(); ++i) {
			if (tessellation) {
				vertex_info[i] = {};
			} else {
				vertex_info[i].stage = {};
			}
		}
		ps_input_info    = {};
		indirect_offsets = offsets;
		indirect_draw    = indirect;
		slice_offset     = 0;
		target_memo      = false;
	}
};

// KYTY_DRAW_PREP (agents/2026-11-20/parallel-cp-design.md, M1/M2): why a pure preparation left a
// draw to the serial path.
enum class DrawPrepRefusal : uint8_t {
	None,
	Tessellation, // the tess layout cache is written on a miss
	MeshDropped,  // beyond the host mesh limits: the serial path drops the draw and warns
	NoProgram,    // first use of the program key: the serial path translates and compiles
	SkipProgram,  // unsupported GPU-selected call target
	CallProgram,  // call programs refresh their source and plan per draw
	Materialize,  // MaterializeResources failed (the serial path exits)
	NoModule,     // new permutation: the serial path compiles a module
	Certificate,  // something the certificate covers changed before the commit
	Count
};

// The caller-owned outputs and the coherence certificate of one draw's program preparation from
// a register snapshot (PipelineCache::PrepareGraphicsPrograms). A prepared DrawRenderState's stage
// runtimes point into `resources`, so this must outlive the draw that uses them.
struct DrawProgramPrep {
	// [0] the vertex (or mesh) stage, [1] the pixel stage. Tessellation is refused.
	std::array<ShaderRecompiler::IR::ResourceSnapshot, 2>       resources;
	std::array<ShaderRecompiler::IR::ResourceSpecialization, 2> specialization;
	// Certificate, loaded before the first guest read: a change of either between the preparation
	// and its commit refuses it. Inline (M2) both are trivially unchanged; with a window (M3) the
	// certificate also needs the read set (see drawprep.md).
	uint64_t guest_map_epoch       = 0;
	uint64_t shader_map_generation = 0;
	// Walker loads on pages with GPU-written bytes during this preparation (statistics: a worker
	// would refuse such a draw).
	uint64_t        guarded_loads = 0;
	DrawPrepRefusal refusal       = DrawPrepRefusal::None;
};

// Writes the shader store on its own thread, running jobs in the order they were posted.
class ShaderStoreWriter {
public:
	ShaderStoreWriter();
	// Keeps the first `kept` bytes of the store at `path`, or starts it over with `header`.
	void Open(const std::filesystem::path& path, std::span<const uint8_t> header, uint64_t kept);
	// Whether the store is open.
	explicit operator bool() const { return !m_file.IsInvalid(); }
	// Compresses `record` and appends it on the store thread, in call order.
	void Append(std::vector<uint8_t>&& record);
	void Post(Common::UniqueFunction<void>&& job);
	// Runs the queued jobs and stops the thread; jobs posted later never run. Safe to call again.
	void Finish() { m_thread = {}; }

private:
	friend struct PipelineCacheTestAccess;

	Common::File                             m_file;
	std::mutex                               m_mutex;
	std::condition_variable_any              m_ready;
	std::queue<Common::UniqueFunction<void>> m_jobs;
	// Declared last, so it drains the queue and joins before the rest is destroyed.
	std::jthread m_thread;
};

// The owning renderer serializes access, including saves while the GPU is running.
class PipelineCache {
public:
	// Without a shader store path nothing is kept across runs.
	PipelineCache(GraphicContext& graphics, ShaderRecompiler::IR::ImageHeapSource* image_heap,
	              const std::filesystem::path& shader_store = {});
	~PipelineCache();
	KYTY_CLASS_NO_COPY(PipelineCache);
	// Finishes the queued writes and stops the store thread, then writes the driver cache if it
	// changed.
	void Save();

	struct Pipeline {
		vk::PipelineLayout      pipeline_layout       = nullptr;
		vk::Pipeline            pipeline              = nullptr;
		vk::DescriptorSetLayout descriptor_set_layout = nullptr;
		// The push constant range's stages, which every update names.
		vk::ShaderStageFlags    push_stages;
		bool                    uses_push_descriptors = false;
		bool                    uses_image_heap       = false;
	};

	// Reads the shaders and registers bound in `command` into the state's shader stages.
	void GetGraphicsPrograms(const CommandBuffer& command, DrawRenderState& state);
	// KYTY_DRAW_PREP: GetGraphicsPrograms as a function of register copies that changes nothing in
	// the cache: no translation or compile, no program, hint or tess-layout insert, no SRT memo, the
	// materialized resources in `prep`. False, with prep.refusal set and `state` partly written, when
	// the draw needs the serial path. GPU thread only for now (walker scratch, m_prep_key).
	bool PrepareGraphicsPrograms(const HW::Context& context, const HW::UserConfig& user_config,
	                             const HW::Shader& shaders, DrawRenderState& state,
	                             DrawProgramPrep& prep);
	// Whether nothing the certificate covers changed since `prep` was prepared.
	[[nodiscard]] static bool DrawPrepCertificateHolds(const DrawProgramPrep& prep);
	// Null skips the dispatch.
	const ShaderModule* GetComputeProgram(const HW::ComputeShaderInfo& regs,
	                                      ShaderComputeInputInfo&      input_info);

	Pipeline& GetGraphicsPipeline(const CommandBuffer& command, const DrawRenderState& state,
	                              const DrawTargets& targets);
	Pipeline& GetComputePipeline(const ShaderComputeInputInfo& input_info);

private:
	friend struct PipelineCacheTestAccess;

	struct Program {
		std::unique_ptr<ShaderRecompiler::ShaderSource> call_source;
		ShaderRecompiler::IR::ResourcePlan           resource_plan;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		std::vector<ShaderModule>                    modules;
		bool                                         skip_dispatch = false;
		// SRT materialization memo and the last module a search found (AbSwitch SRT_REFRESH_MEMO).
		ShaderRecompiler::IR::MaterializationMemo memo;
		uint32_t                                 last_module      = UINT32_MAX;
		uint64_t                                 last_push_cursor = UINT64_MAX;

		[[nodiscard]] std::span<const uint32_t> FunctionCode() const;
		bool RefreshCallSource(const ShaderRecompiler::IR::SrtRuntime& runtime);
		template <bool HasCalls>
		ShaderModule* FindModule(const ShaderRecompiler::IR::SrtRuntime& runtime,
		                         ShaderRecompiler::IR::PushData::Window  push_data);
	};

	struct ProgramKey {
		ShaderType            stage           = ShaderType::Unknown;
		uint64_t              hash            = 0;
		uint32_t              user_data_count = 0;
		std::vector<uint32_t> static_state;

		bool operator==(const ProgramKey&) const = default;
	};

	// Variants of one source share a bucket; ProgramKey equality compares the state.
	struct ProgramKeyHash {
		std::size_t operator()(const ProgramKey& key) const { return key.hash; }
	};

	struct GraphicsPipelineKeyHash {
		std::size_t operator()(const GraphicsPipelineKey& key) const;
	};

	// The last pipeline this cache returned and the key it was found or built with, plus what that
	// key was built from besides the registers and the targets (AbSwitch DRAW_CHAIN_PIPELINE;
	// Bruno Cardenas Torres 9153daf18). A draw that took the previous draw's targets with the same
	// target-state serial and these inputs has the same key, and takes the pipeline without
	// zeroing, filling and hashing one.
	struct LastGraphicsInputs {
		uint64_t               state_serial      = 0;
		vk::PrimitiveTopology  topology {};
		bool                   primitive_restart = false;
		bool                   pixel_active      = false;
		bool                   mesh             = false;
		uint32_t               patch_points     = 0;
		ShaderAlphaBlendSource alpha_source     = ShaderAlphaBlendSource::None;
		bool                   sample_shading   = false;
		bool                   valid            = false;
	};
	GraphicsPipelineKey m_last_graphics_key {};
	Pipeline*           m_last_graphics_pipeline = nullptr;
	LastGraphicsInputs  m_last_graphics_inputs;

	using ModuleMap = std::unordered_map<uint64_t, const ShaderModule*>;

	GraphicContext&                        m_graphics;
	ShaderRecompiler::IR::ImageHeapSource* m_image_heap;
	std::unordered_map<ProgramKey, Program, ProgramKeyHash> m_program_cache;
	// Modules compiled per stage.
	std::array<uint32_t, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1> m_compiled {};
	ProgramKey                             m_lookup_key;
	// KYTY_DRAW_PREP: the lookup key of the pure path, apart from m_lookup_key (one per worker in M3).
	ProgramKey                             m_prep_key;
	// KYTY_PROGRAM_HINT=1: the program each stage found last. Elements of m_program_cache keep their
	// address across rehashes; only WarmUp erases (before any lookup), and an erase must clear these.
	using ProgramEntry = std::pair<const ProgramKey, Program>;
	std::array<ProgramEntry*, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1>
	         m_program_hints {};
	uint64_t m_program_hint_lookups = 0; // KYTY_VERIFY_PROGRAM_HINT statistics
	uint64_t m_program_hint_hits    = 0;
	// KYTY_COMPUTE_HINT=1: the program and the pipeline each compute shader found last, direct-mapped
	// by the shader hash and the module id. The per-stage hint above holds a single compute program,
	// and a frame cycles through ~60 compute shaders. A slot is used only when its stored key equals
	// the lookup key, so a collision is a miss. Same lifetime rules as m_program_hints; elements of
	// m_compute_pipelines are never erased before the destructor.
	struct ComputePipelineHint {
		uint64_t  module_id = 0;
		Pipeline* pipeline  = nullptr;
	};
	static constexpr size_t COMPUTE_HINT_SLOTS = 1024; // ~66 shaders per frame: ~2 colliding pairs
	std::array<ProgramEntry*, COMPUTE_HINT_SLOTS>       m_compute_program_hints {};
	std::array<ComputePipelineHint, COMPUTE_HINT_SLOTS> m_compute_pipeline_hints {};
	uint64_t m_compute_hint_lookups = 0; // KYTY_VERIFY_COMPUTE_HINT statistics
	uint64_t m_compute_hint_hits    = 0;
	vk::PipelineCache             m_driver_cache = nullptr;
	std::filesystem::path         m_driver_cache_path;
	std::chrono::steady_clock::time_point m_next_driver_cache_save;
	bool                          m_driver_cache_dirty = false;
	std::unordered_map<GraphicsPipelineKey, Pipeline, GraphicsPipelineKeyHash> m_graphics_pipelines;
	std::unordered_map<uint64_t, Pipeline>                                     m_compute_pipelines;
	std::unordered_map<uint64_t, ShaderTessellationInputInfo>                  m_tess_layouts;
	// Records everything first created if a shader store is open, and writes the driver cache.
	ShaderStoreWriter m_store;

	template <typename InputInfo>
	const ShaderModule* GetProgram(const ShaderParams& params, InputInfo& input_info,
	                               ShaderRecompiler::IR::PushData::Window& push_data);
	// The register part of GetGraphicsPrograms, shared by the serial path (Pure = false, which
	// always returns true) and PrepareGraphicsPrograms (Pure = true, `prep` non-null).
	template <bool Pure>
	bool GraphicsProgramsImpl(const HW::Context& context, const HW::UserConfig& user_config,
	                          const HW::Shader& shaders, DrawRenderState& state,
	                          DrawProgramPrep* prep);
	// GetProgram without side effects on the cache: an existing program and module only, the
	// resources materialized into `resources`. Null with `refusal` set otherwise.
	template <typename InputInfo>
	const ShaderModule* GetProgramPure(const ShaderParams& params, InputInfo& input_info,
	                                   ShaderRecompiler::IR::PushData::Window& push_data,
	                                   ShaderRecompiler::IR::ResourceSnapshot&       resources,
	                                   ShaderRecompiler::IR::ResourceSpecialization& specialization,
	                                   DrawPrepRefusal&                              refusal);
	ShaderModule& CompileModule(Program& program, const ProgramKey& key,
	                            ShaderRecompiler::CompileOptions       options,
	                            ShaderRecompiler::TranslateResult      translated,
	                            ShaderRecompiler::IR::PushData::Window push_data);
	void InitializeDriverCache();
	// Marks the driver cache changed and, once per interval, has the store thread write it.
	void CheckpointDriverCache();
	void WriteDriverCache() const;

	// pipelineSerialization.cpp
	void WarmUp(const std::filesystem::path& path);
	bool LoadRecord(std::span<const uint8_t>& records);
	void StoreModule(const ProgramKey& key, const Program& program, const ShaderModule& module,
	                 const std::vector<uint32_t>& spirv);
	void StorePipeline(const GraphicsPipelineKey& key);
	void StoreLayout(uint64_t key, const ShaderTessellationInputInfo& layout);
	// Every module by id, with id 0 naming an absent stage.
	[[nodiscard]] ModuleMap MapModules() const;
	// False when a stage id of the key is not in `modules`.
	bool CreateGraphicsPipeline(Pipeline& pipeline, const GraphicsPipelineKey& key,
	                            const ModuleMap& modules);
};

// KYTY_VERIFY_DRAW_PREP: the first field in which two program preparations of one draw differ
// (modules, input infos, materialized resources, mrt mask), or null when they are equal.
[[nodiscard]] const char* DiffGraphicsPrograms(const DrawRenderState& left,
                                               const DrawRenderState& right);

// The shader store of the running title; empty unless --shader-cache is on in a Release build
// without debug dumps, which change the SPIR-V.
std::filesystem::path ShaderStorePath();
// Host capabilities that change the emitted SPIR-V.
ShaderRecompiler::Spirv::Profile SpirvProfile(const GraphicContext& graphics);

// Stages are indexed like GraphicsPipelineKey::stage_ids.
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const GraphicsPipelineKey&                 key,
                            const std::array<const ShaderModule*, 4>& stages,
                            vk::PipelineCache                          driver_cache);
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const ShaderModule& module, vk::PipelineCache driver_cache);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
