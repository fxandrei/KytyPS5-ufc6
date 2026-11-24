#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include "common/abSwitch.h"
#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/pipeline/blendMapping.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/Tessellation.h"
#include "graphics/shader/recompiler/ir/Serialize.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/memory.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <limits>
#include <ranges>
#include <span>
#include <spirv-tools/libspirv.hpp>
#include <utility>
#include <unordered_set>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {
// Both ends determine a stage's push layout; a memo may not alias different windows.
constexpr uint64_t PushWindowKey(ShaderRecompiler::IR::PushData::Window window) {
    return uint64_t{window.begin} | (uint64_t{window.end} << 32u);
}


constexpr auto DriverCacheSaveInterval = std::chrono::minutes {1};

// Set and not "0". Read once at startup, so hot paths test no static guard.
bool EnvFlag(const char* name) {
	const char* value = std::getenv(name);
	return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0;
}
// KYTY_PROGRAM_HINT=1: GetProgram first tries the program its stage found last (same key, no hash
// lookup). KYTY_VERIFY_PROGRAM_HINT=1 implies it, looks up anyway and exits on a different entry.
const bool g_verify_program_hint = EnvFlag("KYTY_VERIFY_PROGRAM_HINT");
// Default on since 2026-11-21 (verified, A/B with bits 43/44): KYTY_PROGRAM_HINT=0 turns it off.
const bool g_program_hint        = g_verify_program_hint || std::getenv("KYTY_PROGRAM_HINT") == nullptr ||
                                   EnvFlag("KYTY_PROGRAM_HINT");
// KYTY_COMPUTE_HINT=1: compute program and pipeline lookups first try the slot of their shader hash
// / module id (PipelineCache::m_compute_program_hints). KYTY_VERIFY_COMPUTE_HINT=1 implies it, also
// does the hash lookups and exits when they find another entry.
const bool g_verify_compute_hint = EnvFlag("KYTY_VERIFY_COMPUTE_HINT");
// Default on since 2026-11-21 (verified: 6.8M lookups, 0 mismatches): KYTY_COMPUTE_HINT=0 turns it off.
const bool g_compute_hint        = g_verify_compute_hint || std::getenv("KYTY_COMPUTE_HINT") == nullptr ||
                                   EnvFlag("KYTY_COMPUTE_HINT");

constexpr size_t ComputeHintSlot(uint64_t hash, size_t slots) {
	return static_cast<size_t>((hash ^ (hash >> 29u) ^ (hash >> 47u)) & (slots - 1u));
}

uint8_t RemapSourceAlphaFactor(uint8_t factor) {
	switch (static_cast<Prospero::BlendFactor>(factor)) {
		case Prospero::BlendFactor::kSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Alpha);
		case Prospero::BlendFactor::kOneMinusSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
		default: return factor;
	}
}

vk::PolygonMode ResolvePolygonMode(const HW::ModeControl& mode, bool cull_front, bool cull_back) {
	// PA_SU_SC_MODE_CNTL.POLY_MODE disables both per-face modes when it is zero.
	if (mode.poly_mode == 0) {
		return vk::PolygonMode::eFill;
	}
	EXIT_NOT_IMPLEMENTED(mode.poly_mode != 1);
	if (cull_front && cull_back) {
		return vk::PolygonMode::eFill;
	}
	if (!cull_front && !cull_back && mode.polymode_front_ptype != mode.polymode_back_ptype) {
		EXIT("Pipeline: different polygon modes for two visible faces are unsupported\n");
	}
	// Vulkan has one polygon mode. A culled face does not constrain that mode.
	const auto polygon_mode = cull_front ? mode.polymode_back_ptype : mode.polymode_front_ptype;
	switch (polygon_mode) {
		case 0: return vk::PolygonMode::ePoint;
		case 1: return vk::PolygonMode::eLine;
		case 2: return vk::PolygonMode::eFill;
		default: EXIT("Pipeline: invalid polygon mode %u\n", polygon_mode);
	}
}

std::string DriverCacheSignature(const vk::PhysicalDeviceProperties& properties) {
	constexpr char hex[] = "0123456789abcdef";
	std::string    uuid(VK_UUID_SIZE * 2, '0');
	for (size_t i = 0; i < VK_UUID_SIZE; i++) {
		uuid[i * 2]     = hex[properties.pipelineCacheUUID[i] >> 4u];
		uuid[i * 2 + 1] = hex[properties.pipelineCacheUUID[i] & 0xfu];
	}
	// Driver cache entries already distinguish shader code and pipeline state across builds.
	return fmt::format("KytyPC2:{:08x}:{:08x}:{:08x}:{}\n", properties.vendorID,
	                   properties.deviceID, properties.driverVersion, uuid);
}

std::string PipelineCacheTitleId() {
	std::string title_id;
	if ((!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) &&
	    (!Loader::SystemContentParamSfoGetString("CONTENT_ID", &title_id) || title_id.empty())) {
		return {};
	}
	if (!std::ranges::all_of(title_id, [](unsigned char c) {
		    return std::isalnum(c) != 0 || c == '-' || c == '_';
	    })) {
		return {};
	}
	return title_id;
}

template <typename... Args>
void PipelineCacheLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}

bool ReadShaderGuestMemory(void*, uint64_t address, std::span<uint32_t> values) {
	// Scalar and unformatted buffer dependencies use the same backing as native raw loads.
	// Image synchronization belongs to formatted buffer bindings, not these reads.
	return !values.empty() && Libs::LibKernel::Memory::TryReadBufferBacking(address, values.data(),
	                                                                        values.size_bytes());
}

void DumpShader(std::span<const uint32_t> words, const std::string& name) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	const auto path = Config::GetShaderLogFolder() / name;
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(words.data(), words.size_bytes());
}

void ValidateShaderSpirv(const char* stage, uint64_t shader_hash,
                         const std::vector<uint32_t>& spirv, bool maintenance8_enabled) {
	if (!Config::ShaderValidationEnabled()) {
		return;
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*,
	                                     const spv_position_t& position, const char* message) {
		messages += fmt::format("{}: {} ({}) {}\n", static_cast<int>(position.line),
		                        static_cast<int>(position.column), static_cast<int>(position.index),
		                        message);
	});
	spvtools::ValidatorOptions validator_options;
	validator_options.SetAllowOffsetTextureOperand(maintenance8_enabled);
	if (tools.Validate(spirv.data(), spirv.size(), validator_options)) {
		return;
	}
	spvtools::SpirvTools disassembler(SPV_ENV_VULKAN_1_2);
	std::string          text;
	disassembler.Disassemble(spirv, &text,
	                         static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COMMENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COLOR));
	LOGF("%s\n", text.c_str());
	EXIT("ShaderRecompiler %s SPIR-V validation failed hash=0x%016" PRIx64 ":\n%s", stage,
	     shader_hash, messages.c_str());
}

} // namespace

ShaderRecompiler::Spirv::Profile SpirvProfile(const GraphicContext& graphics) {
	// KYTY_GDS_FIX=1 (2026-11-21, A/B experiment, default off): DS_INC/DS_DEC with an all-ones
	// limit use one native atomic instead of a load/compare-exchange loop. UFC 6's strand-hair
	// rasterizer (b37644325c6f10c6) takes one GDS ticket per tile from 144 persistent groups; the
	// loop retires about one ticket per L2 round trip under that contention. The store header holds
	// the profile, so changing the variable rebuilds the shader store.
	static const bool inc_dec_add = [] {
		const char* value = std::getenv("KYTY_GDS_FIX");
		return value == nullptr || std::atoi(value) != 0; // default ON since 2026-11-21 23:55 (exact for all-ones limits)
	}();
	// AMD GPUs (vendor 0x1002) run NMin/NMax as v_min/v_max, which give the guest's NaN and
	// signed-zero results; GLSL.std.450 alone does not order -0 below +0.
	return {.maintenance8           = graphics.maintenance8_enabled,
	        .native_float_min_max   = graphics.physical_device_properties.vendorID == 0x1002u,
	        .device_clock_period_ps = graphics.device_clock_period_ps,
	        .robust_buffer_access   = graphics.robust_buffer_access_enabled,
	        .r8_srgb_views          = graphics.r8_srgb_views,
	        .r8g8_srgb_views        = graphics.r8g8_srgb_views,
	        .relaxed_float          = Config::RelaxedShaderFloatEnabled(),
	        .inc_dec_add            = inc_dec_add,
	        .ray_query              = graphics.ray_query_enabled};
}

std::filesystem::path ShaderStorePath() {
	if (KYTY_BUILD != KYTY_BUILD_RELEASE || !Config::ShaderCacheEnabled() ||
	    Config::GraphicsDebugDumpEnabled()) {
		return {};
	}
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return {};
	}
	return std::filesystem::path("_PipelineCache") / (title_id + ".shaders");
}

bool GraphicsPipelineKey::operator==(const GraphicsPipelineKey& other) const {
	return std::memcmp(this, &other, sizeof(*this)) == 0;
}

std::size_t
PipelineCache::GraphicsPipelineKeyHash::operator()(const GraphicsPipelineKey& key) const {
	return XXH3_64bits(&key, sizeof(key));
}

std::span<const uint32_t> PipelineCache::Program::FunctionCode() const {
	return call_source ? std::span<const uint32_t>(call_source->linked->code)
	                         .subspan(call_source->code.size())
	                   : std::span<const uint32_t> {};
}

bool PipelineCache::Program::RefreshCallSource(const ShaderRecompiler::IR::SrtRuntime& runtime) {
	const auto revision = call_source->revision;
	ShaderRecompiler::RefreshShaderSource(*call_source, runtime);
	resource_plan.source_reads = call_source->reads;
	return revision != call_source->revision;
}

template <bool HasCalls>
ShaderModule* PipelineCache::Program::FindModule(const ShaderRecompiler::IR::SrtRuntime& runtime,
                                                 ShaderRecompiler::IR::PushData::Window push_data) {
	// Call programs never memoize: their resource plan is refreshed per draw.
	const bool memo_enabled =
	    !HasCalls && Common::AbSwitch::Enabled(Common::AbSwitch::SRT_REFRESH_MEMO);
	if (!memo_enabled) {
		memo.valid  = false;
		memo.reused = false;
	}
	EXIT_IF(!ShaderRecompiler::IR::MaterializeResources(
	    resource_plan, runtime, resources, specialization, memo_enabled ? &memo : nullptr));
	// A whole-plan memo hit with the same push cursor reuses the last module without a search.
	if (memo.reused && last_module < modules.size() && PushWindowKey(push_data) == last_push_cursor) {
		return &modules[last_module];
	}
	const auto matches = [&](const ShaderModule& candidate) {
		const auto& layout = candidate.info.bindings;
		return layout.push_data_start_dword == push_data.StartFor(layout.ShaderDataDwords(), candidate.info.stage) &&
		       candidate.specialization == specialization &&
		       (!HasCalls || std::ranges::equal(candidate.function_code, FunctionCode()));
	};
	// Modules are append-only, so the last match is tried first on the next refresh.
	if (memo_enabled && last_module < modules.size() && matches(modules[last_module])) {
		last_push_cursor = PushWindowKey(push_data);
		return &modules[last_module];
	}
	for (auto& candidate: modules) {
		if (!matches(candidate)) {
			continue;
		}
		last_module      = static_cast<uint32_t>(&candidate - modules.data());
		last_push_cursor = PushWindowKey(push_data);
		return &candidate;
	}
	return nullptr;
}

ShaderModule& PipelineCache::CompileModule(Program& program, const ProgramKey& key,
                                           ShaderRecompiler::CompileOptions       options,
                                           ShaderRecompiler::TranslateResult      translated,
                                           ShaderRecompiler::IR::PushData::Window push_data) {
	Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::ShadersCompiled,
	                                         Common::DebugCounters::Counter::ShaderCompileNs,
	                                         Common::DebugCounters::Peak::ShaderCompileNs);
	const auto                         profile = SpirvProfile(m_graphics);
	options.profile                            = &profile;
	auto result = ShaderRecompiler::CompileProgram(std::move(translated), options,
	                                               program.specialization, push_data);
	auto info = std::move(result.program).TakeCompiledInfo();

	// The same inputs give the same module id across runs.
	const auto                   function_code = program.FunctionCode();
	ShaderRecompiler::IR::Writer inputs;
	ShaderRecompiler::IR::IoAll(inputs, key.stage, key.hash, key.user_data_count, key.static_state,
	                            info.bindings.push_data_start_dword, program.specialization);
	inputs.Raw(function_code.data(), function_code.size_bytes()); // Last, so it needs no size.
	const auto id = XXH3_64bits(inputs.bytes.data(), inputs.bytes.size());

	const auto* stage = ShaderRecompiler::StageName(key.stage);
	DumpShader(result.spirv, fmt::format("{}_{:016x}_{:016x}.spv", stage, key.hash, id));
	ValidateShaderSpirv(stage, key.hash, result.spirv, profile.maintenance8);
	auto& module = program.modules.emplace_back(ShaderModule {
	    .id             = id,
	    .module         = CompileSPV(result.spirv, m_graphics.device),
	    .info           = std::move(info),
	    .specialization = program.specialization,
	    .function_code  = {function_code.begin(), function_code.end()},
	});
	EXIT_IF(module.module == nullptr);
	program.last_module      = static_cast<uint32_t>(program.modules.size() - 1u);
	program.last_push_cursor = PushWindowKey(push_data);
	m_compiled[static_cast<size_t>(key.stage)]++;
	// Guest geometry shaders are compiled through the host mesh stage.
	std::printf("Shaders: VS %u | PS %u | CS %u | GS %u | LS %u | HS %u | TES %u\n",
	            m_compiled[static_cast<size_t>(ShaderType::Vertex)],
	            m_compiled[static_cast<size_t>(ShaderType::Pixel)],
	            m_compiled[static_cast<size_t>(ShaderType::Compute)],
	            m_compiled[static_cast<size_t>(ShaderType::Mesh)],
	            m_compiled[static_cast<size_t>(ShaderType::Local)],
	            m_compiled[static_cast<size_t>(ShaderType::TessellationControl)],
	            m_compiled[static_cast<size_t>(ShaderType::TessellationEvaluation)]);
	// The inputs of these come from the stage's static key.
	if (key.stage == ShaderType::Pixel &&
	    options.input_info.pixel->parameter_mode == ShaderPixelParameterMode::Rectangle) {
		module.rect_parameters = RectListParameters(*options.input_info.pixel, module.info);
	} else if (key.stage == ShaderType::Mesh) {
		module.mesh_task = options.input_info.vertex->indirect_draw;
	}
	// Programs with GPU-selected calls depend on their callees' current code.
	if (program.call_source == nullptr) {
		StoreModule(key, program, module, result.spirv);
	}
	return module;
}

template <typename InputInfo>
const ShaderModule* PipelineCache::GetProgram(const ShaderParams& params, InputInfo& input_info,
                                              ShaderRecompiler::IR::PushData::Window& push_data) {
	const auto user_data = std::span(params.user_data).first(params.user_data_count);
	ShaderRecompiler::IR::SrtRuntime runtime {
	    .user_data                  = user_data,
	    .shader_base                = params.Base(),
	    .read_specialization_memory = ReadShaderGuestMemory,
	    .image_heap                 = m_image_heap,
	};
	ShaderRecompiler::CompileOptions options {.wave_size = input_info.wave_size};
	if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
		options.stage             = input_info.logical_stage;
		options.input_info.vertex = &input_info;
		options.user_data_base    = 8;
		if (options.stage == ShaderType::Mesh || options.stage == ShaderType::TessellationControl) {
			options.user_data_base = 0;
		}
		if (options.stage == ShaderType::Mesh) {
			options.wave_size = input_info.mesh.wave_size;
		}
	} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
		options.stage            = ShaderType::Pixel;
		options.input_info.pixel = &input_info;
	} else {
		static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
		options.stage                 = ShaderType::Compute;
		options.input_info.compute    = &input_info;
		runtime.workgroup_counts      = input_info.workgroup_counts;
		runtime.indirect_args_address = input_info.indirect_args_address;
	}
	m_lookup_key.stage           = options.stage;
	m_lookup_key.hash            = params.hash;
	m_lookup_key.user_data_count = params.user_data_count;
	BuildStageStaticKey(input_info, m_lookup_key.static_state);
	// The stage's last program when the key is the same (KYTY_PROGRAM_HINT): the hash lookup compares
	// the key with every cached variant of the shader, which all share its bucket.
	const auto    stage_index = static_cast<size_t>(options.stage);
	const bool    use_hint    = g_program_hint && stage_index < m_program_hints.size();
	ProgramEntry* entry       = nullptr;
	// KYTY_COMPUTE_HINT: this compute shader's slot; set to what the lookup below finds.
	ProgramEntry** compute_slot = nullptr;
	if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
		if (g_compute_hint) {
			compute_slot = &m_compute_program_hints[ComputeHintSlot(params.hash, COMPUTE_HINT_SLOTS)];
		}
	}
	if (compute_slot != nullptr && *compute_slot != nullptr && (*compute_slot)->first == m_lookup_key) {
		// Keys are unique in m_program_cache: the entry with an equal key is the one find returns.
		entry = *compute_slot;
		Common::DebugCounters::Add(Common::DebugCounters::Counter::ComputeHintHits);
		if (use_hint) {
			// The per-stage hint stays "the program this stage found last", as without the slots.
			if (m_program_hints[stage_index] == entry) {
				Common::DebugCounters::Add(Common::DebugCounters::Counter::ComputeHintStageHits);
			}
			m_program_hints[stage_index] = entry;
		}
		if (g_verify_compute_hint) [[unlikely]] {
			const auto found = m_program_cache.find(m_lookup_key);
			if (found == m_program_cache.end() || &*found != entry) {
				EXIT("compute hint verify: hash 0x%016" PRIx64 " found another program\n",
				     params.hash);
			}
			m_compute_hint_hits++;
		}
	} else if (use_hint && m_program_hints[stage_index] != nullptr &&
	    m_program_hints[stage_index]->first == m_lookup_key) {
		entry = m_program_hints[stage_index];
		if (g_verify_program_hint) [[unlikely]] {
			const auto found = m_program_cache.find(m_lookup_key);
			if (found == m_program_cache.end() || &*found != entry) {
				EXIT("program hint verify: stage %u hash 0x%016" PRIx64 " found another program\n",
				     static_cast<uint32_t>(stage_index), params.hash);
			}
			m_program_hint_hits++;
		}
	} else {
		const auto found = m_program_cache.find(m_lookup_key);
		entry            = found == m_program_cache.end() ? nullptr : &*found;
		if (use_hint) {
			m_program_hints[stage_index] = entry;
		}
	}
	if (compute_slot != nullptr && *compute_slot != entry) {
		Common::DebugCounters::Add(Common::DebugCounters::Counter::ComputeHintMisses);
		*compute_slot = entry; // nullptr for a new program: set again after the emplace below
	}
	if (compute_slot != nullptr && g_verify_compute_hint &&
	    (++m_compute_hint_lookups & 0x3ffffu) == 0) [[unlikely]] {
		std::printf("compute hint verify: %" PRIu64 " lookups, %" PRIu64 " hint hits, 0 mismatches\n",
		            m_compute_hint_lookups, m_compute_hint_hits);
		std::fflush(stdout);
	}
	if (g_verify_program_hint && (++m_program_hint_lookups & 0xfffffu) == 0) [[unlikely]] {
		std::printf("program hint verify: %" PRIu64 " lookups, %" PRIu64 " hint hits, 0 mismatches\n",
		            m_program_hint_lookups, m_program_hint_hits);
		std::fflush(stdout);
	}
	const auto bind = [&](const ShaderModule& module) {
        // Diagnostic only: compare a retained shader against the same live inputs compiled
        // without the store. This deliberately does extra work and is not a benchmark mode.
        static const bool verify_store = std::getenv("KYTY_VERIFY_SHADER_STORE") != nullptr;
        static std::unordered_set<uint64_t> verified;
        if (verify_store && entry->second.call_source == nullptr && verified.insert(module.id).second) {
            auto verify_options = options;
            auto verify_input = input_info;
            if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
                verify_options.input_info.vertex = &verify_input;
            } else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
                verify_options.input_info.pixel = &verify_input;
            } else {
                verify_options.input_info.compute = &verify_input;
            }
            verify_options.shader_hash = params.hash;
            verify_options.user_data = user_data;
            verify_options.back_code = params.back_code;
            auto translated = ShaderRecompiler::TranslateProgram(params.code, verify_options);
            auto plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
            const auto plan_equal = ShaderRecompiler::IR::SaveResourcePlan(plan) ==
                ShaderRecompiler::IR::SaveResourcePlan(entry->second.resource_plan);
            ShaderRecompiler::IR::ResourceSnapshot resources;
            ShaderRecompiler::IR::ResourceSpecialization specialization;
            EXIT_IF(!ShaderRecompiler::IR::MaterializeResources(plan, runtime, resources, specialization));
            const auto& cached = entry->second.resources;
            const bool resources_equal = resources.buffers == cached.buffers &&
                resources.images == cached.images && resources.samplers == cached.samplers &&
                resources.flattened_srt == cached.flattened_srt && resources.user_data == cached.user_data &&
                resources.specialization_reads == cached.specialization_reads && resources.uniform_fill == cached.uniform_fill;
            const auto profile = SpirvProfile(m_graphics);
            verify_options.profile = &profile;
            auto result = ShaderRecompiler::CompileProgram(std::move(translated), verify_options,
                                                           specialization, push_data);
            const auto info = std::move(result.program).TakeCompiledInfo();
            ShaderRecompiler::IR::Writer fresh_info, cached_info;
            ShaderRecompiler::IR::IoAll(fresh_info, info);
            ShaderRecompiler::IR::IoAll(cached_info, module.info);
            bool auxiliary_equal = true;
            if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
                if (options.stage == ShaderType::Vertex || options.stage == ShaderType::Local) {
                    auxiliary_equal = info.info.vertex_fetch_components == module.info.info.vertex_fetch_components;
                } else if (options.stage == ShaderType::Mesh) {
                    auxiliary_equal = verify_input.indirect_draw == module.mesh_task;
                }
            } else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
                if (verify_input.parameter_mode == ShaderPixelParameterMode::Rectangle) {
                    ShaderRecompiler::IR::Writer fresh_rect, cached_rect;
                    ShaderRecompiler::IR::IoAll(fresh_rect, RectListParameters(verify_input, info));
                    ShaderRecompiler::IR::IoAll(cached_rect, module.rect_parameters);
                    auxiliary_equal = fresh_rect.bytes == cached_rect.bytes;
                }
            }
            PipelineCacheLog("Shader store verify: stage={} hash={:016x} id={:016x} plan={} resources={} specialization={} info={} auxiliary={} spirv={:016x}",
                ShaderRecompiler::StageName(options.stage), params.hash, module.id, plan_equal,
                resources_equal, specialization == module.specialization, fresh_info.bytes == cached_info.bytes,
                auxiliary_equal, XXH3_64bits(result.spirv.data(), result.spirv.size() * sizeof(uint32_t)));
        }
        input_info.stage = {.module = &module, .resources = &entry->second.resources};
        module.info.bindings.AdvancePushData(push_data, module.info.stage);
        return &module;
	};
	bool source_changed = false;
	if (entry != nullptr) {
		auto& cached = entry->second;
		if (cached.skip_dispatch) return nullptr;
		ShaderModule* module = nullptr;
		if (cached.call_source) {
			source_changed = cached.RefreshCallSource(runtime);
			if (!source_changed) module = cached.FindModule<true>(runtime, push_data);
		} else {
			module = cached.FindModule<false>(runtime, push_data);
		}
		if (module != nullptr) return bind(*module);
	}

	options.shader_hash     = params.hash;
	options.user_data       = user_data;
	options.back_code       = params.back_code;
	options.dump_ir         = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
	options.emit_debug_info = Config::GraphicsDebugDumpEnabled();
	ShaderRecompiler::ShaderSource source;
	if (entry == nullptr) {
		DumpShader(params.code,
		           fmt::format("original/{}_{:016x}.bin",
		                       ShaderRecompiler::StageName(options.stage), params.hash));
		source = ShaderRecompiler::PrepareShaderSource(params.code, options);
		if (source.unsupported_call_pc != UINT32_MAX) {
			PipelineCacheLog(
			    "!!! TEMPORARY COMPUTE DISPATCH SKIP !!! shader=0x{:016x} pc=0x{:08x} "
			    "reason=unsupported GPU-selected callable target; rendering is incomplete",
			    params.hash, source.unsupported_call_pc);
			m_program_cache.emplace(m_lookup_key, Program {.skip_dispatch = true});
			return nullptr;
		}
	}
	auto* current = entry == nullptr ? &source : entry->second.call_source.get();
	ShaderRecompiler::TranslateResult translated;
	if (current != nullptr) {
		if (!current->calls.empty()) {
			if (entry == nullptr)
				ShaderRecompiler::RefreshShaderSource(*current, runtime);
			translated = ShaderRecompiler::TranslateProgram(current->linked->decoded, options);
			translated.program.source_reads = current->reads;
		} else {
			translated = ShaderRecompiler::TranslateProgram(current->decoded, options);
		}
	} else {
		translated = ShaderRecompiler::TranslateProgram(params.code, options);
	}
	if (entry == nullptr) {
		auto retained = !source.calls.empty()
		                    ? std::make_unique<ShaderRecompiler::ShaderSource>(std::move(source))
		                    : nullptr;
		entry         = &*m_program_cache
		            .emplace(m_lookup_key,
		                     Program {
		                         .call_source = std::move(retained),
		                         .resource_plan =
		                             ShaderRecompiler::IR::ExtractResourcePlan(translated.program),
		                     })
		            .first;
		if (use_hint) {
			m_program_hints[stage_index] = entry;
		}
		if (compute_slot != nullptr) {
			*compute_slot = entry;
		}
		auto& program = entry->second;
		EXIT_IF(!ShaderRecompiler::IR::MaterializeResources(
		    program.resource_plan, runtime, program.resources, program.specialization,
		    program.call_source == nullptr &&
		            Common::AbSwitch::Enabled(Common::AbSwitch::SRT_REFRESH_MEMO)
		        ? &program.memo
		        : nullptr));
	} else if (source_changed) {
		entry->second.resource_plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
		// The plan changed: any memo key and last-module hint are for the old plan.
		entry->second.memo             = {};
		entry->second.last_module      = UINT32_MAX;
		entry->second.last_push_cursor = UINT64_MAX;
		if (const auto* module = entry->second.FindModule<true>(runtime, push_data)) {
			return bind(*module);
		}
	}
	return bind(CompileModule(entry->second, entry->first, options, std::move(translated),
	                          push_data));
}

// KYTY_DRAW_PREP (M1): GetProgram for an existing program and module, with no write to the
// cache. Differences to GetProgram, each with the reason the result is still the same:
//  - The key goes to m_prep_key; the per-stage hint is read but never written, and a miss uses
//    find. Keys are unique in m_program_cache, so the hint and find give the same entry.
//  - No SRT memo: the memo only skips recomputing what an equal walk gives (KYTY_VERIFY_SRT_MEMO);
//    Program::resources/specialization/memo/last_module stay as the serial path left them, so the
//    memo's "Program::resources matches memo.key" invariant still holds for its next use.
//  - The module search reads last_module as a first guess and scans otherwise; at most one module
//    matches a (push start, specialization) pair, since CompileModule only runs without a match.
//  - The image heap is the same object: Acquire is idempotent within a draw (one refresh per draw,
//    views found again, m_active deduplicated), and the pure path runs after BeginDraw like the
//    serial one. A worker (M3) must refuse plans that can reach Acquire instead.
//  - Never translates, compiles or emplaces: those cases are refusals.
template <typename InputInfo>
const ShaderModule* PipelineCache::GetProgramPure(
    const ShaderParams& params, InputInfo& input_info, ShaderRecompiler::IR::PushData::Window& push_data,
    ShaderRecompiler::IR::ResourceSnapshot&       resources,
    ShaderRecompiler::IR::ResourceSpecialization& specialization, DrawPrepRefusal& refusal) {
	static_assert(std::is_same_v<InputInfo, ShaderVertexInputInfo> ||
	              std::is_same_v<InputInfo, ShaderPixelInputInfo>);
	const auto user_data = std::span(params.user_data).first(params.user_data_count);
	ShaderType stage     = ShaderType::Pixel;
	if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
		stage = input_info.logical_stage;
	}
	m_prep_key.stage           = stage;
	m_prep_key.hash            = params.hash;
	m_prep_key.user_data_count = params.user_data_count;
	BuildStageStaticKey(input_info, m_prep_key.static_state);
	const auto          stage_index = static_cast<size_t>(stage);
	const ProgramEntry* entry       = nullptr;
	if (stage_index < m_program_hints.size() && m_program_hints[stage_index] != nullptr &&
	    m_program_hints[stage_index]->first == m_prep_key) {
		entry = m_program_hints[stage_index];
	} else if (const auto found = m_program_cache.find(m_prep_key); found != m_program_cache.end()) {
		entry = &*found;
	}
	if (entry == nullptr) {
		refusal = DrawPrepRefusal::NoProgram;
		return nullptr;
	}
	const auto& program = entry->second;
	if (program.skip_dispatch) {
		refusal = DrawPrepRefusal::SkipProgram;
		return nullptr;
	}
	if (program.call_source != nullptr) {
		refusal = DrawPrepRefusal::CallProgram;
		return nullptr;
	}
	const ShaderRecompiler::IR::SrtRuntime runtime {
	    .user_data                  = user_data,
	    .shader_base                = params.Base(),
	    .read_specialization_memory = ReadShaderGuestMemory,
	    .image_heap                 = m_image_heap,
	};
	if (!ShaderRecompiler::IR::MaterializeResources(program.resource_plan, runtime, resources,
	                                                specialization, nullptr)) {
		refusal = DrawPrepRefusal::Materialize;
		return nullptr;
	}
	const auto matches = [&](const ShaderModule& candidate) {
		const auto& layout = candidate.info.bindings;
		return layout.push_data_start_dword == push_data.StartFor(layout.ShaderDataDwords(), candidate.info.stage) &&
		       candidate.specialization == specialization;
	};
	const ShaderModule* module = nullptr;
	if (program.last_module < program.modules.size() &&
	    matches(program.modules[program.last_module])) {
		module = &program.modules[program.last_module];
	} else {
		for (const auto& candidate: program.modules) {
			if (matches(candidate)) {
				module = &candidate;
				break;
			}
		}
	}
	if (module == nullptr) {
		refusal = DrawPrepRefusal::NoModule;
		return nullptr;
	}
	input_info.stage = {.module = module, .resources = &resources};
	module->info.bindings.AdvancePushData(push_data, module->info.stage);
	return module;
}

PipelineCache::PipelineCache(GraphicContext&                        graphics,
                             ShaderRecompiler::IR::ImageHeapSource* image_heap,
                             const std::filesystem::path&           shader_store)
    : m_graphics(graphics), m_image_heap(image_heap),
      m_next_driver_cache_save(std::chrono::steady_clock::now() + DriverCacheSaveInterval) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	InitializeDriverCache();
	WarmUp(shader_store);
}

PipelineCache::~PipelineCache() {
	Save();
	auto destroy = [this](const auto& pipelines) {
		for (const auto& pipeline: pipelines | std::views::values) {
			m_graphics.device.destroyPipeline(pipeline.pipeline, nullptr);
			m_graphics.device.destroyPipelineLayout(pipeline.pipeline_layout, nullptr);
			m_graphics.device.destroyDescriptorSetLayout(pipeline.descriptor_set_layout, nullptr);
		}
	};
	destroy(m_graphics_pipelines);
	destroy(m_compute_pipelines);
	for (const auto& program: m_program_cache | std::views::values) {
		for (const auto& module: program.modules) {
			m_graphics.device.destroyShaderModule(module.module, nullptr);
		}
	}
	m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
}

void PipelineCache::InitializeDriverCache() {
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return;
	}
	if (KYTY_BUILD != KYTY_BUILD_RELEASE) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (non-Release build)");
		return;
	}
	m_driver_cache_path     = std::filesystem::path("_PipelineCache") / (title_id + ".bin");
	const auto path         = Common::PathToString(m_driver_cache_path);
	const bool cache_exists = Common::File::IsFileExisting(m_driver_cache_path);
	if (cache_exists) {
		PipelineCacheLog("Vulkan pipeline cache: loading {}", path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initializing {}", path);
	}
	std::vector<uint8_t> initial_data;
	if (cache_exists) {
		Common::File file(m_driver_cache_path, Common::File::Mode::Read);
		const auto   file_size = file.IsInvalid() ? 0 : file.Size();
		const auto   signature = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
		if (file_size >= signature.size() + sizeof(uint64_t) &&
		    file_size <= std::numeric_limits<uint32_t>::max()) {
			std::string cached_signature(signature.size(), '\0');
			uint64_t    payload_hash = 0;
			initial_data.resize(file_size - signature.size() - sizeof(payload_hash));
			uint32_t signature_read = 0;
			uint32_t hash_read      = 0;
			uint32_t payload_read   = 0;
			file.Read(cached_signature.data(), static_cast<uint32_t>(cached_signature.size()),
			          &signature_read);
			file.Read(&payload_hash, sizeof(payload_hash), &hash_read);
			file.Read(initial_data.data(), static_cast<uint32_t>(initial_data.size()),
			          &payload_read);
			file.Close();
			if (signature_read != cached_signature.size() || hash_read != sizeof(payload_hash) ||
			    payload_read != initial_data.size() || cached_signature != signature ||
			    XXH3_64bits(initial_data.data(), initial_data.size()) != payload_hash) {
				initial_data.clear();
				PipelineCacheLog("Vulkan pipeline cache: invalidating {} (driver or data mismatch)",
				                 path);
			}
		} else {
			file.Close();
			PipelineCacheLog("Vulkan pipeline cache: invalidating {} (invalid file size)", path);
		}
	}

	vk::PipelineCacheCreateInfo create {};
	create.initialDataSize = initial_data.size();
	create.pInitialData    = initial_data.empty() ? nullptr : initial_data.data();
	auto result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	if (result != vk::Result::eSuccess && !initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: driver rejected {} ({}); starting empty", path,
		                 vk::to_string(result));
		initial_data.clear();
		create.initialDataSize = 0;
		create.pInitialData    = nullptr;
		result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	}
	if (result != vk::Result::eSuccess) {
		PipelineCacheLog("Vulkan pipeline cache: disabled ({})", vk::to_string(result));
		m_driver_cache = nullptr;
		return;
	}
	if (!initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: loaded {:.2f} MB from {}",
		                 static_cast<double>(initial_data.size()) / 1'000'000.0, path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initialized empty");
	}
}

void PipelineCache::CheckpointDriverCache() {
	if (m_driver_cache == nullptr) {
		return;
	}
	m_driver_cache_dirty = true;
	if (const auto now = std::chrono::steady_clock::now(); now >= m_next_driver_cache_save) {
		m_driver_cache_dirty     = false;
		m_next_driver_cache_save = now + DriverCacheSaveInterval;
		// The cache object is internally synchronized, so pipelines may be created meanwhile.
		m_store.Post([this] { WriteDriverCache(); });
	}
}

void PipelineCache::Save() {
	m_store.Finish();
	if (m_driver_cache_dirty) {
		m_driver_cache_dirty = false;
		WriteDriverCache();
	}
}

void PipelineCache::WriteDriverCache() const {
	size_t               size = 0;
	vk::Result           result;
	std::vector<uint8_t> payload;
	for (uint32_t attempt = 0; attempt < 3; attempt++) {
		size   = 0;
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, nullptr);
		if (result != vk::Result::eSuccess || size == 0 ||
		    size > std::numeric_limits<uint32_t>::max()) {
			break;
		}
		payload.resize(size);
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, payload.data());
		if (result != vk::Result::eIncomplete) {
			break;
		}
	}
	if (result != vk::Result::eSuccess || size == 0 ||
	    size > std::numeric_limits<uint32_t>::max()) {
		PipelineCacheLog("Vulkan pipeline cache: save failed ({}, {:.2f} MB)",
		                 vk::to_string(result), static_cast<double>(size) / 1'000'000.0);
		return;
	}
	payload.resize(size);
	auto       prefix       = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	const auto payload_hash = XXH3_64bits(payload.data(), payload.size());
	prefix.append(reinterpret_cast<const char*>(&payload_hash), sizeof(payload_hash));
	if (!Common::File::CreateDirectories(m_driver_cache_path.parent_path())) {
		PipelineCacheLog("Vulkan pipeline cache: failed to create cache directory");
		return;
	}
	auto temp_path = m_driver_cache_path;
	temp_path += ".tmp";
	Common::File file;
	uint32_t     prefix_written  = 0;
	uint32_t     payload_written = 0;
	if (file.Create(temp_path)) {
		file.Write(prefix.data(), static_cast<uint32_t>(prefix.size()), &prefix_written);
		file.Write(payload.data(), static_cast<uint32_t>(payload.size()), &payload_written);
	}
	const bool flushed = !file.IsInvalid() && file.Flush();
	file.Close();
	const bool written =
	    prefix_written == prefix.size() && payload_written == payload.size() && flushed;
	std::error_code error;
	if (written) {
		std::filesystem::rename(temp_path, m_driver_cache_path, error);
	}
	if (!written || error) {
		PipelineCacheLog("Vulkan pipeline cache: failed to write {}",
		                 Common::PathToString(m_driver_cache_path));
		return;
	}
	PipelineCacheLog("Vulkan pipeline cache: saved {:.2f} MB to {}",
	                 static_cast<double>(payload.size()) / 1'000'000.0,
	                 Common::PathToString(m_driver_cache_path));
}

void PipelineCache::GetGraphicsPrograms(const CommandBuffer& command, DrawRenderState& state) {
	GraphicsProgramsImpl<false>(command.GetRegisters(), command.GetUserConfig(),
	                            command.GetShaders(), state, nullptr);
}

bool PipelineCache::PrepareGraphicsPrograms(const HW::Context&    context,
                                            const HW::UserConfig& user_config,
                                            const HW::Shader& shaders, DrawRenderState& state,
                                            DrawProgramPrep& prep) {
	// The certificate is loaded before the first read of guest memory or the shader map.
	prep.refusal               = DrawPrepRefusal::None;
	prep.guest_map_epoch       = Libs::LibKernel::Memory::GuestMapEpoch();
	prep.shader_map_generation = ShaderMapGeneration();
	const auto guarded_before  = ShaderRecompiler::IR::SrtWalkerGuardedLoads();
	const bool prepared = GraphicsProgramsImpl<true>(context, user_config, shaders, state, &prep);
	prep.guarded_loads  = ShaderRecompiler::IR::SrtWalkerGuardedLoads() - guarded_before;
	return prepared;
}

bool PipelineCache::DrawPrepCertificateHolds(const DrawProgramPrep& prep) {
	return prep.guest_map_epoch == Libs::LibKernel::Memory::GuestMapEpoch() &&
	       prep.shader_map_generation == ShaderMapGeneration();
}

// Pure = false is the serial path, unchanged apart from the `if constexpr (Pure)` branches.
template <bool Pure>
bool PipelineCache::GraphicsProgramsImpl(const HW::Context&    context,
                                         const HW::UserConfig& user_config,
                                         const HW::Shader& shaders, DrawRenderState& state,
                                         [[maybe_unused]] DrawProgramPrep* prep) {
	const auto& sh                  = context.GetShaderRegisters();
	const auto& vertex_regs         = shaders.GetVs();
	const auto& pixel_regs          = shaders.GetPs();
	auto&       vertex_info         = state.vertex_info;
	auto&       pixel_info          = state.ps_input_info;
	const bool  object_primitive_id = context.HasObjectPrimitiveIdOutput();
	const auto  write_mask          = context.GetRenderTargetMask() & sh.m_cbShaderMask;
	uint32_t    color_output_mask   = object_primitive_id ? 1u << 7u : 0;
	for (uint32_t index = 0; index < RENDER_COLOR_ATTACHMENTS_MAX; index++) {
		const auto slot = ShaderPixelExportTarget(sh.m_cbShaderMask, index);
		if (slot >= RENDER_COLOR_ATTACHMENTS_MAX) {
			break;
		}
		if (sh.target_output_mode[index] != 0 && render_target_mask_slot(write_mask, slot) != 0) {
			color_output_mask |= 1u << slot;
		}
	}
	const auto& db = sh.db_shader_control;
	// Without color outputs, a pixel shader still runs for its depth or coverage side effects.
	const bool pixel_active =
	    pixel_regs.ps_regs.data_addr != 0 &&
	    (color_output_mask != 0 || db.shader_kill_enable || db.shader_z_export_enable ||
	     db.shader_mask_export_enable || db.shader_dual_export_enable ||
	     db.shader_execute_on_noop || sh.AlphaToMask());
	state.ps_active        = pixel_active;
	const bool tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	std::array<ShaderParams, 3> vertex_params;
	if constexpr (Pure) {
		if (tess_active) {
			prep->refusal = DrawPrepRefusal::Tessellation;
			return false;
		}
	}
	if (tess_active) {
		// Guest geometry shaders run as host mesh shaders, which cannot follow tessellation.
		if ((context.GetShaderStages() & 0x20u) != 0) {
			static std::atomic_bool warned = false;
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(
				    "Warning: dropping tessellation draws with a geometry shader.\n");
			}
			return true;
		}
		vertex_params = PrepareTessellationPrograms(vertex_regs, context, vertex_info);
		// Reflection decodes both programs; the key also covers the VGT fields in tess.
		const uint64_t layout_key[]   = {vertex_params[0].hash, vertex_params[1].hash,
		                                 uint64_t {sh.m_vgtLsHsConfig & 0xfff00u} << 32u |
		                                     sh.m_vgtTfParam};
		const auto [layout, inserted] = m_tess_layouts.try_emplace(
		    XXH3_64bits(layout_key, sizeof(layout_key)), vertex_info[0].tess);
		if (inserted) {
			ShaderRecompiler::AnalyzeTessellationPrograms(vertex_params[0].code,
			                                              vertex_params[1].code, layout->second);
			if constexpr (!Pure) {
				StoreLayout(layout->first, layout->second);
			}
		}
		for (auto& stage: vertex_info) {
			stage.tess = layout->second;
		}
	} else {
		vertex_params[0] = PrepareProgram(vertex_regs, context, user_config, vertex_info[0]);
	}
	const bool mesh_active = vertex_info[0].logical_stage == ShaderType::Mesh;
	if (!tess_active && !mesh_active) {
		vertex_info[0].indirect_offsets = state.indirect_offsets;
		vertex_info[0].indirect_draw    = state.indirect_draw;
	}
	if (object_primitive_id) {
		// PA_CL_OBJPRIM_ID_CNTL = 2: VGT_OBJECT_ID plus the pipelined 32-bit primitive ID.
		EXIT_NOT_IMPLEMENTED(sh.m_paClObjprimIdCntl != 2u || mesh_active || tess_active ||
		                     !pixel_active || (sh.m_vgtDrawPayloadCntl & 8u) == 0 ||
		                     user_config.GetPrimType() != Prospero::PrimitiveType::kTriList);
		const auto& target = context.GetRenderTarget(7);
		EXIT_NOT_IMPLEMENTED(
		    Prospero::ResolveRenderTargetFormat(target.info.format, target.info.channel_type)
		        .buffer_format != Prospero::BufferFormat::k32UInt);
	}
	if (mesh_active) {
		EXIT_NOT_IMPLEMENTED(!m_graphics.mesh_shader_enabled);
		auto& mesh              = vertex_info[0].mesh;
		mesh.host_subgroup_size = m_graphics.subgroup_size;
		const auto& limits      = m_graphics.mesh_shader_properties;
		const auto  logical_threads =
		    mesh.threads_num[0] * mesh.threads_num[1] * mesh.threads_num[2];
		const auto host_threads = ((logical_threads + mesh.wave_size - 1u) / mesh.wave_size) *
		                          std::min(mesh.host_subgroup_size, mesh.wave_size);
		if (host_threads > limits.maxMeshWorkGroupInvocations ||
		    host_threads > limits.maxMeshWorkGroupSize[0] ||
		    mesh.max_vertices > limits.maxMeshOutputVertices ||
		    mesh.max_primitives > limits.maxMeshOutputPrimitives ||
		    mesh.lds_size_dwords * sizeof(uint32_t) > limits.maxMeshSharedMemorySize) {
			static std::atomic_bool warned = false;
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "Warning: dropping mesh draws the host does not support (shader 0x{:016x}).\n",
				    vertex_params[0].hash));
			}
			if constexpr (Pure) {
				prep->refusal = DrawPrepRefusal::MeshDropped;
				return false;
			}
			return true;
		}
		// A task shader launches each record's groups and passes its draw parameters in a
		// payload beside the LDS, the allocation pair and the Layer array; without room for
		// it, the CPU reads the records.
		if (state.indirect_draw && (mesh.lds_size_dwords + 2u + mesh.max_vertices +
		                            ShaderRecompiler::IR::PushData::MeshTaskPayloadDwordCount) *
		                                   sizeof(uint32_t) <=
		                               limits.maxMeshPayloadAndSharedMemorySize) {
			vertex_info[0].indirect_offsets = state.indirect_offsets;
			vertex_info[0].indirect_draw    = true;
		}
	}
	ShaderParams pixel_params;
	if (pixel_active) {
		std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX>
		    target_export_mapping {};
		for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
			const auto& rt = context.GetRenderTarget(slot);
			if ((color_output_mask & (1u << slot)) != 0 && rt.base.addr != 0) {
				target_export_mapping[slot] =
				    TextureGetRenderTargetFormat(m_graphics, rt.info.format, rt.info.channel_type,
				                                 rt.info.channel_order)
				        .export_mapping;
			}
		}
		pixel_params = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
		pixel_info.object_primitive_id = object_primitive_id;
		for (uint32_t export_index = 0; export_index < 8u; export_index++) {
			const auto slot = ShaderPixelExportTarget(pixel_info.target_shader_mask, export_index);
			if (slot >= 8u || pixel_info.target_output_mode[export_index] == 0u) continue;
			const auto& rt          = context.GetRenderTarget(slot);
			const auto  target_mask = render_target_mask_slot(context.GetRenderTargetMask(), slot);
			if (rt.base.addr == 0 || target_mask == 0u) continue;
			if (rt.info.channel_type == Prospero::ChannelType::kUInt) {
				pixel_info.target_uint_mask |= 1u << export_index;
			} else if (rt.info.channel_type == Prospero::ChannelType::kSInt) {
				pixel_info.target_sint_mask |= 1u << export_index;
			}
			const auto format =
			    Prospero::ResolveRenderTargetFormat(rt.info.format, rt.info.channel_type);
			if (format.buffer_format != Prospero::BufferFormat::k11_11_10UNorm) continue;
			const auto mapping     = pixel_info.target_export_mapping[export_index];
			const auto shader_mask = render_target_mask_slot(pixel_info.target_shader_mask, slot);
			const bool blend       = context.GetBlendControl(slot).enable && !rt.info.blend_bypass;
			if (blend || (mapping.ApplyMask(target_mask) & 7u) != 7u ||
			    (mapping.ApplyMask(shader_mask) & 7u) != 7u ||
			    pixel_info.target_output_mode[export_index] != 9u) {
				EXIT("unsupported packed UNorm render-target export: slot=%u export=%u mode=%u "
				     "target_mask=0x%x shader_mask=0x%x mapping=0x%02x blend=%u\n",
				     slot, export_index, pixel_info.target_output_mode[export_index], target_mask,
				     shader_mask, mapping.packed, static_cast<uint32_t>(blend));
			}
			pixel_info.target_packed_unorm_mask |= 1u << export_index;
			pixel_info.target_round_mode |= static_cast<uint8_t>(rt.info.round_mode)
			                                << export_index;
		}
		if (!mesh_active && !tess_active && Prospero::IsRectList(user_config.GetPrimType())) {
			pixel_info.parameter_mode = ShaderPixelParameterMode::Rectangle;
		} else if (context.GetModeControl().provoking_vtx_last) {
			pixel_info.parameter_mode = ShaderPixelParameterMode::LastVertex;
		} else {
			pixel_info.parameter_mode = ShaderPixelParameterMode::FirstVertex;
		}
		const auto& blend = context.GetBlendControl(0);
		pixel_info.dual_source_blending =
		    blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		    (BlendFactorIsDualSource(blend.color_srcblend) ||
		     BlendFactorIsDualSource(blend.color_destblend) ||
		     (blend.separate_alpha_blend && (BlendFactorIsDualSource(blend.alpha_srcblend) ||
		                                     BlendFactorIsDualSource(blend.alpha_destblend))));
		if (pixel_info.dual_source_blending) {
			// MRT1 supplies the second blend source for target 0; blended targets are not integer.
			pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
			pixel_info.target_export_mapping[1] = pixel_info.target_export_mapping[0];
			pixel_info.target_uint_mask &= ~2u;
			pixel_info.target_sint_mask &= ~2u;
		} else if (blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		           pixel_info.target_output_mode[0] != 0 && pixel_info.target_output_mode[0] != 7 &&
		           std::all_of(std::begin(pixel_info.target_output_mode) + 1,
		                       std::end(pixel_info.target_output_mode),
		                       [](uint8_t mode) { return mode == 0; })) {
			switch (ClassifyBlendMapping(blend, pixel_info.target_export_mapping[0])) {
				case BlendMappingSupport::SourceAlpha:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlpha;
					break;
				case BlendMappingSupport::SourceAlphaOne:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaOne;
					break;
				case BlendMappingSupport::SourceAlphaZero:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaZero;
					break;
				default: break;
			}
			if (pixel_info.alpha_blend_source != ShaderAlphaBlendSource::None) {
				pixel_info.dual_source_blending     = true;
				pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
				pixel_info.target_export_mapping[1] = {};
			}
		}
	}
	auto&       last_stage   = vertex_info[tess_active ? 2u : 0u];
	const auto& clip_control = context.GetClipControl();
	if (clip_control.clip_disable) {
		const auto& viewport = context.GetScreenViewport().viewports[0];
		const auto& limits   = m_graphics.GetPhysicalDeviceProperties().limits;
		auto&       clip     = last_stage.clip_space;
		clip.scale[0]        = viewport.xscale;
		clip.scale[1]        = viewport.yscale;
		clip.offset[0]       = viewport.xoffset;
		clip.offset[1]       = viewport.yoffset;
		clip.half_extent[0] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
		clip.half_extent[1] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
		clip.enabled = true;
	} else if (clip_control.user_clip_planes != 0) {
		// CLIP_DISABLE also disables user planes. How they combine with shader clip and cull
		// distances is unverified; draw push constants only reach VS/mesh and PS, and the
		// generated RECTLIST stages do not forward clip distances.
		EXIT_NOT_IMPLEMENTED((last_stage.pa_cl_vs_out_cntl & 0xffffu) != 0 || tess_active ||
		                     Prospero::IsRectList(user_config.GetPrimType()));
		last_stage.user_clip_planes = clip_control.user_clip_planes;
	}
	ShaderRecompiler::IR::PushData::Window push_data {
	    .begin = ShaderRecompiler::IR::PushData::GraphicsDrawDwords(mesh_active, object_primitive_id,
	                                                                last_stage.user_clip_planes)};
	if (pixel_active) {
		const ShaderModule* pixel_program = nullptr;
		if constexpr (Pure) {
			pixel_program = GetProgramPure(pixel_params, pixel_info, push_data,
			                               prep->resources[1], prep->specialization[1],
			                               prep->refusal);
			if (pixel_program == nullptr) {
				return false;
			}
		} else {
			pixel_program = GetProgram(pixel_params, pixel_info, push_data);
		}
		const auto& pixel_module = *pixel_program;
		for (const auto& output: pixel_module.info.info.outputs) {
			if (output.kind == ShaderRecompiler::IR::StageOutputKind::Mrt ||
			    output.kind == ShaderRecompiler::IR::StageOutputKind::ObjectPrimitiveId) {
				state.mrt_mask |= 1u << output.location;
			}
		}
		state.mrt_mask &= color_output_mask;
	}
	for (uint32_t i = 0; i < (tess_active ? 3u : 1u); i++) {
		if constexpr (Pure) {
			// Tessellation was refused: only stage 0 runs here.
			if (GetProgramPure(vertex_params[i], vertex_info[i], push_data,
			                   prep->resources[0], prep->specialization[0],
			                   prep->refusal) == nullptr) {
				return false;
			}
		} else {
			GetProgram(vertex_params[i], vertex_info[i], push_data);
		}
	}
	return true;
}

template bool PipelineCache::GraphicsProgramsImpl<false>(const HW::Context&, const HW::UserConfig&,
                                                         const HW::Shader&, DrawRenderState&,
                                                         DrawProgramPrep*);
template bool PipelineCache::GraphicsProgramsImpl<true>(const HW::Context&, const HW::UserConfig&,
                                                        const HW::Shader&, DrawRenderState&,
                                                        DrawProgramPrep*);

namespace {

// KYTY_VERIFY_DRAW_PREP comparison helpers. Both sides start from one copy of the reset state and
// are written by the same code, so whole-array comparisons are exact for padding-free elements.
template <typename T>
bool SameArray(const T& left, const T& right) {
	static_assert(std::has_unique_object_representations_v<std::remove_all_extents_t<T>>);
	return std::memcmp(&left, &right, sizeof(T)) == 0;
}

bool SameFloats(const float (&left)[2], const float (&right)[2]) {
	return std::memcmp(left, right, sizeof(left)) == 0;
}

const char* DiffResources(const ShaderRecompiler::IR::ResourceSnapshot* left,
                          const ShaderRecompiler::IR::ResourceSnapshot* right) {
	if ((left == nullptr) != (right == nullptr)) return "stage.resources (null)";
	if (left == nullptr) return nullptr;
	if (left->buffers != right->buffers) return "resources.buffers";
	if (left->images != right->images) return "resources.images";
	if (left->samplers != right->samplers) return "resources.samplers";
	if (left->flattened_srt != right->flattened_srt) return "resources.flattened_srt";
	if (left->user_data != right->user_data) return "resources.user_data";
	if (left->specialization_reads != right->specialization_reads)
		return "resources.specialization_reads";
	if (!(left->uniform_fill == right->uniform_fill)) return "resources.uniform_fill";
	return nullptr;
}

#define KYTY_DRAW_PREP_SAME(field)                                                                 \
	if (!(left.field == right.field)) return KYTY_DP_PREFIX #field
#define KYTY_DRAW_PREP_SAME_ARRAY(field)                                                           \
	if (!SameArray(left.field, right.field)) return KYTY_DP_PREFIX #field

const char* DiffVertexInfo(const ShaderVertexInputInfo& left, const ShaderVertexInputInfo& right) {
#define KYTY_DP_PREFIX "vertex_info."
	KYTY_DRAW_PREP_SAME(stage.module);
	KYTY_DRAW_PREP_SAME(logical_stage);
	KYTY_DRAW_PREP_SAME(resources_num);
	KYTY_DRAW_PREP_SAME(fetch_attrib_reg);
	KYTY_DRAW_PREP_SAME(fetch_buffer_reg);
	KYTY_DRAW_PREP_SAME(buffers_num);
	KYTY_DRAW_PREP_SAME(wave_size);
	KYTY_DRAW_PREP_SAME(scratch_size_dwords);
	KYTY_DRAW_PREP_SAME(lds_size_dwords);
	KYTY_DRAW_PREP_SAME(pa_cl_vs_out_cntl);
	KYTY_DRAW_PREP_SAME(user_clip_planes);
	KYTY_DRAW_PREP_SAME(indirect_offsets);
	KYTY_DRAW_PREP_SAME(indirect_draw);
	KYTY_DRAW_PREP_SAME(has_dynamic_vertex_fetch);
	for (int i = 0; i < ShaderVertexInputInfo::RES_MAX; i++) {
		if (!SameArray(left.resources[i].fields, right.resources[i].fields))
			return KYTY_DP_PREFIX "resources";
		const auto& l = left.resources_dst[i];
		const auto& r = right.resources_dst[i];
		if (l.register_start != r.register_start || l.registers_num != r.registers_num ||
		    l.attr_id != r.attr_id || l.fetch_index != r.fetch_index ||
		    l.buffer_index != r.buffer_index)
			return KYTY_DP_PREFIX "resources_dst";
		const auto& lb = left.buffers[i];
		const auto& rb = right.buffers[i];
		if (lb.addr != rb.addr || lb.stride != rb.stride || lb.num_records != rb.num_records ||
		    lb.fetch_index != rb.fetch_index)
			return KYTY_DP_PREFIX "buffers";
	}
	if (!SameFloats(left.clip_space.scale, right.clip_space.scale) ||
	    !SameFloats(left.clip_space.offset, right.clip_space.offset) ||
	    !SameFloats(left.clip_space.half_extent, right.clip_space.half_extent) ||
	    left.clip_space.enabled != right.clip_space.enabled)
		return KYTY_DP_PREFIX "clip_space";
	KYTY_DRAW_PREP_SAME_ARRAY(mesh.threads_num);
	KYTY_DRAW_PREP_SAME(mesh.lds_size_dwords);
	KYTY_DRAW_PREP_SAME(mesh.scratch_size_dwords);
	KYTY_DRAW_PREP_SAME(mesh.host_subgroup_size);
	KYTY_DRAW_PREP_SAME(mesh.wave_size);
	KYTY_DRAW_PREP_SAME(mesh.input_primitive);
	KYTY_DRAW_PREP_SAME(mesh.primitives_per_group);
	KYTY_DRAW_PREP_SAME(mesh.vertices_per_group);
	KYTY_DRAW_PREP_SAME(mesh.max_vertices);
	KYTY_DRAW_PREP_SAME(mesh.max_primitives);
	KYTY_DRAW_PREP_SAME(mesh.provoking_vertex);
	KYTY_DRAW_PREP_SAME(mesh.fast_launch);
	KYTY_DRAW_PREP_SAME(tess.input_control_points);
	KYTY_DRAW_PREP_SAME(tess.output_control_points);
	KYTY_DRAW_PREP_SAME(tess.ls_stride);
	KYTY_DRAW_PREP_SAME(tess.hs_stride);
	KYTY_DRAW_PREP_SAME(tess.patch_base);
	KYTY_DRAW_PREP_SAME(tess.patch_size);
	KYTY_DRAW_PREP_SAME(tess.domain);
	KYTY_DRAW_PREP_SAME(tess.partitioning);
	KYTY_DRAW_PREP_SAME(tess.output_topology);
#undef KYTY_DP_PREFIX
	std::vector<uint32_t> left_key;
	std::vector<uint32_t> right_key;
	BuildStageStaticKey(left, left_key);
	BuildStageStaticKey(right, right_key);
	if (left_key != right_key) return "vertex_info static key";
	return DiffResources(left.stage.resources, right.stage.resources);
}

const char* DiffPixelInfo(const ShaderPixelInputInfo& left, const ShaderPixelInputInfo& right) {
#define KYTY_DP_PREFIX "ps_input_info."
	KYTY_DRAW_PREP_SAME(stage.module);
	KYTY_DRAW_PREP_SAME_ARRAY(interpolator_settings);
	KYTY_DRAW_PREP_SAME(input_num);
	KYTY_DRAW_PREP_SAME(wave_size);
	KYTY_DRAW_PREP_SAME(ps_system_input_base);
	KYTY_DRAW_PREP_SAME(custom_interpolation_mask);
	KYTY_DRAW_PREP_SAME(ps_perspective_center_vgpr);
	KYTY_DRAW_PREP_SAME(ps_perspective_sample_vgpr);
	KYTY_DRAW_PREP_SAME(ps_perspective_centroid_vgpr);
	KYTY_DRAW_PREP_SAME_ARRAY(target_output_mode);
	KYTY_DRAW_PREP_SAME(target_shader_mask);
	for (size_t i = 0; i < left.target_export_mapping.size(); i++) {
		if (left.target_export_mapping[i].packed != right.target_export_mapping[i].packed)
			return KYTY_DP_PREFIX "target_export_mapping";
	}
	KYTY_DRAW_PREP_SAME(target_packed_unorm_mask);
	KYTY_DRAW_PREP_SAME(target_uint_mask);
	KYTY_DRAW_PREP_SAME(target_sint_mask);
	KYTY_DRAW_PREP_SAME(target_round_mode);
	KYTY_DRAW_PREP_SAME(scratch_size_dwords);
	KYTY_DRAW_PREP_SAME(lds_size_dwords);
	KYTY_DRAW_PREP_SAME(ps_pos_x);
	KYTY_DRAW_PREP_SAME(ps_pos_y);
	KYTY_DRAW_PREP_SAME(ps_pos_z);
	KYTY_DRAW_PREP_SAME(ps_pos_w);
	KYTY_DRAW_PREP_SAME(ps_front_face);
	KYTY_DRAW_PREP_SAME(ps_front_face_all_bits);
	KYTY_DRAW_PREP_SAME(ps_ancillary);
	KYTY_DRAW_PREP_SAME(ps_no_perspective);
	KYTY_DRAW_PREP_SAME(parameter_mode);
	KYTY_DRAW_PREP_SAME(ps_pixel_kill_enable);
	KYTY_DRAW_PREP_SAME(ps_depth_export_enable);
	KYTY_DRAW_PREP_SAME(ps_sample_mask_export_enable);
	KYTY_DRAW_PREP_SAME(ps_sample_shading);
	KYTY_DRAW_PREP_SAME(dual_source_blending);
	KYTY_DRAW_PREP_SAME(object_primitive_id);
	KYTY_DRAW_PREP_SAME(alpha_blend_source);
	KYTY_DRAW_PREP_SAME(ps_early_z);
#undef KYTY_DP_PREFIX
	std::vector<uint32_t> left_key;
	std::vector<uint32_t> right_key;
	BuildStageStaticKey(left, left_key);
	BuildStageStaticKey(right, right_key);
	if (left_key != right_key) return "ps_input_info static key";
	return DiffResources(left.stage.resources, right.stage.resources);
}

#undef KYTY_DRAW_PREP_SAME
#undef KYTY_DRAW_PREP_SAME_ARRAY

} // namespace

const char* DiffGraphicsPrograms(const DrawRenderState& left, const DrawRenderState& right) {
	if (left.mrt_mask != right.mrt_mask) return "mrt_mask";
	if (left.ps_active != right.ps_active) return "ps_active";
	if (left.topology != right.topology) return "topology";
	if (left.primitive_restart_enable != right.primitive_restart_enable)
		return "primitive_restart_enable";
	// Tessellation is refused by the pure path, so only stage 0 is ever prepared; the other
	// stages are compared for their (reset) module pointers.
	for (size_t i = 1; i < left.vertex_info.size(); i++) {
		if (left.vertex_info[i].stage.module != right.vertex_info[i].stage.module)
			return "vertex_info[1..2].stage.module";
	}
	if (const auto* field = DiffVertexInfo(left.vertex_info[0], right.vertex_info[0])) return field;
	return DiffPixelInfo(left.ps_input_info, right.ps_input_info);
}

const ShaderModule* PipelineCache::GetComputeProgram(const HW::ComputeShaderInfo& regs,
                                                     ShaderComputeInputInfo&      input_info) {
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	const auto params             = PrepareProgram(regs, input_info);
	input_info.lds_storage =
	    input_info.lds_size_dwords * 4u >
	    m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	ShaderRecompiler::IR::PushData::Window push_data;
	const auto* module = GetProgram(params, input_info, push_data);
	// HW_ID2 reads are translated with the graphics compute queue's values.
	EXIT_NOT_IMPLEMENTED(module != nullptr && input_info.async_compute &&
	                     module->info.info.reads_queue_id);
	return module;
}

PipelineCache::Pipeline& PipelineCache::GetGraphicsPipeline(const CommandBuffer&   command,
                                                            const DrawRenderState& state,
                                                            const DrawTargets&     targets) {
	const auto* colors        = targets.colors;
	const auto& depth         = targets.depth;
	const auto  vertex_info   = state.VertexStages();
	const auto& ps_input_info = state.ps_input_info;
	const auto& vs_input_info = vertex_info.front();
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Gfx)", profiler::colors::DeepOrangeA200);

	const bool                         ps_active    = state.ps_active;
	const bool                         tessellation = vertex_info.size() == 3;
	std::array<const ShaderModule*, 4> stages {};
	for (uint32_t i = 0; i < vertex_info.size(); i++) {
		stages[i] = vertex_info[i].stage.module;
	}
	if (ps_active) {
		stages[3] = ps_input_info.stage.module;
	}
	const bool mesh        = stages[0]->info.stage == ShaderType::Mesh;
	const auto color_count = targets.color_count;

	auto& ctx = command.GetRegisters();

	// The previous draw's pipeline for a draw with its targets: the key is half a kilobyte to
	// clear, fill, hash and probe, and consecutive draws mostly use the same pipeline (Bruno
	// Cardenas Torres 9153daf18). KYTY_VERIFY_PIPELINE_KEEP=1 builds the key anyway and exits if
	// the kept pipeline is not the one the key finds.
	static const bool keep_verify = std::getenv("KYTY_VERIFY_PIPELINE_KEEP") != nullptr;
	Pipeline*         kept        = nullptr;
	if (Common::AbSwitch::Enabled(Common::AbSwitch::DRAW_CHAIN_PIPELINE) && state.target_memo &&
	    m_last_graphics_inputs.valid && m_last_graphics_pipeline != nullptr &&
	    m_last_graphics_inputs.state_serial == g_target_state_serial &&
	    m_last_graphics_inputs.topology == state.topology &&
	    m_last_graphics_inputs.primitive_restart == state.primitive_restart_enable &&
	    m_last_graphics_inputs.pixel_active == ps_active && m_last_graphics_inputs.mesh == mesh &&
	    m_last_graphics_inputs.patch_points ==
	        (tessellation ? static_cast<uint32_t>(vs_input_info.tess.input_control_points) : 0u) &&
	    (!ps_active ||
	     (m_last_graphics_inputs.alpha_source == ps_input_info.alpha_blend_source &&
	      m_last_graphics_inputs.sample_shading == ps_input_info.ps_sample_shading))) {
		const auto&             last = m_last_graphics_key;
		std::array<uint64_t, 4> ids {};
		for (uint32_t i = 0; i < stages.size(); i++) {
			ids[i] = stages[i] != nullptr ? stages[i]->id : 0;
		}
		bool same = last.stage_ids == ids;
		if (same && !mesh) {
			const auto& input = last.vertex_input;
			same              = input.binding_count == vs_input_info.buffers_num &&
			       input.attribute_count == vs_input_info.resources_num;
			for (int i = 0; i < vs_input_info.buffers_num && same; i++) {
				const auto& buffer = vs_input_info.buffers[i];
				same               = input.bindings[i].stride == buffer.stride &&
				       input.bindings[i].instance == (buffer.fetch_index != 0);
			}
			for (int i = 0; i < vs_input_info.resources_num && same; i++) {
				const auto binding = vs_input_info.resources_dst[i].buffer_index;
				same               = binding >= 0 && binding < vs_input_info.buffers_num &&
				       input.attributes[i].binding == static_cast<uint8_t>(binding) &&
				       input.attributes[i].offset ==
				           static_cast<uint32_t>(vs_input_info.resources[i].Base48() -
				                                 vs_input_info.buffers[binding].addr);
			}
		} else if (same) {
			same = last.vertex_input.binding_count == 0 && last.vertex_input.attribute_count == 0;
		}
		if (same) {
			if (!keep_verify) [[likely]] {
				Common::DebugCounters::Add(Common::DebugCounters::Counter::DrawChainPipeKept);
				return *m_last_graphics_pipeline;
			}
			kept = m_last_graphics_pipeline;
		}
	}

	const HW::ModeControl& mc = ctx.GetModeControl();

	GraphicsPipelineKey key {};
	for (uint32_t i = 0; i < stages.size(); i++) {
		if (stages[i] != nullptr) {
			key.stage_ids[i] = stages[i]->id;
		}
	}
	auto&    static_params      = key.static_params;
	auto&    rendering          = key.rendering;
	uint32_t attachment_samples = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		const auto slot = colors[i].target_slot;
		EXIT_IF(slot >= RENDER_COLOR_ATTACHMENTS_MAX);
		rendering.color_count = std::max(rendering.color_count, slot + 1);
		EXIT_IF(!colors[i].image_id || colors[i].desc.view_info.format == vk::Format::eUndefined);
		static_params.color_mask[slot] = colors[i].export_mapping.ApplyMask(
		    render_target_mask_slot(ctx.GetRenderTargetMask(), colors[i].target_slot));
		if (colors[i].desc.info.guest_format == Prospero::BufferFormat::k11_11_10UNorm) {
			static_params.color_mask[slot] = 1u;
		}
		rendering.color_formats[slot] = colors[i].desc.view_info.format;
		if (attachment_samples == 0) {
			attachment_samples = colors[i].desc.info.samples;
		} else if (attachment_samples != colors[i].desc.info.samples) {
			EXIT("mixed color attachment sample counts are unsupported: %u and %u\n",
			     attachment_samples, colors[i].desc.info.samples);
		}
		const auto& rt           = ctx.GetRenderTarget(colors[i].target_slot);
		const auto& bc           = ctx.GetBlendControl(colors[i].target_slot);
		auto        alpha_source = ShaderAlphaBlendSource::None;
		if (slot == 0 && ps_active) {
			alpha_source = ps_input_info.alpha_blend_source;
		}
		static_params.blend_enable[slot] = bc.enable && !rt.info.blend_bypass;
		if (static_params.blend_enable[slot] && alpha_source == ShaderAlphaBlendSource::None &&
		    ClassifyBlendMapping(bc, colors[i].export_mapping) != BlendMappingSupport::Direct) {
			static_params.blend_enable[slot] = false;
			static std::atomic_bool warned   = false;
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "Warning: blending disabled for unsupported color mapping "
				    "(slot={} mapping=0x{:02x} color={}/{} alpha={}/{} separate={}).\n",
				    slot, colors[i].export_mapping.packed, bc.color_srcblend, bc.color_destblend,
				    bc.alpha_srcblend, bc.alpha_destblend, bc.separate_alpha_blend ? 1 : 0));
			}
		}
		if (static_params.blend_enable[slot]) {
			auto blend = bc;
			switch (alpha_source) {
				case ShaderAlphaBlendSource::SourceAlpha:
					blend.color_srcblend       = RemapSourceAlphaFactor(blend.color_srcblend);
					blend.color_destblend      = RemapSourceAlphaFactor(blend.color_destblend);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::SourceAlphaOne:
				case ShaderAlphaBlendSource::SourceAlphaZero:
					// The second source carries the mapped source factor; its alpha stays logical
					// Sa.
					blend.color_srcblend = static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Color);
					blend.color_destblend =
					    static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::None: break;
			}
			static_params.color_srcblend[slot]       = blend.color_srcblend;
			static_params.color_comb_fcn[slot]       = blend.color_comb_fcn;
			static_params.color_destblend[slot]      = blend.color_destblend;
			static_params.separate_alpha_blend[slot] = blend.separate_alpha_blend;
			if (blend.separate_alpha_blend) {
				static_params.alpha_srcblend[slot]  = blend.alpha_srcblend;
				static_params.alpha_comb_fcn[slot]  = blend.alpha_comb_fcn;
				static_params.alpha_destblend[slot] = blend.alpha_destblend;
			}
		}
	}
	const bool with_depth =
	    depth.desc.view_info.format != vk::Format::eUndefined && static_cast<bool>(depth.image_id);
	if (with_depth) {
		const auto aspects       = depth.desc.view_info.aspect;
		rendering.depth_format   = aspects & vk::ImageAspectFlagBits::eDepth
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		rendering.stencil_format = aspects & vk::ImageAspectFlagBits::eStencil
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		if (attachment_samples == 0) {
			attachment_samples = depth.desc.info.samples;
		} else if (attachment_samples != depth.desc.info.samples) {
			EXIT("mixed color/depth sample counts are unsupported: %u and %u\n", attachment_samples,
			     depth.desc.info.samples);
		}
	}
	if (color_count == 0 && !with_depth) {
		attachment_samples = render_sample_count(ctx.GetAaConfig().msaa_num_samples);
		EXIT_IF(!static_cast<bool>(
		    m_graphics.GetPhysicalDeviceProperties().limits.framebufferNoAttachmentsSampleCounts &
		    vulkan_sample_count(attachment_samples)));
	}
	EXIT_IF(attachment_samples == 0 ||
	        vulkan_sample_count(attachment_samples) == vk::SampleCountFlagBits {});

	const auto& clip_control               = ctx.GetClipControl();
	static_params.negative_one_to_one      = !clip_control.dx_clip_space;
	static_params.depth_clip_enable        = clip_control.IsZClipEnabled();
	static_params.topology                 = state.topology;
	static_params.primitive_restart_enable = state.primitive_restart_enable;
	static_params.samples                  = attachment_samples;
	static_params.sample_shading_enable =
	    ps_active && attachment_samples > 1 && ps_input_info.ps_sample_shading;
	static_params.alpha_to_coverage_enable = ps_active && ctx.GetShaderRegisters().AlphaToMask();
	const bool rect_list             = Prospero::IsRectList(command.GetUserConfig().GetPrimType());
	static_params.provoking_vtx_last = mc.provoking_vtx_last;
	static_params.polygon_mode =
	    ResolvePolygonMode(mc, !rect_list && mc.cull_front, !rect_list && mc.cull_back);
	if (tessellation) {
		static_params.patch_control_points = vs_input_info.tess.input_control_points;
	} else if (!mesh && static_params.topology == vk::PrimitiveTopology::ePatchList) {
		// Rectangle lists expand through generated tessellation stages.
		static_params.patch_control_points = 3;
	}

	if (!mesh) {
		EXIT_IF(vs_input_info.buffers_num < 0 ||
		        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX ||
		        vs_input_info.resources_num < 0 ||
		        vs_input_info.resources_num > ShaderVertexInputInfo::RES_MAX);
		key.vertex_input.binding_count   = static_cast<uint8_t>(vs_input_info.buffers_num);
		key.vertex_input.attribute_count = static_cast<uint8_t>(vs_input_info.resources_num);
		for (int binding = 0; binding < vs_input_info.buffers_num; binding++) {
			const auto& buffer                 = vs_input_info.buffers[binding];
			key.vertex_input.bindings[binding] = {.stride   = buffer.stride,
			                                      .instance = buffer.fetch_index != 0};
		}
		for (int attribute = 0; attribute < vs_input_info.resources_num; attribute++) {
			const auto binding = vs_input_info.resources_dst[attribute].buffer_index;
			EXIT_IF(binding < 0 || binding >= vs_input_info.buffers_num);
			key.vertex_input.attributes[attribute] = {
			    .offset  = static_cast<uint32_t>(vs_input_info.resources[attribute].Base48() -
			                                     vs_input_info.buffers[binding].addr),
			    .binding = static_cast<uint8_t>(binding),
			    .format  = vs_input_info.resources[attribute].RawFormat(),
			};
		}
	}

	auto [iter, inserted] = m_graphics_pipelines.try_emplace(key);
	auto& pipeline        = iter->second;
	if (kept != nullptr && kept != &pipeline) [[unlikely]] {
		EXIT("pipeline keep: the kept pipeline is not the one its key finds\n");
	}
	m_last_graphics_key      = key;
	m_last_graphics_pipeline = &pipeline;
	m_last_graphics_inputs   = {
	      .state_serial      = g_target_state_serial,
	      .topology          = state.topology,
	      .primitive_restart = state.primitive_restart_enable,
	      .pixel_active      = ps_active,
	      .mesh              = mesh,
	      .patch_points =
            tessellation ? static_cast<uint32_t>(vs_input_info.tess.input_control_points) : 0u,
	      .alpha_source = ps_active ? ps_input_info.alpha_blend_source : ShaderAlphaBlendSource::None,
	      .sample_shading = ps_active && ps_input_info.ps_sample_shading,
	      .valid          = true};
	if (!inserted) {
		return pipeline;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(vs_input_info);
		if (ps_active) {
			ShaderDbgDumpInputInfo(ps_input_info);
		}
	}

	{
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::PipelinesCreated,
		                                         Common::DebugCounters::Counter::PipelineCreateNs,
		                                         Common::DebugCounters::Peak::PipelineCreateNs);
		CreatePipelineInternal(m_graphics, pipeline, key, stages, m_driver_cache);
	}
	StorePipeline(key);
	CheckpointDriverCache();

	return pipeline;
}

PipelineCache::Pipeline&
PipelineCache::GetComputePipeline(const ShaderComputeInputInfo& input_info) {
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Compute)", profiler::colors::RedA100);

	const auto& module = *input_info.stage.module;

	// KYTY_COMPUTE_HINT: the pipeline this module id found last. Elements of m_compute_pipelines keep
	// their address, so the slot holds what try_emplace would return for the same id.
	ComputePipelineHint* slot = nullptr;
	if (g_compute_hint) {
		slot = &m_compute_pipeline_hints[ComputeHintSlot(module.id, COMPUTE_HINT_SLOTS)];
		if (slot->pipeline != nullptr && slot->module_id == module.id) {
			Common::DebugCounters::Add(Common::DebugCounters::Counter::ComputePipeMemoHits);
			if (g_verify_compute_hint) [[unlikely]] {
				const auto found = m_compute_pipelines.find(module.id);
				if (found == m_compute_pipelines.end() || &found->second != slot->pipeline) {
					EXIT("compute hint verify: module 0x%016" PRIx64 " found another pipeline\n",
					     module.id);
				}
			}
			return *slot->pipeline;
		}
	}

	auto [iter, inserted] = m_compute_pipelines.try_emplace(module.id);
	auto& pipeline        = iter->second;
	if (slot != nullptr) {
		*slot = {module.id, &pipeline};
	}
	if (!inserted) {
		return pipeline;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(input_info);
	}

	{
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::PipelinesCreated,
		                                         Common::DebugCounters::Counter::PipelineCreateNs,
		                                         Common::DebugCounters::Peak::PipelineCreateNs);
		CreatePipelineInternal(m_graphics, pipeline, module, m_driver_cache);
	}
	CheckpointDriverCache();

	return pipeline;
}
} // namespace Libs::Graphics
