#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include "common/abSwitch.h"
#include "common/abi.h"
#include "common/assert.h"
#include "common/debugCounters.h"
#if defined(TRACY_ENABLE)
// Standalone walker tests do not link the profiling runtime.
#include "common/profiler.h"
#endif
#include "graphics/host_gpu/pageManager.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "kernel/memory.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fmt/format.h>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <xbyak/xbyak.h>

namespace Libs::Graphics::ShaderRecompiler::IR {

SrtRuntime SrtReadCapture::ObservedRuntime() {
	auto runtime = m_source;
	runtime.userdata = this;
	runtime.read_specialization_memory = m_source.read_specialization_memory != nullptr
	                                         ? Read : nullptr;
	return runtime;
}

bool SrtReadCapture::Read(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& capture = *static_cast<SrtReadCapture*>(userdata);
	if (!capture.m_source.read_specialization_memory(capture.m_source.userdata, address, values)) {
		return false;
	}
	capture.m_ranges.emplace_back(address, values.size_bytes());
	return true;
}

namespace {

bool IsRawRead(const ResourcePlan& values, const Inst& inst) {
	const auto op = inst.GetOpcode();
	if (op != ValueOpcode::LoadAddressU32 && op != ValueOpcode::ReadConstBuffer &&
	    op != ValueOpcode::LoadBufferU32) {
		return false;
	}
	const auto index = inst.Flags<MemoryFlags>().index;
	if (index >= values.memory_info.size()) {
		return false;
	}
	const auto& memory = values.memory_info[index];
	if (op != ValueOpcode::LoadBufferU32) {
		return (op == ValueOpcode::LoadAddressU32 && memory.kind == ResourceKind::ScalarAddress) ||
		       (op == ValueOpcode::ReadConstBuffer && memory.kind == ResourceKind::ScalarBuffer);
	}
	if (inst.NumArgs() != 5u || memory.kind != ResourceKind::Buffer || memory.typed ||
	    memory.formatted || memory.coherent || memory.data_bits != 32u ||
	    memory.data_dwords != 1u || memory.idxen || memory.offen || memory.offset != 0u ||
	    inst.Arg(4).GetType() != Type::U1) {
		return false;
	}
	for (uint32_t arg = 1; arg <= 3; ++arg) {
		if (inst.Arg(arg).Resolve() != Value(0u)) return false;
	}
	return true;
}

bool IsDescriptorHandle(ValueOpcode opcode) {
	switch (opcode) {
		case ValueOpcode::GetBufferResource:
		case ValueOpcode::GetAddressResource:
		case ValueOpcode::GetImageResource:
		case ValueOpcode::GetSamplerResource: return true;
		default: return false;
	}
}

bool IsRuntimeSelect(ValueOpcode op) {
	return op == ValueOpcode::SelectU1 || op == ValueOpcode::SelectU32 ||
	       op == ValueOpcode::SelectF32;
}

bool IsRuntimeUniformOp(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::ConditionRef:
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32:
		case ValueOpcode::ConvertU32F32:
		case ValueOpcode::ConvertS32F32:
		case ValueOpcode::ConvertF32U32:
		case ValueOpcode::CompositeConstructU64:
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeConstructU32x2:
		case ValueOpcode::CompositeExtractU32x2:
		case ValueOpcode::BitFieldInsert:
		case ValueOpcode::BitFieldUExtract:
		case ValueOpcode::BitFieldSExtract:
		case ValueOpcode::IAdd32:
		case ValueOpcode::IAdd64:
		case ValueOpcode::ISub32:
		case ValueOpcode::ISub64:
		case ValueOpcode::IMul32:
		case ValueOpcode::UMulHi:
		case ValueOpcode::UMin32:
		case ValueOpcode::UMax32:
		case ValueOpcode::SMin32:
		case ValueOpcode::SMax32:
		case ValueOpcode::ShiftLeftLogical32:
		case ValueOpcode::ShiftLeftLogical64:
		case ValueOpcode::ShiftRightLogical32:
		case ValueOpcode::ShiftRightLogical64:
		case ValueOpcode::ShiftRightArithmetic32:
		case ValueOpcode::ShiftRightArithmetic64:
		case ValueOpcode::BitwiseAnd32:
		case ValueOpcode::BitwiseAnd64:
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::BitwiseXor32:
		case ValueOpcode::BitwiseNot32:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectF32:
		case ValueOpcode::ULessThan32:
		case ValueOpcode::ULessThanEqual32:
		case ValueOpcode::IEqual32:
		case ValueOpcode::IEqual64:
		case ValueOpcode::UGreaterThan32:
		case ValueOpcode::SGreaterThanEqual32:
		case ValueOpcode::INotEqual32:
		case ValueOpcode::INotEqual64:
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalAnd:
		case ValueOpcode::LogicalXor:
		case ValueOpcode::LogicalNot:
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
		case ValueOpcode::FPIsNan32:
		case ValueOpcode::FPMul32:
		case ValueOpcode::FPRecipIFlag32:
		case ValueOpcode::FPTrunc32: return true;
		default: return false;
	}
}

bool ImmediateBits(Value value, uint64_t& bits) {
	switch (value.GetType()) {
		case Type::U1: bits = value.U1(); return true;
		case Type::U8: bits = value.U8(); return true;
		case Type::U16: bits = value.U16(); return true;
		case Type::U32: bits = value.U32(); return true;
		case Type::U64: bits = value.U64(); return true;
		case Type::F32: bits = std::bit_cast<uint32_t>(value.F32Value()); return true;
		default: return false;
	}
}

// Scratch layout in bytes: per descriptor source its eight DWORDs and a known DWORD, one
// activity byte per source, then per image, then one qword per walked value (all ones when
// unknown). The walker's value slots, reachability, states, read bases and captured addresses
// follow.
constexpr uint32_t DescriptorStride = 9u * sizeof(uint32_t);

uint32_t ActiveOffset(const ResourcePlan& plan) {
	return static_cast<uint32_t>(plan.descriptor_sources.size()) * DescriptorStride;
}

uint32_t OutputOffset(const ResourcePlan& plan) {
	const auto active = plan.descriptor_sources.size() + plan.info.images.size();
	return (ActiveOffset(plan) + static_cast<uint32_t>(active) + 7u) & ~7u;
}

const uint8_t* Scratch(const ResourcePlan& plan) {
	return reinterpret_cast<const uint8_t*>(plan.srt_scratch.data());
}

thread_local const Xbyak::CodeGenerator* t_running_walker = nullptr;
// The shader hash and user data of the running walker (KYTY_SRT_DIRTY_STATS diagnostics only; the
// walks that read guest memory run on the GPU thread). Deliberately NOT thread_local: builds that
// added these as thread_local variables rendered the UFC 6 arena with blocky corruption in every
// run (2026-11-19, exes 6A8A5CD1/A4236336/842F4D72), the build without them did not (2F246FD7,
// twice). Cause not understood yet; see the 2026-11-18 HANDOFF.
uint64_t        t_walker_shader     = 0;
const uint32_t* t_walker_user_data  = nullptr;
uint32_t        t_walker_user_count = 0;

// Set by WalkSrt from AbSwitch SRT_LOAD_CHECK; generated code tests it.
uint8_t g_srt_load_check = 0;

// Walker loads from pages that fault on guest reads, on any thread (SrtWalkerGuardedLoads).
// Counted where SrtWalkerFaults is; only these slow paths pay for it.
std::atomic<uint64_t> g_walker_guarded_loads {0};

// A guest load from a page that faults on reads, without the fault: the same completion as the
// fault path of Loader's exception handler (clean bytes from the backing store, else the GPU's
// bytes), and zero where that path skips the load.
uint32_t SrtWalkerReadGuestDword(uint64_t address) noexcept {
#if defined(TRACY_ENABLE)
	KYTY_PROFILER_BLOCK("SRT: guarded guest read");
#endif
	using Counter  = Common::DebugCounters::Counter;
	namespace Mem  = Libs::LibKernel::Memory;
	uint32_t value = 0;
	Common::DebugCounters::Add(Counter::SrtWalkerFaults);
	g_walker_guarded_loads.fetch_add(1, std::memory_order_relaxed);
	if (Common::AbSwitch::Enabled(Common::AbSwitch::SRT_READ_BACKING) &&
	    Mem::TryReadGpuCleanBytes(address, &value, sizeof(value))) {
		Common::DebugCounters::Add(Counter::SrtReadsFromBacking);
		return value;
	}
	Mem::TraceGpuDirtyRead("srt", address);
#if defined(TRACY_ENABLE)
	KYTY_PROFILER_BLOCK("SRT: GPU dirty readback");
#endif
	uint32_t   before      = 0;
	const bool have_before = Mem::TryReadBacking(address, &before, sizeof(before));
	// KYTY_SRT_DIRTY_STATS (diagnostic): the GPU write this read completes, and the time it takes.
	const bool stats = Mem::SrtDirtyStatsEnabled();
	const auto cover = stats ? Mem::SrtDirtyCoverAt(address) : Mem::SrtDirtyCover {};
	const auto start = stats ? std::chrono::steady_clock::now()
	                         : std::chrono::steady_clock::time_point {};
	if (!Mem::HandleGpuFault(PageFaultAccess::Read, address) ||
	    !Mem::TryReadBacking(address, &value, sizeof(value))) {
		return 0;
	}
	if (stats) [[unlikely]] {
		Mem::SrtDirtyRead read;
		read.address      = address;
		read.reader       = t_walker_shader;
		read.before       = before;
		read.after        = value;
		read.known_before = have_before;
		read.wait_ns      = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now() - start)
                           .count();
		Mem::NoteSrtDirtyRead(cover, read);
		static std::atomic_int samples = 0;
		if (cover.in_flight && samples.fetch_add(1, std::memory_order_relaxed) < 40) {
			std::string text = fmt::format("srt-dirty-sample: reader {:016x} addr 0x{:x} {:08x}->{:08x} "
			                               "writer {:016x} size 0x{:x} ud",
			                               read.reader, address, before, value, cover.writer, cover.size);
			for (uint32_t i = 0; i < t_walker_user_count && i < 16; i++) {
				text += fmt::format(" {:08x}", t_walker_user_data[i]);
			}
			std::puts(text.c_str());
			std::fflush(stdout);
		}
	}
	Common::DebugCounters::Add(Counter::GpuFaults);
	if (have_before) {
		Common::DebugCounters::Add(value == before ? Counter::DirtyReadsUnchanged
		                                           : Counter::DirtyReadsChanged);
	}
	return value;
}

// A load only block conditions use (AbSwitch SRT_OPTIONAL_DIRTY_UNKNOWN): the value, or bit 32 set
// (unknown) instead of a wait when a GPU write still in flight covers the DWORD. An unknown
// condition reaches both successors, so the walk stays correct, only less pruned.
// KYTY_SRT_OPTIONAL_DIAG=1 (diagnostic, 2026-11-19): unknown optional loads of the current walk, and
// the forced precise re-walk that compares the resource activity (GPU thread only, see WalkSrt).
uint32_t g_optional_unknowns = 0;
bool     g_force_precise     = false;

uint64_t SrtWalkerReadOptionalDword(uint64_t address) noexcept {
	namespace Mem = Libs::LibKernel::Memory;
	using Counter = Common::DebugCounters::Counter;
	if (!g_force_precise &&
	    Common::AbSwitch::Enabled(Common::AbSwitch::SRT_OPTIONAL_DIRTY_UNKNOWN) &&
	    Mem::GpuWriteInFlightAt(address)) {
		uint32_t value = 0;
		if (Common::AbSwitch::Enabled(Common::AbSwitch::SRT_READ_BACKING) &&
		    Mem::TryReadGpuCleanBytes(address, &value, sizeof(value))) {
			Common::DebugCounters::Add(Counter::SrtWalkerFaults);
			Common::DebugCounters::Add(Counter::SrtReadsFromBacking);
			g_walker_guarded_loads.fetch_add(1, std::memory_order_relaxed);
			return value;
		}
		Common::DebugCounters::Add(Counter::SrtOptionalUnknown);
		g_walker_guarded_loads.fetch_add(1, std::memory_order_relaxed);
		g_optional_unknowns++;
		return uint64_t {1} << 32u;
	}
	return SrtWalkerReadGuestDword(address);
}

// Every guest load is mov eax, [r10] (41 8b 02), so a fault completes it as a zero read. With
// AbSwitch SRT_LOAD_CHECK, the load first looks the page up in the PageManager's table: a page
// that faults on reads (GPU-written) is read by SrtWalkerReadGuestDword instead, which saves the
// exception round trip (about 3.5 us, some 400 times a frame in UFC 6's fights). The walker keeps
// rdi, rsi, rdx and r8 live and r10 for the capture; rcx, r11 and rax are scratch here.
// An optional load (only block conditions use it) calls SrtWalkerReadOptionalDword, whose rax has
// bit 32 set for unknown; the fast path's 32-bit load clears it.
// The slow path of a guest load: rax = the read helper's result for the address in r10.
void EmitGuestSlowCall(Xbyak::CodeGenerator& code, bool optional) {
	// The walker's stack depth varies (values are thunks it calls): align for the Windows x64 call.
	code.push(code.rbp);
	code.mov(code.rbp, code.rsp);
	code.and_(code.rsp, -16);
	code.push(code.rdx);
	code.push(code.r8);
	code.push(code.r10);
	code.push(code.r10);
	code.sub(code.rsp, 32);
	code.mov(code.rcx, code.r10);
	code.mov(code.rax, optional ? reinterpret_cast<uint64_t>(&SrtWalkerReadOptionalDword)
	                            : reinterpret_cast<uint64_t>(&SrtWalkerReadGuestDword));
	code.call(code.rax);
	code.add(code.rsp, 32);
	code.pop(code.r10);
	code.pop(code.r10);
	code.pop(code.r8);
	code.pop(code.rdx);
	code.mov(code.rsp, code.rbp);
	code.pop(code.rbp);
}

void LoadGuest(Xbyak::CodeGenerator& code, const Xbyak::RegExp& address, bool optional = false) {
	constexpr auto Near = Xbyak::CodeGenerator::T_NEAR;
	code.lea(code.r10, code.ptr[address]);
	// The walkers of a loaded shader cache compile before the PageManager exists: read its
	// regions when the walker runs.
	const auto*  lookup = PageManager::AccessLookup();
	Xbyak::Label load, slow, done;
	code.mov(code.rcx, reinterpret_cast<uint64_t>(&g_srt_load_check));
	code.cmp(code.byte[code.rcx], 0);
	code.je(load, Near);
	code.mov(code.r11, reinterpret_cast<uint64_t>(&lookup->regions));
	code.mov(code.r11, code.qword[code.r11]);
	code.test(code.r11, code.r11);
	code.jz(load, Near);
	code.mov(code.rcx, code.r10);
	code.shr(code.rcx, static_cast<int>(lookup->region_shift));
	code.cmp(code.rcx, static_cast<uint32_t>(lookup->region_count));
	code.jae(load, Near);
	code.mov(code.r11, code.qword[code.r11 + code.rcx * 8]);
	code.test(code.r11, code.r11);
	code.jz(load, Near);
	code.mov(code.rcx, code.r10);
	code.shr(code.rcx, static_cast<int>(lookup->page_shift));
	code.and_(code.ecx, lookup->page_mask);
	const auto page = code.rcx * static_cast<int>(lookup->page_stride);
	code.test(code.byte[code.r11 + page + lookup->pages_offset], lookup->access_mask);
	code.jnz(slow, Near);
	code.L(load);
	code.mov(code.eax, code.dword[code.r10]);
	code.jmp(done, Near);
	code.L(slow);
	EmitGuestSlowCall(code, optional);
	code.L(done);
}

// AbSwitch SRT_WALKER_FAST: environment switches read once.
// KYTY_VERIFY_WALKER=1 runs the reference and the fast walker on every walk and exits on a
// difference; KYTY_WALKER_PREFETCH=1 adds the fast walker's entry prefetches.
bool EnvFlag(const char* name) {
	const char* value = std::getenv(name);
	return value != nullptr && *value == '1';
}

bool VerifyWalkerEnabled() {
	static const bool enabled = EnvFlag("KYTY_VERIFY_WALKER");
	return enabled;
}

bool WalkerPrefetchEnabled() {
	static const bool enabled = EnvFlag("KYTY_WALKER_PREFETCH");
	return enabled;
}

// Whether a walker compiled now may run its fast variant: the A/B harness (KYTY_AB_FILE) may set
// SRT_WALKER_FAST at any time, the verify mode runs both; otherwise the mask is fixed at startup.
bool FastWalkerWanted() {
	static const bool forced = [] {
		const char* ab_file = std::getenv("KYTY_AB_FILE");
		return VerifyWalkerEnabled() || (ab_file != nullptr && *ab_file != '\0');
	}();
	return forced || Common::AbSwitch::Enabled(Common::AbSwitch::SRT_WALKER_FAST);
}

class RuntimeValidator {
public:
	explicit RuntimeValidator(const ResourcePlan& program, RuntimeValueType type,
	                          bool wave_uniform = false)
	    : m_program(program), m_type(type), m_wave_uniform(wave_uniform) {}

	bool Run(Value value) { return Validate(value); }

private:
	bool ValidateArguments(const Inst& inst, bool require_uniform) {
		for (size_t index = 0; index < inst.NumArgs(); index++) {
			if (!Validate(inst.Arg(index), require_uniform)) return false;
		}
		return true;
	}

	bool Validate(Value value, bool require_uniform = true) {
		value = value.Resolve();
		// Host floating-point evaluation does not model shader rounding/denormal modes.
		if (m_type != RuntimeValueType::Any &&
		    TypesOverlap(value.GetType(), Type::F16 | Type::F32 | Type::F32x2)) {
			return false;
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			if (!require_uniform) return true;
			switch (value.GetType()) {
				case Type::U1:
				case Type::U8:
				case Type::U16:
				case Type::U32:
				case Type::U64:
				case Type::F32: return true;
				default: return false;
			}
		}
		if (require_uniform && !m_active_mask.IsEmpty() && value == m_active_mask) return true;
		// Integer-only dependency checks do not depend on the active EXEC mask.
		if (!require_uniform && m_validated_dependencies.contains(inst)) return true;
		if (!m_visiting.insert(inst).second) {
			return !require_uniform;
		}
		const auto finish = [&](bool valid) {
			m_visiting.erase(inst);
			if (valid && !require_uniform) m_validated_dependencies.insert(inst);
			return valid;
		};
		const auto op = inst->GetOpcode();
		if (m_type == RuntimeValueType::ImmutableInteger &&
		    (op == ValueOpcode::ReadConst || BufferAccessOf(op) != BufferAccess::None ||
		     AddressOpcodeInfoOf(op).access != AddressAccess::None ||
		     ImageOpcodeInfoOf(op).access != ImageAccess::None ||
		     SharedAccessOf(op) != SharedAccess::None)) {
			return finish(false);
		}
		if (op == ValueOpcode::ReadConst) {
			const auto slot = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (inst->NumArgs() != 2 || inst->Arg(0).Resolve().TryInstruction() == nullptr ||
			    inst->Arg(0).Resolve().TryInstruction()->GetOpcode() !=
			        ValueOpcode::GetSrtResource ||
			    !slot.IsImmediate() || slot.GetType() != Type::U32 ||
			    slot.U32() >= m_program.srt_reads.size()) {
				return finish(false);
			}
			if (m_type != RuntimeValueType::Any) {
				const auto active_mask = m_active_mask;
				m_active_mask          = {};
				const bool valid       = Validate(m_program.srt_reads[slot.U32()].value);
				m_active_mask          = active_mask;
				if (!valid) return finish(false);
			}
		}
		if (!require_uniform) return finish(ValidateArguments(*inst, false));
		if (!m_active_mask.IsEmpty() && IsRuntimeSelect(op) && inst->NumArgs() == 3 &&
		    inst->Arg(0).Resolve() == m_active_mask) {
			// Empty EXEC reads lane zero, so ignored operands still require integer types.
			if (m_type != RuntimeValueType::Any && !Validate(inst->Arg(2), false)) {
				return finish(false);
			}
			return finish(Validate(inst->Arg(1)));
		}
		if (op == ValueOpcode::UndefU1 || op == ValueOpcode::UndefU32 || op == ValueOpcode::Void) {
			return finish(false);
		}
		if (op == ValueOpcode::GetUserData) {
			if (inst->NumArgs() != 1 || inst->Arg(0).GetType() != Type::ScalarReg) {
				return finish(false);
			}
			const auto reg = RegIndex(inst->Arg(0).ScalarRegister());
			if (reg < m_program.user_data_base ||
			    reg - m_program.user_data_base >= m_program.user_data_count) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::GetShaderBase) {
			if (inst->NumArgs() != 0) {
				return finish(false);
			}
			return finish(true);
		}
		if (m_wave_uniform && op == ValueOpcode::GetBuiltin) {
			const auto axis = inst->NumArgs() == 2u ? inst->Arg(1).Resolve() : Value {};
			return finish(m_program.stage == ShaderType::Compute && inst->NumArgs() == 2u &&
			              inst->Arg(0) == Value(static_cast<uint32_t>(StageInputKind::WorkgroupId)) &&
			              axis.IsImmediate() && axis.GetType() == Type::U32 && axis.U32() < 3u);
		}
		if (op == ValueOpcode::Phi) {
			if (m_type != RuntimeValueType::Any && !ValidateArguments(*inst, false)) {
				return finish(false);
			}
			const auto invariant = ResolveInvariantPhi(m_program, value);
			if (invariant.IsEmpty()) {
				return finish(false);
			}
			return finish(Validate(invariant));
		}
		if (op == ValueOpcode::ReadFirstLane) {
			if (inst->NumArgs() != 2 || inst->Arg(0).GetType() != Type::U32 ||
			    inst->Arg(1).GetType() != Type::U1) {
				return finish(false);
			}
			if (m_type != RuntimeValueType::Any && !Validate(inst->Arg(1), false)) {
				return finish(false);
			}
			const auto active_mask = m_active_mask;
			m_active_mask          = inst->Arg(1).Resolve();
			const bool valid       = Validate(inst->Arg(0));
			m_active_mask          = active_mask;
			return finish(valid);
		}
		if (op == ValueOpcode::GetSrtResource) {
			if (inst->NumArgs() != 0) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::LoadAddressU32 || op == ValueOpcode::ReadConstBuffer ||
		    op == ValueOpcode::LoadBufferU32) {
			const auto  expected = op == ValueOpcode::LoadAddressU32
			                           ? ValueOpcode::GetAddressResource
			                           : ValueOpcode::GetBufferResource;
			const auto* handle = inst->NumArgs() != 0 ? inst->Arg(0).ResolveInstruction() : nullptr;
			if (!IsRawRead(m_program, *inst) || handle == nullptr ||
			    handle->GetOpcode() != expected) {
				return finish(false);
			}
			if (op == ValueOpcode::LoadBufferU32) {
				const auto guard = inst->Arg(4).Resolve();
				return finish(Validate(inst->Arg(0)) &&
				              ((!m_active_mask.IsEmpty() && guard == m_active_mask) || Validate(guard)));
			}
		} else if (op == ValueOpcode::CompositeExtractU64) {
			const auto index = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (!index.IsImmediate() || index.GetType() != Type::U32 || index.U32() >= 2u) {
				return finish(false);
			}
		} else if (op == ValueOpcode::CompositeExtractU32x2) {
			const auto* source = inst->NumArgs() == 2 ? inst->Arg(0).ResolveInstruction() : nullptr;
			const auto  index  = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (source == nullptr || !index.IsImmediate() || index.GetType() != Type::U32 ||
			    index.U32() >= 2u || source->GetOpcode() != ValueOpcode::CompositeConstructU32x2) {
				return finish(false);
			}
		}
		if (IsDescriptorHandle(op)) {
			size_t expected = 4u;
			if (op == ValueOpcode::GetImageResource) {
				expected = 8u;
			} else if (op == ValueOpcode::GetAddressResource) {
				expected = 2u;
			}
			if (inst->NumArgs() != expected) {
				return finish(false);
			}
		} else if (op != ValueOpcode::ReadConst && op != ValueOpcode::ReadConstBuffer &&
		           op != ValueOpcode::LoadAddressU32 && !IsRuntimeUniformOp(op)) {
			return finish(false);
		}
		return finish(ValidateArguments(*inst, true));
	}

	const ResourcePlan&             m_program;
	RuntimeValueType                m_type;
	bool                            m_wave_uniform;
	Value                           m_active_mask;
	std::unordered_set<const Inst*> m_visiting;
	std::unordered_set<const Inst*> m_validated_dependencies;
};

// Emits one x86 function per plan. Each value, per EXEC-mask context, is a memoized thunk with a
// state byte (0 unevaluated, 1 unknown or in progress, 2 known, 3 a control Phi holding its first
// value) and a qword slot; a reference is `call thunk; cmp state, 2`. A value the walk body already
// referenced is tested first and its call runs out of line. Operands that are unknown, constant or
// plain user data are folded at emission. Registers: rdi user data, rsi flat buffer, rdx scratch
// plus m_bias, r8 capture cursor.
// The walker's code plus what AbSwitch SRT_PREWALK_MEMO needs to know about its inputs.
struct WalkerCode final : Xbyak::CodeGenerator {
	WalkerCode(): Xbyak::CodeGenerator(4096, Xbyak::AutoGrow) {}
	uint64_t user_data_mask       = 0;     // user-data DWORDs the walker reads
	bool     user_data_mask_valid = false; // all of them are below 64
	// AbSwitch SRT_WALKER_FAST: code offset of the fast variant's entry (0: not compiled). The
	// reference variant starts at offset 0 of the same buffer, so SrtWalkerLoadLength covers both.
	size_t fast_entry = 0;
};

// The fast variant (AbSwitch SRT_WALKER_FAST) emits the same thunks, states, slots, captures and
// scratch stores as the reference, so both produce identical scratch, flat SRT and captures. It
// differs in:
// - rbx (saved at entry) holds the page number (address >> page_shift) of the last guest page this
//   walk found readable without the slow path, or -1. A load from that page skips the lookup. Only
//   verdicts that cannot change during the walk are kept: the load check is off, or the page's
//   region exists and its access-watcher byte is zero (read watchers are only added on the GPU
//   thread, which is running this walk; every slow-path call forgets the page anyway). Pages that
//   are cached lie below region_count << region_shift (2^44), so a hit also passes
//   LoadAddressU32's 48-bit test, which therefore moves to the miss path.
// - The lookup and the slow path run out of line; the guest load stays mov eax, [r10] in the thunk.
// - State bytes and the flat SRT are zeroed with SSE stores (WalkSrt resizes the flat SRT only).
// - KYTY_WALKER_PREFETCH=1: prefetcht0 of every 16-byte group of constant offsets read from each
//   user-data base pair, right after the base is formed (no architectural effect).
class WalkerCompiler {
public:
	WalkerCompiler(ResourcePlan& plan, Xbyak::CodeGenerator& code, bool fast = false)
	    : m_plan(plan), m_code(code), m_fast(fast) {}

	void Compile(std::span<const Value> values);

private:
	enum class Kind : uint8_t { Unknown, Constant, UserData, ShaderBase, Node };
	struct Operand {
		Kind     kind  = Kind::Unknown;
		uint64_t value = 0;

		bool operator==(const Operand& other) const = default;
	};
	// masked: the operand depends on its context's EXEC mask, so it is not the root context's.
	struct Ref {
		Operand operand;
		bool    masked = false;
	};
	struct Node {
		const Inst*          inst    = nullptr;
		uint32_t             context = 0;
		Operand              result;
		std::vector<Operand> args;
		uint32_t             slot = 0, state = 0;
		Xbyak::Label         label;
		bool                 pending = true, referenced = false, masked = false, called = false;
		// Only block conditions use the value, so it may read as unknown (EmitRead).
		bool optional = false;
	};
	// An out-of-line thunk call for a walk body reference to a value the body referenced before.
	struct Stub {
		Operand             operand;
		const Xbyak::Label* unknown = nullptr;
		Xbyak::Label        entry, back;
	};
	// The fast variant's out-of-line page lookup of one guest load: miss is entered with r10 = the
	// address; it continues at load (the thunk's mov eax, [r10]) or, after the slow path, at done.
	struct ColdLoad {
		Xbyak::Label miss, load, done;
		bool         optional = false;
		bool         check48  = false; // LoadAddressU32's 48-bit test, implied on the hot path
	};

	static constexpr auto Near = Xbyak::CodeGenerator::T_NEAR;

	// The scratch address at offset; offsets m_reach to m_reach + 255 take 8-bit displacements.
	Xbyak::RegExp At(size_t offset) const { return m_code.rdx - m_bias + offset; }

	bool Thunk(uint32_t index) const {
		const auto& node = m_nodes[index];
		return node.referenced || node.result == Operand {Kind::Node, index};
	}
	static bool MayBeUnknown(const Operand& operand) {
		return operand.kind == Kind::Unknown || operand.kind == Kind::Node;
	}

	uint32_t Context(Value mask) {
		mask = mask.Resolve();
		const auto found = std::ranges::find(m_masks, mask);
		if (found != m_masks.end()) return static_cast<uint32_t>(found - m_masks.begin());
		m_masks.push_back(mask);
		return static_cast<uint32_t>(m_masks.size() - 1u);
	}

	void AddOutput(Value value, uint32_t gate);
	void Consume(std::vector<std::vector<uint32_t>>& gates);
	Ref Evaluate(Value value, uint32_t context);
	Operand Analyze(uint32_t index);
	Operand AnalyzeRead(Node& node, const auto& arg, Operand self);
	void Layout();
	void Gate(std::vector<uint32_t>& gates, const Xbyak::Label& skip);
	void Zero(uint32_t offset, uint32_t size);
	void Branch(const Operand& operand, const Xbyak::Label& unknown);
	void Load(const Xbyak::Reg64& reg, const Operand& operand);
	void Base(const Operand& low, const Operand& high);
	void Fetch(const Operand& operand, const Xbyak::Label& unknown);
	void Set(void (Xbyak::CodeGenerator::*condition)(const Xbyak::Operand&));
	void Store(const Node& node);
	void EmitNode(uint32_t index);
	void EmitRead(const Node& node);
	void EmitStrict(const Node& node);
	// Fast variant (m_fast) helpers.
	template <class Address>
	void ZeroFast(const Address& at, uint32_t offset, uint32_t size);
	void LoadGuestFast(bool optional, bool check48);
	void EmitColdLoads();
	void EmitPrefetches(uint64_t low, uint64_t high);
	// Whether a cached page (below region_count << region_shift) implies a clear bit 48+ address.
	static bool CachedPageImplies48();

	ResourcePlan&                                     m_plan;
	Xbyak::CodeGenerator&                             m_code;
	bool                                              m_fast = false;
	std::deque<ColdLoad>                              m_cold_loads;
	uint32_t                                          m_prefetches = 0;
	std::vector<Value>                                m_masks {Value {}};
	std::deque<Node>                                  m_nodes;
	std::map<std::pair<const Inst*, uint32_t>, uint32_t> m_memo;
	std::vector<Value>                                m_outputs;
	std::vector<std::vector<uint32_t>>                m_output_gates;
	// Per user-data base pair of a read, the scratch slot holding the base for the walk.
	std::map<std::pair<uint64_t, uint64_t>, uint32_t> m_bases;
	std::deque<Stub>                                  m_stubs;
	Xbyak::Label                                      m_ret;
	uint32_t m_shader_base = 0, m_reach = 0, m_states = 0, m_states_end = 0, m_bias = 0;
	uint32_t m_user_data = 0, m_captures = 0;
	uint64_t m_user_mask = 0;     // user-data DWORDs Load reads (bit i = DWORD i), when all are < 64
	bool     m_user_wide = false; // a read DWORD is >= 64: m_user_mask is incomplete
	bool     m_body = true;
};

void WalkerCompiler::AddOutput(Value value, uint32_t gate) {
	const auto* inst = value.Resolve().TryInstruction();
	if (inst == nullptr) return;
	auto index = inst->WalkOutput();
	if (index >= m_outputs.size() || m_outputs[index].TryInstruction() != inst) {
		index = static_cast<uint32_t>(m_outputs.size());
		inst->SetWalkOutput(index);
		m_outputs.push_back(value.Resolve());
		m_output_gates.emplace_back();
	}
	m_output_gates[index].push_back(gate);
}

// Mirrors MaterializeResources: each descriptor it reads back is gated by the activity of the
// resource root that needs it, and so are an indirect root's selector values.
void WalkerCompiler::Consume(std::vector<std::vector<uint32_t>>& gates) {
	const auto& sources = m_plan.descriptor_sources;
	const auto  consume = [&](uint32_t root) {
		if (root >= sources.size()) return;
		const auto& indirect = sources[root].indirect_descriptor;
		if (!indirect) {
			gates[root].push_back(root);
			return;
		}
		const auto add = [&](uint32_t source) {
			if (source < sources.size()) gates[source].push_back(root);
		};
		if (indirect->sources.empty()) {
			add(indirect->table_source);
			if (indirect->selector) {
				add(indirect->selector->source);
				if (indirect->selector->rows) add(indirect->selector->rows->source);
			}
		}
		for (const auto source: indirect->sources) add(source);
		AddOutput(indirect->key_count, root);
		AddOutput(indirect->selector_first, root);
		AddOutput(indirect->selector_mask, root);
	};
	for (const auto& buffer: m_plan.info.buffers) consume(buffer.source);
	for (const auto& image: m_plan.info.images) consume(image.source);
	for (const auto& sampler: m_plan.info.samplers) {
		const auto* indirect = sampler.source < sources.size() && sources[sampler.source].indirect_descriptor
		    ? &*sources[sampler.source].indirect_descriptor : nullptr;
		if (indirect == nullptr || indirect->sources.empty()) {
			consume(sampler.source);
		} else if (indirect->sources[0] < sources.size()) {
			gates[indirect->sources[0]].push_back(indirect->sources[0]);
		}
	}
	const auto& fill = m_plan.uniform_fill;
	if (fill.fill.words != 0) {
		AddOutput(fill.pattern_size, UINT32_MAX);
		AddOutput(fill.element_count, UINT32_MAX);
		for (const auto value: fill.values) AddOutput(value, UINT32_MAX);
	}
}

WalkerCompiler::Ref WalkerCompiler::Evaluate(Value value, uint32_t context) {
	value = value.Resolve();
	if (value.IsEmpty()) return {};
	if (value.IsImmediate()) {
		uint64_t bits = 0;
		return ImmediateBits(value, bits) ? Ref {{Kind::Constant, bits}} : Ref {};
	}
	const auto mask = m_masks[context];
	if (context != 0 && value == mask) return {{Kind::Constant, 1u}, true};
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) return {};
	const auto op = inst->GetOpcode();
	if (context != 0 && IsRuntimeSelect(op) && inst->NumArgs() == 3 && inst->Arg(0).Resolve() == mask) {
		return {Evaluate(inst->Arg(1), context).operand, true};
	}
	switch (op) {
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32:
		case ValueOpcode::ConditionRef: return Evaluate(inst->Arg(0), context);
		case ValueOpcode::CompositeExtractU32x2: {
			const auto* source = inst->Arg(0).ResolveInstruction();
			const auto  index  = inst->Arg(1).Resolve();
			if (source == nullptr || source->GetOpcode() != ValueOpcode::CompositeConstructU32x2 ||
			    !index.IsImmediate() || index.GetType() != Type::U32 || index.U32() >= 2u) return {};
			return Evaluate(source->Arg(index.U32()), context);
		}
		case ValueOpcode::ReadFirstLane:
			// The lane read does not depend on the EXEC mask it is read under.
			return {Evaluate(inst->Arg(0), Context(inst->Arg(1))).operand};
		case ValueOpcode::GetUserData: {
			const auto reg = RegIndex(inst->Arg(0).ScalarRegister());
			if (reg < m_plan.user_data_base || reg - m_plan.user_data_base >= m_plan.user_data_count) return {};
			return {{Kind::UserData, reg - m_plan.user_data_base}};
		}
		case ValueOpcode::GetShaderBase: return {{Kind::ShaderBase}};
		case ValueOpcode::Phi:
			if (!m_plan.control_phi_predecessors.contains(inst)) {
				const auto invariant = ResolveInvariantPhi(m_plan, value);
				return invariant.IsEmpty() ? Ref {} : Evaluate(invariant, context);
			}
			break;
		default: break;
	}
	const auto [found, inserted] = m_memo.try_emplace({inst, context}, static_cast<uint32_t>(m_nodes.size()));
	const auto index = found->second;
	if (!inserted) {
		auto& node = m_nodes[index];
		if (!node.pending) return {node.result, node.masked};
		// A cycle: the thunk reads as unknown while it runs.
		node.referenced = true;
		return {{Kind::Node, index}, context != 0};
	}
	auto& node   = m_nodes.emplace_back();
	node.inst    = inst;
	node.context = context;
	node.result  = Analyze(index);
	node.pending = false;
	if (context != 0 && !node.masked && !node.referenced) {
		node.result = Evaluate(value, 0).operand;
		return {node.result};
	}
	return {node.result, node.masked};
}

WalkerCompiler::Operand WalkerCompiler::Analyze(uint32_t index) {
	auto&       node = m_nodes[index];
	const auto& inst = *node.inst;
	const auto  self = Operand {Kind::Node, index};
	const auto  arg  = [&](Value value) {
		const auto ref = Evaluate(value, node.context);
		node.masked |= ref.masked;
		return ref.operand;
	};
	const auto constant = [](uint64_t value) { return Operand {Kind::Constant, value}; };
	switch (inst.GetOpcode()) {
		case ValueOpcode::Phi: {
			bool known = false;
			for (size_t i = 0; i < inst.NumArgs(); ++i) {
				node.args.push_back(arg(inst.Arg(i)));
				known |= node.args.back().kind != Kind::Unknown;
			}
			return known ? self : Operand {};
		}
		case ValueOpcode::LoadAddressU32:
		case ValueOpcode::ReadConstBuffer:
		case ValueOpcode::LoadBufferU32: return AnalyzeRead(node, arg, self);
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectF32: {
			const auto predicate = arg(inst.Arg(0));
			if (predicate.kind == Kind::Unknown) return {};
			if (predicate.kind == Kind::Constant) return arg(inst.Arg(predicate.value != 0u ? 1 : 2));
			node.args = {predicate, arg(inst.Arg(1)), arg(inst.Arg(2))};
			return node.args[1].kind == Kind::Unknown && node.args[2].kind == Kind::Unknown ? Operand {} : self;
		}
		case ValueOpcode::ULessThan32:
		case ValueOpcode::ULessThanEqual32: {
			// No U32 is below zero and every U32 is at least zero, including an unknown loop-carried
			// or varying lane value. args[0] is that zero-tested bound.
			const bool less  = inst.GetOpcode() == ValueOpcode::ULessThan32;
			const auto bound = arg(inst.Arg(less ? 1 : 0));
			if (bound.kind == Kind::Unknown) return {};
			if (bound.kind == Kind::Constant && static_cast<uint32_t>(bound.value) == 0u) return constant(!less);
			node.args = {bound, arg(inst.Arg(less ? 0 : 1))};
			return node.args[1].kind == Kind::Unknown && bound.kind == Kind::Constant ? Operand {} : self;
		}
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalAnd: {
			// A deciding operand, nonzero for Or and zero for And, fixes the result.
			const bool conjunction = inst.GetOpcode() == ValueOpcode::LogicalAnd;
			const auto decides     = [&](const Operand& x) {
				return x.kind == Kind::Constant && (x.value == 0u) == conjunction;
			};
			const auto a = arg(inst.Arg(0));
			if (decides(a)) return constant(!conjunction);
			const auto b = arg(inst.Arg(1));
			if (decides(b)) return constant(!conjunction);
			if (a.kind == Kind::Constant && b.kind == Kind::Constant) return constant(conjunction);
			if (a.kind == Kind::Constant && b.kind == Kind::Unknown) return {};
			// Only lanes satisfying the left EXEC factor can make a conjunction true, so an
			// unknown left side retries the right one under that factor as its mask.
			Operand refined;
			if (conjunction && MayBeUnknown(a)) {
				refined = Evaluate(inst.Arg(1), Context(inst.Arg(0))).operand;
				if (refined == b || (refined.kind == Kind::Constant && refined.value != 0u)) refined = {};
				if (a.kind == Kind::Unknown && refined.kind == Kind::Constant) return constant(0u);
			}
			if (a.kind == Kind::Unknown && refined.kind == Kind::Unknown &&
			    (b.kind == Kind::Unknown || b.kind == Kind::Constant)) return {};
			node.args = {a, b, refined};
			return self;
		}
		case ValueOpcode::CompositeExtractU64: {
			const auto component = inst.Arg(1).Resolve();
			if (!component.IsImmediate() || component.GetType() != Type::U32 || component.U32() >= 2u) return {};
			[[fallthrough]];
		}
		default:
			// The other runtime-uniform ops are strict; a U32x2 pair is only read through its extracts.
			if (!IsRuntimeUniformOp(inst.GetOpcode()) || inst.GetOpcode() == ValueOpcode::CompositeConstructU32x2)
				return {};
			for (size_t i = 0; i < inst.NumArgs(); ++i) {
				node.args.push_back(arg(inst.Arg(i)));
				if (node.args.back().kind == Kind::Unknown) return {};
			}
			return self;
	}
}

WalkerCompiler::Operand WalkerCompiler::AnalyzeRead(Node& node, const auto& arg, Operand self) {
	const auto& inst = *node.inst;
	const auto* handle = inst.Arg(0).ResolveInstruction();
	if (!IsRawRead(m_plan, inst) || handle == nullptr) return {};
	const bool vector = inst.GetOpcode() == ValueOpcode::LoadBufferU32;
	if (vector) {
		const auto guard = arg(inst.Arg(4));
		if (guard.kind == Kind::Unknown) return {};
		if (guard.kind == Kind::Constant && guard.value == 0u) return {Kind::Constant, 0u};
		node.args.push_back(guard);
	}
	const bool buffer = inst.GetOpcode() != ValueOpcode::LoadAddressU32;
	const auto offset = static_cast<int32_t>(m_plan.memory_info[inst.Flags<MemoryFlags>().index].offset);
	bool failed = buffer && (handle->NumArgs() != 4u || offset < 0);
	for (const auto& word: {handle->Arg(0), handle->Arg(1), inst.Arg(1)}) {
		node.args.push_back(arg(word));
		failed |= node.args.back().kind == Kind::Unknown;
	}
	for (uint32_t word = 2; buffer && !failed && word < 4u; ++word) {
		node.args.push_back(arg(handle->Arg(word)));
		failed |= node.args.back().kind == Kind::Unknown;
	}
	if (!failed) return self;
	// A disabled vector lane still reads zero.
	if (!vector || node.args[0].kind == Kind::Constant) return {};
	node.args.resize(1);
	return self;
}

void WalkerCompiler::Layout() {
	uint32_t offset = OutputOffset(m_plan) + static_cast<uint32_t>(m_outputs.size()) * 8u;
	m_shader_base = offset;
	offset += 8u;
	for (uint32_t index = 0; index < m_nodes.size(); ++index) {
		if (!Thunk(index)) continue;
		auto& node = m_nodes[index];
		node.slot  = offset;
		offset += 8u;
		// EmitRead forms a read's base from the two operands after LoadBufferU32's leading test.
		const auto   op   = node.inst->GetOpcode();
		const size_t skip = op == ValueOpcode::LoadBufferU32 ? 1u : 0u;
		const bool read = op == ValueOpcode::LoadAddressU32 || op == ValueOpcode::ReadConstBuffer ||
		                  op == ValueOpcode::LoadBufferU32;
		if (read && node.result == Operand {Kind::Node, index} && node.args.size() > skip + 2u &&
		    node.args[skip].kind == Kind::UserData && node.args[skip + 1].kind == Kind::UserData) {
			m_bases.try_emplace({node.args[skip].value, node.args[skip + 1].value}, 0u);
		}
	}
	m_reach = offset;
	m_bias  = m_reach + 128u;
	offset += static_cast<uint32_t>(m_plan.control_flow.size());
	m_states = offset;
	for (uint32_t index = 0; index < m_nodes.size(); ++index) {
		if (Thunk(index)) m_nodes[index].state = offset++;
	}
	m_states_end = offset;
	offset       = (offset + 7u) & ~7u;
	for (auto& [pair, slot]: m_bases) {
		slot = offset;
		offset += 8u;
	}
	// Captured addresses come last; EmitRead counts them.
	m_plan.srt_captures = offset;
}

// Falls through when any gating source is active. The list is not empty.
void WalkerCompiler::Gate(std::vector<uint32_t>& gates, const Xbyak::Label& skip) {
	auto& c = m_code;
	std::ranges::sort(gates);
	if (gates.back() == UINT32_MAX) return;
	gates.erase(std::unique(gates.begin(), gates.end()), gates.end());
	Xbyak::Label active;
	for (size_t i = 0; i < gates.size(); ++i) {
		c.cmp(c.byte[At(ActiveOffset(m_plan) + gates[i])], 0);
		if (i + 1u < gates.size()) {
			c.jne(active, Near);
		} else {
			c.je(skip, Near);
		}
	}
	c.L(active);
}

void WalkerCompiler::Zero(uint32_t offset, uint32_t size) {
	auto& c = m_code;
	if (size == 0) return;
	if (m_fast && size <= 512u) {
		ZeroFast([this](uint32_t at) { return At(at); }, offset, size);
		return;
	}
	c.mov(c.r11, c.rdi);
	c.lea(c.rdi, c.ptr[At(offset)]);
	c.mov(c.ecx, size);
	c.xor_(c.eax, c.eax);
	c.rep();
	c.stosb();
	c.mov(c.rdi, c.r11);
}

// Zeroes exactly [offset, offset + size) of the memory that at(offset) addresses; the last store
// may overlap the one before it. Clobbers rax and xmm0.
template <class Address>
void WalkerCompiler::ZeroFast(const Address& at, uint32_t offset, uint32_t size) {
	auto& c = m_code;
	if (size == 0) return;
	if (size >= 16u) {
		c.pxor(c.xmm0, c.xmm0);
		uint32_t done = 0;
		for (; done + 16u <= size; done += 16u) c.movdqu(c.xword[at(offset + done)], c.xmm0);
		if (done != size) c.movdqu(c.xword[at(offset + size - 16u)], c.xmm0);
		return;
	}
	c.xor_(c.eax, c.eax);
	if (size >= 8u) {
		c.mov(c.qword[at(offset)], c.rax);
		if (size > 8u) c.mov(c.qword[at(offset + size - 8u)], c.rax);
	} else if (size >= 4u) {
		c.mov(c.dword[at(offset)], c.eax);
		if (size > 4u) c.mov(c.dword[at(offset + size - 4u)], c.eax);
	} else if (size >= 2u) {
		c.mov(c.word[at(offset)], c.ax);
		if (size > 2u) c.mov(c.word[at(offset + size - 2u)], c.ax);
	} else {
		c.mov(c.byte[at(offset)], c.al);
	}
}

bool WalkerCompiler::CachedPageImplies48() {
	const auto* lookup = PageManager::AccessLookup();
	return lookup->page_shift <= lookup->region_shift && lookup->region_shift <= 48u &&
	       lookup->region_count <= (uint64_t {1} << (48u - lookup->region_shift));
}

// The fast variant's guest load, r10 = the address: on the cached page, mov eax, [r10] (41 8b 02,
// the form SrtWalkerLoadLength and the fault handler complete); otherwise the out-of-line lookup.
void WalkerCompiler::LoadGuestFast(bool optional, bool check48) {
	auto&       c      = m_code;
	const auto* lookup = PageManager::AccessLookup();
	auto&       cold   = m_cold_loads.emplace_back();
	cold.optional      = optional;
	cold.check48       = check48;
	c.mov(c.rcx, c.r10);
	c.shr(c.rcx, static_cast<int>(lookup->page_shift));
	c.cmp(c.rcx, c.rbx);
	c.jne(cold.miss, Near);
	c.L(cold.load);
	c.mov(c.eax, c.dword[c.r10]);
	c.L(cold.done);
}

// The out-of-line part of LoadGuestFast, after every thunk. Same decisions as LoadGuest: a direct
// load unless the load check is on, the PageManager exists, the address is in its table, the region
// exists and the page has a read watcher (then the slow-path call). The region range test comes
// first because a cached page must lie in the table; it only reorders tests that all lead to the
// direct load.
void WalkerCompiler::EmitColdLoads() {
	auto&       c      = m_code;
	const auto* lookup = PageManager::AccessLookup();
	const int   shift  = static_cast<int>(lookup->page_shift);
	for (auto& cold: m_cold_loads) {
		Xbyak::Label slow, cache;
		c.L(cold.miss);
		if (cold.check48) {
			// LoadAddressU32's 48-bit test (EmitRead), unknown when it fails: the jump returns from
			// the thunk, whose frame this code still runs in.
			c.mov(c.rcx, c.r10);
			c.shr(c.rcx, 48);
			c.jnz(m_ret, Near);
		}
		c.mov(c.rcx, c.r10);
		c.shr(c.rcx, static_cast<int>(lookup->region_shift));
		c.cmp(c.rcx, static_cast<uint32_t>(lookup->region_count));
		c.jae(cold.load, Near);
		c.mov(c.r11, reinterpret_cast<uint64_t>(&g_srt_load_check));
		c.cmp(c.byte[c.r11], 0);
		c.je(cache, Near);
		c.mov(c.r11, reinterpret_cast<uint64_t>(&lookup->regions));
		c.mov(c.r11, c.qword[c.r11]);
		c.test(c.r11, c.r11);
		c.jz(cold.load, Near);
		c.mov(c.r11, c.qword[c.r11 + c.rcx * 8]);
		c.test(c.r11, c.r11);
		c.jz(cold.load, Near);
		c.mov(c.rcx, c.r10);
		c.shr(c.rcx, shift);
		c.and_(c.ecx, lookup->page_mask);
		const auto page = c.rcx * static_cast<int>(lookup->page_stride);
		c.test(c.byte[c.r11 + page + lookup->pages_offset], lookup->access_mask);
		c.jnz(slow, Near);
		c.L(cache);
		c.mov(c.rbx, c.r10);
		c.shr(c.rbx, shift);
		c.jmp(cold.load, Near);
		c.L(slow);
		EmitGuestSlowCall(c, cold.optional);
		// The read helper may have changed page states: forget the cached page.
		c.mov(c.rbx, ~uint64_t {0});
		c.jmp(cold.done, Near);
	}
}

// KYTY_WALKER_PREFETCH: rax = the base of user-data pair (low, high). Prefetches each 16-byte group
// of the constant offsets that read thunks load from it (at most 32 per walker).
void WalkerCompiler::EmitPrefetches(uint64_t low, uint64_t high) {
	auto&                c = m_code;
	std::vector<int64_t> constants;
	for (uint32_t index = 0; index < m_nodes.size() && m_prefetches + constants.size() < 32u; ++index) {
		if (!Thunk(index)) continue;
		const auto& node = m_nodes[index];
		const auto  op   = node.inst->GetOpcode();
		if (op != ValueOpcode::LoadAddressU32 && op != ValueOpcode::ReadConstBuffer &&
		    op != ValueOpcode::LoadBufferU32) continue;
		const size_t skip = op == ValueOpcode::LoadBufferU32 ? 1u : 0u;
		if (node.result != Operand {Kind::Node, index} || node.args.size() <= skip + 2u ||
		    node.args[skip] != Operand {Kind::UserData, low} ||
		    node.args[skip + 1] != Operand {Kind::UserData, high} ||
		    node.args[skip + 2].kind != Kind::Constant) continue;
		const auto immediate =
		    static_cast<int64_t>(static_cast<int32_t>(m_plan.memory_info[node.inst->Flags<MemoryFlags>().index].offset)) &
		    ~int64_t {3};
		const auto constant = int64_t {static_cast<uint32_t>(node.args[skip + 2].value) & ~3u} + immediate;
		if (constant + 4 > INT32_MAX || constant < INT32_MIN) continue;
		if (std::ranges::none_of(constants, [&](int64_t other) { return (other >> 4) == (constant >> 4); }))
			constants.push_back(constant);
	}
	for (const auto constant: constants) c.prefetcht0(c.ptr[c.rax + static_cast<size_t>(constant)]);
	m_prefetches += static_cast<uint32_t>(constants.size());
}

// Jumps to unknown unless the operand has a value at run time.
void WalkerCompiler::Branch(const Operand& operand, const Xbyak::Label& unknown) {
	auto& c = m_code;
	if (operand.kind == Kind::Unknown) {
		c.jmp(unknown, Near);
	} else if (operand.kind == Kind::Node) {
		auto& node = m_nodes[operand.value];
		if (m_body && node.called) {
			auto& stub   = m_stubs.emplace_back();
			stub.operand = operand;
			stub.unknown = &unknown;
			c.cmp(c.byte[At(node.state)], 2);
			c.jne(stub.entry, Near);
			c.L(stub.back);
			return;
		}
		node.called |= m_body;
		c.call(node.label);
		c.cmp(c.byte[At(node.state)], 2);
		c.jne(unknown, Near);
	}
}

void WalkerCompiler::Load(const Xbyak::Reg64& reg, const Operand& operand) {
	auto& c = m_code;
	switch (operand.kind) {
		case Kind::Constant: c.mov(reg, operand.value); break;
		case Kind::UserData:
			m_user_data = std::max(m_user_data, static_cast<uint32_t>(operand.value) + 1u);
			if (operand.value < 64u) {
				m_user_mask |= uint64_t {1} << operand.value;
			} else {
				m_user_wide = true;
			}
			c.mov(reg.cvt32(), c.dword[c.rdi + operand.value * sizeof(uint32_t)]);
			break;
		case Kind::ShaderBase: c.mov(reg, c.qword[At(m_shader_base)]); break;
		case Kind::Node: c.mov(reg, c.qword[At(m_nodes[operand.value].slot)]); break;
		case Kind::Unknown: break;
	}
}

// rax = the 48-bit base (high << 32 | low) with its low two bits clear.
void WalkerCompiler::Base(const Operand& low, const Operand& high) {
	auto& c = m_code;
	Load(c.rax, high);
	c.shl(c.rax, 48);
	c.shr(c.rax, 16);
	Load(c.rcx, low);
	c.and_(c.ecx, ~3u);
	c.or_(c.rax, c.rcx);
}

// Loads the operand into rax, or jumps to unknown when it has no value at run time.
void WalkerCompiler::Fetch(const Operand& operand, const Xbyak::Label& unknown) {
	Branch(operand, unknown);
	Load(m_code.rax, operand);
}

// Sets eax to 1 when the condition holds, else 0.
void WalkerCompiler::Set(void (Xbyak::CodeGenerator::*condition)(const Xbyak::Operand&)) {
	(m_code.*condition)(m_code.al);
	m_code.movzx(m_code.eax, m_code.al);
}

void WalkerCompiler::Store(const Node& node) {
	auto& c = m_code;
	c.mov(c.qword[At(node.slot)], c.rax);
	c.mov(c.byte[At(node.state)], 2);
	c.ret();
}

void WalkerCompiler::EmitNode(uint32_t index) {
	auto&       c     = m_code;
	auto&       node  = m_nodes[index];
	const auto& args  = node.args;
	const auto  state = c.byte[At(node.state)];
	c.L(node.label);
	c.cmp(state, 0);
	c.jne(m_ret, Near);
	c.mov(state, 1);
	if (node.result != Operand {Kind::Node, index}) {
		// A cycle referenced this value before it folded to another operand.
		Fetch(node.result, m_ret);
		Store(node);
		return;
	}
	switch (node.inst->GetOpcode()) {
		case ValueOpcode::Phi: {
			// Only predecessors reached by this walk contribute, and they must agree. The state is 3
			// once one of them has stored the value; callers read it like 1, as unknown.
			const auto& blocks = m_plan.control_phi_predecessors.at(node.inst);
			for (size_t i = 0; i < args.size(); ++i) {
				Xbyak::Label next, first;
				c.cmp(c.byte[At(m_reach + blocks[i])], 0);
				c.je(next, Near);
				Fetch(args[i], m_ret);
				c.cmp(state, 1);
				c.je(first, Near);
				c.cmp(c.rax, c.qword[At(node.slot)]);
				c.jne(m_ret, Near);
				c.jmp(next, Near);
				c.L(first);
				c.mov(c.qword[At(node.slot)], c.rax);
				c.mov(state, 3);
				c.L(next);
			}
			c.cmp(state, 3);
			c.jne(m_ret, Near);
			c.mov(state, 2);
			c.ret();
			return;
		}
		case ValueOpcode::LoadAddressU32:
		case ValueOpcode::ReadConstBuffer:
		case ValueOpcode::LoadBufferU32: EmitRead(node); return;
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectF32: {
			Xbyak::Label other;
			Fetch(args[0], m_ret);
			c.test(c.rax, c.rax);
			c.jz(other, Near);
			Fetch(args[1], m_ret);
			Store(node);
			c.L(other);
			Fetch(args[2], m_ret);
			Store(node);
			return;
		}
		case ValueOpcode::ULessThan32:
		case ValueOpcode::ULessThanEqual32: {
			const bool less = node.inst->GetOpcode() == ValueOpcode::ULessThan32;
			Xbyak::Label bound_zero;
			Fetch(args[0], m_ret);
			c.test(c.eax, c.eax);
			c.jz(bound_zero, Near);
			Branch(args[1], m_ret);
			Load(c.rax, args[0]);
			Load(c.rcx, args[1]);
			// ULT tests other < bound, ULE bound <= other.
			c.cmp(c.ecx, c.eax);
			Set(less ? &Xbyak::CodeGenerator::setb : &Xbyak::CodeGenerator::setae);
			Store(node);
			c.L(bound_zero);
			c.mov(c.eax, uint32_t {!less});
			Store(node);
			return;
		}
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalAnd: {
			const bool conjunction = node.inst->GetOpcode() == ValueOpcode::LogicalAnd;
			Xbyak::Label unknown_left, refine, decided;
			const auto decide = [&] {
				if (conjunction) {
					c.jz(decided, Near);
				} else {
					c.jnz(decided, Near);
				}
			};
			Branch(args[0], unknown_left);
			if (args[0].kind != Kind::Unknown) {
				Load(c.rax, args[0]);
				c.test(c.rax, c.rax);
				decide();
				Fetch(args[1], m_ret);
				c.test(c.rax, c.rax);
				Set(&Xbyak::CodeGenerator::setne);
				Store(node);
			}
			c.L(unknown_left);
			if (MayBeUnknown(args[0])) {
				// Without the left side only a deciding right side, or a conjunction's zero
				// refined right side, is known.
				if (args[1].kind != Kind::Unknown) {
					Fetch(args[1], refine);
					c.test(c.rax, c.rax);
					decide();
				}
				c.L(refine);
				Fetch(args[2], m_ret);
				c.test(c.rax, c.rax);
				c.jnz(m_ret, Near);
			}
			c.L(decided);
			c.mov(c.eax, uint32_t {!conjunction});
			Store(node);
			return;
		}
		default: EmitStrict(node); return;
	}
}

// Raw scalar and uniform DWORD buffer loads; out-of-bounds reads are zero.
void WalkerCompiler::EmitRead(const Node& node) {
	auto&      c    = m_code;
	const auto& inst = *node.inst;
	const auto op = inst.GetOpcode();
	const bool vector = op == ValueOpcode::LoadBufferU32;
	auto args = std::span(node.args);
	Xbyak::Label zero;
	if (vector) {
		Fetch(args[0], m_ret);
		c.test(c.rax, c.rax);
		c.jz(zero, Near);
		args = args.subspan(1);
		if (args.empty()) c.jmp(m_ret, Near);
	}
	if (!args.empty()) {
		for (const auto& arg: args) Branch(arg, m_ret);
		const auto immediate =
		    static_cast<int64_t>(static_cast<int32_t>(m_plan.memory_info[inst.Flags<MemoryFlags>().index].offset)) &
		    ~int64_t {3};
		// A constant offset folds into the displacement; r10 = the address.
		const auto constant = int64_t {static_cast<uint32_t>(args[2].value) & ~3u} + immediate;
		const bool folded   = args[2].kind == Kind::Constant && constant + 4 <= INT32_MAX;
		const auto offset   = folded ? Xbyak::RegExp(static_cast<size_t>(constant))
		                             : c.r9 + static_cast<size_t>(immediate);
		if (args[0].kind == Kind::UserData && args[1].kind == Kind::UserData) {
			c.mov(c.rax, c.qword[At(m_bases.at({args[0].value, args[1].value}))]);
		} else {
			Base(args[0], args[1]);
		}
		if (!folded) {
			Load(c.r9, args[2]);
			c.and_(c.r9d, ~3u);
		}
		c.lea(c.r10, c.ptr[c.rax + offset]);
		// The fast variant tests bit 48+ on the lookup's miss path only: a cached page implies it.
		const bool check48_on_miss = m_fast && op == ValueOpcode::LoadAddressU32 && CachedPageImplies48();
		if (op == ValueOpcode::LoadAddressU32) {
			if (!check48_on_miss) {
				c.mov(c.rcx, c.r10);
				c.shr(c.rcx, 48);
				c.jnz(m_ret, Near);
			}
		} else {
			Load(c.r11, args[1]);
			c.shr(c.r11d, 16);
			if (vector) {
				// Only the uniform, unswizzled structured DWORD address is evaluated on the host.
				c.test(c.r11d, 1u << 15u);
				c.jnz(m_ret, Near);
				Load(c.rcx, args[4]);
				c.test(c.ecx, (1u << 23u) | 0xf0000000u);
				c.jnz(m_ret, Near);
				c.test(c.r11d, 0x3fffu);
				c.jz(zero, Near);
				c.test(c.ecx, 0x7fu << 12u);
				c.jz(zero, Near);
			}
			c.and_(c.r11d, 0x3fffu);
			Load(c.rcx, args[3]);
			if (vector) {
				c.test(c.ecx, c.ecx);
				c.jz(zero, Near);
			}
			Xbyak::Label sized;
			c.test(c.r11d, c.r11d);
			c.jz(sized, Near);
			c.imul(c.rcx, c.r11);
			c.L(sized);
			c.lea(c.r11, c.ptr[offset + 4u]);
			c.cmp(c.r11, c.rcx);
			c.ja(zero, Near);
		}
		// The guest load (mov eax, [r10]) with the local SRT_LOAD_CHECK page test, see LoadGuest.
		if (m_fast) {
			LoadGuestFast(node.optional, check48_on_miss);
		} else {
			LoadGuest(c, c.r10, node.optional);
		}
		if (node.optional) {
			// Bit 32: unknown, a GPU write still in flight covers the DWORD (the state stays 1).
			c.bt(c.rax, 32);
			c.jc(m_ret, Near);
		}
		++m_captures;
		c.mov(c.qword[c.r8], c.r10);
		c.add(c.r8, 8);
		Store(node);
	}
	// Only buffer reads have a zero path.
	if (op == ValueOpcode::LoadAddressU32) return;
	c.L(zero);
	c.xor_(c.eax, c.eax);
	Store(node);
}

void WalkerCompiler::EmitStrict(const Node& node) {
	auto&       c    = m_code;
	const auto& inst = *node.inst;
	for (const auto& arg: node.args) Branch(arg, m_ret);
	const std::array regs {c.rax, c.rcx, c.r9, c.r11};
	for (size_t i = 0; i < node.args.size() && i < regs.size(); ++i) Load(regs[i], node.args[i]);
	// Field ops require offset + width <= 32. Leaves (1 << width) - 1 in r11, the offset in
	// cl and the former rcx (the offset or the inserted value) in r10.
	const auto field = [&](const Xbyak::Reg32& offset, const Xbyak::Reg32& width) {
		c.lea(c.r10, c.ptr[offset.cvt64() + width.cvt64()]);
		c.cmp(c.r10, 32);
		c.ja(m_ret, Near);
		c.mov(c.r10, c.rcx);
		c.mov(c.ecx, width);
		c.mov(c.r11d, 1);
		c.shl(c.r11, c.cl);
		c.dec(c.r11);
		c.mov(c.ecx, offset.getIdx() == Xbyak::Operand::RCX ? c.r10d : offset);
	};
	switch (inst.GetOpcode()) {
		case ValueOpcode::CompositeConstructU64:
			c.shl(c.rcx, 32);
			c.or_(c.rax, c.rcx);
			break;
		case ValueOpcode::CompositeExtractU64:
			if (inst.Arg(1).Resolve().U32() != 0u) c.shr(c.rax, 32);
			c.mov(c.eax, c.eax);
			break;
		case ValueOpcode::IAdd32: c.add(c.eax, c.ecx); break;
		case ValueOpcode::IAdd64: c.add(c.rax, c.rcx); break;
		case ValueOpcode::ISub32: c.sub(c.eax, c.ecx); break;
		case ValueOpcode::ISub64: c.sub(c.rax, c.rcx); break;
		case ValueOpcode::IMul32: c.imul(c.eax, c.ecx); break;
		case ValueOpcode::UMulHi:
			c.imul(c.rax, c.rcx);
			c.shr(c.rax, 32);
			break;
		case ValueOpcode::UMin32:
			c.cmp(c.eax, c.ecx);
			c.cmova(c.eax, c.ecx);
			break;
		case ValueOpcode::UMax32:
			c.cmp(c.eax, c.ecx);
			c.cmovb(c.eax, c.ecx);
			break;
		case ValueOpcode::SMin32:
			c.cmp(c.eax, c.ecx);
			c.cmovg(c.eax, c.ecx);
			break;
		case ValueOpcode::SMax32:
			c.cmp(c.eax, c.ecx);
			c.cmovl(c.eax, c.ecx);
			break;
		case ValueOpcode::ConvertF32U32:
			c.cvtsi2ss(c.xmm0, c.rax);
			c.movd(c.eax, c.xmm0);
			break;
		case ValueOpcode::ConvertU32F32: {
			// The conversion saturates and maps NaN to zero.
			Xbyak::Label done;
			c.movd(c.xmm0, c.eax);
			c.xorps(c.xmm1, c.xmm1);
			c.xor_(c.eax, c.eax);
			c.ucomiss(c.xmm0, c.xmm1);
			c.jbe(done, Near);
			c.mov(c.ecx, 0x4f800000u);
			c.movd(c.xmm1, c.ecx);
			c.mov(c.eax, UINT32_MAX);
			c.ucomiss(c.xmm0, c.xmm1);
			c.jae(done, Near);
			c.cvttss2si(c.rax, c.xmm0);
			c.L(done);
			break;
		}
		case ValueOpcode::ConvertS32F32: {
			// The conversion saturates and maps NaN to zero. cvttss2si's out-of-range result,
			// INT32_MIN, is already right for negative overflow.
			Xbyak::Label done;
			c.movd(c.xmm0, c.eax);
			c.xor_(c.eax, c.eax);
			c.ucomiss(c.xmm0, c.xmm0);
			c.jp(done, Near);
			c.mov(c.ecx, 0x4f000000u);
			c.movd(c.xmm1, c.ecx);
			c.mov(c.eax, INT32_MAX);
			c.ucomiss(c.xmm0, c.xmm1);
			c.jae(done, Near);
			c.cvttss2si(c.eax, c.xmm0);
			c.L(done);
			break;
		}
		case ValueOpcode::FPMul32:
			c.movd(c.xmm0, c.eax);
			c.movd(c.xmm1, c.ecx);
			c.mulss(c.xmm0, c.xmm1);
			c.movd(c.eax, c.xmm0);
			break;
		case ValueOpcode::FPTrunc32:
			c.movd(c.xmm0, c.eax);
			c.roundss(c.xmm0, c.xmm0, 3);
			c.movd(c.eax, c.xmm0);
			break;
		case ValueOpcode::FPRecipIFlag32:
			// Normal positive powers of two have exact normal reciprocals in every FP mode.
			c.test(c.eax, 0x807fffffu);
			c.jnz(m_ret, Near);
			c.shr(c.eax, 23);
			c.jz(m_ret, Near);
			c.cmp(c.eax, 254);
			c.jae(m_ret, Near);
			c.neg(c.eax);
			c.add(c.eax, 254);
			c.shl(c.eax, 23);
			break;
		case ValueOpcode::FPIsNan32:
			c.and_(c.eax, 0x7fffffffu);
			c.cmp(c.eax, 0x7f800000u);
			Set(&Xbyak::CodeGenerator::seta);
			break;
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
			if (inst.Flags<FPCompareFlags>().flush_input_denorms) {
				for (const auto& reg: {c.eax, c.ecx}) {
					Xbyak::Label normal;
					c.mov(c.r11d, reg);
					c.and_(c.r11d, 0x7fffffffu);
					c.cmp(c.r11d, 0x00800000u);
					c.jae(normal, Near);
					c.and_(reg, 0x80000000u);
					c.L(normal);
				}
			}
			c.movd(c.xmm0, c.eax);
			c.movd(c.xmm1, c.ecx);
			// Unordered operands set CF, so AE is false for NaN.
			if (inst.GetOpcode() == ValueOpcode::FPOrdLessThanEqual32) {
				c.ucomiss(c.xmm1, c.xmm0);
			} else {
				c.ucomiss(c.xmm0, c.xmm1);
			}
			Set(&Xbyak::CodeGenerator::setae);
			break;
		case ValueOpcode::BitwiseAnd32: c.and_(c.eax, c.ecx); break;
		case ValueOpcode::BitwiseAnd64: c.and_(c.rax, c.rcx); break;
		case ValueOpcode::BitwiseOr32: c.or_(c.eax, c.ecx); break;
		case ValueOpcode::BitwiseXor32: c.xor_(c.eax, c.ecx); break;
		case ValueOpcode::BitwiseNot32: c.not_(c.eax); break;
		case ValueOpcode::ShiftLeftLogical32: c.shl(c.eax, c.cl); break;
		case ValueOpcode::ShiftLeftLogical64: c.shl(c.rax, c.cl); break;
		case ValueOpcode::ShiftRightLogical32: c.shr(c.eax, c.cl); break;
		case ValueOpcode::ShiftRightLogical64: c.shr(c.rax, c.cl); break;
		case ValueOpcode::ShiftRightArithmetic32: c.sar(c.eax, c.cl); break;
		case ValueOpcode::ShiftRightArithmetic64: c.sar(c.rax, c.cl); break;
		case ValueOpcode::BitFieldUExtract:
			field(c.ecx, c.r9d);
			c.shr(c.rax, c.cl);
			c.and_(c.rax, c.r11);
			break;
		case ValueOpcode::BitFieldSExtract: {
			Xbyak::Label done;
			field(c.ecx, c.r9d);
			c.shr(c.rax, c.cl);
			c.and_(c.rax, c.r11);
			// Width zero tests bit 63, which is clear.
			c.lea(c.ecx, c.ptr[c.r9 - 1]);
			c.bt(c.rax, c.rcx);
			c.jnc(done, Near);
			c.not_(c.r11);
			c.or_(c.rax, c.r11);
			c.mov(c.eax, c.eax);
			c.L(done);
			break;
		}
		case ValueOpcode::BitFieldInsert:
			field(c.r9d, c.r11d);
			c.shl(c.r11, c.cl);
			c.shl(c.r10, c.cl);
			c.and_(c.r10, c.r11);
			c.not_(c.r11);
			c.and_(c.rax, c.r11);
			c.or_(c.rax, c.r10);
			c.mov(c.eax, c.eax);
			break;
		case ValueOpcode::IEqual32:
			c.cmp(c.eax, c.ecx);
			Set(&Xbyak::CodeGenerator::sete);
			break;
		case ValueOpcode::INotEqual32:
			c.cmp(c.eax, c.ecx);
			Set(&Xbyak::CodeGenerator::setne);
			break;
		case ValueOpcode::IEqual64:
			c.cmp(c.rax, c.rcx);
			Set(&Xbyak::CodeGenerator::sete);
			break;
		case ValueOpcode::INotEqual64:
			c.cmp(c.rax, c.rcx);
			Set(&Xbyak::CodeGenerator::setne);
			break;
		case ValueOpcode::UGreaterThan32:
			c.cmp(c.eax, c.ecx);
			Set(&Xbyak::CodeGenerator::seta);
			break;
		case ValueOpcode::SGreaterThanEqual32:
			c.cmp(c.eax, c.ecx);
			Set(&Xbyak::CodeGenerator::setge);
			break;
		case ValueOpcode::LogicalXor:
			c.test(c.rax, c.rax);
			c.setne(c.al);
			c.test(c.rcx, c.rcx);
			c.setne(c.cl);
			c.xor_(c.al, c.cl);
			c.movzx(c.eax, c.al);
			break;
		case ValueOpcode::LogicalNot:
			c.test(c.rax, c.rax);
			Set(&Xbyak::CodeGenerator::sete);
			break;
		default: EXIT("unexpected SRT walker opcode\n");
	}
	Store(node);
}

void WalkerCompiler::Compile(std::span<const Value> values) {
	auto&       c         = m_code;
	const auto& sources   = m_plan.descriptor_sources;
	const auto& blocks    = m_plan.control_flow;
	const bool  resources = values.empty();
	std::vector<std::vector<uint32_t>> gates(sources.size());
	if (resources) Consume(gates);
	for (const auto value: values) AddOutput(value, UINT32_MAX);
	// Analyze every root first, so thunk bodies only reuse finished operands.
	std::vector<Operand>              conditions, reads, outputs;
	std::vector<std::vector<Operand>> descriptors(sources.size());
	if (resources) {
		for (const auto& block: blocks) conditions.push_back(Evaluate(block.condition, 0).operand);
		for (const auto& read: m_plan.srt_reads) reads.push_back(Evaluate(read.value, 0).operand);
		for (uint32_t source = 0; source < sources.size(); ++source) {
			if (gates[source].empty()) continue;
			for (uint32_t word = 0; word < sources[source].dword_count; ++word)
				descriptors[source].push_back(Evaluate(sources[source].dwords[word], 0).operand);
		}
	}
	for (const auto value: m_outputs) outputs.push_back(Evaluate(value, 0).operand);
	// SRT reads, descriptors and outputs must be known; a value only block conditions use may read
	// as unknown, which reaches both successors (AbSwitch SRT_OPTIONAL_DIRTY_UNKNOWN, EmitRead).
	{
		std::vector<bool>     required(m_nodes.size(), false);
		std::vector<uint32_t> pending;
		const auto            require = [&](const Operand& operand) {
            if (operand.kind == Kind::Node && operand.value < required.size() &&
                !required[operand.value]) {
                required[operand.value] = true;
                pending.push_back(static_cast<uint32_t>(operand.value));
            }
		};
		for (const auto& read: reads) require(read);
		for (const auto& words: descriptors) {
			for (const auto& word: words) require(word);
		}
		for (const auto& output: outputs) require(output);
		while (!pending.empty()) {
			const auto& node = m_nodes[pending.back()];
			pending.pop_back();
			require(node.result);
			for (const auto& arg: node.args) require(arg);
		}
		for (uint32_t index = 0; index < m_nodes.size(); ++index) {
			m_nodes[index].optional = resources && !required[index];
		}
	}
	Layout();

	Xbyak::Label pass, fail, reset;
	// Unknown targets of walk body references; stubs jump to them after the body.
	std::deque<Xbyak::Label> exits;
	if (m_fast) {
		// rbx: the cached guest page (none yet); callee-saved in the walker's SysV ABI.
		c.push(c.rbx);
		c.mov(c.rbx, ~uint64_t {0});
	}
	c.add(c.rdx, m_bias);
	c.mov(c.qword[At(m_shader_base)], c.rcx);
	for (const auto& [pair, slot]: m_bases) {
		Base({Kind::UserData, pair.first}, {Kind::UserData, pair.second});
		c.mov(c.qword[At(slot)], c.rax);
		if (m_fast && WalkerPrefetchEnabled()) EmitPrefetches(pair.first, pair.second);
	}
	Zero(m_reach, m_states_end - m_reach);
	if (m_fast) {
		// The flat SRT (WalkSrt only sizes it): zero, as vector::assign did before the walk.
		const auto bytes = static_cast<uint32_t>(m_plan.srt_reads.size() * sizeof(uint32_t));
		if (bytes <= 1024u) {
			ZeroFast([this](uint32_t at) { return m_code.rsi + at; }, 0u, bytes);
		} else {
			c.mov(c.r11, c.rdi);
			c.mov(c.rdi, c.rsi);
			c.mov(c.ecx, bytes);
			c.xor_(c.eax, c.eax);
			c.rep();
			c.stosb();
			c.mov(c.rdi, c.r11);
		}
	}
	c.lea(c.r8, c.ptr[At(m_plan.srt_captures)]);
	if (!resources) {
		for (uint32_t block = 0; block < blocks.size(); ++block)
			c.mov(c.byte[At(m_reach + block)], 1);
	} else if (!blocks.empty()) {
		// Reachability grows monotonically. A newly reached predecessor of an evaluated
		// control Phi restarts the walk with fresh values; a retreating edge resumes at its target.
		std::vector<std::vector<uint32_t>> rechecks(blocks.size());
		for (uint32_t index = 0; index < m_nodes.size(); ++index) {
			const auto& node = m_nodes[index];
			if (node.inst->GetOpcode() != ValueOpcode::Phi || node.result != Operand {Kind::Node, index}) continue;
			for (const auto block: m_plan.control_phi_predecessors.at(node.inst)) rechecks[block].push_back(node.state);
		}
		std::vector<Xbyak::Label> starts(blocks.size());
		c.mov(c.byte[At(m_reach)], 1);
		c.L(pass);
		for (uint32_t block = 0; block < blocks.size(); ++block) {
			Xbyak::Label next;
			const auto mark = [&](uint32_t target) {
				const auto reach = c.byte[At(m_reach + target)];
				// Reaching a later block without control Phis to recheck only sets its byte.
				if (target > block && rechecks[target].empty()) {
					c.mov(reach, 1);
					return;
				}
				Xbyak::Label reached;
				c.cmp(reach, 0);
				c.jne(reached, Near);
				c.mov(reach, 1);
				for (const auto state: rechecks[target]) {
					c.cmp(c.byte[At(state)], 0);
					c.jne(reset, Near);
				}
				if (target <= block) c.jmp(starts[target], Near);
				c.L(reached);
			};
			const auto& successors = blocks[block].successors;
			const auto  condition  = conditions[block];
			c.L(starts[block]);
			c.cmp(c.byte[At(m_reach + block)], 0);
			c.je(next, Near);
			if (condition.kind == Kind::Constant) {
				mark(successors[condition.value != 0u ? 0 : 1]);
			} else {
				if (condition.kind != Kind::Unknown) {
					auto&        both = exits.emplace_back();
					Xbyak::Label other;
					Fetch(condition, both);
					c.test(c.rax, c.rax);
					c.jz(other, Near);
					mark(successors[0]);
					c.jmp(next, Near);
					c.L(other);
					mark(successors[1]);
					c.jmp(next, Near);
					c.L(both);
				}
				for (const auto successor: successors) mark(successor);
			}
			c.L(next);
		}
	}
	if (resources) {
		const auto flat = [&](uint32_t slot) {
			const auto offset = slot < reads.size() ? m_plan.srt_reads[slot].flat_offset : UINT32_MAX;
			if (offset >= reads.size()) {
				c.jmp(fail, Near);
				return;
			}
			Fetch(reads[slot], fail);
			c.mov(c.dword[c.rsi + offset * sizeof(uint32_t)], c.eax);
		};
		if (blocks.empty()) {
			for (uint32_t slot = 0; slot < reads.size(); ++slot) flat(slot);
		}
		for (uint32_t block = 0; block < blocks.size(); ++block) {
			if (blocks[block].srt_reads.empty()) continue;
			Xbyak::Label skip;
			c.cmp(c.byte[At(m_reach + block)], 0);
			c.je(skip, Near);
			for (const auto slot: blocks[block].srt_reads) flat(slot);
			c.L(skip);
		}
		// A source or image no block lists is always active.
		for (uint32_t index = 0; index < sources.size() + m_plan.info.images.size(); ++index) {
			const bool     image = index >= sources.size();
			const uint32_t id    = image ? index - static_cast<uint32_t>(sources.size()) : index;
			bool listed = false;
			for (uint32_t block = 0; block < blocks.size(); ++block) {
				const auto& list = image ? blocks[block].images : blocks[block].sources;
				if (std::ranges::find(list, id) == list.end()) continue;
				if (listed) {
					c.or_(c.al, c.byte[At(m_reach + block)]);
				} else {
					c.mov(c.al, c.byte[At(m_reach + block)]);
				}
				listed = true;
			}
			const auto active = c.byte[At(ActiveOffset(m_plan) + index)];
			if (listed) {
				c.mov(active, c.al);
			} else {
				c.mov(active, 1);
			}
		}
		for (uint32_t source = 0; source < sources.size(); ++source) {
			if (gates[source].empty()) continue;
			auto&      skip  = exits.emplace_back();
			const auto known = c.dword[At(source * DescriptorStride + 8u * sizeof(uint32_t))];
			c.mov(known, 0);
			Gate(gates[source], skip);
			for (uint32_t word = 0; word < descriptors[source].size(); ++word) {
				Fetch(descriptors[source][word], skip);
				c.mov(c.dword[At(source * DescriptorStride + word * sizeof(uint32_t))], c.eax);
			}
			c.mov(known, 1);
			c.L(skip);
		}
	}
	for (uint32_t index = 0; index < outputs.size(); ++index) {
		auto&      skip   = exits.emplace_back();
		const auto output = c.qword[At(OutputOffset(m_plan) + index * 8u)];
		c.mov(output, -1);
		Gate(m_output_gates[index], skip);
		Fetch(outputs[index], skip);
		c.mov(c.eax, c.eax);
		c.mov(output, c.rax);
		c.L(skip);
	}
	c.mov(c.rax, c.r8);
	if (m_fast) {
		// The walk's own exits restore rbx; m_ret stays the thunks' plain return.
		c.pop(c.rbx);
		c.ret();
		c.L(fail);
		c.xor_(c.eax, c.eax);
		c.pop(c.rbx);
		c.ret();
		c.L(m_ret);
		c.ret();
	} else {
		c.L(m_ret);
		c.ret();
		c.L(fail);
		c.xor_(c.eax, c.eax);
		c.ret();
	}
	if (resources && !blocks.empty()) {
		c.L(reset);
		Zero(m_states, m_states_end - m_states);
		c.lea(c.r8, c.ptr[At(m_plan.srt_captures)]);
		c.jmp(pass, Near);
	}
	m_body = false;
	for (uint32_t index = 0; index < m_nodes.size(); ++index) {
		if (Thunk(index)) EmitNode(index);
	}
	for (auto& stub: m_stubs) {
		c.L(stub.entry);
		Branch(stub.operand, *stub.unknown);
		c.jmp(stub.back, Near);
	}
	if (m_fast) EmitColdLoads();
	m_plan.srt_scratch.assign(m_plan.srt_captures / 8u + m_captures, 0u);
	m_plan.srt_user_data = m_user_data;
	if (auto* code = dynamic_cast<WalkerCode*>(&m_code); code != nullptr) {
		code->user_data_mask = m_user_wide ? 0u : m_user_mask;
		code->user_data_mask_valid = !m_user_wide;
	}
	// CompileSrtWalker makes the code executable once every variant is emitted.
}

// KYTY_DUMP_PLAN=<hash>[,<hash>...] (diagnostic, 2026-11-19): WalkSrt prints each listed shader's
// plan once (descriptor sources, SRT reads, uniform fill, block conditions, buffers) as value trees.
std::string PlanValueString(Value value, std::unordered_map<const Inst*, uint32_t>& ids,
                            uint32_t depth) {
	value = value.Resolve();
	if (value.IsEmpty()) return "<null>";
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		switch (value.GetType()) {
			case Type::ScalarReg: return fmt::format("s{}", RegIndex(value.ScalarRegister()));
			case Type::VectorReg: return fmt::format("v{}", RegIndex(value.VectorRegister()));
			case Type::U1: return value.U1() ? "true" : "false";
			case Type::U32: return fmt::format("0x{:x}", value.U32());
			case Type::U64: return fmt::format("0x{:x}", value.U64());
			default: return fmt::format("<{}>", TypeName(value.GetType()));
		}
	}
	if (const auto known = ids.find(inst); known != ids.end()) return fmt::format("%{}", known->second);
	if (depth > 32) return "...";
	const auto id = static_cast<uint32_t>(ids.size() + 1);
	ids.emplace(inst, id);
	std::string text = fmt::format("%{}={}(", id, ValueOpcodeName(inst->GetOpcode()));
	for (size_t i = 0; i < inst->NumArgs(); i++) {
		if (i != 0) text += ",";
		text += PlanValueString(inst->Arg(i), ids, depth + 1);
	}
	return text + ")";
}

void DumpPlanOnce(const ResourcePlan& plan) {
	static const std::vector<uint64_t> wanted = [] {
		std::vector<uint64_t> hashes;
		if (const char* list = std::getenv("KYTY_DUMP_PLAN"); list != nullptr) {
			for (const char* cursor = list; *cursor != '\0';) {
				char*      end  = nullptr;
				const auto hash = std::strtoull(cursor, &end, 16);
				if (end == cursor) break;
				hashes.push_back(hash);
				// ',' or ';' (the run scripts pass environment lists split at commas).
				cursor = *end == ',' || *end == ';' ? end + 1 : end;
			}
		}
		return hashes;
	}();
	if (wanted.empty() || std::ranges::find(wanted, plan.shader_hash) == wanted.end()) return;
	static std::mutex                              mutex;
	static std::unordered_set<const ResourcePlan*> printed;
	std::scoped_lock                               lock(mutex);
	if (!printed.insert(&plan).second) return;
	std::unordered_map<const Inst*, uint32_t> ids;
	std::string text = fmt::format(
	    "plan {:016x}: stage {} user_data {}+{} walker_user_data {} sources {} srt_reads {} "
	    "blocks {} fill words {} kind {} bounded {}\n",
	    plan.shader_hash, static_cast<int>(plan.stage), plan.user_data_base, plan.user_data_count,
	    plan.srt_user_data, plan.descriptor_sources.size(), plan.srt_reads.size(),
	    plan.control_flow.size(), plan.uniform_fill.fill.words,
	    static_cast<int>(plan.uniform_fill.fill.kind), plan.uniform_fill.fill.bounded_elements);
	for (size_t i = 0; i < plan.descriptor_sources.size(); i++) {
		const auto& source = plan.descriptor_sources[i];
		text += fmt::format("  source {} dwords {}{}:", i, source.dword_count,
		                    source.indirect_descriptor ? " indirect" : "");
		for (uint32_t word = 0; word < source.dword_count; word++) {
			text += " " + PlanValueString(source.dwords[word], ids, 0);
		}
		text += "\n";
	}
	for (size_t i = 0; i < plan.srt_reads.size(); i++) {
		text += fmt::format("  srt_read {} flat {}: {}\n", i, plan.srt_reads[i].flat_offset,
		                    PlanValueString(plan.srt_reads[i].value, ids, 0));
	}
	const auto& fill = plan.uniform_fill;
	if (fill.fill.words != 0 || !fill.element_count.IsEmpty()) {
		text += "  uniform_fill values:";
		for (const auto& value: fill.values) text += " " + PlanValueString(value, ids, 0);
		text += " pattern_size " + PlanValueString(fill.pattern_size, ids, 0) + " element_count " +
		        PlanValueString(fill.element_count, ids, 0) + "\n";
	}
	for (size_t i = 0; i < plan.control_flow.size(); i++) {
		if (!plan.control_flow[i].condition.IsEmpty()) {
			text += fmt::format("  block {} condition {}\n", i,
			                    PlanValueString(plan.control_flow[i].condition, ids, 0));
		}
	}
	for (const auto& [address, size]: plan.source_reads) {
		text += fmt::format("  source_read 0x{:x}+{}\n", address, size);
	}
	for (size_t i = 0; i < plan.info.buffers.size(); i++) {
		const auto& buffer = plan.info.buffers[i];
		text += fmt::format("  buffer {} source {} read {} written {} atomic {} scalar {} extent {}\n",
		                    i, buffer.source, buffer.read, buffer.written, buffer.atomic,
		                    buffer.scalar, buffer.max_byte_extent);
	}
	std::fputs(text.c_str(), stdout);
	std::fflush(stdout);
}

} // namespace

void CompileSrtWalker(ResourcePlan& plan, std::span<const Value> values) {
	auto  owned = std::make_unique<WalkerCode>();
	auto& code  = *owned;
	plan.srt_code = std::move(owned);
	WalkerCompiler(plan, code).Compile(values);
	if (FastWalkerWanted()) {
		// AbSwitch SRT_WALKER_FAST: the fast variant follows in the same buffer. Its analysis and
		// layout repeat the reference's exactly (same plan, deterministic passes); check that.
		const auto captures  = plan.srt_captures;
		const auto scratch   = plan.srt_scratch.size();
		const auto user_data = plan.srt_user_data;
		const auto mask      = code.user_data_mask;
		const bool valid     = code.user_data_mask_valid;
		code.fast_entry      = code.getSize();
		WalkerCompiler(plan, code, true).Compile(values);
		if (plan.srt_captures != captures || plan.srt_scratch.size() != scratch ||
		    plan.srt_user_data != user_data || code.user_data_mask != mask ||
		    code.user_data_mask_valid != valid || code.fast_entry == 0) {
			EXIT("SRT walker: the fast variant's layout differs (hash=%016" PRIx64 ")\n",
			     plan.shader_hash);
		}
	}
	code.readyRE();
	EXIT_IF(Xbyak::GetError() != 0);
}

bool SrtWalkerUserDataMask(const ResourcePlan& plan, uint64_t& mask) noexcept {
	const auto* code = dynamic_cast<const WalkerCode*>(plan.srt_code.get());
	if (code == nullptr || !code->user_data_mask_valid) return false;
	mask = code->user_data_mask;
	return true;
}

// KYTY_SRT_OPTIONAL_DIAG: the sources and images that a walk with unknown optional loads made active
// but the precise re-walk did not (printed with their descriptor words), and a 5 s summary.
static bool SrtOptionalDiagEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SRT_OPTIONAL_DIAG");
		return value != nullptr && *value == '1';
	}();
	return enabled;
}

static void LogOptionalDifferences(const ResourcePlan& plan, std::span<const uint8_t> optimistic) {
	static std::mutex mutex;
	static uint64_t   diff_walks = 0, unknown_walks = 0, logged = 0;
	static int64_t    window_start = 0;
	// At most 3 lines per (plan, index), 600 in all.
	static std::map<std::pair<uint64_t, uint32_t>, uint32_t> per_key;
	std::scoped_lock  lock(mutex);
	unknown_walks++;
	const auto* precise = Scratch(plan);
	const auto  sources = static_cast<uint32_t>(plan.descriptor_sources.size());
	const auto  count   = sources + static_cast<uint32_t>(plan.info.images.size());
	bool        differs = false;
	for (uint32_t index = 0; index < count; ++index) {
		const auto offset = ActiveOffset(plan) + index;
		const bool opt    = optimistic[offset] != 0;
		const bool exact  = precise[offset] != 0;
		if (opt == exact) continue;
		differs = true;
		if (logged >= 600 || per_key[{plan.shader_hash, index}]++ >= 3) continue;
		logged++;
		const bool     image  = index >= sources;
		const uint32_t source = image ? plan.info.images[index - sources].source : index;
		std::string    words;
		if (source < sources) {
			uint32_t dwords[9] {};
			std::memcpy(dwords, optimistic.data() + source * DescriptorStride, sizeof(dwords));
			for (uint32_t w = 0; w < 9; ++w) words += fmt::format(" {:08x}", dwords[w]);
		}
		std::printf("srt-optional-diff: plan %016" PRIx64 " %s %u (source %u) active optimistic %d "
		            "precise %d words%s\n",
		            plan.shader_hash, image ? "image" : "source",
		            image ? index - sources : index, source, opt ? 1 : 0, exact ? 1 : 0,
		            words.c_str());
	}
	diff_walks += differs ? 1 : 0;
	const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
	                        std::chrono::steady_clock::now().time_since_epoch())
	                        .count();
	if (window_start == 0) window_start = now;
	if (now - window_start >= 5'000'000'000) {
		const double s = static_cast<double>(now - window_start) / 1e9;
		std::printf("srt-optional-diag: %.1fs unknown_walks/s=%.1f diff_walks/s=%.1f\n", s,
		            static_cast<double>(unknown_walks) / s, static_cast<double>(diff_walks) / s);
		std::fflush(stdout);
		unknown_walks = diff_walks = 0;
		window_start  = now;
	}
}

namespace {

using WalkerEntry = const uint64_t* KYTY_SYSV_ABI (*)(const uint32_t*, uint32_t*, uint64_t*, uint64_t);

// One call of a walker variant; the reference starts at the code buffer, the fast one at fast_entry.
const uint64_t* CallWalker(const ResourcePlan& plan, const WalkerCode& code, bool fast,
                           const SrtRuntime& runtime, std::vector<uint32_t>& flat) {
	const auto entry = reinterpret_cast<uintptr_t>(code.getCode()) + (fast ? code.fast_entry : 0u);
	return reinterpret_cast<WalkerEntry>(entry)(runtime.user_data.data(), flat.data(),
	                                            plan.srt_scratch.data(), runtime.shader_base);
}

// KYTY_VERIFY_WALKER: one walker run's complete result.
struct WalkerRun {
	size_t                end = SIZE_MAX; // capture end in scratch qwords; SIZE_MAX: the walk failed
	std::vector<uint64_t> scratch;
	std::vector<uint32_t> flat;
	uint32_t              unknowns = 0; // g_optional_unknowns after the run

	bool operator==(const WalkerRun& other) const = default;
};

// GPU-thread counters of the verify mode (plain globals: see the thread_local note above).
uint64_t g_verify_walker_equal    = 0;
uint64_t g_verify_walker_unstable = 0;
uint64_t g_verify_walker_skipped  = 0;

void ReportWalkerDifference(const ResourcePlan& plan, const WalkerRun& reference, const WalkerRun& fast) {
	const auto signed_end = [](size_t end) { return end == SIZE_MAX ? -1ll : static_cast<long long>(end); };
	std::printf("srt-walker-verify: plan %016" PRIx64 " stage %u differs: end %lld/%lld (qwords, -1 failed) "
	            "optional unknowns %u/%u (reference/fast)\n",
	            plan.shader_hash, static_cast<uint32_t>(plan.stage), signed_end(reference.end),
	            signed_end(fast.end), reference.unknowns, fast.unknowns);
	const auto* left  = reinterpret_cast<const uint8_t*>(reference.scratch.data());
	const auto* right = reinterpret_cast<const uint8_t*>(fast.scratch.data());
	const auto  bytes = std::min(reference.scratch.size(), fast.scratch.size()) * sizeof(uint64_t);
	for (size_t byte = 0, shown = 0; byte < bytes && shown < 16; ++byte) {
		if (left[byte] == right[byte]) continue;
		const char* region = byte < ActiveOffset(plan)      ? "descriptor words"
		                     : byte < OutputOffset(plan)    ? "activity"
		                     : byte < plan.srt_captures     ? "outputs/slots/reach/states/bases"
		                                                    : "captures";
		std::printf("  scratch byte %zu (%s): %02x/%02x\n", byte, region, left[byte], right[byte]);
		shown++;
	}
	for (size_t slot = 0, shown = 0; slot < reference.flat.size() && slot < fast.flat.size() && shown < 16; ++slot) {
		if (reference.flat[slot] == fast.flat[slot]) continue;
		std::printf("  flat %zu: %08x/%08x\n", slot, reference.flat[slot], fast.flat[slot]);
		shown++;
	}
	std::fflush(stdout);
}

// Runs the reference and the fast walker from the same scratch contents and optional-unknown count
// and exits when they differ while the reference reproduces its own result (a difference the
// reference does not reproduce comes from guest memory or GPU state changing between the runs,
// e.g. an optional load whose GPU write completed; such a walk is retried, at most three times).
// The result left in place is the last run's.
const uint64_t* VerifyWalk(const ResourcePlan& plan, const WalkerCode& code, const SrtRuntime& runtime,
                           std::vector<uint32_t>& flat) {
	const auto                  count    = plan.srt_reads.size();
	const auto                  unknowns = g_optional_unknowns;
	const std::vector<uint64_t> before   = plan.srt_scratch;
	size_t                      last_end = SIZE_MAX;
	const auto                  run      = [&](bool fast) {
		std::copy(before.begin(), before.end(), plan.srt_scratch.begin());
		g_optional_unknowns = unknowns;
		// The fast walker zeroes the flat SRT itself: start it from garbage to check that.
		flat.assign(count, fast ? 0xcdcdcdcdu : 0u);
		const auto* end = CallWalker(plan, code, fast, runtime, flat);
		WalkerRun   result;
		result.end   = end == nullptr ? SIZE_MAX : static_cast<size_t>(end - plan.srt_scratch.data());
		last_end     = result.end;
		result.scratch  = plan.srt_scratch;
		result.flat     = flat;
		result.unknowns = g_optional_unknowns;
		return result;
	};
	const auto finish = [&]() -> const uint64_t* {
		return last_end == SIZE_MAX ? nullptr : plan.srt_scratch.data() + last_end;
	};
	for (int attempt = 0; attempt < 3; ++attempt) {
		const auto reference = run(false);
		const auto fast      = run(true);
		if (fast == reference) {
			if ((g_verify_walker_equal++ & ((uint64_t {1} << 20u) - 1u)) == 0) {
				std::printf("srt-walker-verify: %" PRIu64 " walks equal, %" PRIu64 " unstable retries, %" PRIu64
				            " unverified\n",
				            g_verify_walker_equal, g_verify_walker_unstable, g_verify_walker_skipped);
				std::fflush(stdout);
			}
			return finish();
		}
		const auto again = run(false);
		if (again == reference) {
			ReportWalkerDifference(plan, reference, fast);
			EXIT("SRT walker verify: the fast walker differs from the reference (hash=%016" PRIx64 ")\n",
			     plan.shader_hash);
		}
		g_verify_walker_unstable++;
	}
	g_verify_walker_skipped++;
	return finish();
}

// One walk with the selected variant (both under KYTY_VERIFY_WALKER).
const uint64_t* RunWalker(const ResourcePlan& plan, const WalkerCode& code, bool fast,
                          const SrtRuntime& runtime, std::vector<uint32_t>& flat) {
	if (code.fast_entry != 0 && VerifyWalkerEnabled()) [[unlikely]] {
		return VerifyWalk(plan, code, runtime, flat);
	}
	if (fast) {
		// The fast walker zeroes the flat SRT at entry.
		flat.resize(plan.srt_reads.size());
	} else {
		flat.assign(plan.srt_reads.size(), 0u);
	}
	return CallWalker(plan, code, fast, runtime, flat);
}

} // namespace

bool WalkSrt(const ResourcePlan& plan, const SrtRuntime& runtime, std::vector<uint32_t>& flat,
             std::vector<std::pair<uint64_t, uint64_t>>* reads) {
	if (plan.srt_code == nullptr || runtime.user_data.size() < plan.srt_user_data) return false;
	DumpPlanOnce(plan);
	// CompileSrtWalker is the only producer of srt_code.
	const auto& code = static_cast<const WalkerCode&>(*plan.srt_code);
	const bool  fast = code.fast_entry != 0 &&
	                  Common::AbSwitch::Enabled(Common::AbSwitch::SRT_WALKER_FAST);
	t_running_walker    = &code;
	t_walker_shader     = plan.shader_hash;
	t_walker_user_data  = runtime.user_data.data();
	t_walker_user_count = static_cast<uint32_t>(runtime.user_data.size());
	g_srt_load_check = Common::AbSwitch::Enabled(Common::AbSwitch::SRT_LOAD_CHECK) ? 1 : 0;
	g_optional_unknowns = 0;
	const auto* end = RunWalker(plan, code, fast, runtime, flat);
	if (end != nullptr && g_optional_unknowns != 0 && SrtOptionalDiagEnabled()) [[unlikely]] {
		// Keep this walk's descriptors and activity, then walk again waiting for the GPU (the
		// precise result is the one used, so the diagnostic behaves like bit 59 off).
		const auto* scratch = Scratch(plan);
		const std::vector<uint8_t> optimistic(scratch, scratch + OutputOffset(plan));
		g_force_precise = true;
		end = RunWalker(plan, code, fast, runtime, flat);
		g_force_precise = false;
		if (end != nullptr) LogOptionalDifferences(plan, optimistic);
	}
	t_running_walker = nullptr;
	if (end == nullptr) return false;
	if (reads != nullptr) {
		// Resizing first lets the copy vectorize; emplace_back reloads the vector's end per read.
		const std::span captured {plan.srt_scratch.data() + plan.srt_captures / 8u, end};
		const auto      first = reads->size();
		reads->resize(first + captured.size());
		std::ranges::transform(captured, reads->data() + first, [](uint64_t address) {
			return std::pair {address, uint64_t {sizeof(uint32_t)}};
		});
	}
	return true;
}

bool WalkedValue(const ResourcePlan& plan, Value value, uint32_t& result) {
	value = value.Resolve();
	uint64_t bits = 0;
	if (value.IsImmediate()) {
		if (!ImmediateBits(value, bits)) return false;
		result = static_cast<uint32_t>(bits);
		return true;
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) return false;
	EXIT_IF(inst->WalkOutput() == UINT32_MAX);
	std::memcpy(&bits, Scratch(plan) + OutputOffset(plan) + inst->WalkOutput() * 8u, sizeof(bits));
	result = static_cast<uint32_t>(bits);
	return bits <= UINT32_MAX;
}

bool WalkedDescriptor(const ResourcePlan& plan, uint32_t source, DescriptorValue& result) {
	if (source >= plan.descriptor_sources.size()) return false;
	std::array<uint32_t, 9> words {};
	std::memcpy(words.data(), Scratch(plan) + source * DescriptorStride, DescriptorStride);
	if (words[8] == 0u) return false;
	result = {};
	result.dword_count = plan.descriptor_sources[source].dword_count;
	std::copy_n(words.begin(), result.dword_count, result.dwords.begin());
	return true;
}

bool WalkedActive(const ResourcePlan& plan, uint32_t source) {
	return Scratch(plan)[ActiveOffset(plan) + source] != 0u;
}

bool WalkedImageActive(const ResourcePlan& plan, uint32_t image) {
	return Scratch(plan)[ActiveOffset(plan) + plan.descriptor_sources.size() + image] != 0u;
}

bool ValidateRuntimeValue(const ResourcePlan& program, Value value, RuntimeValueType type) {
	return RuntimeValidator(program, type).Run(value);
}

bool ValidateWaveUniformValue(const ResourcePlan& program, Value value) {
	return RuntimeValidator(program, RuntimeValueType::Any, true).Run(value);
}

namespace {
uint64_t g_walker_skipped_loads = 0;
} // namespace

uint32_t SrtWalkerLastOptionalUnknowns() noexcept {
	return g_optional_unknowns;
}

uint64_t SrtWalkerSkippedLoads() noexcept {
	return g_walker_skipped_loads;
}

void NoteSrtWalkerSkippedLoad() noexcept {
	g_walker_skipped_loads++;
}

uint64_t SrtWalkerGuardedLoads() noexcept {
	return g_walker_guarded_loads.load(std::memory_order_relaxed);
}

void NoteSrtWalkerGuardedLoad() noexcept {
	g_walker_guarded_loads.fetch_add(1, std::memory_order_relaxed);
}

bool SrtPageReadWatched(uint64_t address) noexcept {
	// The same test LoadGuest emits.
	const auto* lookup = PageManager::AccessLookup();
	const void* table  = lookup->regions;
	if (table == nullptr) return false;
	const uint64_t region = address >> lookup->region_shift;
	if (region >= lookup->region_count) return false;
	const auto* base = static_cast<const uint8_t* const*>(table)[region];
	if (base == nullptr) return false;
	const uint64_t page = (address >> lookup->page_shift) & lookup->page_mask;
	return (base[page * lookup->page_stride + lookup->pages_offset] & lookup->access_mask) != 0;
}

uint32_t SrtWalkerLoadLength(uint64_t rip) {
	if (t_running_walker == nullptr ||
	    rip - reinterpret_cast<uint64_t>(t_running_walker->getCode()) >= t_running_walker->getSize())
		return 0;
	const auto* code = reinterpret_cast<const uint8_t*>(rip);
	return code[0] == 0x41u && code[1] == 0x8bu && code[2] == 0x02u ? 3 : 0;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
