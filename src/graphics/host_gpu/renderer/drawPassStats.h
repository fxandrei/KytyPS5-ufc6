#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPASSSTATS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPASSSTATS_H_

#include <cstdint>

namespace Libs::Graphics {

class CommandBuffer;
struct DrawRenderState;
struct DrawTargets;

// KYTY_DRAW_PASS_STATS=1 (diagnostic, 2026-11-20, game-side draw cuts for UFC 6): the GPU thread
// counts the draws it emits per render-target set ("class": color formats and extents, depth
// format, extent and test state; viewport sizes and distinct views listed per class) and per
// program group (class + guest VS/PS addresses and shader hashes), plus dispatches per compute
// shader. Every KYTY_DRAW_PASS_STATS_SECONDS (default 5) it prints the busiest classes, groups and
// compute shaders at a guest flip as "draw-pass-stats*:" lines on stdout.
// CPU only: no GPU timers, readbacks or waits. Off: one predictable branch per draw, dispatch and
// flip. On: a few hash lookups per draw (~0.1-0.2 us), so FPS of such a run is not an FPS figure.
namespace DrawPassStats {

[[nodiscard]] bool ReadActive() noexcept;

// Whether any option of this module is set; the hooks call into it only then.
[[nodiscard]] inline bool Active() noexcept {
	static const bool active = ReadActive();
	return active;
}

// Every RenderExecutor::DrawIndex / DrawAuto call, emitted or not.
void NoteCall() noexcept;
// A draw ExecutePreparedDraw recorded; indirect_records is the GPU-read record count (0 if direct).
void NoteDraw(const CommandBuffer& buffer, const DrawRenderState& state, const DrawTargets& targets,
              uint32_t index_count, uint32_t instance_count, uint32_t indirect_records, bool mesh);
// A recorded dispatch; indirect dispatches have unknown (0) group counts.
void NoteDispatch(uint64_t cs_address, uint64_t cs_hash, uint32_t x, uint32_t y, uint32_t z,
                  bool indirect);
// A guest flip (CpOpFlip or a CPU flip): ends a frame; prints when the window is full.
void NoteFlip();

// KYTY_SKIP_DRAW_* (MEASUREMENT ONLY, 2026-11-20): drops matching draws to measure what removing
// them (e.g. one shadow cascade or the local-light shadow maps) would change, without patching the
// game. Skipping changes the rendering: an FPS from such a run is never an emulator FPS result.
//   KYTY_SKIP_DRAW_PS / _VS=<hex>[;<hex>]  guest PS / VS addresses (as printed by the stats)
//   KYTY_SKIP_DRAW_CLASS=<16 hex>[;...]    render-target classes (hash= of draw-pass-stats-class)
//   KYTY_SKIP_DRAW_VP=<W>x<H>[;...]         viewport sizes
//   KYTY_SKIP_DRAW_KEEP_VIEWS=<n>            keep the first n distinct views per frame, skip later
//   KYTY_SKIP_DRAW_COLOR=1                   also draws with color targets (default: depth-only)
//   KYTY_SKIP_DRAW_AB_BIT=<bit>              skip only while that (unused) AbSwitch bit is set
//   KYTY_SKIP_DRAW_AFTER=<seconds>           skip only that long after the first flip
// A draw is skipped when it matches every list that is set. Never skipped: draws whose programs
// write buffers, depth/stencil clears and copies, and non-normal color-control modes.
// Before the programs are looked up (pure PS/VS rules, programs already seen without buffer
// writes or color targets): the draw costs almost nothing, as if the game had not submitted it.
[[nodiscard]] bool SkipBeforePrograms(const CommandBuffer& buffer);
// With programs and targets resolved: every rule.
[[nodiscard]] bool SkipResolved(const CommandBuffer& buffer, const DrawRenderState& state,
                                const DrawTargets& targets);

} // namespace DrawPassStats

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPASSSTATS_H_
