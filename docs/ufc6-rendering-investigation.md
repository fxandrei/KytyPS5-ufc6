# UFC 6 rendering investigation

## Current entry point, 2026-11-24

The owner authorized refresh of all known repositories, push to the personal fork and a published
experimental release. Upstream `b446af9d2` is integrated with the local snapshot `8d40d8b2a` preserved.
Exact reviewed tips are in `docs/releases/2026-11-24-repositories.json`; integration notes are in
`docs/releases/2026-11-24-integration.md`. Detailed evidence remains in
`D:/Projects/kyty-dev/handoff/2026-11-24/HANDOFF.txt` and the logs beside it.

Emulator/launcher and the selected test targets build. Eight isolated GPU functional controls pass:
resolve cache, native indirect dispatch, stencil storage aliases, bounded image copies, GDS with both
allocation modes, BDA with diagnostic host visibility and ordered dispatch. Integration fixes preserve
actual resolve stamps, let upstream color results take precedence, restore the ordinary image-copy
end-layout contract and allow S8 plane records to move between depth owners. A redundant indirect
argument barrier is superseded by the upstream writer dependencies. No FPS gain is inferred.

Eleven CPU/resource/kernel CTests passed during integration. The full GPU suite still stops at the
previously recorded adjacent-download-stride assertion. The filesystem suite stops in an upstream
Linux `/proc` waiter helper on Windows; its separate network mode fails the initial NP callback state
assertion. Those paths remain unverified. No Vulkan-validation pass: RTSS remains active.
Live merged-source control and release publication are still pending. Known watchdog/corruption,
replay and BAR menu issues require a prerelease. No 60 FPS combat claim.

## Previous entry point, 2026-11-23

**Final checkpoint22:06:** read `D:\Projects\kyty-dev\handoff\2026-11-23\RESULTS_2205.md`. The fixed-corner host-barrier A/B was neutral (13pairs,1.000715 ratio, near60FPS in both phases). BAR allocation still reproduced slowdown/watchdog with per-shader timers absent and direct BAR reads disabled. Unmapped allocation reached observed clean UFC APEX combat and KO/results, but also printed a pre-fight guest watchdog. The bounded active-fight interval gives59 compilation-free samples, mean39.21/median39.28FPS at3840x2160 depth; no60FPS combat or accepted improvement. Next is a new fight-only CPU profile with the now-populated cache, plus a separate allocation/residency diagnosis. All own test/monitor processes are closed. Final upstream fetch is4e9065450 (POSIX rename export, reviewed), still not merged. Earlier entries below retain chronology and methodological caveats.

Read `D:\Projects\kyty-dev\handoff\2026-11-23\HANDOFF.txt` and `investigation.json`, then the final 2026-11-22 entry of `handoff\2026-11-18\HANDOFF.txt`. Older sections below preserve the investigation; their implementation status is superseded where noted there.

The latest inherited tree contains the adopted CPU/BDA/SRT changes, GDS increment/decrement optimization and pressure-dependent texture-cache GC, plus experimental Streamline/parallel-preparation/unmap work. HEAD is still `2882ea1a9`; about 50 tracked files and additional new sources are uncommitted. Preserve them. The target selected with the previous agent is native 4K. Reported prior controls: UFC Apex approximately 42-46 FPS, T-Mobile 17-30 FPS; these scenes must not be compared as an optimization A/B. Close-up hair menus remain slow/unstable. Details and rejected experiments are in the November 18 handoff and November 21 agent reports.

On resumption, `origin/main` was fetched to `7d0c350e2`; it has not been integrated. `run-series.ps1` was corrected to keep active caches by default. It previously restored the November 14 shader store before *every* launch; the new GDS shader profile invalidated that old store, causing repeated cold controls. `-RestoreLegacyCacheSet` now explicitly requests the historical diagnostic behavior. Syntax verification passed. Do not bypass emitter/profile changes by stamping an old cache.

The existing executable (`ECA7578D7D1F60EEAE42239164FD646FCD074710F79A529C96F9F8CB651C7E18`) loaded 642 programs / 756 modules / 715 pipelines in the first resumed 4K corner control (`20261123-204117-perf-resume-warm-corner-20261123-1-20261118`). It subsequently compiled additional 4K content and printed a guest GPU-watchdog breadcrumb report near `worldHalfRes`. The test was closed normally by the agent after about 102 seconds when free system RAM fell to 1.49 GiB. Exit 0 is a manual close, not a healthy-run assertion. No FPS conclusion. RTSS was running, so this was not a Vulkan validation control.

November 23 continuation: read `MILESTONE_2144.md` in the current handoff. The original hair ISA has now decoded successfully (1014 instructions; actual GDS DS_INC_RTN at 0x218, no DS_ORDERED_COUNT). Compute binding diagnostics identify 37.7MB guest ranges inside native allocations that grow to 1.03GB. These allocation flags do not establish physical WDDM residency.

An initial isolated-timing implementation lacked explicit gates after each timestamp; its large intervals cannot be assigned exclusively to a shader. With four gates, completed selected samples remain roughly 0.005-0.32ms despite a 2.4 FPS corner menu. Timer-ring exhaustion and completion-window attribution limit coverage. This shifts the next investigation to work/visibility/resource handling around dispatches; it does not prove all hair dispatches are cheap.

The opt-in unmapped guest-buffer control recovered spontaneously; a BAR control remained slow even with 2-3.5GiB free system RAM. This is not an accepted FPS improvement: the allocation choice also changes blanket host visibility barriers. A same-process diagnostic now varies that barrier with bit 11 while retaining unmapped guest buffers. GDS CPU reads receive an explicit buffer-specific visibility barrier. Build B913385F and five focused GDS/BDA/ordered/stencil functional controls pass. Full GPU CTest still fails the adjacent-download-stride assertion in BufferCacheDirtyGarbageCollection; do not claim a suite or Vulkan-validation pass. See the current handoff for the live A/B result. No new accepted performance change, deployment or push.

Resolution qualification: these November 23 controls configure **3840x2160 output** in a 1280x720 window. The game still adjusts internal rendering resolution: stable corner controls report depth targets of 3200x1800, while slow controls vary. Do not label these fixed native-4K rendering or compare differing internal work as identical workloads.

Snapshot: 2026-11-14, ongoing. Read the latest `HANDOFF_CURRENT.txt` and working-tree diff; this document is a portable entry point, not a claim that the target game is fixed.

## Objective and preserved baseline

The objective is correct, stable rendering/gameplay and eventually 60 FPS in fights. Five small generic fixes were committed before the current experiment:

- `27684570e`: initialize the image heap before pipeline-cache warmup; independent heap-cache regression fails before and passes after.
- `43a7d5bd0`: avoid stores to untouched read-only depth/stencil aspects; mixed-aspect native validation case.
- `ed874194a`: restrict sampled-view inherited attachment usage to format support, including RGB9E5.
- `37fc0f043`: record BDA fault bitmap bits with device-scope atomic OR; concurrent 32-page regression fails before and passes after.
- `184cc181a`: bound 3D/2D-array copies to available array layers; native transfer/readback regression and isolated validation pass.

`fd3d07110` preserves diagnostics. `2882ea1a` adds the opt-in stencil-storage regression. Current stencil-storage implementation is experimental and has not been deployed to `D:\games\_kytyps5`.

Latest upstream fetch recorded before this experiment: `origin/main` at `58ca8d1d7`, already integrated. Check again when resuming; do not overwrite the working tree to sync.

## Confirmed missing compatibility behavior

A full-warm control preloaded 952 programs, 1165 modules and 1132 native pipelines, then failed before corner selection:

- Shader `0x3880e09f634cd734`, module `7106699877482241273`, resource 1.
- A general R8_UINT storage descriptor was resolved to a native D32_SFLOAT_S8_UINT stencil view.
- The native depth/stencil image had no STORAGE usage. Vulkan reported `VUID-VkImageViewCreateInfo-pNext-02662`.
- This is a general image write, not a proven uniform clear. Dropping the usage flag, adding unsupported STORAGE to depth, or forcing a clear is invalid.

The original minimal `--stencil-storage-alias-only` test reproduced this without the game and failed before creating an invalid view.

Original warm-cache visual corruption is not conclusively attributed to this single defect. Compilation/logging timing changes can mask failures.

## Current prototype

The existing stencil association can acquire its own R8_UINT native backing while retaining its link to the depth owner. Stencil bytes are copied S8 -> R8 on the GPU before acquisition and R8 -> S8 after storage dispatches. Native depth is preserved. CPU uploads still target the owner's stencil; the materialized R8 plane can publish to CPU memory.

Shared sampled/storage bindings are rediscovered to use the same R8 backing. Proven uniform clears retain the native stencil-clear path. Changing stencil address or retiring an owner freezes the latest GPU bytes into a standalone R8 image; re-adoption can transfer current GPU R8 data into native stencil.

Transfers have a separate lazy 4 MiB scratch buffer rather than reusing the allocation bound as emulated LDS/shared memory. Scratch barriers are buffer-specific. The native test verifies that the shared utility allocation remains intact.

Supported scope is deliberately checked: compute, R8_UINT, non-heap/non-atomic, full single-mip/single-layer/single-sample uncompressed 2D stencil, without a concurrently bound depth attachment. Other layouts are rejected explicitly. Do not relax these guards without data and coverage.

## Test evidence and unresolved live behavior

The expanded opt-in native test passed with exit 0 and no Vulkan validation errors:

- Acquire current GPU stencil `0x91` despite stale CPU data.
- Execute a real PS5 IMAGE_STORE dispatch into a 3x2 plane; change only pixel (1,0) to `0x35`.
- Preserve the other five stencil texels and all depth values (`0.625`).
- Publish new bytes to CPU memory.
- Preserve the allocation used for shader shared/LDS storage.
- Freeze old GPU data when stencil address changes.
- Resolve two sampled bindings and one storage binding to the same physical R8 image.
- Preserve stencil after native owner retirement and re-adopt newer GPU R8 bytes (`0x4d`) rather than stale CPU bytes (`0x29`).

Logs: `stencil-storage-lifecycle-validation.log` and metadata under the local handoff. Test executable SHA: `7C2AA8654A105DDE601EC51D4327B1304892ECBD59CCDBBA68ECB72EB3F09EB4`. The functional GPU CTest passed in 26.45 seconds. The new alias test was run separately; integrate it into default coverage once the feature is accepted.

Live controls are mixed, so the prototype is **not yet accepted as stable**:

- An earlier prototype with shared scratch reached a GPU stall (waiting tick 72554, completed 72547, record consumer idle). Root cause not proven.
- A private-scratch warm run stopped at the explicit layout guard, address `0x11c8030000`; that binary lacked detailed predicate logging. Do not guess which condition failed.
- A later diagnostic build, SHA `887D8012CCCCFCF777BA8B9AAFAC918FC8D092FA81395BC0B92EF53FAB4A3E1E`, reached clean active fight through 4:10 with four attack inputs under confirmed synchronization validation and GPU checkpoints. It preloaded the full fight cache and compiled only one extra shader/two pipelines. No old view error or stall in that control.
- The non-validation control `20261114-002439-perf-stencil-private-baseline-no-validation-20261114` ended itself after about 225 seconds with a GPU stall and fatal wait/presentation errors (exit 321). Last completed tick was 641149; waits were for 641157/641160. The recording consumer had consumed all 23949893552 bytes and was sleeping. No per-operation GPU checkpoints were enabled. Its 20.52 FPS median covered a heavier ~6122 draws/frame scene including the introduction; it is neither a fight baseline nor an improvement/regression comparison. The clean introduction capture does not establish active-fight correctness.
- The intermittent layout case and longer combat/replay behavior remain unresolved. A new non-validation control disables only record-thread bit 37 and enables GPU checkpoints: `20261114-003012-perf-stencil-private-record-off-checkpoints-20261114`. Same executable, preserved shader store and frozen native cache; result pending.

Do not claim 60 FPS, an FPS improvement or complete artifact/replay stability from these controls. Validation produced approximately 6-8 FPS and is not a performance benchmark.

## Reproduction and evidence

Build/test/run helpers: `_Build/diagnostics/upstream-20261107/{build,run-tests,start-run,control-game}.ps1`.

The native case is `shader_recompiler_compute_tests.exe --stencil-storage-alias-only`. The local wrapper `run-gpu-validation.ps1` forces the installed validation layer and rejects VUIDs even if native exit is 0. `start-run.ps1 -Checkpoints` enables GPU checkpoints without the bounds-probe instrumentation.

The post-atomic complete fight shader cache is preserved as `PPSA23566-atomic-clean-fight-225537.shaders`, SHA `B334DEDDF7F589F30D10431B14A287816E0516E6889609F9A2E41C4493D61BC8`. Native-driver cache control is the preserved `native-driver-cache-before-depth-fix.bin`, SHA `E41364C2E7E17B2EC287D9087E3984CFE92BDD1ABEBB2D3A350CB183DB89A630`.

For host-only rebuilds, diagnostics sometimes replace the shader-cache executable timestamp in a fresh backup copy. This is recorded per run and does not change the original backup. **Never use that bypass for changed shader-emitter semantics**; generate a new cache instead.

Detailed local evidence: `D:\Projects\kyty-dev\handoff\2026-11-13\{HANDOFF.txt,investigation.json,STENCIL_STORAGE_PLAN.txt}` and `D:\Projects\kyty-dev\runs`. Preserve these before cleanup or model changes.

## Continuity update, 2026-11-14 00:31

The user selected Extra High and reiterated autonomous testing/documentation. The local v2 architecture guide was read (SHA-256 `EBE65C63FDC34BEEEF1F8F9CE8CA9EF882F64ADD4F34E80AABF27D0A4E84A713`). Its baseline snapshot predates the uncommitted stencil-storage prototype; use current logs/diff for implementation status. Its CPU model description also differs from the live emulator title, which reports Ryzen 7 7800X3D; verify hardware rather than copying the guide's 9800X3D claim.

Upstream was fetched again at 00:30: `origin/main` remains `58ca8d1d7fd621196a60c0a2bda664f4672815d7`, already an ancestor of HEAD. No additional upstream source update is pending. C: had 12.7 GiB free, D: 232.64 GiB. No deployment or push.

## Continuity update, 2026-11-18 (Claude Code session)

Details: `D:\Projects\kyty-dev\handoff\2026-11-18\HANDOFF.txt`. All changes below are uncommitted, on top of the prototype.

- **Intermittent layout guard identified.** Run `20261118-175612-perf-indirect-note-record-off-20261118` hit `unsupported stencil storage layout` with the full predicate logged: every field in range except `compressed=true` (owner stencil is HTILE-compressed; plane 0x11c84b0000, 1600x904). The guard is widened: compressed owners are accepted; `TextureCache::CheckStencilStorageMeta` exits explicitly if the owner's HTILE clear is still pending (clear_mask bit 0), at materialization and at every acquisition. Rationale: the native D32S8 image holds logical stencil values; only a pending HTILE clear, applied at the next depth-target bind, would make them stale. No native test with a compressed owner yet.
- **New stall diagnostic, no GPU commands added.** With `KYTY_GPU_CHECKPOINTS` (or `start-run.ps1 -Checkpoints`), a stall report also prints, for indirect draws of unfinished ticks, the argument words read through the buffer host mapping (`gpu[mapped]`) and the guest CPU backing (`cpu[ok]`). Purpose: test whether the hung filter draws (PS `26ec5a6f42265b1f` / `6869ba5373a429b9`, indirect) read huge counts. Not yet exercised (no stall since).
- **Undocumented 14 Nov run reviewed.** `20261114-004155-perf-filter-gpu-word-record-off-20261114`: loop-bound word = 8 on the GPU in all 4096 samples, but the probe filled up before the stall-prone region and perturbs timing (EndRendering per sample). Static SPIR-V review: the loop terminates for any sane bound.
- **Driver log.** Only one nvlddmkm event 153 (engine reset) on 14 Nov, at 00:28:21 = the record-ON stall of run 002439. The record-OFF stall (003012) has no reset event.
- **Live results with the widened guard** (exe `D3FC10E9…`, same cache set as the stalled 003012 run: store FDDAB16B restamped, native E413):
  - `20261118-180532-…-record-off-…`: 578 s, no guard, no stall.
  - `20261118-181650-…-record-on-…` (default mask): 546 s, no guard, no stall; screenshot of the active fight is clean.
  - Two clean runs are not proof for an intermittent defect (≥5 needed if p≈0.5). No FPS claim (RTSS was injected).
- Tests: functional `shader_recompiler_compute` passes; `--stencil-storage-alias-only` passes functionally. Validation runs were not possible because the user's RTSS was running (the wrapper refuses; do not close RTSS yourself).

## 2026-11-18: stale BDA reads cause the GPU stall episodes (bit 58)

- `BufferCache::QueueBdaSync(..., is_write=true)` queued only pages in `m_dma_accessed`. A CPU write to a page that no BDA shader had read yet was never synced, so the first BDA read returned the stale GPU copy.
- UFC 6 compute shader `b059f33ca3077aac` (per-pixel list walk over BDA memory) then followed stale links: 56-130 ms per dispatch, the game's watchdog printed "GPU Stopped Working" (localLightShadow), FPS fell to ~1 for seconds.
- AbSwitch bit 58 `BDA_SYNC_ALL_WRITES` queues every CPU write. Watchdog reports 0/9 runs with the bit, 5/9 without; shader cost 1.25 ms/call vs 2.5-17 ms; CP-bound FPS neutral (same-process A/B median 1.015). Adopted into `DEFAULT_MASK` (491701838748053327).
- Limitation: more pages are uploaded at sync. An upstream-quality fix may sync on first fault instead; check upstream 9651a14e8 first.

## 2026-11-19: SRT walker port, readback timing, vblank quantization, PGO refresh (paused)

- Adopted in the working tree: port of upstream 46213dfd9 + 657a94e72 (shorter SRT walkers), keeping the local `LoadGuest` with `SRT_LOAD_CHECK`. Cross-run exe A/B with crowd: +2-4% FPS; every port run was faster than every valid base run.
- `KYTY_READBACK_TIMING=1` / `KYTY_QUEUE_TIMING=1` (diagnostic): readback waits are GPU execution of ~3 in-flight command buffers (~500 us per waiting readback, ~10% of the GPU thread), not record-thread lag; async compute queues are ~6% of the GPU thread.
- Fight frame times are vblank-quantized (median 25.0 ms without crowd, 33.3 ms with crowd at 120 Hz vblank). 60 FPS needs the CP frame below 16.7 ms: -32% without crowd, about -48% with crowd.
- New PGO profile collected (`pgo\20261119-intro-crowd+fight-nocrowd.profdata`), not yet built or measured. Two TDRs occurred only with the instrumented build; cause unknown.


## 2026-11-19 evening: SRT walker readback waits explained, bit 59, intermittent arena corruption

- New env-gated diagnostics (zero cost when off): `KYTY_SRT_DIRTY_STATS=1` prints `srt-dirty:` (walker loads of GPU-dirty memory by covering GPU write: writer/reader shader, size, in flight or done, changed/unchanged, wait) and `write-stats:` (shader-written bindings by writer); `KYTY_DUMP_PLAN=<hash>[;<hash>]` prints a shader's resource plan (descriptor sources, SRT reads, uniform fill, block conditions) once. Runs pass them with `start-run.ps1`/`run-series.ps1 -ExtraEnv`.
- Finding: ~10% of the GPU thread in the no-crowd fight is the JIT SRT walker waiting for compute shaders that are still writing small counter buffers (writers `ad1a9c7709e5b8f7`, `085e9fdd34251ae7`, `f9c1e0b0d34742fe`; readers `b54f4ecc141f6e16`, `9ad619ecb118aaaf`, `a2932b1ddf6e6544`). The loads only decide block conditions such as `!(buf1[buf0[0]] <= Undef)`, which are unknown unless the value is 0, so both successors are reached anyway.
- AbSwitch bit 59 `SRT_OPTIONAL_DIRTY_UNKNOWN` (uncommitted, not in `DEFAULT_MASK`): condition-only walker loads read as unknown instead of waiting when an in-flight GPU write covers them. Same-process A/B without crowd: FPS median 1.041 (15/17 pairs), 39.5 -> 42.2 FPS, but that run was visually corrupted in both phases, so no FPS claim yet.
- Intermittent blocky corruption of the no-crowd arena appeared in 4 of 7 runs on 2026-11-19 (also with bit 59 off; the same executable was corrupt once and clean once). Crowd runs and all no-crowd runs of the previous night were clean. Cause unknown; check screenshots before trusting any FPS number.
- The refreshed PGO profile was neutral (work/s 0.99) and is not used.
