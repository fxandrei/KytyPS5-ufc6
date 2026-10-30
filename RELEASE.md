# KytyPS5 UFC 6 performance build

**Windows x64 binary release of the `KytyPS5-ufc6` fork.**
Built from the `ufc6-perf` branch of
[fxandrei/KytyPS5-ufc6](https://github.com/fxandrei/KytyPS5-ufc6), on top of upstream
KytyPS5 `4919fdb6` (2026-10-30). Source tag for this build: **`ufc6-v20261030`**.

This is **not** an official KytyPS5 release. It is a community fork focused on EA SPORTS UFC 6
(`PPSA23566`, v01.005.000), distributed under **GPL-2.0-only**. It contains no game files.

---

## What this fork adds over upstream

The fork keeps upstream's emulator and adds a set of measured performance switches for the GPU
command processor (the emulator thread that turns the game's PM4 command stream into Vulkan calls).
In UFC 6 the fight is command-processor-bound: the GPU is busy only about 16-17 ms per frame while
the frame takes 30-32 ms.

| Area | Change | Measured effect |
|---|---|---|
| Command recording | The GPU thread's Vulkan commands are deep-copied into a ring and recorded on the submit thread, which also submits in order (`RECORD_THREAD`, bit 37). | In-fight A/B over 16 pairs: **+18% FPS and +19% draws/s** (25.7 -> 30.8 FPS). ~895k packets/s, ~155 MB/s, zero direct calls/drains in the fight. |
| Draw chain | A draw keeps the previous draw's resolved render targets while no command that could change them ran and the same rendering instance is open; it then skips resolving the registers and acquiring the targets again (`DRAW_CHAIN`, bit 45). | **+2-3%** in-fight A/B (median ratio 1.023-1.029, faster in 69-75% of pairs). About 56% of the fight's draws take the previous targets (~79k of ~142k draws/s). |
| Submission and readback | Queued submits, urgent submit, eager flushes, lazy GPU polling, spin-before-wait, direct readback through the resizable BAR, per-range write ticks, labels written in command order. | Cumulative; removes most of the CPU-side wait for short GPU readbacks. |
| Caching and deduplication | Image pool, target/texture/LRU memos, shader map cache, image start index, exact draw state, pipeline/compute bind deduplication. | Cumulative; the command processor spends less time on repeated lookups and binds. |
| Stability | Render-area clamp, depth-copy clamp at 1080p, depth-target recreation, HTile/stencil ownership fixes. | Fixes the `3D WIDTH ZT Violation` TDR in long fights and the 1080p device loss from the depth copy; also the fighter-select blocks/green screen. |
| Game speed | `_Patches/PPSA23566.json` (not code in the emulator). | Removes the game's 25 ms gameplay slow motion below 40 FPS; the 100 ms catch-up covers down to 10 FPS. |
| PS5 Pro | Decoder fixes for the `--trinity` path. | Experimental only; the Pro path still renders black 3D scenes. Leave it off. |
| Diagnostics | A/B mask (`KYTY_AB`), per-second CSV, record-thread and draw-chain counters in the performance panel/CSV. | For testing; the CSV and overlay are off by default, and the counters are cheap. |

The cumulative result on the test machine is **about 31-33 FPS standing at the start of the fight**
with a 1080p internal render and a 4K output. Details in [Performance](#performance).

The performance changes above are individual A/B switches in `src/common/abSwitch.h`. The ones that
measured faster and showed no problem are in `DEFAULT_MASK`; the rejected and diagnostic ones are
off. `KYTY_AB=0` disables them all for comparison, and `KYTY_AB=<mask>` enables a specific set; see
[Troubleshooting](#troubleshooting) before changing it. The build also includes small
command-processor cleanups from the same session (batched counters, cheaper dirty-range checks,
buffer-cache fast paths); they are not individually user-visible.

---

## Requirements

- Windows 10 version 1803 or newer, or Windows 11, x64.
- A Vulkan 1.3 GPU with a current driver. This build was tested on an NVIDIA RTX 4090; other
  Vulkan 1.3 GPUs are untested with these settings.
- **32 GB of RAM recommended (16 GB minimum).** The emulator reserves about 12.7 GB of direct
  memory for the game at boot; with 16 GB, close browsers and other memory-heavy applications
  first, or startup fails with "could not reserve ... direct memory".
- A legally obtained dump of **EA SPORTS UFC 6, title ID `PPSA23566`, version `01.005.000`**. The
  game-speed patch is version-specific and matches this version only.
- A few hundred MB of free disk space for the shader cache (`_PipelineCache`).

No PlayStation firmware, keys, game files, or saves are included or required beyond your own dump.

---

## Install

1. Download the release zip and **extract it to a writable folder** (for example
   `D:\KytyPS5-ufc6`). Do not run it from inside the zip, and do not put it in `Program Files`:
   the emulator writes its shader cache and logs next to the executables, and the launcher reads
   `_Patches` from there.
2. Keep the folder structure intact. `kyty_launcher.exe` needs the Qt DLLs, the plugin folders
   (`platforms`, `styles`, `imageformats`, `iconengines`, `networkinformation`, `tls`, `generic`),
   `assets`, and `_Patches`.
3. Run `kyty_launcher.exe`.
4. Add the folder that contains your game dump (the launcher scans it for `eboot.bin` and
   `param.json`), then select **EA SPORTS UFC 6** in the list.
5. Open the game's settings and apply the values in [Recommended settings](#recommended-settings).
   Save.
6. Start the game. The first launch compiles shaders and may stutter for several minutes; see
   [First run: shader cache](#first-run-shader-cache).
7. The game is offline: dismiss the "Failed to download Roster Update" and PSN messages with
   **Cross** (keyboard: **J**). The title screen and main menu follow; the round starts after the
   walkout.

This build is **unsigned**, so Windows SmartScreen may show a warning. Verify the SHA-256 of the
zip against the one on the release page before running it.

This is a **Source/Fork build**, not an Official build: the launcher's upstream update checker is
disabled in non-Official builds. There are no automatic updates; download new releases from this
repository manually.

---

## Recommended settings

These are the settings behind the performance numbers below. They are deliberately conservative:
image correctness first, then frame rate.

### Launcher

| Setting | Value | Why |
|---|---|---|
| Screen resolution | `3840x2160` | 4K output for recording. Use `1920x1080` if you only need a 1080p capture. |
| Run in fullscreen | On (borderless) | Stable 4K presentation; no window chrome in the capture. |
| Present mode | `Mailbox` | Low-latency presentation; avoids the FIFO queue adding a frame. |
| Vblank frequency | `120` | The game sees a 119.88 Hz display, so its 30 Hz menu cap presents at 60 FPS; the fight also gets ~4% from less frame-time quantisation. Do not use 60. |
| GPU | The RTX 4090 (index `0`), or Auto if you have a single GPU | Keeps the emulator on the discrete GPU. |
| Shader optimization | `Performance` | Optimise generated shaders for speed. |
| Cache shaders | On | Without it, every launch recompiles pipelines. |
| Enable tessellation support (experimental) | On | Correct UFC 6 geometry; the fight is CPU-bound, so turning it off is not a useful speed-up. |
| Enable occlusion queries | Off | Costs GPU/CPU time; UFC 6 does not need it for correctness here. Turn it on only if you see hidden lights/objects popping. |
| Enable HDR display support | Off | SDR capture; HDR also needs a compatible host display and adds work. |
| Show performance panel | Off | It costs a little and appears in the capture. Press **F2** to toggle it temporarily, **Shift+F2** for details. |
| Tracy profiler | Off | Costs several percent. |
| Vulkan validation | Off | Currently exits on `E5B9G9R9`; see [Known issues](#known-issues). |
| Shader validation | Off | Debug tool; costs time. |
| RenderDoc capture | Off | Debug tool. |
| Command buffer dump | Off | Debug tool; huge files. |
| Shader logging / Printf output | `Silent` | `File` only while diagnosing a problem; logging costs time. |
| Skip boot logos | On | Faster startup. |
| Enable trophy notifications | Off | Clean capture; no pop-ups over the fight. |
| Auto-hide cursor | On | The cursor leaves the capture. |
| Stretch to window | Off | The game is 16:9; stretching distorts it. |
| Pro mode (experimental) | Off | The PS5 Pro path is not usable yet. |
| Enable network connection | Off | The game is offline; fewer online timeouts. |
| Windows SysV red zone crash protection (experimental) | On | Default on Windows; keep it. |
| Accurate texture synchronization | Off | Compatibility option; costs performance. Enable only if you see missing textures. |

### NVIDIA Control Panel

1. **Manage 3D settings -> Program Settings -> Add -> `kyty_emulator.exe`** (the one in your
   extracted folder).
2. Set **Power management mode = Prefer maximum performance**.
3. Apply. Restart the emulator.
4. Check the actual state during the first minute of a fight:

   ```powershell
   nvidia-smi --query-gpu=pstate,clocks.gr,power.draw --format=csv -l 1
   ```

   The RTX 4090 should be in **P0** at roughly 2.6-2.8 GHz. In **P3** (about 1.5-1.8 GHz) the
   driver halves the effective readback speed and you lose about 10% FPS. If it stays in P3,
   set the same option globally or reboot; some drivers ignore the per-program setting until the
   next start.

Optional: set the Windows power plan to High performance and close GPU-monitoring overlays
(RTSS, Afterburner) for the final recording. RTSS costs a little and its overlay may end up in the
capture.

---

## The game-speed patch

The release includes `_Patches/PPSA23566.json`. `kyty_launcher.exe` applies it automatically when
it launches UFC 6, because the file name matches the title ID. You do not need to pass any command
line option. If you start `kyty_emulator.exe` directly, pass `--game-patch _Patches\PPSA23566.json`
yourself. Delete the file (or disable the entry in the launcher's **Cheats (experimental)** dialog)
to disable it.

**Why it is needed.** UFC 6 advances at most **25 ms of game time per rendered frame**. At 31-33 FPS
a frame takes 30-32 ms, so without the patch the fight would run at about **0.78-0.83x** real time;
at 24 FPS it runs at 0.61x. The patch hooks the game's frame-sync code and, when a frame takes
longer than 25 ms, scales the gameplay step by `elapsed / 25 ms`. At 31-33 FPS the scale is about
1.2-1.3, so the round clock advances in real time. At 40 FPS or more the patch is a no-op.

**The 100 ms catch-up.** The elapsed time used for the scale is capped at **100 ms**, not 50 ms.
The previous 50 ms cap lost time on hitches below 20 FPS; 100 ms keeps the game real-time down to
10 FPS. Below 10 FPS the cap limits catch-up to 4x, so a very long hitch cannot teleport the fight.

The patch is a gameplay modification, not an emulator speed-up: it does not change FPS or render
resolution. It matches only `PPSA23566` v01.005.000; if your dump has another version, the emulator
logs `cheat file does not match the loaded game` and skips it. To confirm it applied, temporarily
set **Printf output** to `File` and look for:

```
Successfully applied cheat: Real-time game speed: ...
```

in `_kyty.txt`. Set Printf back to `Silent` afterwards.

---

## First run: shader cache

The emulator compiles the game's shaders and Vulkan pipelines on first use and stores them in
`_PipelineCache`. **The first run after installing a new build is cold:** expect stutter for
several minutes while pipelines compile (individual compiles can take hundreds of milliseconds,
with occasional multi-second peaks). Do not judge FPS, record, or benchmark during this warm-up.
Play a fight, then restart the emulator for a clean measurement.

The cache is keyed to the executable's timestamp. Replacing `kyty_emulator.exe` (a new release)
invalidates it; an old `_PipelineCache` next to a new executable is simply ignored and the first run
is cold again. The release zip deliberately contains no `_PipelineCache`: a cache from another build
can be stale, and it is large.

If the first run stalls, see the cold-cache issue below; retry once. The cache written before the
stall is kept, so the retry is warmer.

---

## Performance

Measured on **Ryzen 7 9800X3D + GeForce RTX 4090**, 3840x2160 output, 1080p internal render,
Windows 11, current NVIDIA driver:

| Scene | FPS (median) | Notes |
|---|---|---|
| Fight, standing at the start | **31-33** | ~4,400 draws/frame; the number this release targets. |
| Fight, ground/clinch | 40-48 | Fewer draws; the command processor is less loaded. |
| Main menus | 50-60 | The game caps the menu at 30 Hz; with vblank 120 the emulator presents 60. |
| Title screen | 60 | Not representative of the fight. |
| Post-fight/replay | Lower | Very heavy scene (~6,900 draws/frame); not a benchmark. |

Expect **±5-8% between fights**: the AI fights differently and the camera cuts change the draw
count. Differences of 1-2% are noise unless measured A/B in the same fight.

This is a **CPU-bound** result, not a GPU-bound one. The GPU is busy about 16-17 ms per frame while
the frame takes 30-32 ms; the command processor is the limit. A faster GPU will not move the fight
much; a CPU with a large cache (9800X3D class) does. 60 FPS in the standing phases is not reached by
this build; 31-33 FPS is the realistic result at these settings.

The **25 ms** in this document is the game's gameplay simulation cap (see
[The game-speed patch](#the-game-speed-patch)), not a frame-rate cap. The fight itself asks for
60 Hz; the emulator runs as fast as the command processor allows.

The game's **dynamic resolution** (Frostbite DRS) lowers the internal render size when the frame
time rises. Kyty reports guest GPU timestamps at command-processor time, so DRS reacts to CPU frame
time, waits included. In the fight the internal buffer sits around **1920x1080** (occasionally
1600x900); the 3840x2160 output is an upscale of that. Do not expect native 4K pixels, and raising
the screen resolution alone will not add detail that the game is not rendering.

If you see about 27-28 FPS instead of 31-33, check, in this order: the GPU is in **P0**, the shader
cache is warm (not the first run), the performance panel and Tracy profiler are off, the game-speed
patch is applied, and you are on the latest build. A cold cache or P3 alone explains the difference.

---

## Known issues

- **Cold-cache GPU stall.** Seen twice: on the first run after a shader-cache reset, the GPU can
  stall during the first fight (the picture freezes and the emulator stops presenting). Close the
  emulator and start it again **once**; the cache written before the stall is kept, and the second
  run is normal. If it repeats, delete `_PipelineCache`, reproduce once more, and report it with
  the log; do not retry in a loop.
- **Vulkan validation exits on `E5B9G9R9`.** With **Vulkan validation** enabled, the emulator exits
  during boot on a `vkCreateImageView` for the `E5B9G9R9` format: `FindView` asks for
  `COLOR_ATTACHMENT` usage regardless of the usage the caller requested. This is a
  validation-only bug; the normal path is fine. Keep validation **off**.
- **PS5 Pro path is experimental.** **Pro mode (experimental)** (`--trinity`) may boot, but the 3D
  scenes are black (PSSR 2 entry points are missing and stores through GPU-built descriptors are not
  emulated), and it can lose the device. Leave it off for UFC 6.
- **GPU power state.** If the NVIDIA driver keeps the card in P3, expect about 10% lower FPS and
  longer readback waits. Set **Prefer maximum performance**; see above.
- **Menu is capped at 30 Hz by the game.** 50-60 FPS in the menu is the emulator presenting the
  game's 30 Hz flips at 120 Hz vblank, not a 60 FPS menu. Do not use the menu as the fight
  benchmark.
- **Offline pop-ups.** "Failed to download Roster Update" and the PSN sign-in message are normal
  for an offline dump; dismiss them with Cross.
- **Minor visual artifacts.** Some hair/skinning speckles and occasional UI blotches remain in
  certain scenes; they are upstream renderer limitations, not caused by the performance switches.
- **Unsigned executable.** SmartScreen may warn. Compare the zip's SHA-256 with the release page.
- **This is a UFC 6-focused build.** The switches are tuned and validated on UFC 6. Other titles
  may work, but use upstream KytyPS5 for general compatibility and bug reports.

---

## Troubleshooting

- **Stutter for minutes, then smooth:** shader cache warm-up; let it finish and restart.
- **FPS lower than 31-33:** check P0, warm cache, overlays off, patch applied, no background
  memory pressure; see [Performance](#performance).
- **Fight feels slow:** the game-speed patch is missing or did not match the game version. Check
  `_Patches/PPSA23566.json` is next to `kyty_launcher.exe`, the game is v01.005.000, and the log
  says the cheat applied.
- **Rendering corruption:** try `KYTY_AB=0` in the environment to disable all performance switches
  and compare. If the corruption disappears, report the scene and the build with the mask; keep the
  default mask for normal play.
- **Black 3D scenes:** make sure **Pro mode (experimental)** is off.
- **Validation abort:** keep Vulkan validation off.
- **Startup fails with "could not reserve ... direct memory":** close memory-heavy applications;
  12.7 GB of commit must be free.

For bug reports, include the build label from the window title or crash report (it contains the git
hash), `_Logs`, the GPU model and driver, the NVIDIA P-state, and whether the shader cache was warm.
Open issues at [fxandrei/KytyPS5-ufc6/issues](https://github.com/fxandrei/KytyPS5-ufc6/issues).

---

## License and attribution

KytyPS5 is free software under the **GNU General Public License version 2 only**
(`GPL-2.0-only`). This fork and this binary are distributed under the same license.
The full text is in **`LICENSE`** at the root of the zip.

**Source availability.** The complete corresponding source for this binary is the public tag
**`ufc6-v20261030`** in
[fxandrei/KytyPS5-ufc6](https://github.com/fxandrei/KytyPS5-ufc6). GitHub also attaches
`Source code (zip)` and `Source code (tar.gz)` to the release page. Those archives contain the
repository at the tag; the `3rdparty` submodules are pinned by `.gitmodules` to specific commits and
are fetched by the build. Keep the repository and the tag public for as long as you distribute the
binary. The changes relative to upstream are the commits on `ufc6-perf` between upstream `4919fdb6`
and this tag; the binary was built from that tag, not from a dirty tree.

**If you redistribute this zip**, you must:

- keep `LICENSE` and this notice;
- keep the source available under GPL-2.0 (the public tag, or a written offer valid for three
  years);
- keep the copyright and attribution notices, including the original Kyty MIT notice
  (`licenses/Kyty-MIT.txt`) and the third-party notices in `licenses/`;
- not add restrictions beyond GPL-2.0, and not present the build as an official KytyPS5, Sony, or
  EA release.

**No game files.** The zip contains no PlayStation game data, no `eboot.bin`, no `sce_sys`, no
saves, no keys, and no firmware. `_Patches/PPSA23566.json` contains only guest addresses and byte
patterns for the user's own dump; it is not a game asset. You must own UFC 6 to use it.

**Third-party components.** The zip includes components from Qt 6.10.3 (LGPLv3; unmodified runtime
DLLs, replaceable with your own Qt 6.10.3 build), FFmpeg, Opus, Noto Sans, and the OpenXR loader.
Their licenses and notices are in `licenses/`. Keep that folder in any redistribution.

KytyPS5 is not affiliated with Sony Interactive Entertainment or Electronic Arts. PlayStation is a
trademark of Sony Interactive Entertainment; EA SPORTS UFC is a trademark of Electronic Arts. This
project does not distribute games or copyrighted system software.

**No warranty.** The binary is provided "as is", without any warranty, to the extent permitted by
law.

---

## Credits

- [KytyPS5](https://github.com/KytyPS5/KytyPS5) upstream developers, and
  [InoriRus/Kyty](https://github.com/InoriRus/Kyty), the original project (MIT).
- BryanKAdams, for the public GPL-2.0 command-stream/record-thread work that the record thread is
  ported from.
- Bruno Cardenas Torres, for the public draw-chain design (`28073f9ed`) that `DRAW_CHAIN` follows.
- chenxiao07, for the image-start-index idea (`05287b245`), and Cozmaaa for the buffer-idle-GC
  experiment (measured and rejected for UFC 6, but kept in the source as a switch).
- Everyone who tested UFC 6 builds, reported scenes, and sat through the A/B runs.

Maintained by **fxandrei**. This is a community fork; please report UFC 6 results and regressions
on the fork's issue tracker, not upstream.
