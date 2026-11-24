# UFC 6 timing experiment: use the game's frame budget

**Experimental; disabled by default. Animation and physics validation pending.**
This JSON is an alternative to `realtime-100ms.json`, not a patch to combine
with it. On 2026-11-01, a separate enabled copy completed a controlled runtime
test. Its nominal budget tracked elapsed time in the measured windows; that
does not establish normal animation speed. The tracked JSON remains disabled
and the installed patch has not been replaced. This experiment has not
established a rendering-FPS gain and does not introduce independent simulation
updates.

## What static analysis establishes

For PPSA23566, eboot 01.005.000, module base `0x900000000`:

- `0x900035100` is the frame-sync function. Its output pointer is R14.
- The mode-2 path writes these four signed 64-bit durations in nanoseconds:

  | Offset | Value |
  |---|---|
  | `0x00` | nominal frame budget multiplied by the incoming float time scale |
  | `0x08` | nominal frame budget before the time-scale multiplication |
  | `0x10` | elapsed time since the previous sync completed |
  | `0x18` | elapsed time since this sync began |

- The existing hook starts at `0x9000353db`, branches from `0x9000353e1`
  into `0x90f000000`, and returns at `0x9000353e9`. The cave replays
  the two displaced stores before changing output `[0]`.
- Mode 0/1 exits via `0x90003519a` and never executes this hook.
- `0x900033530` copies the timing object to the loop object at `+0x4b0`
  and `+0x708`; `0x900033620` also copies it to `+0x530` and `+0x770`.
- One downstream consumer, `0x900033803`, converts timing `[0]` to a
  floating-point duration and passes it through virtual calls at `+0x50`.
  This does not establish that every animation/physics system uses this field.
- `0x900033530` also converts elapsed `[0x10]` and wait `[0x18]` to seconds
  before calling `0x903dd9ea0`. That destination is a bare `ret` in this
  executable; this particular elapsed-time call does not advance gameplay.
- The loop's virtual `+0x1f0` resolves to `0x90002f710`, then
  `0x90002f780`, which schedules several jobs with the whole timing object.
  One observed mode-2 loop iteration is not therefore proof that every
  gameplay subsystem executed one nominal simulation step.
- Existing `runs/time-dump7.txt` has timing objects at `+0x4b0` and `+0x708`
  with `[0] = [8] = 16666666`, even for examples with `[0x10] > 25000000`.
  These old samples alone do not identify the current scene or active patch.

The earlier `ufc6-game-speed.md` notes infer a 25 ms gameplay cap from a round
clock advancing at 0.609x wall time around 24.5 FPS. They explicitly state that
the cap itself was not located. `RoundTimerScale` appears in combat API metadata,
including `api_health_stamina_manager`, next to `RoundTimeRemaining`,
`RoundTimeElapsed`, `CurrentRound` and `RoundDuration`. Its active value is
unknown. `UseRealTimeClock` belongs to the generic
`PropertyInterpolatorEntityData`; its presence does not establish a selector
for the fight's round clock. Similarly, the identified `EnableFixedTimeStep`
property belongs to `EmitterSystemSettings`, not a global simulation setting.
These names alone do not validate any gameplay timing model.

UFC 6 also has offline slow-motion and situational time-dilation settings,
documented by [EA's accessibility guide](https://www.frostbite.com/able/resources/ufc/ufc-6).
Their state must be checked before evaluating emulator timing changes.

## Controlled observations, 2026-11-01

The `runs/` paths below are relative to `D:\Projects\kyty-dev`; `_Build/`
paths are relative to this repository.

The Settings > Gameplay > Combat Style screen shows **Competitive** and
**Clock Speed: Accelerated**. This was checked in
`runs/20261101-122433-perf-speed-settings-adjust/shots/0078s.png` and again
in `0244s.png`. Time Dilation Assists was On, while Constant Slow Motion was
Off. The assists master was temporarily disabled during inspection, then
restored and verified before continuing. A master setting of On does not
mean that a slowdown is continuously active; situational effects must be
considered separately.

The accelerated-clock setting is direct evidence that one displayed round
second must not be assumed to equal one real second. Its exact multiplier
has not been measured. The proposed factor 1.5 remains an explanation to test,
not a measured round-clock scale or proof of animation speed.

With the vertex/index probe disabled, the installed elapsed/25 ms patch (A2)
and the temporary base-budget candidate (B) report:

| Variant | Scene | Mode-2 updates/s | Elapsed p50 | Timing scale p50 | Nominal/elapsed p50 |
|---|---|---:|---:|---:|---:|
| A2: elapsed/25 ms | Fighter selection | 57.667 | 17.254816 ms | 1.000000 | 0.965914 |
| A2: elapsed/25 ms | Standing fight | 29.776 | 33.275922 ms | 1.331037 | 0.666667 |
| B: elapsed/base budget | Fighter selection | 57.982 | 17.252386 ms | 1.035143 | 1.000000 |
| B: elapsed/base budget | Standing fight | 29.131 | 33.745078 ms | 2.024705 | 1.000000 |

The base budget was 16,666,666 ns. All 922 selected selection dumps and 863
fight dumps passed the layout checks, with zero mismatches against the
installed patch's arithmetic. The summaries are
`_Build/diagnostics/speed-auto-20261101/selection-vi-off-timing.json` and
`fight-vi-off-timing.json`, derived from `timing-control-vi-off.txt` with an
explicit GameSimulationLoop thread ID.

The A2 fight's two-thirds ratio establishes the relationship between these
engine fields under the installed patch. It is not a direct measurement of
animation or physics speed. Conversely, the earlier title screen ran about
120 mode-2 updates/s with nominal budget 16.67 ms and elapsed about 8.33 ms;
that observation rules out treating nominal budget times loop frequency as a
universal gameplay clock.

The candidate's separate enabled copy completed
`runs/20261101-122935-perf-speed-base-budget` normally with exit code 0:
228 seconds total, including approximately 100 seconds of fighting. Its trace
is `_Build/diagnostics/speed-auto-20261101/timing-base-budget.txt`. In the
comparison window from process second 122 through 150, the mode-2 interval
p50 was 33.841 ms, the nominal/elapsed ratio was 1, and the analyzer reported
zero patch-arithmetic mismatches and zero unattributed dumps. The table's B
fight values refer to this window.

The runtime test therefore confirms that the candidate changes the measured
engine budget from the installed patch's two-thirds relationship to elapsed
time. It does not yet establish animation or physics speed, nor a rendering-FPS
gain. Whole-fight FPS medians of A2 29.24 and B 30.26 came from different scene
histories and are not a controlled estimate of the patch's performance effect.

Combat Preset remained Competitive and Clock Speed remained Accelerated.
Time Dilation Assists was restored to On and Constant Slow Motion remained
Off. Human feedback about the actual movement and animation speed is pending.
Only the temporary `_Build` JSON was enabled; the tracked source JSON remains
disabled and the installed patch remains unchanged.

## Candidate semantics

Preserve the original timing value unless the game's budget is positive and
`min(elapsed, 100 ms) > budget`. Otherwise:

```text
timing[0] = trunc(original_timing[0] * min(elapsed, 100 ms) / timing[8])
```

The full-width signed IMUL/IDIV preserves an incoming pause or time-scale
multiplier. The budget-zero guard prevents division by zero. Negative budgets
also skip the change. The cap is applied before the comparison, so a nominal
budget above 100 ms is not reduced accidentally.

For an incoming scale of 1 and budget 16,666,666 ns:

| FPS | Elapsed | Existing elapsed/25 ms output | Candidate output |
|---|---:|---:|---:|
| 20 | 50 ms | 33.333332 ms | 50 ms |
| 30 | 33.333333 ms | 22.222221 ms | 33.333333 ms |
| 40 | 25 ms | 16.666666 ms | 25 ms |
| 60 | 16.666666 ms | 16.666666 ms | 16.666666 ms |

The candidate is about 1.5x stronger than the current patch in the 20-40 FPS
range. It may therefore run gameplay too fast if the existing 25 ms assumption
is correct. If timing `[0]` is the relevant fixed simulation step, its proposed
relationship to wall time is the reason to test this alternative.

## Reproduce the offline verification

```powershell
& .\scripts\patches\build-ufc6-realtime-base-budget.ps1
```

This assembles the tracked `.s` source using Clang's LLVM integrated assembler,
links both sections at their actual guest addresses using LLD, extracts the
bytes with llvm-objcopy, and verifies both jump destinations through
llvm-objdump. It regenerates the JSON with `enabled: false` and writes build
artifacts below `_Build/diagnostics/base-budget-patch` only. It never deploys a
patch or starts the emulator.

Verified output: hook 14 bytes; cave 53 bytes. The original hook bytes are
`4c89f84c29e0498946184c897b38`; both JSON byte sequences are generated from the
assembled ELF. Eleven arithmetic reference cases pass, including 20/30/40/60
FPS, a 150 ms hitch, zero/negative budget, a pause, and a half initial scale.
These are formula/reference checks, not execution tests inside the game.

## Visual feedback: 2026-11-01

A later user-review run used the installed release-d executable with the
experimental patch supplied explicitly, with the timing and vertex/index
probes off. The user reported that the fight was faster at about 30 FPS,
but **too fast**, and closed the game. The base-budget candidate is therefore
not accepted as normal gameplay speed. Matching the engine budget to elapsed
time did not establish the desired animation or gameplay rate.

The review macro did not recognize the fight HUD and timed out; its automatic
pause step was never reached. Subsequent automation stops at fighter selection,
then checks test-window captures and uses the emulator virtual-pad protocol.
The user explicitly authorized autonomous selection and occasional face-button
presses in fights on November 1; no user intervention is required for navigation.

The installed executable and patch remain unchanged. Contextual Time Dilation
was restored to its original enabled value; constant slow motion remains off.
The next prepared experiment uses a fixed 20 ms divisor: about 1.25 times the
legacy patch in the 20-40 FPS range, between legacy25 and this base-budget
candidate. It is a calibration experiment, not a validated 1x simulation fix.

## November 1 follow-up: replay control and automatic HUD clock

The user played two fights with the 20 ms candidate and reported a slightly
fast round clock, then a freeze of the fighters during strike replays twice.
Crowd and menu animation continued. Read-only captures in
`_Build/diagnostics/replay-freeze-20261101-1736` include host/guest stacks and
four timing-object samples. The sampled replay updates had positive changing
elapsed values (16.86-17.57 ms) and a nominal 16.67 ms budget; the 20 ms patch
was inactive for those particular updates. This does not rule out a transition
or earlier state affected by the patch. No root cause is established.

The original installed 25 ms patch was used for the next control fights. The
installed executable and patch hashes remain unchanged. After an accidental
window close, autonomous navigation restarted the game in run
`D:/Projects/kyty-dev/runs/20261101-181049-perf-replay-ocr-25ms`. The control
advanced into the second round; this is not proof that the problematic strike
replay was exercised. The user also freed disk space, which confounds comparison
with the earlier failures; no disk-full error was found in the captured logs.

`../Ufc6RoundClock.psm1` reads the standing and ground HUD regions with local
Windows OCR. It widens the condensed font, requires agreement of at least two
variants, and rejects conflicting readings. Seventeen labeled screenshot checks
passed, including menus with no timer. `../watch-ufc6-round-clock.ps1` captures
only the emulator test window and records the HUD value, monotonic capture time,
phase, FPS and measurement cost. Missing HUD, phase changes and resets split
measurement segments. `-CheckExpectedRate` is opt-in: **Accelerated clock mode
does not imply one displayed second per real second**. RealTime mode and a
verified reference are still needed to label physical gameplay speed as normal.

A first 60-second live OCR run accepted 26/31 samples and measured standing and
ground clock rates. Its FPS and clock-rate results are not calibration evidence:
an unrelated LM Studio server occupied roughly 16-17 GiB of dedicated GPU
memory, with the emulator using another 4-7 GiB and spilling into shared memory.
Menus fell to 2 FPS and fighting ran near 13 FPS. The 100 ms timing cap and
memory pressure make this unsuitable for normal-speed or FPS A/B conclusions.

## Validation still required

1. Both variants' timing values are captured above. Any follow-up capture must
   use the same verified settings and an explicitly selected scene window.
   At the clock call with return address `0x9014ff59c`, the virtual-clock
   wrapper clobbers R14/R15. To recover the frame-sync output pointer, walk to
   the RBP frame whose return address `[rbp+8]` is `0x9000351b3`, then read
   the wrapper's saved R14 at `[rbp-8]`. This first clock call of a new
   mode-2 update can expose the previous update's completed timing values.
   The final clock parent `0x9000353bb` occurs before the patch and mixes a
   newly written base-scale value with older elapsed fields; do not treat it
   as evidence of the completed patch result.
2. Keep the verified clock-speed and slow-motion/time-dilation settings
   constant across variants; account for intentional situational effects.
3. Compare repeatable animation duration, movement distance and round-clock
   rate separately across rendering rates and both patch variants. A console
   reference or another known-normal animation reference is needed to label
   either result as 1x gameplay speed.
4. Validate striking, grappling, hit detection, audio and replays before
   accepting the candidate. Neither formula adds fixed-step subupdates, so
   matching wall time does not imply console-equivalent physics cadence.
