KytyPS5 UFC 6 - experimental Windows x64 build, 2026-11-24

This preview combines current upstream fixes with the local UFC 6 development
work. It is not a stable release and does not promise 60 FPS combat. GPU watchdog
warnings, graphical corruption, severe menu slowdowns and replay problems remain
under investigation. Do not use the accelerated round clock to measure physics.

Extract the complete archive, preferably on D:, and start the included launcher.
No game files, saves, keys, personal settings or shader caches are included.
Use your own extracted game. New compiler/profile changes require cache rebuilds;
first-run shader/pipeline compilation is not a performance benchmark.

Optional UFC 6 helper (PowerShell, example path; replace it with your own):

  powershell -ExecutionPolicy Bypass -File .\Run-UFC6.ps1 -GamePath "D:\games\UFC6"

The helper reports 4K output in a 1280x720 window; the game's dynamic internal
resolution remains active. Add -Output1080p for the lower-output mode. Window size
does not establish internal resolution.

The helper applies the previously tested elapsed/25ms timing adjustment, for
PPSA23566 version 01.005.000 only. It changes in-memory gameplay timing and is not
proof of correct wall-clock physics or replay behavior. Add -OriginalTiming to
disable it. The normal launcher does not automatically apply this patch.

Experimental allocation control:

  powershell -ExecutionPolicy Bypass -File .\Run-UFC6.ps1 -GamePath "D:\games\UFC6" -ExperimentalUnmappedBuffers

This selects GPU buffers without a CPU mapping. Previous controls avoided the
worst menu slowdown, but a pre-fight watchdog still occurred and combat did not
reach 60 FPS. It is opt-in; the normal allocation default is unchanged.

Research scripts in the source repository include rejected/unvalidated timing
and crowd experiments. They are not enabled by this package. Optional Streamline
frame generation is not part of this binary package's supported feature set;
no NVIDIA SDK/runtime or generated-frame FPS claim is included.

See the GitHub release notes, BUILD-INFO.json and SHA256SUMS.txt for the exact
source/build provenance and validation. Detailed investigation entry point:
docs/ufc6-rendering-investigation.md in the source repository.
