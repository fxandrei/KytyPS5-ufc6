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

  powershell -ExecutionPolicy Bypass -File .\Run-UFC 6.ps1 -GamePath "D:\games\UFC 6"

Latest integrated live controls: default BAR allocation reached one visually
clean 1080p fight and accepted four attack buttons. The bounded compilation-free
sample was about 47 FPS; this is not an A/B improvement or a 60 FPS result.
A pre-fight guest watchdog still occurred. The 4K control showed major graphical
corruption and severe slowdown before combat. Use this preview for testing.
For the configuration that reached combat, add -Output1080p to the example above.

The helper reports 4K output in a 1280x720 window; the game's dynamic internal
resolution remains active. Add -Output1080p for the lower-output mode. Window size
does not establish internal resolution.

The helper applies the previously tested elapsed/25ms timing adjustment, for
PPSA23566 version 01.005.000 only. It changes in-memory gameplay timing and is not
proof of correct wall-clock physics or replay behavior. Add -OriginalTiming to
disable it. The normal launcher does not automatically apply this patch.

Experimental allocation control:

  powershell -ExecutionPolicy Bypass -File .\Run-UFC 6.ps1 -GamePath "D:\games\UFC 6" -ExperimentalUnmappedBuffers

This selects GPU buffers without a CPU mapping. The current integrated 4K and
1080p controls using this experiment ended with fatal Vulkan semaphore waits
before combat. Earlier revisions behaved differently; it is not recommended as
a performance option for this snapshot. The normal allocation default is unchanged.

Research scripts in the source repository include rejected/unvalidated timing
and crowd experiments. They are not enabled by this package. Optional Streamline
frame generation is not part of this binary package's supported feature set;
no NVIDIA SDK/runtime or generated-frame FPS claim is included.

See the GitHub release notes, BUILD-INFO.json and SHA256SUMS.txt for the exact
source/build provenance and validation. Detailed investigation entry point:
docs/ufc6-rendering-investigation.md in the source repository.
