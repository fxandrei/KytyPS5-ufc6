# UFC 6 development preview, 2026-11-24

This update preserves the local UFC 6 investigation and integrates upstream through `b446af9d26e000b5dd93ecfc8bf5403ad500e9fd`. The pre-integration source is retained in commit `8d40d8b2a` and backup branch `backup/ufc6-local-20261124`. Publication to `fxandrei/KytyPS5-ufc6` was explicitly requested by the owner. The release is experimental; 60 FPS combat and GPU/replay stability are not established.

The integration includes upstream stencil-only/shared-stencil rendering, descriptor resolve caching, per-draw-window stream-copy reuse, SRT backing reads, separate pixel/vertex push-data windows, and kernel/audio/shader compatibility changes. Local record-thread, BDA-slice, resource-memo, read-binding and GDS work is retained. GDS CPU reads have explicit host visibility; optional buffer-allocation and timing diagnostics remain opt-in. The old bit58 selective-write behavior is superseded by unconditional queuing of CPU writes, matching the previous local default and current upstream correctness policy.

Push-data memo keys now cover both ends of the upstream stage-specific window. Metadata tracks both guest surface identity and native image ownership, preserving pending HTile clears when an attachment grows. The label-based flip interface retains nonblocking guest-queue suspension. Clang/MSVC OpenAL builds use a small root-owned compatibility preinclude for function-effect annotations; the third-party submodule is unmodified.

The large GPU test file is based on the coherent current upstream fixture APIs, with the local isolated stencil-storage, bounded volume/array-copy and GDS-readback regressions and offline ISA decoder retained. Old BDA accessed-bit tests are superseded by the upstream fault-list representation. Unit-test memory stubs match the local optional walker diagnostics. A historical failing stride assertion and all new validation results must be reported accurately; a functional pass is not a Vulkan-validation pass.

Known repositories refreshed/reviewed: upstream, the personal fork, chenxiao07, todojuegos67-bot, yumlevi, Jetsku, Cozmaaa, AstroHybrid, Senaxx, ElyFura, joshweible and NHL27PC. Chenxiao's optional WEAVE/mode4 work is a substantial Demon's Souls-specific parallel architecture; AstroHybrid's new launcher/dual-GPU/frame-generation changes target a different runtime. They are not blindly enabled here. Senaxx's 64MiB VMA block setting is a follow-up lead for allocator pressure, not a proven UFC 6 fix. Earlier rejected speculation/memo/batching experiments remain rejected unless new evidence justifies a controlled test.

The November23 unmapped-buffer Apex control measured approximately39 FPS in a bounded active-fight interval without new compilations. That is historical evidence for a different source revision, not a performance claim for this integration. Previous watchdogs, graphical corruption and severe BAR-path menu slowdowns remain material limitations. The game uses an Accelerated round-clock mode.

Build/test/package results and final commit, executable hashes and release URL are recorded in the published release assets and the current handoff. The portable package excludes game dumps, saves, keys, personal configuration, caches, SDK runtimes and user uploads. The generic launcher applies no game patch automatically. The optional UFC helper uses only the previously tested25ms timing patch and exposes the unmapped-buffer backend explicitly as an experiment.

Validation checkpoint before live control:

- Emulator, launcher, GPU test executable and three kernel regression targets build successfully.
  Compiler warnings remain; no warning-free-build claim.
- Eight isolated GPU functional controls pass, including resolve-cache invalidation, native indirect
  arguments, stencil-storage round trips, bounded volume/array copies and GDS/BDA/ordered visibility.
- Eleven CPU/resource/kernel CTests pass. The complete GPU suite still fails the inherited adjacent
  fault-download stride assertion; checks after it are not claimed. The kernel filesystem suite uses
  a Linux `/proc` sleeping-thread probe that cannot work on Windows. Its existing network-only mode
  also fails the initial NP callback state assertion. These are explicit remaining limitations.
- Texture and color memos now honor actual upstream resolve stamps; zero/ambiguous results cannot
  bypass clear-key invalidation. Buffered stencil bridges retain GENERAL, ordinary transfers use the
  upstream layout contract. S8 plane reparenting and indirect writer dependencies pass their tests.
- Functional execution is not a Vulkan validation run. RTSS remains active and was left alone.
- Merged-source live controls used executable `BB09611E57159C758EC4834EC38C20950E352FB455DD86B035E18BA276D49C59`
  from clean integration commit `46a5657e2`. Default BAR + 1080p reached visually clean combat and accepted
  Square/Triangle/Cross/Circle. The screenshot-bounded 20.2 s interval contains18 compilation-free samples:
  mean 46.82 / median 46.39 FPS,1920x1080 depth. No matched A/B or 60 FPS claim. Pre-fight watchdogs occurred.
- Default BAR +4K produced visible black bands/white regions and severe pre-fight slowdown; memory
  pressure was observed. This is a correctness failure, not a usable FPS benchmark. Unmapped-buffer
  controls at both output sizes failed in Vulkan semaphore waits before combat. The cause and any
  regression relative to older, differently configured controls remain unisolated.
- Final artifact hashes, package verification and publication state are supplied with the release at
  https://github.com/fxandrei/KytyPS5-ufc6/releases/tag/ufc6-v20261124-preview.1 . Native 4K correctness,
  allocator pressure and fatal waits take priority over further performance changes.
