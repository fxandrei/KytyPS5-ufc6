# Kyty / UFC 6 development continuity

- Continue the existing investigation; do not restart the project analysis.
- Read `HANDOFF_CURRENT.txt` and `docs/ufc6-rendering-investigation.md` first. The detailed local history, test matrix, logs and captures are in `D:\Projects\kyty-dev\handoff\2026-11-13\HANDOFF.txt` and `investigation.json`. Later dates in that directory are continuations of the same investigation.
- Inspect `git status`, HEAD and the current diff before editing. Preserve user uploads and existing work. The uploaded UFC build/patch directories and the architecture guide are user files.
- Put new build artifacts, captures, downloads and TEMP/TMP on **D:**. Use the existing `_Build/diagnostics/upstream-20261107` helpers; they enforce C: >= 5 GiB and D: >= 10 GiB before starting another task. Windows/driver dumps may still use C:; monitor free space. Do not perform broad cleanup of user/system files.
- Preserve facts, hypotheses, failed attempts and unverified limitations separately. Record executable SHA, source revision/diff, cache provenance, settings, phase and evidence for live controls.
- A functional CTest pass is not proof of a clean Vulkan validation run. Existing full-suite validation caveats are documented. Validate affected isolated GPU cases separately.
- Do not claim an FPS improvement from validation, shader compilation, diagnostic logging or a visibly corrupted frame. Round-clock mode is Accelerated; it is not a wall-clock physics measurement.
- Do not re-run rejected memo/speculation/batching experiments without new evidence. Prior performance attribution and the user's architecture guide are linked in the handoff.
- Update the repository summary and detailed local handoff after significant milestones, including work left in the working tree. Do not deploy an experimental emulator or push changes unless authorized for that action.
