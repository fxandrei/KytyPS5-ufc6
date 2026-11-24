# EXPERIMENT ONLY: UFC 6 PPSA23566 01.005.000.
# Intermediate calibration after the elapsed/game-budget experiment felt too fast.
# This replaces the elapsed/25ms patch; do not combine the two patches.
# It is not a validated real-time speed or an FPS optimization.
#
# timing[0] keeps its original value for elapsed <= 20ms. Otherwise:
# timing[0] = trunc(original_timing[0] * min(elapsed, 100ms) / 20ms).
# The initial time scale (including pause/zero) remains multiplicative. At
# 20-40 FPS this supplies approximately 1.25x the legacy elapsed/25ms budget,
# with at most a one-nanosecond difference from integer truncation.
#
# Only caller-saved RAX, RCX, RDX and flags are clobbered, as in the legacy
# patch. R14 is the output-timing pointer; RBX/R15 are the original frame-sync
# object and timestamp. The original canary epilogue follows return_to_game.
.intel_syntax noprefix

.section .hook,"ax",@progbits
.global ufc6_timing_hook
ufc6_timing_hook:
    mov rax, r15
    sub rax, r12
    jmp ufc6_timing_cave
    nop
    nop
    nop

.section .cave,"ax",@progbits
.global ufc6_timing_cave
ufc6_timing_cave:
    # Replay the two displaced stores before modifying the scaled budget.
    mov qword ptr [r14 + 0x18], rax
    mov qword ptr [rbx + 0x38], r15

    mov rax, qword ptr [r14 + 0x10]
    cmp rax, 20000000
    jle timing_done
    cmp rax, 100000000
    jle elapsed_capped
    mov eax, 100000000
elapsed_capped:
    # Signed full-width multiplication avoids intermediate 64-bit overflow.
    imul qword ptr [r14]
    mov ecx, 20000000
    idiv rcx
    mov qword ptr [r14], rax
timing_done:
    jmp return_to_game

.section .note.GNU-stack,"",@progbits
