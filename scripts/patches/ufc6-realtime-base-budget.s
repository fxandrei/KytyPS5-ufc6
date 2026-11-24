# EXPERIMENT ONLY: UFC 6 PPSA23566 01.005.000.
# This is an alternative to the existing elapsed/25ms patch, not an add-on.
# It has not been validated for animation, physics, audio, or game-clock speed.
#
# The first two displaced instructions remain at the original hook. The cave
# replays the remaining instructions, then adjusts timing[0] only when the
# capped elapsed time exceeds the game's own positive timing[8] budget.
#
# Clobbers only caller-saved RAX, RCX, RDX and flags, all dead before the original
# canary epilogue at return_to_game. R14 is the output-timing pointer; RBX/R15 are
# the original frame-sync object's pointer and current timestamp.
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
    mov qword ptr [r14 + 0x18], rax
    mov qword ptr [rbx + 0x38], r15

    mov rax, qword ptr [r14 + 0x10]
    cmp rax, 100000000
    jle elapsed_capped
    mov eax, 100000000
elapsed_capped:
    mov rcx, qword ptr [r14 + 0x08]
    test rcx, rcx
    jle timing_done
    cmp rax, rcx
    jle timing_done

    # Signed full-width multiply and divide preserve the initial timing[0]
    # multiplier; the 128-bit product avoids intermediate 64-bit overflow.
    imul qword ptr [r14]
    idiv rcx
    mov qword ptr [r14], rax
timing_done:
    jmp return_to_game

.section .note.GNU-stack,"",@progbits
