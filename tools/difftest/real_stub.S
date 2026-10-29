// run_case(const State* in, State* out, void* code): runs code with every GPR, the flags and ymm0-15 loaded from
// `in`; the code ends with an absolute jump to exit_stub, which stores the same state to `out`.
// State: gpr[16] (rax rcx rdx rbx rsp rbp rsi rdi r8-r15) at 0, flags at 128, ymm[16][32] at 136.
#ifdef __APPLE__
#define SYM(x) _##x
#else
#define SYM(x) x
#endif
.intel_syntax noprefix
.data
.p2align 3
g_out: .quad 0
host_rsp: .quad 0
code_ptr: .quad 0
save_rax: .quad 0
mxcsr_default: .long 0x1f80
.text
.globl SYM(run_case)
SYM(run_case):
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 8
    mov [rip + g_out], rsi
    mov [rip + code_ptr], rdx
    mov [rip + host_rsp], rsp
    ldmxcsr [rip + mxcsr_default]
    fninit
    vmovdqu ymm0, [rdi + 136]
    vmovdqu ymm1, [rdi + 168]
    vmovdqu ymm2, [rdi + 200]
    vmovdqu ymm3, [rdi + 232]
    vmovdqu ymm4, [rdi + 264]
    vmovdqu ymm5, [rdi + 296]
    vmovdqu ymm6, [rdi + 328]
    vmovdqu ymm7, [rdi + 360]
    vmovdqu ymm8, [rdi + 392]
    vmovdqu ymm9, [rdi + 424]
    vmovdqu ymm10, [rdi + 456]
    vmovdqu ymm11, [rdi + 488]
    vmovdqu ymm12, [rdi + 520]
    vmovdqu ymm13, [rdi + 552]
    vmovdqu ymm14, [rdi + 584]
    vmovdqu ymm15, [rdi + 616]
    push qword ptr [rdi + 128]
    popfq
    mov rax, [rdi + 0]
    mov rcx, [rdi + 8]
    mov rdx, [rdi + 16]
    mov rbx, [rdi + 24]
    mov rsp, [rdi + 32]
    mov rbp, [rdi + 40]
    mov rsi, [rdi + 48]
    mov r8, [rdi + 64]
    mov r9, [rdi + 72]
    mov r10, [rdi + 80]
    mov r11, [rdi + 88]
    mov r12, [rdi + 96]
    mov r13, [rdi + 104]
    mov r14, [rdi + 112]
    mov r15, [rdi + 120]
    mov rdi, [rdi + 56]
    jmp [rip + code_ptr]

.globl SYM(exit_stub)
SYM(exit_stub):
    mov [rip + save_rax], rax
    mov rax, [rip + g_out]
    mov [rax + 8], rcx
    mov [rax + 16], rdx
    mov [rax + 24], rbx
    mov [rax + 32], rsp
    mov [rax + 40], rbp
    mov [rax + 48], rsi
    mov [rax + 56], rdi
    mov [rax + 64], r8
    mov [rax + 72], r9
    mov [rax + 80], r10
    mov [rax + 88], r11
    mov [rax + 96], r12
    mov [rax + 104], r13
    mov [rax + 112], r14
    mov [rax + 120], r15
    mov rsp, [rip + host_rsp]
    pushfq
    pop qword ptr [rax + 128]
    mov rcx, [rip + save_rax]
    mov [rax], rcx
    vmovdqu [rax + 136], ymm0
    vmovdqu [rax + 168], ymm1
    vmovdqu [rax + 200], ymm2
    vmovdqu [rax + 232], ymm3
    vmovdqu [rax + 264], ymm4
    vmovdqu [rax + 296], ymm5
    vmovdqu [rax + 328], ymm6
    vmovdqu [rax + 360], ymm7
    vmovdqu [rax + 392], ymm8
    vmovdqu [rax + 424], ymm9
    vmovdqu [rax + 456], ymm10
    vmovdqu [rax + 488], ymm11
    vmovdqu [rax + 520], ymm12
    vmovdqu [rax + 552], ymm13
    vmovdqu [rax + 584], ymm14
    vmovdqu [rax + 616], ymm15
    vzeroupper
    add rsp, 8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
