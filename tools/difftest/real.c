// Runs each Case on the host CPU (x86-64 under Rosetta) and writes a Result per case.
// Usage: real <cases.bin> <results.bin> [first case]
// After a SIGILL Rosetta no longer runs AVX in the process, so the harness re-executes itself to continue.
#include "common.h"
#include <setjmp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#pragma pack(push, 1)
typedef struct { uint64_t gpr[16]; uint64_t flags; uint8_t ymm[16][32]; } State;
#pragma pack(pop)
void run_case(const State* in, State* out, void* code);
void exit_stub(void);

static sigjmp_buf g_jmp;
static volatile int g_sig;
static void OnSignal(int sig) { g_sig = sig; siglongjmp(g_jmp, 1); }

int main(int argc, char** argv) {
    const long first = argc > 3 ? atol(argv[3]) : 0;
    FILE* in = fopen(argv[1], "rb");
    FILE* out = fopen(argv[2], first ? "ab" : "wb");
    fseek(in, first * (long)sizeof(Case), SEEK_SET);
    uint8_t* code = mmap((void*)CODE_ADDR, 0x1000, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);
    uint8_t* data = mmap((void*)DATA_ADDR, DATA_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);
    if (code != (void*)CODE_ADDR || data != (void*)DATA_ADDR) { perror("mmap"); return 1; }
    static uint8_t altstack[1 << 16];
    stack_t ss = {altstack, 0, sizeof(altstack)};
    sigaltstack(&ss, 0);
    struct sigaction sa = {0};
    sa.sa_handler = OnSignal;
    sa.sa_flags = SA_ONSTACK | SA_NODEFER;
    int sigs[] = {SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGTRAP};
    for (int i = 0; i < 5; i++) sigaction(sigs[i], &sa, 0);
    Case c;
    long n = first;
    while (fread(&c, sizeof(c), 1, in) == 1) {
        Result r;
        memset(&r, 0, sizeof(r));
        fill_data(data, c.seed);
        mprotect(code, 0x1000, PROT_READ | PROT_WRITE);
        memcpy(code, c.code, c.len);
        uint8_t jump[14] = {0xff, 0x25, 0, 0, 0, 0};
        uint64_t target = (uint64_t)exit_stub;
        memcpy(jump + 6, &target, 8);
        memcpy(code + c.len, jump, sizeof(jump));
        mprotect(code, 0x1000, PROT_READ | PROT_EXEC);
        State s, o;
        memcpy(s.gpr, c.gpr, sizeof(s.gpr));
        s.flags = c.flags;
        memcpy(s.ymm, c.ymm, sizeof(s.ymm));
        g_sig = 0;
        if (sigsetjmp(g_jmp, 1) == 0) {
            run_case(&s, &o, code);
            memcpy(r.gpr, o.gpr, sizeof(r.gpr));
            r.flags = o.flags;
            memcpy(r.ymm, o.ymm, sizeof(r.ymm));
            diff_data(data, c.seed, &r);
        } else {
            r.status = g_sig == SIGILL ? 2 : 1;
        }
        fwrite(&r, sizeof(r), 1, out);
        n++;
        if (g_sig) {
            fclose(out);
            char next[32];
            snprintf(next, sizeof(next), "%ld", n);
            execl(argv[0], argv[0], argv[1], argv[2], next, (char*)0);
            perror("execl");
            return 1;
        }
    }
    fclose(out);
    fprintf(stderr, "%ld cases\n", n);
    return 0;
}
