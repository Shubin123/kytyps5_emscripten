// Runs each Case in the x86 interpreter and writes a Result per case. Usage: interp <cases.bin> <results.bin>
#include "common.h"
#include "x86_interpreter.h"
#include <cstdio>
#include <cstring>
#include <vector>
using namespace KytyWeb;

int main(int argc, char** argv) {
    FILE* in = fopen(argv[1], "rb");
    FILE* out = fopen(argv[2], "wb");
    const uint64_t size = DATA_ADDR + DATA_SIZE - CODE_ADDR;
    std::vector<uint8_t> mem(size);
    uint8_t* data = mem.data() + (DATA_ADDR - CODE_ADDR);
    Case c;
    long n = 0;
    while (fread(&c, sizeof(c), 1, in) == 1) {
        Result r;
        memset(&r, 0, sizeof(r));
        fill_data(data, c.seed);
        memset(mem.data(), 0x90, 0x1000);
        memcpy(mem.data(), c.code, c.len);
        X86Interpreter cpu;
        cpu.SetMemory(mem.data(), size, CODE_ADDR);
        X86Regs& g = cpu.Regs();
        uint64_t* order[16] = {&g.rax, &g.rcx, &g.rdx, &g.rbx, &g.rsp, &g.rbp, &g.rsi, &g.rdi,
                               &g.r8, &g.r9, &g.r10, &g.r11, &g.r12, &g.r13, &g.r14, &g.r15};
        for (int i = 0; i < 16; i++) *order[i] = c.gpr[i];
        g.rflags = c.flags;
        for (int i = 0; i < 16; i++) { memcpy(g.xmm[i], c.ymm[i], 16); memcpy(g.ymm_hi[i], c.ymm[i] + 16, 16); }
        g.rip = CODE_ADDR;
        X86Exception e = X86Exception::None;
        for (int steps = 0; steps < 100000 && g.rip != CODE_ADDR + c.len; steps++) {
            e = cpu.Step();
            if (e != X86Exception::None && e != X86Exception::HLECall && e != X86Exception::Yield) break;
        }
        if (e != X86Exception::None && e != X86Exception::HLECall && e != X86Exception::Yield) {
            r.status = e == X86Exception::InvalidOpcode ? 3 : 1;
        } else {
            for (int i = 0; i < 16; i++) r.gpr[i] = *order[i];
            r.flags = g.rflags;
            for (int i = 0; i < 16; i++) { memcpy(r.ymm[i], g.xmm[i], 16); memcpy(r.ymm[i] + 16, g.ymm_hi[i], 16); }
            diff_data(data, c.seed, &r);
        }
        fwrite(&r, sizeof(r), 1, out);
        n++;
    }
    fclose(out);
    fprintf(stderr, "%ld cases\n", n);
    return 0;
}
