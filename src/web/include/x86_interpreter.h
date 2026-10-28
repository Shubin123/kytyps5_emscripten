#ifndef KYTY_X86_INTERPRETER_H
#define KYTY_X86_INTERPRETER_H

#include <cstdint>
#include <cstddef>
#include <functional>
#include <string>
#include <unordered_map>

namespace KytyWeb {

struct X86Regs {
    uint64_t rax, rbx, rcx, rdx;
    uint64_t rsi, rdi, rbp, rsp;
    uint64_t r8, r9, r10, r11;
    uint64_t r12, r13, r14, r15;
    uint64_t rip;
    uint64_t rflags;

    // SSE registers (simplified)
    __attribute__((aligned(16))) uint8_t xmm[16][16];

    // Segment registers (unused in flat model but kept for compatibility)
    uint16_t cs, ds, es, fs, gs, ss;

    // x87 FPU state (simplified)
    double st[8];
    int st_top;
};

enum class X86Exception {
    None = 0,
    DivideByZero,
    Debug,
    Breakpoint,
    Overflow,
    BoundRange,
    InvalidOpcode,
    DeviceNotAvailable,
    DoubleFault,
    GeneralProtection,
    PageFault,
    Syscall,
    HLECall
};

using HLEHandler = std::function<void(X86Regs&, uint64_t func_id)>;

class X86Interpreter {
public:
    X86Interpreter();
    ~X86Interpreter();

    void Reset();
    void SetMemory(uint8_t* base, size_t size, uint64_t base_addr = 0);
    void SetStackPointer(uint64_t sp);
    void SetInstructionPointer(uint64_t ip);

    void RegisterHLEFunction(uint64_t address, HLEHandler handler);
    void RegisterSyscallHandler(HLEHandler handler);

    X86Exception Step();
    X86Exception Run(uint64_t max_instructions = 0);

    X86Regs& Regs() { return regs_; }
    const X86Regs& Regs() const { return regs_; }

    uint64_t ReadMemory64(uint64_t addr) const;
    uint32_t ReadMemory32(uint64_t addr) const;
    uint16_t ReadMemory16(uint64_t addr) const;
    uint8_t  ReadMemory8(uint64_t addr) const;

    void WriteMemory64(uint64_t addr, uint64_t val);
    void WriteMemory32(uint64_t addr, uint32_t val);
    void WriteMemory16(uint64_t addr, uint16_t val);
    void WriteMemory8(uint64_t addr, uint8_t val);

    void Push64(uint64_t val);
    uint64_t Pop64();

    bool IsRunning() const { return running_; }
    void Stop() { running_ = false; }

    uint64_t GetInstructionCount() const { return insn_count_; }

private:
    X86Exception DecodeAndExecute();
    void UpdateFlags(uint64_t result, int bits);
    void UpdateFlagsAdd(uint64_t a, uint64_t b, uint64_t result, int bits);
    void UpdateFlagsSub(uint64_t a, uint64_t b, uint64_t result, int bits);

    bool GetCF() const { return (regs_.rflags >> 0) & 1; }
    bool GetPF() const { return (regs_.rflags >> 2) & 1; }
    bool GetAF() const { return (regs_.rflags >> 4) & 1; }
    bool GetZF() const { return (regs_.rflags >> 6) & 1; }
    bool GetSF() const { return (regs_.rflags >> 7) & 1; }
    bool GetOF() const { return (regs_.rflags >> 11) & 1; }

    void SetCF(bool v) { regs_.rflags = (regs_.rflags & ~(1ULL << 0)) | ((uint64_t)v << 0); }
    void SetPF(bool v) { regs_.rflags = (regs_.rflags & ~(1ULL << 2)) | ((uint64_t)v << 2); }
    void SetAF(bool v) { regs_.rflags = (regs_.rflags & ~(1ULL << 4)) | ((uint64_t)v << 4); }
    void SetZF(bool v) { regs_.rflags = (regs_.rflags & ~(1ULL << 6)) | ((uint64_t)v << 6); }
    void SetSF(bool v) { regs_.rflags = (regs_.rflags & ~(1ULL << 7)) | ((uint64_t)v << 7); }
    void SetOF(bool v) { regs_.rflags = (regs_.rflags & ~(1ULL << 11)) | ((uint64_t)v << 11); }

    uint64_t& RegByIndex(int idx);
    uint64_t ResolveModRM(uint8_t modrm, uint8_t* sib_out, int addr_size, int* disp_bytes);

    X86Regs regs_;
    uint8_t* mem_base_;
    size_t mem_size_;
    uint64_t mem_base_addr_;
    bool running_;
    uint64_t insn_count_;
    std::unordered_map<uint64_t, HLEHandler> hle_handlers_;
    HLEHandler syscall_handler_;
};

} // namespace KytyWeb

#endif
