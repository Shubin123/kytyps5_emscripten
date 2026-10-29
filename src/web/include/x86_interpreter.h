#ifndef KYTY_X86_INTERPRETER_H
#define KYTY_X86_INTERPRETER_H

#include <cstdint>
#include <cstddef>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace KytyWeb {

struct X86Regs {
    uint64_t rax, rbx, rcx, rdx;
    uint64_t rsi, rdi, rbp, rsp;
    uint64_t r8, r9, r10, r11;
    uint64_t r12, r13, r14, r15;
    uint64_t rip;
    uint64_t rflags;

    // SSE registers; ymm_hi holds bits 255:128 of the AVX registers.
    __attribute__((aligned(16))) uint8_t xmm[16][16];
    __attribute__((aligned(16))) uint8_t ymm_hi[16][16];

    // Segment registers (unused in flat model but kept for compatibility)
    uint16_t cs, ds, es, fs, gs, ss;
    // FS holds the thread pointer (TLS); the selectors above are ignored.
    uint64_t fs_base, gs_base;

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
    HLECall,
    Yield  // the current thread gave up its time slice (blocked, slept, or yielded)
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

    void SetTraceEnabled(bool enable) { trace_enabled_ = enable; }
    // Logs the first hits on an address (with the return address at [rsp]), for debugging.
    void AddWatch(uint64_t addr) { watch_[addr] = 0; }
    // Logs the instructions that change the quadword at addr (checked between instructions, so HLE writes count).
    void AddMemoryWatch(uint64_t addr) { mem_watch_.push_back({addr, ReadMemory64(addr), 0}); }
    // Names code addresses in watch logs (e.g. which SDK function an import stub belongs to).
    void SetAddressNamer(std::function<std::string(uint64_t)> namer) { address_namer_ = std::move(namer); }

    // From an HLE handler: end the time slice. With retry, the call's RET is not executed, so RIP stays
    // on the import stub and the same call runs again when the thread is next scheduled.
    void RequestYield(bool retry) { yield_ = true; yield_retry_ = retry; }

    // After Step() returns PageFault: the unmapped address and the faulting instruction.
    uint64_t FaultAddress() const { return fault_addr_; }
    uint64_t FaultRip() const { return fault_rip_; }
    // True if [addr, addr + n) is mapped; otherwise records a page fault.
    bool Mapped(uint64_t addr, uint64_t n) const;
    void SetTraceLimit(int limit) { trace_limit_ = limit; }

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
    bool trace_enabled_ = false;
    int trace_limit_ = 200;
    int trace_count_ = 0;
    std::unordered_map<uint64_t, HLEHandler> hle_handlers_;
    std::unordered_map<uint64_t, int> watch_;
    struct MemoryWatch { uint64_t addr, value; int hits; };
    std::vector<MemoryWatch> mem_watch_;
    uint64_t prev_rip_ = 0;
    std::function<std::string(uint64_t)> address_namer_;
    bool yield_ = false, yield_retry_ = false;
    mutable bool fault_ = false;
    mutable uint64_t fault_addr_ = 0;
    uint64_t fault_rip_ = 0;
    HLEHandler syscall_handler_;
};

} // namespace KytyWeb

#endif
