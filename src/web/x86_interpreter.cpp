#include "x86_interpreter.h"
#include <cstring>
#include <cstdio>
#include <cmath>
#include <algorithm>

namespace KytyWeb {

static uint8_t parity_table[256];
static bool parity_init_done = false;

static void init_parity() {
    if (parity_init_done) return;
    for (int i = 0; i < 256; i++) {
        int bits = 0;
        for (int b = 0; b < 8; b++) bits += (i >> b) & 1;
        parity_table[i] = (bits & 1) == 0 ? 1 : 0;
    }
    parity_init_done = true;
}

static inline uint64_t sign_extend(uint64_t val, int from_bits) {
    uint64_t mask = 1ULL << (from_bits - 1);
    return (val ^ mask) - mask;
}

static inline uint64_t truncate(uint64_t val, int bits) {
    if (bits >= 64) return val;
    return val & ((1ULL << bits) - 1);
}

X86Interpreter::X86Interpreter() : mem_base_(nullptr), mem_size_(0), mem_base_addr_(0), running_(false), insn_count_(0) {
    init_parity();
    Reset();
}

X86Interpreter::~X86Interpreter() = default;

void X86Interpreter::Reset() {
    memset(&regs_, 0, sizeof(regs_));
    regs_.rflags = 0x202;
    running_ = false;
    insn_count_ = 0;
}

void X86Interpreter::SetMemory(uint8_t* base, size_t size, uint64_t base_addr) {
    mem_base_ = base;
    mem_size_ = size;
    mem_base_addr_ = base_addr;
}

void X86Interpreter::SetStackPointer(uint64_t sp) { regs_.rsp = sp; }
void X86Interpreter::SetInstructionPointer(uint64_t ip) { regs_.rip = ip; }

void X86Interpreter::RegisterHLEFunction(uint64_t address, HLEHandler handler) {
    hle_handlers_[address] = std::move(handler);
}

void X86Interpreter::RegisterSyscallHandler(HLEHandler handler) {
    syscall_handler_ = std::move(handler);
}

uint8_t X86Interpreter::ReadMemory8(uint64_t addr) const {
    if (addr < mem_base_addr_) return 0;
    uint64_t off = addr - mem_base_addr_;
    if (off < mem_size_) return mem_base_[off];
    return 0;
}

uint16_t X86Interpreter::ReadMemory16(uint64_t addr) const {
    if (addr < mem_base_addr_) return 0;
    uint64_t off = addr - mem_base_addr_;
    if (off + 2 > mem_size_) return 0;
    uint16_t v;
    memcpy(&v, mem_base_ + off, 2);
    return v;
}

uint32_t X86Interpreter::ReadMemory32(uint64_t addr) const {
    if (addr < mem_base_addr_) return 0;
    uint64_t off = addr - mem_base_addr_;
    if (off + 4 > mem_size_) return 0;
    uint32_t v;
    memcpy(&v, mem_base_ + off, 4);
    return v;
}

uint64_t X86Interpreter::ReadMemory64(uint64_t addr) const {
    if (addr < mem_base_addr_) return 0;
    uint64_t off = addr - mem_base_addr_;
    if (off + 8 > mem_size_) return 0;
    uint64_t v;
    memcpy(&v, mem_base_ + off, 8);
    return v;
}

void X86Interpreter::WriteMemory8(uint64_t addr, uint8_t val) {
    if (addr < mem_base_addr_) return;
    uint64_t off = addr - mem_base_addr_;
    if (off < mem_size_) mem_base_[off] = val;
}

void X86Interpreter::WriteMemory16(uint64_t addr, uint16_t val) {
    if (addr < mem_base_addr_) return;
    uint64_t off = addr - mem_base_addr_;
    if (off + 2 <= mem_size_) memcpy(mem_base_ + off, &val, 2);
}

void X86Interpreter::WriteMemory32(uint64_t addr, uint32_t val) {
    if (addr < mem_base_addr_) return;
    uint64_t off = addr - mem_base_addr_;
    if (off + 4 <= mem_size_) memcpy(mem_base_ + off, &val, 4);
}

void X86Interpreter::WriteMemory64(uint64_t addr, uint64_t val) {
    if (addr < mem_base_addr_) return;
    uint64_t off = addr - mem_base_addr_;
    if (off + 8 <= mem_size_) memcpy(mem_base_ + off, &val, 8);
}

void X86Interpreter::Push64(uint64_t val) {
    regs_.rsp -= 8;
    WriteMemory64(regs_.rsp, val);
}

uint64_t X86Interpreter::Pop64() {
    uint64_t val = ReadMemory64(regs_.rsp);
    regs_.rsp += 8;
    return val;
}

uint64_t& X86Interpreter::RegByIndex(int idx) {
    switch (idx & 0xF) {
        case 0:  return regs_.rax;
        case 1:  return regs_.rcx;
        case 2:  return regs_.rdx;
        case 3:  return regs_.rbx;
        case 4:  return regs_.rsp;
        case 5:  return regs_.rbp;
        case 6:  return regs_.rsi;
        case 7:  return regs_.rdi;
        case 8:  return regs_.r8;
        case 9:  return regs_.r9;
        case 10: return regs_.r10;
        case 11: return regs_.r11;
        case 12: return regs_.r12;
        case 13: return regs_.r13;
        case 14: return regs_.r14;
        case 15: return regs_.r15;
    }
    return regs_.rax;
}

void X86Interpreter::UpdateFlags(uint64_t result, int bits) {
    uint64_t r = truncate(result, bits);
    SetZF(r == 0);
    SetSF((r >> (bits - 1)) & 1);
    SetPF(parity_table[r & 0xFF]);
}

void X86Interpreter::UpdateFlagsAdd(uint64_t a, uint64_t b, uint64_t result, int bits) {
    uint64_t r = truncate(result, bits);
    uint64_t ta = truncate(a, bits);
    uint64_t tb = truncate(b, bits);
    SetZF(r == 0);
    SetSF((r >> (bits - 1)) & 1);
    SetPF(parity_table[r & 0xFF]);
    SetCF(r < ta);
    SetAF(((ta ^ tb ^ r) >> 4) & 1);
    uint64_t sign_mask = 1ULL << (bits - 1);
    SetOF(((~(ta ^ tb)) & (ta ^ r) & sign_mask) != 0);
}

void X86Interpreter::UpdateFlagsSub(uint64_t a, uint64_t b, uint64_t result, int bits) {
    uint64_t r = truncate(result, bits);
    uint64_t ta = truncate(a, bits);
    uint64_t tb = truncate(b, bits);
    SetZF(r == 0);
    SetSF((r >> (bits - 1)) & 1);
    SetPF(parity_table[r & 0xFF]);
    SetCF(ta < tb);
    SetAF(((ta ^ tb ^ r) >> 4) & 1);
    uint64_t sign_mask = 1ULL << (bits - 1);
    SetOF(((ta ^ tb) & (ta ^ r) & sign_mask) != 0);
}

uint64_t X86Interpreter::ResolveModRM(uint8_t modrm, uint8_t* sib_out, int addr_size, int* disp_bytes) {
    int mod = (modrm >> 6) & 3;
    int rm  = modrm & 7;
    *disp_bytes = 0;

    if (mod == 3) {
        *disp_bytes = 0;
        return 0; // register direct — caller handles
    }

    uint64_t addr = 0;

    auto read_disp32_at_rip = [&]() -> int32_t {
        int32_t disp32;
        uint64_t off = regs_.rip - mem_base_addr_;
        if (off + 4 <= mem_size_) {
            memcpy(&disp32, mem_base_ + off, 4);
        } else {
            disp32 = 0;
        }
        regs_.rip += 4;
        return disp32;
    };

    if (rm == 4) {
        // SIB byte follows
        uint8_t sib = ReadMemory8(regs_.rip);
        regs_.rip++;
        if (sib_out) *sib_out = sib;

        int scale = (sib >> 6) & 3;
        int index = (sib >> 3) & 7;
        int base  = sib & 7;

        // base
        if (base == 5 && mod == 0) {
            int32_t disp32 = read_disp32_at_rip();
            addr = (uint64_t)(int64_t)disp32;
            *disp_bytes = 4;
        } else {
            addr = RegByIndex(base);
        }

        // index (index==4 means no index in standard encoding)
        if (index != 4) {
            addr += RegByIndex(index) << scale;
        }
    } else if (rm == 5 && mod == 0) {
        // RIP-relative addressing
        int32_t disp32 = read_disp32_at_rip();
        addr = regs_.rip + (int64_t)disp32;
        *disp_bytes = 4;
        return addr;
    } else {
        addr = RegByIndex(rm);
    }

    if (mod == 1) {
        int8_t disp8 = (int8_t)ReadMemory8(regs_.rip);
        regs_.rip++;
        addr += (int64_t)disp8;
        *disp_bytes = 1;
    } else if (mod == 2) {
        int32_t disp32 = read_disp32_at_rip();
        addr += (int64_t)disp32;
        *disp_bytes = 4;
    }

    return addr;
}

X86Exception X86Interpreter::Step() {
    if (trace_enabled_ && trace_count_ < trace_limit_) {
        uint8_t b0 = ReadMemory8(regs_.rip);
        uint8_t b1 = ReadMemory8(regs_.rip + 1);
        uint8_t b2 = ReadMemory8(regs_.rip + 2);
        uint8_t b3 = ReadMemory8(regs_.rip + 3);
        printf("[trace] #%d RIP=0x%llx bytes=%02x %02x %02x %02x  RAX=%llx RDI=%llx RSI=%llx RSP=%llx\n",
               trace_count_, (unsigned long long)regs_.rip,
               b0, b1, b2, b3,
               (unsigned long long)regs_.rax, (unsigned long long)regs_.rdi,
               (unsigned long long)regs_.rsi, (unsigned long long)regs_.rsp);
        trace_count_++;
    }

    auto it = hle_handlers_.find(regs_.rip);
    if (it != hle_handlers_.end()) {
        it->second(regs_, regs_.rip);
        insn_count_++;
        DecodeAndExecute();
        insn_count_++;
        return X86Exception::HLECall;
    }

    X86Exception ex = DecodeAndExecute();
    insn_count_++;
    return ex;
}

X86Exception X86Interpreter::Run(uint64_t max_instructions) {
    running_ = true;
    uint64_t count = 0;

    while (running_) {
        if (regs_.rip < mem_base_addr_ || regs_.rip >= mem_base_addr_ + mem_size_) {
            running_ = false;
            return X86Exception::GeneralProtection;
        }

        X86Exception ex = Step();

        if (ex != X86Exception::None && ex != X86Exception::HLECall && ex != X86Exception::Syscall) {
            running_ = false;
            return ex;
        }

        count++;
        if (max_instructions > 0 && count >= max_instructions) break;
    }
    return X86Exception::None;
}

// The main decode-and-execute loop. Handles x86-64 long mode instructions.
X86Exception X86Interpreter::DecodeAndExecute() {
    // Prefixes
    bool rex = false;
    int rex_w = 0, rex_r = 0, rex_x = 0, rex_b = 0;
    bool has_lock = false;
    bool has_rep = false, has_repne = false;
    bool has_op_size = false;  // 0x66
    bool has_addr_size = false; // 0x67
    int seg_override = -1;

    // Consume prefix bytes
    for (;;) {
        uint8_t b = ReadMemory8(regs_.rip);
        if (b >= 0x40 && b <= 0x4F) {
            rex = true;
            rex_w = (b >> 3) & 1;
            rex_r = (b >> 2) & 1;
            rex_x = (b >> 1) & 1;
            rex_b = b & 1;
            regs_.rip++;
        } else if (b == 0xF0) { has_lock = true; regs_.rip++; }
        else if (b == 0xF2) { has_repne = true; regs_.rip++; }
        else if (b == 0xF3) { has_rep = true; regs_.rip++; }
        else if (b == 0x66) { has_op_size = true; regs_.rip++; }
        else if (b == 0x67) { has_addr_size = true; regs_.rip++; }
        else if (b == 0x26 || b == 0x2E || b == 0x36 || b == 0x3E || b == 0x64 || b == 0x65) {
            seg_override = b; regs_.rip++;
        }
        else break;
    }

    (void)has_lock;
    (void)has_addr_size;
    (void)seg_override;

    int op_size = rex_w ? 64 : (has_op_size ? 16 : 32);
    int op_bytes = op_size / 8;

    uint8_t opcode = ReadMemory8(regs_.rip++);

    // Helper lambdas for reading ModRM operands
    auto read_modrm = [&]() -> uint8_t {
        return ReadMemory8(regs_.rip++);
    };

    auto reg_field = [&](uint8_t modrm) -> int {
        return ((modrm >> 3) & 7) | (rex_r << 3);
    };

    auto rm_field = [&](uint8_t modrm) -> int {
        return (modrm & 7) | (rex_b << 3);
    };

    // Read effective address or register from ModRM
    // Returns: if mod==3, returns register index | 0x100 flag; else memory address
    auto decode_rm = [&](uint8_t modrm, uint64_t* addr_out, bool* is_reg) {
        int mod = (modrm >> 6) & 3;
        int rm_raw = modrm & 7;
        if (mod == 3) {
            *is_reg = true;
            *addr_out = rm_raw | (rex_b << 3);
            return;
        }
        *is_reg = false;

        // Save rip, then resolve
        uint64_t saved_rip = regs_.rip;
        (void)saved_rip;

        uint64_t ea = 0;
        auto fetch_disp32 = [&]() -> int32_t {
            int32_t d32 = 0;
            uint64_t off = regs_.rip - mem_base_addr_;
            if (off + 4 <= mem_size_) memcpy(&d32, mem_base_ + off, 4);
            regs_.rip += 4;
            return d32;
        };

        if (rm_raw == 4) {
            // SIB
            uint8_t sib = ReadMemory8(regs_.rip++);
            int scale = (sib >> 6) & 3;
            int index = ((sib >> 3) & 7) | (rex_x << 3);
            int base = (sib & 7) | (rex_b << 3);

            if ((sib & 7) == 5 && mod == 0) {
                int32_t d32 = fetch_disp32();
                ea = (uint64_t)(int64_t)d32;
            } else {
                ea = RegByIndex(base);
            }
            if (((sib >> 3) & 7) != 4) {
                ea += RegByIndex(index) << scale;
            }
        } else if (rm_raw == 5 && mod == 0) {
            int32_t d32 = fetch_disp32();
            ea = regs_.rip + (int64_t)d32;
            *addr_out = ea;
            return;
        } else {
            ea = RegByIndex(rm_raw | (rex_b << 3));
        }

        if (mod == 1) {
            int8_t d8 = (int8_t)ReadMemory8(regs_.rip++);
            ea += (int64_t)d8;
        } else if (mod == 2) {
            int32_t d32 = fetch_disp32();
            ea += (int64_t)d32;
        }
        *addr_out = ea;
    };

    // Read value from rm operand
    auto read_rm_val = [&](uint8_t modrm, int bits) -> uint64_t {
        uint64_t addr; bool is_reg;
        decode_rm(modrm, &addr, &is_reg);
        if (is_reg) {
            uint64_t v = RegByIndex((int)addr);
            return truncate(v, bits);
        }
        switch (bits) {
            case 8:  return ReadMemory8(addr);
            case 16: return ReadMemory16(addr);
            case 32: return ReadMemory32(addr);
            case 64: return ReadMemory64(addr);
        }
        return 0;
    };

    // Write value to rm operand
    auto write_rm_val = [&](uint8_t modrm_saved, uint64_t val, int bits) {
        // We need to re-decode to get address; the caller must have saved rip before first decode
        // Instead, we use a different approach: caller passes resolved addr/is_reg
    };
    (void)write_rm_val;

    // ALU helper: performs op and writes result
    // op: 0=ADD 1=OR 2=ADC 3=SBB 4=AND 5=SUB 6=XOR 7=CMP
    auto alu_op = [&](int op, uint64_t& dst, uint64_t src, int bits) -> uint64_t {
        uint64_t a = truncate(dst, bits);
        uint64_t b = truncate(src, bits);
        uint64_t result = 0;
        switch (op) {
            case 0: // ADD
                result = a + b;
                UpdateFlagsAdd(a, b, result, bits);
                break;
            case 1: // OR
                result = a | b;
                UpdateFlags(result, bits);
                SetCF(false); SetOF(false); SetAF(false);
                break;
            case 2: // ADC
                result = a + b + (uint64_t)GetCF();
                UpdateFlagsAdd(a, b + (uint64_t)GetCF(), result, bits);
                break;
            case 3: // SBB
                result = a - b - (uint64_t)GetCF();
                UpdateFlagsSub(a, b + (uint64_t)GetCF(), result, bits);
                break;
            case 4: // AND
                result = a & b;
                UpdateFlags(result, bits);
                SetCF(false); SetOF(false); SetAF(false);
                break;
            case 5: // SUB
                result = a - b;
                UpdateFlagsSub(a, b, result, bits);
                break;
            case 6: // XOR
                result = a ^ b;
                UpdateFlags(result, bits);
                SetCF(false); SetOF(false); SetAF(false);
                break;
            case 7: // CMP
                result = a - b;
                UpdateFlagsSub(a, b, result, bits);
                return a; // CMP does not change dst
        }
        result = truncate(result, bits);
        if (bits == 32 && rex_w == 0) {
            dst = result; // zero-extend 32-bit result to 64-bit register
        } else if (bits < 64) {
            uint64_t mask = (1ULL << bits) - 1;
            dst = (dst & ~mask) | result;
        } else {
            dst = result;
        }
        return result;
    };

    // Condition code evaluation
    auto eval_cc = [&](int cc) -> bool {
        switch (cc) {
            case 0x0: return GetOF();                    // O
            case 0x1: return !GetOF();                   // NO
            case 0x2: return GetCF();                    // B/NAE/C
            case 0x3: return !GetCF();                   // NB/AE/NC
            case 0x4: return GetZF();                    // E/Z
            case 0x5: return !GetZF();                   // NE/NZ
            case 0x6: return GetCF() || GetZF();         // BE/NA
            case 0x7: return !GetCF() && !GetZF();       // NBE/A
            case 0x8: return GetSF();                    // S
            case 0x9: return !GetSF();                   // NS
            case 0xA: return GetPF();                    // P/PE
            case 0xB: return !GetPF();                   // NP/PO
            case 0xC: return GetSF() != GetOF();         // L/NGE
            case 0xD: return GetSF() == GetOF();         // NL/GE
            case 0xE: return GetZF() || (GetSF() != GetOF()); // LE/NG
            case 0xF: return !GetZF() && (GetSF() == GetOF()); // NLE/G
        }
        return false;
    };

    // Helper to read immediate values
    auto read_imm8 = [&]() -> uint8_t {
        uint8_t v = ReadMemory8(regs_.rip); regs_.rip++; return v;
    };
    auto read_imm16 = [&]() -> uint16_t {
        uint16_t v = ReadMemory16(regs_.rip); regs_.rip += 2; return v;
    };
    auto read_imm32 = [&]() -> uint32_t {
        uint32_t v = ReadMemory32(regs_.rip); regs_.rip += 4; return v;
    };
    auto read_imm64 = [&]() -> uint64_t {
        uint64_t v = ReadMemory64(regs_.rip); regs_.rip += 8; return v;
    };

    // Helper to write result back to rm
    auto write_to_rm = [&](uint64_t addr, bool is_reg, uint64_t val, int bits) {
        if (is_reg) {
            uint64_t& r = RegByIndex((int)addr);
            if (bits == 32) {
                r = truncate(val, 32); // zero-extend
            } else if (bits < 64) {
                uint64_t mask = (1ULL << bits) - 1;
                r = (r & ~mask) | (val & mask);
            } else {
                r = val;
            }
        } else {
            switch (bits) {
                case 8:  WriteMemory8(addr, (uint8_t)val); break;
                case 16: WriteMemory16(addr, (uint16_t)val); break;
                case 32: WriteMemory32(addr, (uint32_t)val); break;
                case 64: WriteMemory64(addr, val); break;
            }
        }
    };

    auto read_from_rm = [&](uint64_t addr, bool is_reg, int bits) -> uint64_t {
        if (is_reg) return truncate(RegByIndex((int)addr), bits);
        switch (bits) {
            case 8:  return ReadMemory8(addr);
            case 16: return ReadMemory16(addr);
            case 32: return ReadMemory32(addr);
            case 64: return ReadMemory64(addr);
        }
        return 0;
    };

    // --- Main opcode dispatch ---

    switch (opcode) {

    // 0x00-0x05: ADD r/m, r  |  ADD r, r/m  |  ADD al/rax, imm
    case 0x00: case 0x01: case 0x02: case 0x03: case 0x04: case 0x05: {
        int direction = (opcode >> 1) & 1; // 0 = rm<-r, 1 = r<-rm
        int is_byte = !(opcode & 1) && opcode < 0x04;
        int bits;
        if (opcode == 0x00 || opcode == 0x02) bits = 8;
        else if (opcode == 0x04) bits = 8;
        else if (opcode == 0x05) bits = op_size;
        else bits = op_size;

        if (opcode == 0x04) {
            uint8_t imm = read_imm8();
            alu_op(0, regs_.rax, imm, 8);
        } else if (opcode == 0x05) {
            uint64_t imm = (op_size == 64) ? (uint64_t)(int64_t)(int32_t)read_imm32() : read_imm32();
            if (op_size == 16) imm = (uint16_t)imm;
            alu_op(0, regs_.rax, imm, op_size);
        } else {
            uint8_t modrm = read_modrm();
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            int reg = reg_field(modrm);

            if (direction == 0 || opcode < 0x02) {
                // rm <- rm OP reg
                uint64_t rm_val = read_from_rm(rm_addr, rm_is_reg, bits);
                uint64_t reg_val = truncate(RegByIndex(reg), bits);
                uint64_t dummy = rm_val;
                alu_op(0, dummy, reg_val, bits);
                write_to_rm(rm_addr, rm_is_reg, dummy, bits);
            } else {
                // reg <- reg OP rm
                uint64_t rm_val = read_from_rm(rm_addr, rm_is_reg, bits);
                alu_op(0, RegByIndex(reg), rm_val, bits);
            }
        }
        break;
    }

    // 0x08-0x0D: OR
    case 0x08: case 0x09: case 0x0A: case 0x0B: case 0x0C: case 0x0D: {
        int sub_op = opcode - 0x08;
        int bits = (sub_op == 0 || sub_op == 2 || sub_op == 4) ? 8 : op_size;
        if (sub_op == 4) bits = 8;
        if (sub_op == 5) bits = op_size;

        if (sub_op >= 4) {
            uint64_t imm = (bits == 8) ? read_imm8() : ((bits == 16) ? read_imm16() : (uint64_t)(int64_t)(int32_t)read_imm32());
            alu_op(1, regs_.rax, imm, bits);
        } else {
            uint8_t modrm = read_modrm();
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            int reg = reg_field(modrm);
            if (sub_op < 2) {
                uint64_t rm_val = read_from_rm(rm_addr, rm_is_reg, bits);
                uint64_t dummy = rm_val;
                alu_op(1, dummy, truncate(RegByIndex(reg), bits), bits);
                write_to_rm(rm_addr, rm_is_reg, dummy, bits);
            } else {
                uint64_t rm_val = read_from_rm(rm_addr, rm_is_reg, bits);
                alu_op(1, RegByIndex(reg), rm_val, bits);
            }
        }
        break;
    }

    // 0x10-0x15: ADC
    case 0x10: case 0x11: case 0x12: case 0x13: case 0x14: case 0x15: {
        int sub_op = opcode - 0x10;
        int bits = (sub_op == 0 || sub_op == 2 || sub_op == 4) ? 8 : op_size;
        if (sub_op >= 4) {
            uint64_t imm = (bits == 8) ? read_imm8() : ((bits == 16) ? read_imm16() : (uint64_t)(int64_t)(int32_t)read_imm32());
            alu_op(2, regs_.rax, imm, bits);
        } else {
            uint8_t modrm = read_modrm();
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            int reg = reg_field(modrm);
            if (sub_op < 2) {
                uint64_t v = read_from_rm(rm_addr, rm_is_reg, bits);
                alu_op(2, v, truncate(RegByIndex(reg), bits), bits);
                write_to_rm(rm_addr, rm_is_reg, v, bits);
            } else {
                alu_op(2, RegByIndex(reg), read_from_rm(rm_addr, rm_is_reg, bits), bits);
            }
        }
        break;
    }

    // 0x18-0x1D: SBB
    case 0x18: case 0x19: case 0x1A: case 0x1B: case 0x1C: case 0x1D: {
        int sub_op = opcode - 0x18;
        int bits = (sub_op == 0 || sub_op == 2 || sub_op == 4) ? 8 : op_size;
        if (sub_op >= 4) {
            uint64_t imm = (bits == 8) ? read_imm8() : ((bits == 16) ? read_imm16() : (uint64_t)(int64_t)(int32_t)read_imm32());
            alu_op(3, regs_.rax, imm, bits);
        } else {
            uint8_t modrm = read_modrm();
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            int reg = reg_field(modrm);
            if (sub_op < 2) {
                uint64_t v = read_from_rm(rm_addr, rm_is_reg, bits);
                alu_op(3, v, truncate(RegByIndex(reg), bits), bits);
                write_to_rm(rm_addr, rm_is_reg, v, bits);
            } else {
                alu_op(3, RegByIndex(reg), read_from_rm(rm_addr, rm_is_reg, bits), bits);
            }
        }
        break;
    }

    // 0x20-0x25: AND
    case 0x20: case 0x21: case 0x22: case 0x23: case 0x24: case 0x25: {
        int sub_op = opcode - 0x20;
        int bits = (sub_op == 0 || sub_op == 2 || sub_op == 4) ? 8 : op_size;
        if (sub_op >= 4) {
            uint64_t imm = (bits == 8) ? read_imm8() : ((bits == 16) ? read_imm16() : (uint64_t)(int64_t)(int32_t)read_imm32());
            alu_op(4, regs_.rax, imm, bits);
        } else {
            uint8_t modrm = read_modrm();
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            int reg = reg_field(modrm);
            if (sub_op < 2) {
                uint64_t v = read_from_rm(rm_addr, rm_is_reg, bits);
                alu_op(4, v, truncate(RegByIndex(reg), bits), bits);
                write_to_rm(rm_addr, rm_is_reg, v, bits);
            } else {
                alu_op(4, RegByIndex(reg), read_from_rm(rm_addr, rm_is_reg, bits), bits);
            }
        }
        break;
    }

    // 0x28-0x2D: SUB
    case 0x28: case 0x29: case 0x2A: case 0x2B: case 0x2C: case 0x2D: {
        int sub_op = opcode - 0x28;
        int bits = (sub_op == 0 || sub_op == 2 || sub_op == 4) ? 8 : op_size;
        if (sub_op >= 4) {
            uint64_t imm = (bits == 8) ? read_imm8() : ((bits == 16) ? read_imm16() : (uint64_t)(int64_t)(int32_t)read_imm32());
            alu_op(5, regs_.rax, imm, bits);
        } else {
            uint8_t modrm = read_modrm();
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            int reg = reg_field(modrm);
            if (sub_op < 2) {
                uint64_t v = read_from_rm(rm_addr, rm_is_reg, bits);
                alu_op(5, v, truncate(RegByIndex(reg), bits), bits);
                write_to_rm(rm_addr, rm_is_reg, v, bits);
            } else {
                alu_op(5, RegByIndex(reg), read_from_rm(rm_addr, rm_is_reg, bits), bits);
            }
        }
        break;
    }

    // 0x30-0x35: XOR
    case 0x30: case 0x31: case 0x32: case 0x33: case 0x34: case 0x35: {
        int sub_op = opcode - 0x30;
        int bits = (sub_op == 0 || sub_op == 2 || sub_op == 4) ? 8 : op_size;
        if (sub_op >= 4) {
            uint64_t imm = (bits == 8) ? read_imm8() : ((bits == 16) ? read_imm16() : (uint64_t)(int64_t)(int32_t)read_imm32());
            alu_op(6, regs_.rax, imm, bits);
        } else {
            uint8_t modrm = read_modrm();
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            int reg = reg_field(modrm);
            if (sub_op < 2) {
                uint64_t v = read_from_rm(rm_addr, rm_is_reg, bits);
                alu_op(6, v, truncate(RegByIndex(reg), bits), bits);
                write_to_rm(rm_addr, rm_is_reg, v, bits);
            } else {
                alu_op(6, RegByIndex(reg), read_from_rm(rm_addr, rm_is_reg, bits), bits);
            }
        }
        break;
    }

    // 0x38-0x3D: CMP
    case 0x38: case 0x39: case 0x3A: case 0x3B: case 0x3C: case 0x3D: {
        int sub_op = opcode - 0x38;
        int bits = (sub_op == 0 || sub_op == 2 || sub_op == 4) ? 8 : op_size;
        if (sub_op >= 4) {
            uint64_t imm = (bits == 8) ? read_imm8() : ((bits == 16) ? read_imm16() : (uint64_t)(int64_t)(int32_t)read_imm32());
            alu_op(7, regs_.rax, imm, bits);
        } else {
            uint8_t modrm = read_modrm();
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            int reg = reg_field(modrm);
            if (sub_op < 2) {
                uint64_t v = read_from_rm(rm_addr, rm_is_reg, bits);
                alu_op(7, v, truncate(RegByIndex(reg), bits), bits);
            } else {
                uint64_t v = RegByIndex(reg);
                alu_op(7, v, read_from_rm(rm_addr, rm_is_reg, bits), bits);
            }
        }
        break;
    }

    // 0x50-0x57: PUSH reg
    case 0x50: case 0x51: case 0x52: case 0x53:
    case 0x54: case 0x55: case 0x56: case 0x57: {
        int reg = (opcode - 0x50) | (rex_b << 3);
        Push64(RegByIndex(reg));
        break;
    }

    // 0x58-0x5F: POP reg
    case 0x58: case 0x59: case 0x5A: case 0x5B:
    case 0x5C: case 0x5D: case 0x5E: case 0x5F: {
        int reg = (opcode - 0x58) | (rex_b << 3);
        RegByIndex(reg) = Pop64();
        break;
    }

    // 0x63: MOVSXD r64, r/m32
    case 0x63: {
        uint8_t modrm = read_modrm();
        uint64_t rm_addr; bool rm_is_reg;
        decode_rm(modrm, &rm_addr, &rm_is_reg);
        int reg = reg_field(modrm);
        int32_t val = (int32_t)read_from_rm(rm_addr, rm_is_reg, 32);
        if (rex_w) {
            RegByIndex(reg) = (uint64_t)(int64_t)val;
        } else {
            RegByIndex(reg) = (uint32_t)val;
        }
        break;
    }

    // 0x68: PUSH imm32/16
    case 0x68: {
        if (has_op_size) {
            Push64((uint64_t)(int64_t)(int16_t)read_imm16());
        } else {
            Push64((uint64_t)(int64_t)(int32_t)read_imm32());
        }
        break;
    }

    // 0x69: IMUL r, r/m, imm32
    case 0x69: {
        uint8_t modrm = read_modrm();
        uint64_t rm_addr; bool rm_is_reg;
        decode_rm(modrm, &rm_addr, &rm_is_reg);
        int reg = reg_field(modrm);
        int64_t rm_val = (int64_t)sign_extend(read_from_rm(rm_addr, rm_is_reg, op_size), op_size);
        int64_t imm = (int64_t)(int32_t)read_imm32();
        __int128 wide = (__int128)rm_val * (__int128)imm;
        uint64_t result = truncate((uint64_t)wide, op_size);
        RegByIndex(reg) = (op_size == 32) ? (uint32_t)result : result;
        int64_t sresult = (int64_t)sign_extend(result, op_size);
        SetCF(sresult != (int64_t)wide);
        SetOF(sresult != (int64_t)wide);
        break;
    }

    // 0x6A: PUSH imm8 (sign-extended)
    case 0x6A: {
        Push64((uint64_t)(int64_t)(int8_t)read_imm8());
        break;
    }

    // 0x6B: IMUL r, r/m, imm8
    case 0x6B: {
        uint8_t modrm = read_modrm();
        uint64_t rm_addr; bool rm_is_reg;
        decode_rm(modrm, &rm_addr, &rm_is_reg);
        int reg = reg_field(modrm);
        int64_t rm_val = (int64_t)sign_extend(read_from_rm(rm_addr, rm_is_reg, op_size), op_size);
        int64_t imm = (int64_t)(int8_t)read_imm8();
        __int128 wide = (__int128)rm_val * (__int128)imm;
        uint64_t result = truncate((uint64_t)wide, op_size);
        RegByIndex(reg) = (op_size == 32) ? (uint32_t)result : result;
        int64_t sresult = (int64_t)sign_extend(result, op_size);
        SetCF(sresult != (int64_t)wide);
        SetOF(sresult != (int64_t)wide);
        break;
    }

    // 0x70-0x7F: Jcc rel8
    case 0x70: case 0x71: case 0x72: case 0x73:
    case 0x74: case 0x75: case 0x76: case 0x77:
    case 0x78: case 0x79: case 0x7A: case 0x7B:
    case 0x7C: case 0x7D: case 0x7E: case 0x7F: {
        int8_t rel = (int8_t)read_imm8();
        if (eval_cc(opcode - 0x70)) {
            regs_.rip += (int64_t)rel;
        }
        break;
    }

    // 0x80: ALU r/m8, imm8
    case 0x80: {
        uint8_t modrm = read_modrm();
        int op = reg_field(modrm) & 7;
        uint64_t rm_addr; bool rm_is_reg;
        decode_rm(modrm, &rm_addr, &rm_is_reg);
        uint64_t val = read_from_rm(rm_addr, rm_is_reg, 8);
        uint8_t imm = read_imm8();
        alu_op(op, val, imm, 8);
        if (op != 7) write_to_rm(rm_addr, rm_is_reg, val, 8);
        break;
    }

    // 0x81: ALU r/m, imm32
    case 0x81: {
        uint8_t modrm = read_modrm();
        int op = reg_field(modrm) & 7;
        uint64_t rm_addr; bool rm_is_reg;
        decode_rm(modrm, &rm_addr, &rm_is_reg);
        uint64_t val = read_from_rm(rm_addr, rm_is_reg, op_size);
        uint64_t imm;
        if (op_size == 16) imm = read_imm16();
        else imm = (rex_w) ? (uint64_t)(int64_t)(int32_t)read_imm32() : read_imm32();
        alu_op(op, val, imm, op_size);
        if (op != 7) write_to_rm(rm_addr, rm_is_reg, val, op_size);
        break;
    }

    // 0x83: ALU r/m, imm8 (sign-extended)
    case 0x83: {
        uint8_t modrm = read_modrm();
        int op = reg_field(modrm) & 7;
        uint64_t rm_addr; bool rm_is_reg;
        decode_rm(modrm, &rm_addr, &rm_is_reg);
        uint64_t val = read_from_rm(rm_addr, rm_is_reg, op_size);
        uint64_t imm = (uint64_t)(int64_t)(int8_t)read_imm8();
        alu_op(op, val, imm, op_size);
        if (op != 7) write_to_rm(rm_addr, rm_is_reg, val, op_size);
        break;
    }

    // 0x84: TEST r/m8, r8
    case 0x84: {
        uint8_t modrm = read_modrm();
        uint64_t rm_addr; bool rm_is_reg;
        decode_rm(modrm, &rm_addr, &rm_is_reg);
        uint8_t a = (uint8_t)read_from_rm(rm_addr, rm_is_reg, 8);
        uint8_t b = (uint8_t)truncate(RegByIndex(reg_field(modrm)), 8);
        uint8_t result = a & b;
        UpdateFlags(result, 8);
        SetCF(false); SetOF(false);
        break;
    }

    // 0x85: TEST r/m, r
    case 0x85: {
        uint8_t modrm = read_modrm();
        uint64_t rm_addr; bool rm_is_reg;
        decode_rm(modrm, &rm_addr, &rm_is_reg);
        uint64_t a = read_from_rm(rm_addr, rm_is_reg, op_size);
        uint64_t b = truncate(RegByIndex(reg_field(modrm)), op_size);
        uint64_t result = a & b;
        UpdateFlags(result, op_size);
        SetCF(false); SetOF(false);
        break;
    }

    // 0x86: XCHG r/m8, r8
    case 0x86: {
        uint8_t modrm = read_modrm();
        uint64_t rm_addr; bool rm_is_reg;
        decode_rm(modrm, &rm_addr, &rm_is_reg);
        int reg = reg_field(modrm);
        uint8_t rv = (uint8_t)read_from_rm(rm_addr, rm_is_reg, 8);
        uint8_t regv = (uint8_t)truncate(RegByIndex(reg), 8);
        write_to_rm(rm_addr, rm_is_reg, regv, 8);
        RegByIndex(reg) = (RegByIndex(reg) & ~0xFFULL) | rv;
        break;
    }

    // 0x87: XCHG r/m, r
    case 0x87: {
        uint8_t modrm = read_modrm();
        uint64_t rm_addr; bool rm_is_reg;
        decode_rm(modrm, &rm_addr, &rm_is_reg);
        int reg = reg_field(modrm);
        uint64_t rv = read_from_rm(rm_addr, rm_is_reg, op_size);
        uint64_t regv = truncate(RegByIndex(reg), op_size);
        write_to_rm(rm_addr, rm_is_reg, regv, op_size);
        if (op_size == 32) RegByIndex(reg) = (uint32_t)rv;
        else RegByIndex(reg) = rv;
        break;
    }

    // 0x88: MOV r/m8, r8
    case 0x88: {
        uint8_t modrm = read_modrm();
        uint64_t rm_addr; bool rm_is_reg;
        decode_rm(modrm, &rm_addr, &rm_is_reg);
        write_to_rm(rm_addr, rm_is_reg, truncate(RegByIndex(reg_field(modrm)), 8), 8);
        break;
    }

    // 0x89: MOV r/m, r
    case 0x89: {
        uint8_t modrm = read_modrm();
        uint64_t rm_addr; bool rm_is_reg;
        decode_rm(modrm, &rm_addr, &rm_is_reg);
        write_to_rm(rm_addr, rm_is_reg, truncate(RegByIndex(reg_field(modrm)), op_size), op_size);
        break;
    }

    // 0x8A: MOV r8, r/m8
    case 0x8A: {
        uint8_t modrm = read_modrm();
        uint64_t rm_addr; bool rm_is_reg;
        decode_rm(modrm, &rm_addr, &rm_is_reg);
        int reg = reg_field(modrm);
        uint8_t val = (uint8_t)read_from_rm(rm_addr, rm_is_reg, 8);
        RegByIndex(reg) = (RegByIndex(reg) & ~0xFFULL) | val;
        break;
    }

    // 0x8B: MOV r, r/m
    case 0x8B: {
        uint8_t modrm = read_modrm();
        uint64_t rm_addr; bool rm_is_reg;
        decode_rm(modrm, &rm_addr, &rm_is_reg);
        int reg = reg_field(modrm);
        uint64_t val = read_from_rm(rm_addr, rm_is_reg, op_size);
        if (op_size == 32) RegByIndex(reg) = (uint32_t)val;
        else RegByIndex(reg) = val;
        break;
    }

    // 0x8D: LEA r, m
    case 0x8D: {
        uint8_t modrm = read_modrm();
        uint64_t rm_addr; bool rm_is_reg;
        decode_rm(modrm, &rm_addr, &rm_is_reg);
        int reg = reg_field(modrm);
        if (op_size == 32) RegByIndex(reg) = (uint32_t)rm_addr;
        else RegByIndex(reg) = rm_addr;
        break;
    }

    // 0x8F: POP r/m
    case 0x8F: {
        uint8_t modrm = read_modrm();
        uint64_t rm_addr; bool rm_is_reg;
        decode_rm(modrm, &rm_addr, &rm_is_reg);
        write_to_rm(rm_addr, rm_is_reg, Pop64(), 64);
        break;
    }

    // 0x90: NOP (or XCHG rax, rax with REX.B)
    case 0x90: {
        if (rex_b) {
            uint64_t tmp = regs_.rax;
            regs_.rax = regs_.r8;
            regs_.r8 = tmp;
        }
        // else NOP
        break;
    }

    // 0x91-0x97: XCHG rax, reg
    case 0x91: case 0x92: case 0x93: case 0x94: case 0x95: case 0x96: case 0x97: {
        int reg = (opcode - 0x90) | (rex_b << 3);
        uint64_t tmp = regs_.rax;
        regs_.rax = RegByIndex(reg);
        RegByIndex(reg) = tmp;
        break;
    }

    // 0x98: CBW / CWDE / CDQE
    case 0x98: {
        if (rex_w) regs_.rax = (uint64_t)(int64_t)(int32_t)(uint32_t)regs_.rax;
        else if (has_op_size) regs_.rax = (regs_.rax & ~0xFFFFULL) | (uint16_t)(int16_t)(int8_t)(uint8_t)regs_.rax;
        else regs_.rax = (uint32_t)(int32_t)(int16_t)(uint16_t)regs_.rax;
        break;
    }

    // 0x99: CWD / CDQ / CQO
    case 0x99: {
        if (rex_w) {
            regs_.rdx = ((int64_t)regs_.rax < 0) ? 0xFFFFFFFFFFFFFFFFULL : 0;
        } else if (has_op_size) {
            regs_.rdx = (regs_.rdx & ~0xFFFFULL) | (((int16_t)(uint16_t)regs_.rax < 0) ? 0xFFFFULL : 0);
        } else {
            regs_.rdx = ((int32_t)(uint32_t)regs_.rax < 0) ? 0xFFFFFFFFULL : 0;
        }
        break;
    }

    // 0x9C: PUSHFQ
    case 0x9C: Push64(regs_.rflags); break;

    // 0x9D: POPFQ
    case 0x9D: regs_.rflags = Pop64() | 0x2; break;

    // 0xA0: MOV AL, moffs8
    case 0xA0: {
        uint64_t addr = (rex_w || !has_addr_size) ? read_imm64() : read_imm32();
        regs_.rax = (regs_.rax & ~0xFFULL) | ReadMemory8(addr);
        break;
    }

    // 0xA1: MOV rAX, moffs
    case 0xA1: {
        uint64_t addr = (rex_w || !has_addr_size) ? read_imm64() : read_imm32();
        if (op_size == 64) regs_.rax = ReadMemory64(addr);
        else if (op_size == 16) regs_.rax = (regs_.rax & ~0xFFFFULL) | ReadMemory16(addr);
        else regs_.rax = ReadMemory32(addr);
        break;
    }

    // 0xA2: MOV moffs8, AL
    case 0xA2: {
        uint64_t addr = (rex_w || !has_addr_size) ? read_imm64() : read_imm32();
        WriteMemory8(addr, (uint8_t)regs_.rax);
        break;
    }

    // 0xA3: MOV moffs, rAX
    case 0xA3: {
        uint64_t addr = (rex_w || !has_addr_size) ? read_imm64() : read_imm32();
        if (op_size == 64) WriteMemory64(addr, regs_.rax);
        else if (op_size == 16) WriteMemory16(addr, (uint16_t)regs_.rax);
        else WriteMemory32(addr, (uint32_t)regs_.rax);
        break;
    }

    // 0xA4: MOVSB
    case 0xA4: {
        auto do_one = [&]() {
            WriteMemory8(regs_.rdi, ReadMemory8(regs_.rsi));
            int64_t delta = (regs_.rflags & (1 << 10)) ? -1 : 1; // DF
            regs_.rsi += delta;
            regs_.rdi += delta;
        };
        if (has_rep) {
            while (regs_.rcx) { do_one(); regs_.rcx--; }
        } else {
            do_one();
        }
        break;
    }

    // 0xA5: MOVSW/MOVSD/MOVSQ
    case 0xA5: {
        int sz = op_size / 8;
        auto do_one = [&]() {
            if (op_size == 64) WriteMemory64(regs_.rdi, ReadMemory64(regs_.rsi));
            else if (op_size == 32) WriteMemory32(regs_.rdi, ReadMemory32(regs_.rsi));
            else WriteMemory16(regs_.rdi, ReadMemory16(regs_.rsi));
            int64_t delta = (regs_.rflags & (1 << 10)) ? -(int64_t)sz : (int64_t)sz;
            regs_.rsi += delta;
            regs_.rdi += delta;
        };
        if (has_rep) {
            while (regs_.rcx) { do_one(); regs_.rcx--; }
        } else {
            do_one();
        }
        break;
    }

    // 0xA6: CMPSB
    case 0xA6: {
        auto do_one = [&]() -> bool {
            uint8_t a = ReadMemory8(regs_.rsi);
            uint8_t b = ReadMemory8(regs_.rdi);
            uint64_t tmp = a;
            alu_op(7, tmp, b, 8);
            int64_t delta = (regs_.rflags & (1 << 10)) ? -1 : 1;
            regs_.rsi += delta;
            regs_.rdi += delta;
            return GetZF();
        };
        if (has_rep) {
            while (regs_.rcx) { regs_.rcx--; if (!do_one()) break; }
        } else if (has_repne) {
            while (regs_.rcx) { regs_.rcx--; if (do_one()) break; }
        } else {
            do_one();
        }
        break;
    }

    // 0xA8: TEST AL, imm8
    case 0xA8: {
        uint8_t imm = read_imm8();
        uint8_t result = (uint8_t)regs_.rax & imm;
        UpdateFlags(result, 8);
        SetCF(false); SetOF(false);
        break;
    }

    // 0xA9: TEST rAX, imm
    case 0xA9: {
        uint64_t imm = (op_size == 64) ? (uint64_t)(int64_t)(int32_t)read_imm32() : read_imm32();
        if (op_size == 16) imm = (uint16_t)imm;
        uint64_t result = truncate(regs_.rax, op_size) & truncate(imm, op_size);
        UpdateFlags(result, op_size);
        SetCF(false); SetOF(false);
        break;
    }

    // 0xAA: STOSB
    case 0xAA: {
        auto do_one = [&]() {
            WriteMemory8(regs_.rdi, (uint8_t)regs_.rax);
            regs_.rdi += (regs_.rflags & (1 << 10)) ? -1 : 1;
        };
        if (has_rep) {
            while (regs_.rcx) { do_one(); regs_.rcx--; }
        } else {
            do_one();
        }
        break;
    }

    // 0xAB: STOSW/STOSD/STOSQ
    case 0xAB: {
        int sz = op_size / 8;
        auto do_one = [&]() {
            if (op_size == 64) WriteMemory64(regs_.rdi, regs_.rax);
            else if (op_size == 32) WriteMemory32(regs_.rdi, (uint32_t)regs_.rax);
            else WriteMemory16(regs_.rdi, (uint16_t)regs_.rax);
            regs_.rdi += (regs_.rflags & (1 << 10)) ? -(int64_t)sz : (int64_t)sz;
        };
        if (has_rep) {
            while (regs_.rcx) { do_one(); regs_.rcx--; }
        } else {
            do_one();
        }
        break;
    }

    // 0xAC: LODSB
    case 0xAC: {
        auto do_one = [&]() {
            regs_.rax = (regs_.rax & ~0xFFULL) | ReadMemory8(regs_.rsi);
            regs_.rsi += (regs_.rflags & (1 << 10)) ? -1 : 1;
        };
        if (has_rep) { while (regs_.rcx) { do_one(); regs_.rcx--; } } else do_one();
        break;
    }

    // 0xAD: LODSW/LODSD/LODSQ
    case 0xAD: {
        int sz = op_size / 8;
        auto do_one = [&]() {
            if (op_size == 64) regs_.rax = ReadMemory64(regs_.rsi);
            else if (op_size == 32) regs_.rax = ReadMemory32(regs_.rsi);
            else regs_.rax = (regs_.rax & ~0xFFFFULL) | ReadMemory16(regs_.rsi);
            regs_.rsi += (regs_.rflags & (1 << 10)) ? -(int64_t)sz : (int64_t)sz;
        };
        if (has_rep) { while (regs_.rcx) { do_one(); regs_.rcx--; } } else do_one();
        break;
    }

    // 0xAE: SCASB
    case 0xAE: {
        auto do_one = [&]() -> bool {
            uint8_t mem = ReadMemory8(regs_.rdi);
            uint64_t al = regs_.rax & 0xFF;
            alu_op(7, al, mem, 8);
            regs_.rdi += (regs_.rflags & (1 << 10)) ? -1 : 1;
            return GetZF();
        };
        if (has_rep) { while (regs_.rcx) { regs_.rcx--; if (!do_one()) break; } }
        else if (has_repne) { while (regs_.rcx) { regs_.rcx--; if (do_one()) break; } }
        else do_one();
        break;
    }

    // 0xB0-0xB7: MOV r8, imm8
    case 0xB0: case 0xB1: case 0xB2: case 0xB3:
    case 0xB4: case 0xB5: case 0xB6: case 0xB7: {
        int reg = (opcode - 0xB0) | (rex_b << 3);
        uint8_t imm = read_imm8();
        if (rex || reg < 4) {
            RegByIndex(reg) = (RegByIndex(reg) & ~0xFFULL) | imm;
        } else {
            // High byte registers (AH, CH, DH, BH) without REX
            int real_reg = reg - 4;
            RegByIndex(real_reg) = (RegByIndex(real_reg) & ~0xFF00ULL) | ((uint64_t)imm << 8);
        }
        break;
    }

    // 0xB8-0xBF: MOV r, imm
    case 0xB8: case 0xB9: case 0xBA: case 0xBB:
    case 0xBC: case 0xBD: case 0xBE: case 0xBF: {
        int reg = (opcode - 0xB8) | (rex_b << 3);
        if (rex_w) {
            RegByIndex(reg) = read_imm64();
        } else if (has_op_size) {
            RegByIndex(reg) = (RegByIndex(reg) & ~0xFFFFULL) | read_imm16();
        } else {
            RegByIndex(reg) = read_imm32(); // zero-extends to 64
        }
        break;
    }

    // 0xC0: shift/rotate r/m8, imm8
    case 0xC0: {
        uint8_t modrm = read_modrm();
        int sub = reg_field(modrm) & 7;
        uint64_t rm_addr; bool rm_is_reg;
        decode_rm(modrm, &rm_addr, &rm_is_reg);
        uint8_t imm = read_imm8() & 0x1F;
        uint64_t val = read_from_rm(rm_addr, rm_is_reg, 8);
        if (imm == 0) break;
        switch (sub) {
            case 0: val = (val << imm) | (val >> (8 - imm)); break; // ROL
            case 1: val = (val >> imm) | (val << (8 - imm)); break; // ROR
            case 4: val <<= imm; break; // SHL
            case 5: val >>= imm; break; // SHR
            case 7: val = (uint64_t)((int8_t)(uint8_t)val >> imm); break; // SAR
            default: break;
        }
        val = truncate(val, 8);
        if (sub >= 4) UpdateFlags(val, 8);
        write_to_rm(rm_addr, rm_is_reg, val, 8);
        break;
    }

    // 0xC1: shift/rotate r/m, imm8
    case 0xC1: {
        uint8_t modrm = read_modrm();
        int sub = reg_field(modrm) & 7;
        uint64_t rm_addr; bool rm_is_reg;
        decode_rm(modrm, &rm_addr, &rm_is_reg);
        uint64_t val = read_from_rm(rm_addr, rm_is_reg, op_size);
        uint8_t count = read_imm8() & (op_size == 64 ? 0x3F : 0x1F);
        if (count == 0) break;
        switch (sub) {
            case 0: // ROL
                val = (val << count) | (val >> (op_size - count));
                break;
            case 1: // ROR
                val = (val >> count) | (val << (op_size - count));
                break;
            case 4: // SHL
                if (count < op_size) { SetCF((val >> (op_size - count)) & 1); val <<= count; }
                else { SetCF(0); val = 0; }
                break;
            case 5: // SHR
                if (count < op_size) { SetCF((val >> (count - 1)) & 1); val >>= count; }
                else { SetCF(0); val = 0; }
                break;
            case 7: { // SAR
                int64_t sval = (int64_t)sign_extend(val, op_size);
                SetCF((sval >> (count - 1)) & 1);
                val = (uint64_t)(sval >> count);
                break;
            }
            default: break;
        }
        val = truncate(val, op_size);
        if (sub >= 4) UpdateFlags(val, op_size);
        write_to_rm(rm_addr, rm_is_reg, val, op_size);
        break;
    }

    // 0xC3: RET
    case 0xC3: {
        regs_.rip = Pop64();
        break;
    }

    // 0xC6: MOV r/m8, imm8
    case 0xC6: {
        uint8_t modrm = read_modrm();
        uint64_t rm_addr; bool rm_is_reg;
        decode_rm(modrm, &rm_addr, &rm_is_reg);
        write_to_rm(rm_addr, rm_is_reg, read_imm8(), 8);
        break;
    }

    // 0xC7: MOV r/m, imm32
    case 0xC7: {
        uint8_t modrm = read_modrm();
        uint64_t rm_addr; bool rm_is_reg;
        decode_rm(modrm, &rm_addr, &rm_is_reg);
        uint64_t imm;
        if (op_size == 16) imm = read_imm16();
        else imm = rex_w ? (uint64_t)(int64_t)(int32_t)read_imm32() : read_imm32();
        write_to_rm(rm_addr, rm_is_reg, imm, op_size);
        break;
    }

    // 0xC8: ENTER
    case 0xC8: {
        uint16_t size = read_imm16();
        uint8_t nesting = read_imm8();
        Push64(regs_.rbp);
        uint64_t frame = regs_.rsp;
        if (nesting > 0) {
            for (int i = 1; i < nesting; i++) {
                regs_.rbp -= 8;
                Push64(ReadMemory64(regs_.rbp));
            }
            Push64(frame);
        }
        regs_.rbp = frame;
        regs_.rsp -= size;
        break;
    }

    // 0xC9: LEAVE
    case 0xC9: {
        regs_.rsp = regs_.rbp;
        regs_.rbp = Pop64();
        break;
    }

    // 0xCC: INT 3
    case 0xCC: return X86Exception::Breakpoint;

    // 0xCD: INT imm8
    case 0xCD: {
        uint8_t vec = read_imm8();
        (void)vec;
        return X86Exception::Breakpoint;
    }

    // 0xD0: shift/rotate r/m8, 1
    case 0xD0: {
        uint8_t modrm = read_modrm();
        int sub = reg_field(modrm) & 7;
        uint64_t rm_addr; bool rm_is_reg;
        decode_rm(modrm, &rm_addr, &rm_is_reg);
        uint64_t val = read_from_rm(rm_addr, rm_is_reg, 8);
        uint8_t imm = 1;
        switch (sub) {
            case 0: val = (val << imm) | (val >> (8 - imm)); break;
            case 1: val = (val >> imm) | (val << (8 - imm)); break;
            case 4: val <<= imm; break;
            case 5: val >>= imm; break;
            case 7: val = (uint64_t)((int8_t)(uint8_t)val >> imm); break;
            default: break;
        }
        val = truncate(val, 8);
        if (sub >= 4) UpdateFlags(val, 8);
        write_to_rm(rm_addr, rm_is_reg, val, 8);
        break;
    }

    // 0xD1: shift/rotate r/m, 1
    case 0xD1: {
        uint8_t modrm = read_modrm();
        int sub = reg_field(modrm) & 7;
        uint64_t rm_addr; bool rm_is_reg;
        decode_rm(modrm, &rm_addr, &rm_is_reg);
        uint64_t val = read_from_rm(rm_addr, rm_is_reg, op_size);
        uint8_t count = 1;
        switch (sub) {
            case 0: val = (val << 1) | (val >> (op_size - 1)); break;
            case 1: val = (val >> 1) | (val << (op_size - 1)); break;
            case 4: SetCF((val >> (op_size - 1)) & 1); val <<= 1; break;
            case 5: SetCF(val & 1); val >>= 1; break;
            case 7: { int64_t s = (int64_t)sign_extend(val, op_size); SetCF(s & 1); val = (uint64_t)(s >> 1); break; }
            default: break;
        }
        (void)count;
        val = truncate(val, op_size);
        if (sub >= 4) UpdateFlags(val, op_size);
        write_to_rm(rm_addr, rm_is_reg, val, op_size);
        break;
    }

    // 0xD2: shift/rotate r/m8, CL
    case 0xD2: {
        uint8_t modrm = read_modrm();
        int sub = reg_field(modrm) & 7;
        uint64_t rm_addr; bool rm_is_reg;
        decode_rm(modrm, &rm_addr, &rm_is_reg);
        uint64_t val = read_from_rm(rm_addr, rm_is_reg, 8);
        uint8_t imm = (uint8_t)regs_.rcx & 0x1F;
        if (imm == 0) break;
        switch (sub) {
            case 0: val = (val << imm) | (val >> (8 - imm)); break;
            case 1: val = (val >> imm) | (val << (8 - imm)); break;
            case 4: val <<= imm; break;
            case 5: val >>= imm; break;
            case 7: val = (uint64_t)((int8_t)(uint8_t)val >> imm); break;
            default: break;
        }
        val = truncate(val, 8);
        if (sub >= 4) UpdateFlags(val, 8);
        write_to_rm(rm_addr, rm_is_reg, val, 8);
        break;
    }

    // 0xD3: shift/rotate r/m, CL
    case 0xD3: {
        uint8_t modrm = read_modrm();
        int sub = reg_field(modrm) & 7;
        uint64_t rm_addr; bool rm_is_reg;
        decode_rm(modrm, &rm_addr, &rm_is_reg);
        uint64_t val = read_from_rm(rm_addr, rm_is_reg, op_size);
        uint8_t count = (uint8_t)regs_.rcx & (op_size == 64 ? 0x3F : 0x1F);
        if (count == 0) break;
        switch (sub) {
            case 0: val = (val << count) | (val >> (op_size - count)); break;
            case 1: val = (val >> count) | (val << (op_size - count)); break;
            case 4:
                if (count < (uint8_t)op_size) { SetCF((val >> (op_size - count)) & 1); val <<= count; }
                else { SetCF(0); val = 0; }
                break;
            case 5:
                if (count < (uint8_t)op_size) { SetCF((val >> (count - 1)) & 1); val >>= count; }
                else { SetCF(0); val = 0; }
                break;
            case 7: {
                int64_t s = (int64_t)sign_extend(val, op_size);
                SetCF((s >> (count - 1)) & 1);
                val = (uint64_t)(s >> count);
                break;
            }
            default: break;
        }
        val = truncate(val, op_size);
        if (sub >= 4) UpdateFlags(val, op_size);
        write_to_rm(rm_addr, rm_is_reg, val, op_size);
        break;
    }

    // 0xE8: CALL rel32
    case 0xE8: {
        int32_t rel = (int32_t)read_imm32();
        uint64_t target = regs_.rip + (int64_t)rel;
        Push64(regs_.rip);
        regs_.rip = target;
        break;
    }

    // 0xE9: JMP rel32
    case 0xE9: {
        int32_t rel = (int32_t)read_imm32();
        regs_.rip += (int64_t)rel;
        break;
    }

    // 0xEB: JMP rel8
    case 0xEB: {
        int8_t rel = (int8_t)read_imm8();
        regs_.rip += (int64_t)rel;
        break;
    }

    // 0xF5: CMC
    case 0xF5: SetCF(!GetCF()); break;

    // 0xF6: group 3 byte
    case 0xF6: {
        uint8_t modrm = read_modrm();
        int sub = reg_field(modrm) & 7;
        uint64_t rm_addr; bool rm_is_reg;
        decode_rm(modrm, &rm_addr, &rm_is_reg);
        uint64_t val = read_from_rm(rm_addr, rm_is_reg, 8);

        switch (sub) {
            case 0: case 1: { // TEST r/m8, imm8
                uint8_t imm = read_imm8();
                UpdateFlags(val & imm, 8);
                SetCF(false); SetOF(false);
                break;
            }
            case 2: // NOT
                write_to_rm(rm_addr, rm_is_reg, ~val & 0xFF, 8);
                break;
            case 3: { // NEG
                uint64_t result = (~val + 1) & 0xFF;
                SetCF(val != 0);
                UpdateFlagsSub(0, val, result, 8);
                write_to_rm(rm_addr, rm_is_reg, result, 8);
                break;
            }
            case 4: { // MUL
                uint16_t result = (uint16_t)(uint8_t)regs_.rax * (uint16_t)(uint8_t)val;
                regs_.rax = (regs_.rax & ~0xFFFFULL) | result;
                SetCF((result >> 8) != 0);
                SetOF((result >> 8) != 0);
                break;
            }
            case 5: { // IMUL
                int16_t result = (int16_t)(int8_t)(uint8_t)regs_.rax * (int16_t)(int8_t)(uint8_t)val;
                regs_.rax = (regs_.rax & ~0xFFFFULL) | (uint16_t)result;
                SetCF((int8_t)result != result);
                SetOF((int8_t)result != result);
                break;
            }
            case 6: { // DIV
                if (val == 0) return X86Exception::DivideByZero;
                uint16_t ax = (uint16_t)regs_.rax;
                regs_.rax = (regs_.rax & ~0xFFFFULL) | ((ax % (uint8_t)val) << 8) | (ax / (uint8_t)val);
                break;
            }
            case 7: { // IDIV
                if (val == 0) return X86Exception::DivideByZero;
                int16_t ax = (int16_t)(uint16_t)regs_.rax;
                int8_t divisor = (int8_t)(uint8_t)val;
                regs_.rax = (regs_.rax & ~0xFFFFULL) | ((uint8_t)(ax % divisor) << 8) | (uint8_t)(ax / divisor);
                break;
            }
        }
        break;
    }

    // 0xF7: group 3 word/dword/qword
    case 0xF7: {
        uint8_t modrm = read_modrm();
        int sub = reg_field(modrm) & 7;
        uint64_t rm_addr; bool rm_is_reg;
        decode_rm(modrm, &rm_addr, &rm_is_reg);
        uint64_t val = read_from_rm(rm_addr, rm_is_reg, op_size);

        switch (sub) {
            case 0: case 1: { // TEST r/m, imm
                uint64_t imm = (op_size == 16) ? read_imm16() : (rex_w ? (uint64_t)(int64_t)(int32_t)read_imm32() : read_imm32());
                UpdateFlags(truncate(val, op_size) & truncate(imm, op_size), op_size);
                SetCF(false); SetOF(false);
                break;
            }
            case 2: // NOT
                write_to_rm(rm_addr, rm_is_reg, truncate(~val, op_size), op_size);
                break;
            case 3: { // NEG
                uint64_t result = truncate(-val, op_size);
                SetCF(val != 0);
                UpdateFlagsSub(0, val, result, op_size);
                write_to_rm(rm_addr, rm_is_reg, result, op_size);
                break;
            }
            case 4: { // MUL
                if (op_size == 32) {
                    uint64_t result = (uint64_t)(uint32_t)regs_.rax * (uint64_t)(uint32_t)val;
                    regs_.rax = (uint32_t)result;
                    regs_.rdx = (uint32_t)(result >> 32);
                    SetCF(regs_.rdx != 0); SetOF(regs_.rdx != 0);
                } else if (op_size == 64) {
                    unsigned __int128 result = (unsigned __int128)regs_.rax * (unsigned __int128)val;
                    regs_.rax = (uint64_t)result;
                    regs_.rdx = (uint64_t)(result >> 64);
                    SetCF(regs_.rdx != 0); SetOF(regs_.rdx != 0);
                } else {
                    uint32_t result = (uint16_t)regs_.rax * (uint16_t)val;
                    regs_.rax = (regs_.rax & ~0xFFFFFFFFULL) | result;
                    regs_.rdx = (regs_.rdx & ~0xFFFFULL) | (result >> 16);
                    SetCF((result >> 16) != 0); SetOF((result >> 16) != 0);
                }
                break;
            }
            case 5: { // IMUL
                if (op_size == 32) {
                    int64_t result = (int64_t)(int32_t)(uint32_t)regs_.rax * (int64_t)(int32_t)(uint32_t)val;
                    regs_.rax = (uint32_t)result;
                    regs_.rdx = (uint32_t)((uint64_t)result >> 32);
                    SetCF((int32_t)regs_.rax != result); SetOF((int32_t)regs_.rax != result);
                } else if (op_size == 64) {
                    __int128 result = (__int128)(int64_t)regs_.rax * (__int128)(int64_t)val;
                    regs_.rax = (uint64_t)result;
                    regs_.rdx = (uint64_t)((unsigned __int128)result >> 64);
                    SetCF((int64_t)regs_.rax != result); SetOF((int64_t)regs_.rax != result);
                } else {
                    int32_t result = (int16_t)(uint16_t)regs_.rax * (int16_t)(uint16_t)val;
                    regs_.rax = (regs_.rax & ~0xFFFFFFFFULL) | (uint32_t)result;
                    regs_.rdx = (regs_.rdx & ~0xFFFFULL) | (uint16_t)((uint32_t)result >> 16);
                    SetCF((int16_t)(uint16_t)regs_.rax != result); SetOF((int16_t)(uint16_t)regs_.rax != result);
                }
                break;
            }
            case 6: { // DIV
                if (val == 0) return X86Exception::DivideByZero;
                if (op_size == 32) {
                    uint64_t dividend = ((uint64_t)(uint32_t)regs_.rdx << 32) | (uint32_t)regs_.rax;
                    regs_.rax = (uint32_t)(dividend / (uint32_t)val);
                    regs_.rdx = (uint32_t)(dividend % (uint32_t)val);
                } else if (op_size == 64) {
                    unsigned __int128 dividend = ((unsigned __int128)regs_.rdx << 64) | regs_.rax;
                    regs_.rax = (uint64_t)(dividend / val);
                    regs_.rdx = (uint64_t)(dividend % val);
                } else {
                    uint32_t dividend = ((uint32_t)(uint16_t)regs_.rdx << 16) | (uint16_t)regs_.rax;
                    regs_.rax = (regs_.rax & ~0xFFFFULL) | (uint16_t)(dividend / (uint16_t)val);
                    regs_.rdx = (regs_.rdx & ~0xFFFFULL) | (uint16_t)(dividend % (uint16_t)val);
                }
                break;
            }
            case 7: { // IDIV
                if (val == 0) return X86Exception::DivideByZero;
                if (op_size == 32) {
                    int64_t dividend = ((int64_t)(int32_t)(uint32_t)regs_.rdx << 32) | (uint32_t)regs_.rax;
                    int32_t divisor = (int32_t)(uint32_t)val;
                    regs_.rax = (uint32_t)(dividend / divisor);
                    regs_.rdx = (uint32_t)(dividend % divisor);
                } else if (op_size == 64) {
                    __int128 dividend = ((__int128)(int64_t)regs_.rdx << 64) | regs_.rax;
                    int64_t divisor = (int64_t)val;
                    regs_.rax = (uint64_t)(dividend / divisor);
                    regs_.rdx = (uint64_t)(dividend % divisor);
                } else {
                    int32_t dividend = ((int32_t)(int16_t)(uint16_t)regs_.rdx << 16) | (uint16_t)regs_.rax;
                    int16_t divisor = (int16_t)(uint16_t)val;
                    regs_.rax = (regs_.rax & ~0xFFFFULL) | (uint16_t)(dividend / divisor);
                    regs_.rdx = (regs_.rdx & ~0xFFFFULL) | (uint16_t)(dividend % divisor);
                }
                break;
            }
        }
        break;
    }

    // 0xF8: CLC
    case 0xF8: SetCF(false); break;
    // 0xF9: STC
    case 0xF9: SetCF(true); break;
    // 0xFA: CLI (no-op in user mode)
    case 0xFA: break;
    // 0xFB: STI (no-op in user mode)
    case 0xFB: break;
    // 0xFC: CLD
    case 0xFC: regs_.rflags &= ~(1ULL << 10); break;
    // 0xFD: STD
    case 0xFD: regs_.rflags |= (1ULL << 10); break;

    // 0xFE: INC/DEC r/m8
    case 0xFE: {
        uint8_t modrm = read_modrm();
        int sub = reg_field(modrm) & 7;
        uint64_t rm_addr; bool rm_is_reg;
        decode_rm(modrm, &rm_addr, &rm_is_reg);
        uint64_t val = read_from_rm(rm_addr, rm_is_reg, 8);
        bool old_cf = GetCF();
        if (sub == 0) { // INC
            uint64_t result = (val + 1) & 0xFF;
            UpdateFlagsAdd(val, 1, result, 8);
        } else { // DEC
            uint64_t result = (val - 1) & 0xFF;
            UpdateFlagsSub(val, 1, result, 8);
            val = result;
        }
        if (sub == 0) val = (val + 1) & 0xFF; // re-compute for write (INC already set flags)
        // Fix: compute once
        SetCF(old_cf);
        if (sub == 0) write_to_rm(rm_addr, rm_is_reg, (read_from_rm(rm_addr, rm_is_reg, 8) + 1) & 0xFF, 8);
        else write_to_rm(rm_addr, rm_is_reg, (read_from_rm(rm_addr, rm_is_reg, 8) - 1) & 0xFF, 8);
        break;
    }

    // 0xFF: group 5
    case 0xFF: {
        uint8_t modrm = read_modrm();
        int sub = reg_field(modrm) & 7;
        uint64_t rm_addr; bool rm_is_reg;
        decode_rm(modrm, &rm_addr, &rm_is_reg);

        switch (sub) {
            case 0: { // INC
                uint64_t val = read_from_rm(rm_addr, rm_is_reg, op_size);
                bool old_cf = GetCF();
                uint64_t result = truncate(val + 1, op_size);
                UpdateFlagsAdd(val, 1, result, op_size);
                SetCF(old_cf);
                write_to_rm(rm_addr, rm_is_reg, result, op_size);
                break;
            }
            case 1: { // DEC
                uint64_t val = read_from_rm(rm_addr, rm_is_reg, op_size);
                bool old_cf = GetCF();
                uint64_t result = truncate(val - 1, op_size);
                UpdateFlagsSub(val, 1, result, op_size);
                SetCF(old_cf);
                write_to_rm(rm_addr, rm_is_reg, result, op_size);
                break;
            }
            case 2: { // CALL r/m
                uint64_t target = read_from_rm(rm_addr, rm_is_reg, 64);
                Push64(regs_.rip);
                regs_.rip = target;
                break;
            }
            case 4: { // JMP r/m
                regs_.rip = read_from_rm(rm_addr, rm_is_reg, 64);
                break;
            }
            case 6: { // PUSH r/m
                Push64(read_from_rm(rm_addr, rm_is_reg, 64));
                break;
            }
            default:
                return X86Exception::InvalidOpcode;
        }
        break;
    }

    // --- Two-byte opcodes (0x0F prefix) ---
    case 0x0F: {
        uint8_t op2 = ReadMemory8(regs_.rip++);
        switch (op2) {

        // 0x0F 0x05: SYSCALL
        case 0x05: {
            if (syscall_handler_) {
                syscall_handler_(regs_, regs_.rax);
                return X86Exception::Syscall;
            }
            return X86Exception::InvalidOpcode;
        }

        // 0x0F 0x10: MOVUPS xmm, xmm/m128 (or MOVSS/MOVSD with prefixes)
        case 0x10: {
            uint8_t modrm = read_modrm();
            int reg = reg_field(modrm);
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            if (rm_is_reg) {
                memcpy(regs_.xmm[reg], regs_.xmm[(int)rm_addr], 16);
            } else {
                uint64_t off = rm_addr - mem_base_addr_;
                if (off + 16 <= mem_size_) memcpy(regs_.xmm[reg], mem_base_ + off, 16);
            }
            break;
        }

        // 0x0F 0x11: MOVUPS xmm/m128, xmm
        case 0x11: {
            uint8_t modrm = read_modrm();
            int reg = reg_field(modrm);
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            if (rm_is_reg) {
                memcpy(regs_.xmm[(int)rm_addr], regs_.xmm[reg], 16);
            } else {
                uint64_t off = rm_addr - mem_base_addr_;
                if (off + 16 <= mem_size_) memcpy(mem_base_ + off, regs_.xmm[reg], 16);
            }
            break;
        }

        // 0x0F 0x1F: NOP (multi-byte)
        case 0x1F: {
            uint8_t modrm = read_modrm();
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            break;
        }

        // 0x0F 0x28: MOVAPS xmm, xmm/m128
        case 0x28: {
            uint8_t modrm = read_modrm();
            int reg = reg_field(modrm);
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            if (rm_is_reg) memcpy(regs_.xmm[reg], regs_.xmm[(int)rm_addr], 16);
            else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(regs_.xmm[reg], mem_base_ + off, 16); }
            break;
        }

        // 0x0F 0x29: MOVAPS xmm/m128, xmm
        case 0x29: {
            uint8_t modrm = read_modrm();
            int reg = reg_field(modrm);
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            if (rm_is_reg) memcpy(regs_.xmm[(int)rm_addr], regs_.xmm[reg], 16);
            else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(mem_base_ + off, regs_.xmm[reg], 16); }
            break;
        }

        // 0x0F 0x2E: UCOMISS/UCOMISD
        case 0x2E: {
            uint8_t modrm = read_modrm();
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            // Simplified: treat as NOP with flag clear
            SetCF(false); SetZF(false); SetPF(false); SetOF(false); SetSF(false); SetAF(false);
            break;
        }

        // 0x0F 0x31: RDTSC
        case 0x31: {
            uint64_t tsc = insn_count_ * 100;
            regs_.rax = tsc & 0xFFFFFFFF;
            regs_.rdx = tsc >> 32;
            break;
        }

        // 0x0F 0x40-0x4F: CMOVcc r, r/m
        case 0x40: case 0x41: case 0x42: case 0x43:
        case 0x44: case 0x45: case 0x46: case 0x47:
        case 0x48: case 0x49: case 0x4A: case 0x4B:
        case 0x4C: case 0x4D: case 0x4E: case 0x4F: {
            uint8_t modrm = read_modrm();
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            int reg = reg_field(modrm);
            uint64_t val = read_from_rm(rm_addr, rm_is_reg, op_size);
            if (eval_cc(op2 - 0x40)) {
                if (op_size == 32) RegByIndex(reg) = (uint32_t)val;
                else RegByIndex(reg) = val;
            }
            break;
        }

        // 0x0F 0x57: XORPS/XORPD xmm, xmm/m128
        case 0x57: {
            uint8_t modrm = read_modrm();
            int reg = reg_field(modrm);
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            uint8_t src[16];
            if (rm_is_reg) memcpy(src, regs_.xmm[(int)rm_addr], 16);
            else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
            for (int i = 0; i < 16; i++) regs_.xmm[reg][i] ^= src[i];
            break;
        }

        // 0x0F 0x6F: MOVDQA/MOVDQU xmm, xmm/m128
        case 0x6F: {
            uint8_t modrm = read_modrm();
            int reg = reg_field(modrm);
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            if (rm_is_reg) memcpy(regs_.xmm[reg], regs_.xmm[(int)rm_addr], 16);
            else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(regs_.xmm[reg], mem_base_ + off, 16); }
            break;
        }

        // 0x0F 0x7F: MOVDQA/MOVDQU xmm/m128, xmm
        case 0x7F: {
            uint8_t modrm = read_modrm();
            int reg = reg_field(modrm);
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            if (rm_is_reg) memcpy(regs_.xmm[(int)rm_addr], regs_.xmm[reg], 16);
            else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(mem_base_ + off, regs_.xmm[reg], 16); }
            break;
        }

        // 0x0F 0x80-0x8F: Jcc rel32
        case 0x80: case 0x81: case 0x82: case 0x83:
        case 0x84: case 0x85: case 0x86: case 0x87:
        case 0x88: case 0x89: case 0x8A: case 0x8B:
        case 0x8C: case 0x8D: case 0x8E: case 0x8F: {
            int32_t rel = (int32_t)read_imm32();
            if (eval_cc(op2 - 0x80)) {
                regs_.rip += (int64_t)rel;
            }
            break;
        }

        // 0x0F 0x90-0x9F: SETcc r/m8
        case 0x90: case 0x91: case 0x92: case 0x93:
        case 0x94: case 0x95: case 0x96: case 0x97:
        case 0x98: case 0x99: case 0x9A: case 0x9B:
        case 0x9C: case 0x9D: case 0x9E: case 0x9F: {
            uint8_t modrm = read_modrm();
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            write_to_rm(rm_addr, rm_is_reg, eval_cc(op2 - 0x90) ? 1 : 0, 8);
            break;
        }

        // 0x0F 0xA2: CPUID
        case 0xA2: {
            uint32_t leaf = (uint32_t)regs_.rax;
            regs_.rax = regs_.rbx = regs_.rcx = regs_.rdx = 0;
            switch (leaf) {
                case 0:
                    regs_.rax = 0x16; // max leaf
                    regs_.rbx = 0x756E6547; // "Genu"
                    regs_.rdx = 0x49656E69; // "ineI"
                    regs_.rcx = 0x6C65746E; // "ntel"
                    break;
                case 1:
                    regs_.rax = 0x000906EA; // family/model/stepping
                    regs_.rbx = 0x00100800;
                    regs_.rcx = 0x7FFAFBBF; // SSE4.2, POPCNT, etc.
                    regs_.rdx = 0xBFEBFBFF; // SSE2, etc.
                    break;
                case 0x80000000:
                    regs_.rax = 0x80000008;
                    break;
                case 0x80000001:
                    regs_.rcx = 0x00000121; // LAHF, ABM
                    regs_.rdx = 0x2C100800; // NX, LM, RDTSCP
                    break;
            }
            break;
        }

        // 0x0F 0xA3: BT r/m, r
        case 0xA3: {
            uint8_t modrm = read_modrm();
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            uint64_t val = read_from_rm(rm_addr, rm_is_reg, op_size);
            int bit = truncate(RegByIndex(reg_field(modrm)), op_size) % op_size;
            SetCF((val >> bit) & 1);
            break;
        }

        // 0x0F 0xAB: BTS r/m, r
        case 0xAB: {
            uint8_t modrm = read_modrm();
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            uint64_t val = read_from_rm(rm_addr, rm_is_reg, op_size);
            int bit = truncate(RegByIndex(reg_field(modrm)), op_size) % op_size;
            SetCF((val >> bit) & 1);
            val |= (1ULL << bit);
            write_to_rm(rm_addr, rm_is_reg, truncate(val, op_size), op_size);
            break;
        }

        // 0x0F 0xAF: IMUL r, r/m
        case 0xAF: {
            uint8_t modrm = read_modrm();
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            int reg = reg_field(modrm);
            int64_t a = (int64_t)sign_extend(RegByIndex(reg), op_size);
            int64_t b = (int64_t)sign_extend(read_from_rm(rm_addr, rm_is_reg, op_size), op_size);
            __int128 wide = (__int128)a * (__int128)b;
            uint64_t result = truncate((uint64_t)wide, op_size);
            if (op_size == 32) RegByIndex(reg) = (uint32_t)result;
            else RegByIndex(reg) = result;
            int64_t sr = (int64_t)sign_extend(result, op_size);
            SetCF(sr != (int64_t)wide);
            SetOF(sr != (int64_t)wide);
            break;
        }

        // 0x0F 0xB3: BTR r/m, r
        case 0xB3: {
            uint8_t modrm = read_modrm();
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            uint64_t val = read_from_rm(rm_addr, rm_is_reg, op_size);
            int bit = truncate(RegByIndex(reg_field(modrm)), op_size) % op_size;
            SetCF((val >> bit) & 1);
            val &= ~(1ULL << bit);
            write_to_rm(rm_addr, rm_is_reg, truncate(val, op_size), op_size);
            break;
        }

        // 0x0F 0xB6: MOVZX r, r/m8
        case 0xB6: {
            uint8_t modrm = read_modrm();
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            int reg = reg_field(modrm);
            uint8_t val = (uint8_t)read_from_rm(rm_addr, rm_is_reg, 8);
            if (op_size == 32) RegByIndex(reg) = (uint32_t)val;
            else RegByIndex(reg) = val;
            break;
        }

        // 0x0F 0xB7: MOVZX r, r/m16
        case 0xB7: {
            uint8_t modrm = read_modrm();
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            int reg = reg_field(modrm);
            uint16_t val = (uint16_t)read_from_rm(rm_addr, rm_is_reg, 16);
            if (op_size == 32 || rex_w) RegByIndex(reg) = (uint32_t)val;
            else RegByIndex(reg) = val;
            break;
        }

        // 0x0F 0xBA: group 8 — BT/BTS/BTR/BTC r/m, imm8
        case 0xBA: {
            uint8_t modrm = read_modrm();
            int sub = reg_field(modrm) & 7;
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            uint64_t val = read_from_rm(rm_addr, rm_is_reg, op_size);
            int bit = read_imm8() % op_size;
            SetCF((val >> bit) & 1);
            switch (sub) {
                case 4: break; // BT
                case 5: val |= (1ULL << bit); break;  // BTS
                case 6: val &= ~(1ULL << bit); break; // BTR
                case 7: val ^= (1ULL << bit); break;  // BTC
                default: break;
            }
            if (sub >= 5) write_to_rm(rm_addr, rm_is_reg, truncate(val, op_size), op_size);
            break;
        }

        // 0x0F 0xBC: BSF r, r/m
        case 0xBC: {
            uint8_t modrm = read_modrm();
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            int reg = reg_field(modrm);
            uint64_t val = read_from_rm(rm_addr, rm_is_reg, op_size);
            val = truncate(val, op_size);
            if (val == 0) {
                SetZF(true);
            } else {
                SetZF(false);
                int bit = 0;
                while (!((val >> bit) & 1)) bit++;
                if (op_size == 32) RegByIndex(reg) = (uint32_t)bit;
                else RegByIndex(reg) = bit;
            }
            break;
        }

        // 0x0F 0xBD: BSR r, r/m
        case 0xBD: {
            uint8_t modrm = read_modrm();
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            int reg = reg_field(modrm);
            uint64_t val = read_from_rm(rm_addr, rm_is_reg, op_size);
            val = truncate(val, op_size);
            if (val == 0) {
                SetZF(true);
            } else {
                SetZF(false);
                int bit = op_size - 1;
                while (bit >= 0 && !((val >> bit) & 1)) bit--;
                if (op_size == 32) RegByIndex(reg) = (uint32_t)bit;
                else RegByIndex(reg) = bit;
            }
            break;
        }

        // 0x0F 0xBE: MOVSX r, r/m8
        case 0xBE: {
            uint8_t modrm = read_modrm();
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            int reg = reg_field(modrm);
            int8_t val = (int8_t)read_from_rm(rm_addr, rm_is_reg, 8);
            if (op_size == 32) RegByIndex(reg) = (uint32_t)(int32_t)val;
            else if (rex_w) RegByIndex(reg) = (uint64_t)(int64_t)val;
            else RegByIndex(reg) = (RegByIndex(reg) & ~0xFFFFULL) | (uint16_t)(int16_t)val;
            break;
        }

        // 0x0F 0xBF: MOVSX r, r/m16
        case 0xBF: {
            uint8_t modrm = read_modrm();
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            int reg = reg_field(modrm);
            int16_t val = (int16_t)read_from_rm(rm_addr, rm_is_reg, 16);
            if (rex_w) RegByIndex(reg) = (uint64_t)(int64_t)val;
            else RegByIndex(reg) = (uint32_t)(int32_t)val;
            break;
        }

        // 0x0F 0xC8-0xCF: BSWAP r32/r64
        case 0xC8: case 0xC9: case 0xCA: case 0xCB:
        case 0xCC: case 0xCD: case 0xCE: case 0xCF: {
            int reg = (op2 - 0xC8) | (rex_b << 3);
            if (rex_w) {
                uint64_t v = RegByIndex(reg);
                v = ((v & 0x00000000000000FFULL) << 56) |
                    ((v & 0x000000000000FF00ULL) << 40) |
                    ((v & 0x0000000000FF0000ULL) << 24) |
                    ((v & 0x00000000FF000000ULL) << 8)  |
                    ((v & 0x000000FF00000000ULL) >> 8)  |
                    ((v & 0x0000FF0000000000ULL) >> 24) |
                    ((v & 0x00FF000000000000ULL) >> 40) |
                    ((v & 0xFF00000000000000ULL) >> 56);
                RegByIndex(reg) = v;
            } else {
                uint32_t v = (uint32_t)RegByIndex(reg);
                v = ((v & 0xFF) << 24) | ((v & 0xFF00) << 8) | ((v & 0xFF0000) >> 8) | ((v >> 24) & 0xFF);
                RegByIndex(reg) = v; // zero-extends
            }
            break;
        }

        // 0x0F 0xEF: PXOR xmm, xmm/m128
        case 0xEF: {
            uint8_t modrm = read_modrm();
            int reg = reg_field(modrm);
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            uint8_t src[16];
            if (rm_is_reg) memcpy(src, regs_.xmm[(int)rm_addr], 16);
            else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
            for (int i = 0; i < 16; i++) regs_.xmm[reg][i] ^= src[i];
            break;
        }

        // 0x0F 0xC7: CMPXCHG8B/CMPXCHG16B m64/m128
        case 0xC7: {
            uint8_t modrm = read_modrm();
            int sub = reg_field(modrm);
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            if (sub == 1 && !rm_is_reg) {
                if (rex_w) {
                    // CMPXCHG16B
                    uint64_t lo = ReadMemory64(rm_addr);
                    uint64_t hi = ReadMemory64(rm_addr + 8);
                    if (lo == regs_.rax && hi == regs_.rdx) {
                        SetZF(true);
                        WriteMemory64(rm_addr, regs_.rbx);
                        WriteMemory64(rm_addr + 8, regs_.rcx);
                    } else {
                        SetZF(false);
                        regs_.rax = lo;
                        regs_.rdx = hi;
                    }
                } else {
                    // CMPXCHG8B
                    uint64_t val = ReadMemory64(rm_addr);
                    uint64_t cmp = ((uint64_t)regs_.rdx << 32) | (regs_.rax & 0xFFFFFFFF);
                    if (val == cmp) {
                        SetZF(true);
                        uint64_t newval = ((uint64_t)regs_.rcx << 32) | (regs_.rbx & 0xFFFFFFFF);
                        WriteMemory64(rm_addr, newval);
                    } else {
                        SetZF(false);
                        regs_.rax = val & 0xFFFFFFFF;
                        regs_.rdx = val >> 32;
                    }
                }
            } else if (sub == 6) {
                // RDRAND — return random value
                RegByIndex(modrm & 7) = insn_count_ * 6364136223846793005ULL + 1442695040888963407ULL;
                SetCF(true);
            }
            break;
        }

        // 0x0F 0xD5: PMULLW — packed multiply low words (with 66 prefix)
        case 0xD5: {
            uint8_t modrm = read_modrm();
            int reg = reg_field(modrm);
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            break;
        }

        default:
            fprintf(stderr, "[x86] Unimplemented 0F opcode: 0x%02X at RIP=0x%llX\n", op2,
                    (unsigned long long)(regs_.rip - 2));
            return X86Exception::InvalidOpcode;
        }
        break;
    } // end 0x0F

    // 0xE0: LOOPNE rel8
    case 0xE0: {
        int8_t rel = (int8_t)read_imm8();
        regs_.rcx--;
        if (regs_.rcx != 0 && !GetZF()) regs_.rip += (int64_t)rel;
        break;
    }
    // 0xE1: LOOPE rel8
    case 0xE1: {
        int8_t rel = (int8_t)read_imm8();
        regs_.rcx--;
        if (regs_.rcx != 0 && GetZF()) regs_.rip += (int64_t)rel;
        break;
    }
    // 0xE2: LOOP rel8
    case 0xE2: {
        int8_t rel = (int8_t)read_imm8();
        regs_.rcx--;
        if (regs_.rcx != 0) regs_.rip += (int64_t)rel;
        break;
    }

    // --- EVEX prefix (AVX-512) — decode and skip ---
    case 0x62: {
        // EVEX is 4 bytes: 62 [P0] [P1] [P2] opcode modrm ...
        // We decode enough to skip the instruction
        uint8_t p0 = ReadMemory8(regs_.rip++);
        uint8_t p1 = ReadMemory8(regs_.rip++);
        uint8_t p2 = ReadMemory8(regs_.rip++);
        uint8_t evex_op = ReadMemory8(regs_.rip++);
        (void)p0; (void)p1; (void)p2;
        // Consume modrm + operands
        uint8_t modrm = read_modrm();
        uint64_t rm_addr; bool rm_is_reg;
        decode_rm(modrm, &rm_addr, &rm_is_reg);
        // Some EVEX instructions have an immediate
        // 0F3A map instructions typically have an imm8
        int mmm = p0 & 7;
        if (mmm == 3) read_imm8(); // 0F3A map: has imm8
        static int evex_count = 0;
        if (evex_count < 5)
            fprintf(stderr, "[x86] EVEX instruction (op=0x%02X map=%d) at RIP=0x%llX — treated as NOP\n",
                    evex_op, mmm, (unsigned long long)(regs_.rip));
        evex_count++;
        break;
    }

    // --- VEX-encoded instructions (AVX) ---
    case 0xC5: case 0xC4: {
        // VEX prefix decoding
        int vex_R = 1, vex_X = 1, vex_B = 1;
        int vex_W = 0, vex_L = 0, vex_pp = 0, vex_vvvv = 0;
        int vex_mmmmm = 1; // default 0F map
        uint8_t vex_opcode;

        if (opcode == 0xC5) {
            // 2-byte VEX: C5 [RvvvvLpp]
            uint8_t b1 = ReadMemory8(regs_.rip++);
            vex_R = (b1 >> 7) & 1;
            vex_vvvv = (~b1 >> 3) & 0xF;
            vex_L = (b1 >> 2) & 1;
            vex_pp = b1 & 3;
        } else {
            // 3-byte VEX: C4 [RXBmmmmm] [WvvvvLpp]
            uint8_t b1 = ReadMemory8(regs_.rip++);
            uint8_t b2 = ReadMemory8(regs_.rip++);
            vex_R = (b1 >> 7) & 1;
            vex_X = (b1 >> 6) & 1;
            vex_B = (b1 >> 5) & 1;
            vex_mmmmm = b1 & 0x1F;
            vex_W = (b2 >> 7) & 1;
            vex_vvvv = (~b2 >> 3) & 0xF;
            vex_L = (b2 >> 2) & 1;
            vex_pp = b2 & 3;
        }

        vex_opcode = ReadMemory8(regs_.rip++);

        // Override REX fields from VEX
        rex_r = vex_R ? 0 : 1;
        rex_x = vex_X ? 0 : 1;
        rex_b = vex_B ? 0 : 1;
        rex_w = vex_W;
        rex = true;

        // VEX.pp maps to legacy prefix: 0=none, 1=66, 2=F3, 3=F2
        bool vex_66 = (vex_pp == 1);
        bool vex_f3 = (vex_pp == 2);
        bool vex_f2 = (vex_pp == 3);
        (void)vex_66; (void)vex_f2;
        int vec_len = vex_L ? 32 : 16; // YMM (256) or XMM (128)

        if (vex_mmmmm == 1) {
            // 0F opcode map (most AVX instructions)
            switch (vex_opcode) {

            // VMOVUPS / VMOVSS / VMOVSD / VMOVUPD
            case 0x10: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);

                if (vex_f3) {
                    // VMOVSS xmm, xmm, xmm  or  VMOVSS xmm, m32
                    if (rm_is_reg) {
                        // 3-operand: merge vvvv with scalar from rm
                        if (reg < 16) memcpy(regs_.xmm[reg], regs_.xmm[vex_vvvv < 16 ? vex_vvvv : 0], 16);
                        uint32_t val;
                        memcpy(&val, regs_.xmm[(int)rm_addr & 15], 4);
                        memcpy(regs_.xmm[reg], &val, 4);
                    } else {
                        memset(regs_.xmm[reg], 0, 16);
                        uint32_t val = ReadMemory32(rm_addr);
                        memcpy(regs_.xmm[reg], &val, 4);
                    }
                } else {
                    // VMOVUPS xmm/ymm, xmm/m128/m256
                    if (rm_is_reg) {
                        memcpy(regs_.xmm[reg], regs_.xmm[(int)rm_addr & 15], 16);
                    } else {
                        uint64_t off = rm_addr - mem_base_addr_;
                        if (off + 16 <= mem_size_) memcpy(regs_.xmm[reg], mem_base_ + off, 16);
                        else memset(regs_.xmm[reg], 0, 16);
                    }
                }
                break;
            }
            case 0x11: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);

                if (vex_f3) {
                    // VMOVSS xmm/m32, xmm
                    if (rm_is_reg) {
                        uint32_t val;
                        memcpy(&val, regs_.xmm[reg], 4);
                        memcpy(regs_.xmm[(int)rm_addr & 15], &val, 4);
                    } else {
                        uint32_t val;
                        memcpy(&val, regs_.xmm[reg], 4);
                        WriteMemory32(rm_addr, val);
                    }
                } else {
                    // VMOVUPS xmm/m128, xmm
                    if (rm_is_reg) {
                        memcpy(regs_.xmm[(int)rm_addr & 15], regs_.xmm[reg], 16);
                    } else {
                        uint64_t off = rm_addr - mem_base_addr_;
                        if (off + 16 <= mem_size_) memcpy(mem_base_ + off, regs_.xmm[reg], 16);
                    }
                }
                break;
            }

            // VMOVAPS / VMOVAPD
            case 0x28: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                if (rm_is_reg) memcpy(regs_.xmm[reg], regs_.xmm[(int)rm_addr & 15], 16);
                else {
                    uint64_t off = rm_addr - mem_base_addr_;
                    if (off + 16 <= mem_size_) memcpy(regs_.xmm[reg], mem_base_ + off, 16);
                }
                break;
            }
            case 0x29: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                if (rm_is_reg) memcpy(regs_.xmm[(int)rm_addr & 15], regs_.xmm[reg], 16);
                else {
                    uint64_t off = rm_addr - mem_base_addr_;
                    if (off + 16 <= mem_size_) memcpy(mem_base_ + off, regs_.xmm[reg], 16);
                }
                break;
            }

            // VUCOMISS / VUCOMISD
            case 0x2E: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                float a, b_val;
                memcpy(&a, regs_.xmm[reg], 4);
                if (rm_is_reg) memcpy(&b_val, regs_.xmm[(int)rm_addr & 15], 4);
                else { uint32_t tmp = ReadMemory32(rm_addr); memcpy(&b_val, &tmp, 4); }
                SetCF(false); SetZF(false); SetPF(false); SetOF(false); SetSF(false); SetAF(false);
                if (a != a || b_val != b_val) { SetCF(true); SetZF(true); SetPF(true); }
                else if (a < b_val) SetCF(true);
                else if (a == b_val) SetZF(true);
                break;
            }

            // VXORPS / VXORPD / VPXOR
            case 0x57: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t src[16];
                if (rm_is_reg) memcpy(src, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
                uint8_t vvvv_data[16];
                memcpy(vvvv_data, regs_.xmm[vex_vvvv & 15], 16);
                for (int i = 0; i < 16; i++) regs_.xmm[reg][i] = vvvv_data[i] ^ src[i];
                break;
            }

            // VMULPS / VMULSS
            case 0x59: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);

                if (vex_f3) {
                    // VMULSS
                    float a, b_val;
                    memcpy(&a, regs_.xmm[vex_vvvv & 15], 4);
                    if (rm_is_reg) memcpy(&b_val, regs_.xmm[(int)rm_addr & 15], 4);
                    else { uint32_t tmp = ReadMemory32(rm_addr); memcpy(&b_val, &tmp, 4); }
                    float r = a * b_val;
                    memcpy(regs_.xmm[reg], regs_.xmm[vex_vvvv & 15], 16);
                    memcpy(regs_.xmm[reg], &r, 4);
                } else {
                    // VMULPS
                    float src[4], dst[4];
                    memcpy(dst, regs_.xmm[vex_vvvv & 15], 16);
                    if (rm_is_reg) memcpy(src, regs_.xmm[(int)rm_addr & 15], 16);
                    else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
                    for (int i = 0; i < 4; i++) dst[i] *= src[i];
                    memcpy(regs_.xmm[reg], dst, 16);
                }
                break;
            }

            // VADDPS / VADDSS
            case 0x58: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);

                if (vex_f3) {
                    float a, b_val;
                    memcpy(&a, regs_.xmm[vex_vvvv & 15], 4);
                    if (rm_is_reg) memcpy(&b_val, regs_.xmm[(int)rm_addr & 15], 4);
                    else { uint32_t tmp = ReadMemory32(rm_addr); memcpy(&b_val, &tmp, 4); }
                    float r = a + b_val;
                    memcpy(regs_.xmm[reg], regs_.xmm[vex_vvvv & 15], 16);
                    memcpy(regs_.xmm[reg], &r, 4);
                } else {
                    float src_f[4], dst_f[4];
                    memcpy(dst_f, regs_.xmm[vex_vvvv & 15], 16);
                    if (rm_is_reg) memcpy(src_f, regs_.xmm[(int)rm_addr & 15], 16);
                    else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src_f, mem_base_ + off, 16); else memset(src_f, 0, 16); }
                    for (int i = 0; i < 4; i++) dst_f[i] += src_f[i];
                    memcpy(regs_.xmm[reg], dst_f, 16);
                }
                break;
            }

            // VSUBPS / VSUBSS
            case 0x5C: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);

                if (vex_f3) {
                    float a, b_val;
                    memcpy(&a, regs_.xmm[vex_vvvv & 15], 4);
                    if (rm_is_reg) memcpy(&b_val, regs_.xmm[(int)rm_addr & 15], 4);
                    else { uint32_t tmp = ReadMemory32(rm_addr); memcpy(&b_val, &tmp, 4); }
                    float r = a - b_val;
                    memcpy(regs_.xmm[reg], regs_.xmm[vex_vvvv & 15], 16);
                    memcpy(regs_.xmm[reg], &r, 4);
                } else {
                    float src_f[4], dst_f[4];
                    memcpy(dst_f, regs_.xmm[vex_vvvv & 15], 16);
                    if (rm_is_reg) memcpy(src_f, regs_.xmm[(int)rm_addr & 15], 16);
                    else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src_f, mem_base_ + off, 16); else memset(src_f, 0, 16); }
                    for (int i = 0; i < 4; i++) dst_f[i] -= src_f[i];
                    memcpy(regs_.xmm[reg], dst_f, 16);
                }
                break;
            }

            // VDIVPS / VDIVSS
            case 0x5E: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);

                if (vex_f3) {
                    float a, b_val;
                    memcpy(&a, regs_.xmm[vex_vvvv & 15], 4);
                    if (rm_is_reg) memcpy(&b_val, regs_.xmm[(int)rm_addr & 15], 4);
                    else { uint32_t tmp = ReadMemory32(rm_addr); memcpy(&b_val, &tmp, 4); }
                    float r = (b_val != 0.0f) ? a / b_val : 0.0f;
                    memcpy(regs_.xmm[reg], regs_.xmm[vex_vvvv & 15], 16);
                    memcpy(regs_.xmm[reg], &r, 4);
                } else {
                    float src_f[4], dst_f[4];
                    memcpy(dst_f, regs_.xmm[vex_vvvv & 15], 16);
                    if (rm_is_reg) memcpy(src_f, regs_.xmm[(int)rm_addr & 15], 16);
                    else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src_f, mem_base_ + off, 16); else memset(src_f, 0, 16); }
                    for (int i = 0; i < 4; i++) dst_f[i] = (src_f[i] != 0.0f) ? dst_f[i] / src_f[i] : 0.0f;
                    memcpy(regs_.xmm[reg], dst_f, 16);
                }
                break;
            }

            // VANDPS
            case 0x54: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t src_b[16], vvvv_b[16];
                memcpy(vvvv_b, regs_.xmm[vex_vvvv & 15], 16);
                if (rm_is_reg) memcpy(src_b, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src_b, mem_base_ + off, 16); else memset(src_b, 0, 16); }
                for (int i = 0; i < 16; i++) regs_.xmm[reg][i] = vvvv_b[i] & src_b[i];
                break;
            }

            // VANDNPS
            case 0x55: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t src_b[16], vvvv_b[16];
                memcpy(vvvv_b, regs_.xmm[vex_vvvv & 15], 16);
                if (rm_is_reg) memcpy(src_b, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src_b, mem_base_ + off, 16); else memset(src_b, 0, 16); }
                for (int i = 0; i < 16; i++) regs_.xmm[reg][i] = (~vvvv_b[i]) & src_b[i];
                break;
            }

            // VORPS
            case 0x56: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t src_b[16], vvvv_b[16];
                memcpy(vvvv_b, regs_.xmm[vex_vvvv & 15], 16);
                if (rm_is_reg) memcpy(src_b, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src_b, mem_base_ + off, 16); else memset(src_b, 0, 16); }
                for (int i = 0; i < 16; i++) regs_.xmm[reg][i] = vvvv_b[i] | src_b[i];
                break;
            }

            // VCVTSI2SS
            case 0x2A: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                if (vex_f3) {
                    int64_t ival;
                    if (rm_is_reg) ival = (int64_t)RegByIndex((int)rm_addr);
                    else ival = (int64_t)(int32_t)ReadMemory32(rm_addr);
                    float fval = rex_w ? (float)ival : (float)(int32_t)ival;
                    memcpy(regs_.xmm[reg], regs_.xmm[vex_vvvv & 15], 16);
                    memcpy(regs_.xmm[reg], &fval, 4);
                }
                break;
            }

            // VCVTTSS2SI
            case 0x2C: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                if (vex_f3) {
                    float fval;
                    if (rm_is_reg) memcpy(&fval, regs_.xmm[(int)rm_addr & 15], 4);
                    else { uint32_t tmp = ReadMemory32(rm_addr); memcpy(&fval, &tmp, 4); }
                    if (rex_w) RegByIndex(reg) = (uint64_t)(int64_t)fval;
                    else RegByIndex(reg) = (uint64_t)(uint32_t)(int32_t)fval;
                }
                break;
            }

            // VMOVD / VMOVQ xmm, r/m32/64
            case 0x6E: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                memset(regs_.xmm[reg], 0, 16);
                if (rex_w) {
                    uint64_t val = rm_is_reg ? RegByIndex((int)rm_addr) : ReadMemory64(rm_addr);
                    memcpy(regs_.xmm[reg], &val, 8);
                } else {
                    uint32_t val = rm_is_reg ? (uint32_t)RegByIndex((int)rm_addr) : ReadMemory32(rm_addr);
                    memcpy(regs_.xmm[reg], &val, 4);
                }
                break;
            }

            // VMOVD / VMOVQ r/m32/64, xmm
            case 0x7E: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                if (rex_w) {
                    uint64_t val;
                    memcpy(&val, regs_.xmm[reg], 8);
                    if (rm_is_reg) RegByIndex((int)rm_addr) = val;
                    else WriteMemory64(rm_addr, val);
                } else {
                    uint32_t val;
                    memcpy(&val, regs_.xmm[reg], 4);
                    if (rm_is_reg) RegByIndex((int)rm_addr) = val;
                    else WriteMemory32(rm_addr, val);
                }
                break;
            }

            // VMOVDQA / VMOVDQU (66/F3 prefix variants)
            case 0x6F: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                if (rm_is_reg) memcpy(regs_.xmm[reg], regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(regs_.xmm[reg], mem_base_ + off, 16); }
                break;
            }
            case 0x7F: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                if (rm_is_reg) memcpy(regs_.xmm[(int)rm_addr & 15], regs_.xmm[reg], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(mem_base_ + off, regs_.xmm[reg], 16); }
                break;
            }

            // VSQRTPS / VSQRTSS
            case 0x51: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                if (vex_f3) {
                    float val;
                    if (rm_is_reg) memcpy(&val, regs_.xmm[(int)rm_addr & 15], 4);
                    else { uint32_t tmp = ReadMemory32(rm_addr); memcpy(&val, &tmp, 4); }
                    float r = sqrtf(val);
                    memcpy(regs_.xmm[reg], regs_.xmm[vex_vvvv & 15], 16);
                    memcpy(regs_.xmm[reg], &r, 4);
                } else {
                    float src_f[4];
                    if (rm_is_reg) memcpy(src_f, regs_.xmm[(int)rm_addr & 15], 16);
                    else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src_f, mem_base_ + off, 16); else memset(src_f, 0, 16); }
                    for (int i = 0; i < 4; i++) src_f[i] = sqrtf(src_f[i]);
                    memcpy(regs_.xmm[reg], src_f, 16);
                }
                break;
            }

            // VMINPS / VMINSS
            case 0x5D: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                if (vex_f3) {
                    float a, b_val;
                    memcpy(&a, regs_.xmm[vex_vvvv & 15], 4);
                    if (rm_is_reg) memcpy(&b_val, regs_.xmm[(int)rm_addr & 15], 4);
                    else { uint32_t tmp = ReadMemory32(rm_addr); memcpy(&b_val, &tmp, 4); }
                    float r = (a < b_val) ? a : b_val;
                    memcpy(regs_.xmm[reg], regs_.xmm[vex_vvvv & 15], 16);
                    memcpy(regs_.xmm[reg], &r, 4);
                } else {
                    float src_f[4], dst_f[4];
                    memcpy(dst_f, regs_.xmm[vex_vvvv & 15], 16);
                    if (rm_is_reg) memcpy(src_f, regs_.xmm[(int)rm_addr & 15], 16);
                    else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src_f, mem_base_ + off, 16); else memset(src_f, 0, 16); }
                    for (int i = 0; i < 4; i++) dst_f[i] = (dst_f[i] < src_f[i]) ? dst_f[i] : src_f[i];
                    memcpy(regs_.xmm[reg], dst_f, 16);
                }
                break;
            }

            // VMAXPS / VMAXSS
            case 0x5F: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                if (vex_f3) {
                    float a, b_val;
                    memcpy(&a, regs_.xmm[vex_vvvv & 15], 4);
                    if (rm_is_reg) memcpy(&b_val, regs_.xmm[(int)rm_addr & 15], 4);
                    else { uint32_t tmp = ReadMemory32(rm_addr); memcpy(&b_val, &tmp, 4); }
                    float r = (a > b_val) ? a : b_val;
                    memcpy(regs_.xmm[reg], regs_.xmm[vex_vvvv & 15], 16);
                    memcpy(regs_.xmm[reg], &r, 4);
                } else {
                    float src_f[4], dst_f[4];
                    memcpy(dst_f, regs_.xmm[vex_vvvv & 15], 16);
                    if (rm_is_reg) memcpy(src_f, regs_.xmm[(int)rm_addr & 15], 16);
                    else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src_f, mem_base_ + off, 16); else memset(src_f, 0, 16); }
                    for (int i = 0; i < 4; i++) dst_f[i] = (dst_f[i] > src_f[i]) ? dst_f[i] : src_f[i];
                    memcpy(regs_.xmm[reg], dst_f, 16);
                }
                break;
            }

            // VSHUFPS
            case 0xC6: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t imm = read_imm8();
                float src_f[4], vvvv_f[4], dst_f[4];
                memcpy(vvvv_f, regs_.xmm[vex_vvvv & 15], 16);
                if (rm_is_reg) memcpy(src_f, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src_f, mem_base_ + off, 16); else memset(src_f, 0, 16); }
                dst_f[0] = vvvv_f[(imm >> 0) & 3];
                dst_f[1] = vvvv_f[(imm >> 2) & 3];
                dst_f[2] = src_f[(imm >> 4) & 3];
                dst_f[3] = src_f[(imm >> 6) & 3];
                memcpy(regs_.xmm[reg], dst_f, 16);
                break;
            }

            // VCMPPS / VCMPSS
            case 0xC2: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t imm = read_imm8();
                if (vex_f3) {
                    float a, b_val;
                    memcpy(&a, regs_.xmm[vex_vvvv & 15], 4);
                    if (rm_is_reg) memcpy(&b_val, regs_.xmm[(int)rm_addr & 15], 4);
                    else { uint32_t tmp = ReadMemory32(rm_addr); memcpy(&b_val, &tmp, 4); }
                    bool cmp = false;
                    switch (imm & 7) {
                        case 0: cmp = (a == b_val); break;
                        case 1: cmp = (a < b_val); break;
                        case 2: cmp = (a <= b_val); break;
                        case 4: cmp = (a != b_val); break;
                        case 5: cmp = !(a < b_val); break;
                        case 6: cmp = !(a <= b_val); break;
                        default: break;
                    }
                    uint32_t result = cmp ? 0xFFFFFFFF : 0;
                    memcpy(regs_.xmm[reg], regs_.xmm[vex_vvvv & 15], 16);
                    memcpy(regs_.xmm[reg], &result, 4);
                } else {
                    // VCMPPS — packed comparison
                    float src_f[4], dst_f[4];
                    uint32_t out[4];
                    memcpy(dst_f, regs_.xmm[vex_vvvv & 15], 16);
                    if (rm_is_reg) memcpy(src_f, regs_.xmm[(int)rm_addr & 15], 16);
                    else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src_f, mem_base_ + off, 16); else memset(src_f, 0, 16); }
                    for (int i = 0; i < 4; i++) {
                        bool c = false;
                        switch (imm & 7) {
                            case 0: c = (dst_f[i] == src_f[i]); break;
                            case 1: c = (dst_f[i] < src_f[i]); break;
                            case 2: c = (dst_f[i] <= src_f[i]); break;
                            case 4: c = (dst_f[i] != src_f[i]); break;
                            case 5: c = !(dst_f[i] < src_f[i]); break;
                            case 6: c = !(dst_f[i] <= src_f[i]); break;
                            default: break;
                        }
                        out[i] = c ? 0xFFFFFFFF : 0;
                    }
                    memcpy(regs_.xmm[reg], out, 16);
                }
                break;
            }

            // VMOVMSKPS
            case 0x50: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint32_t vals[4];
                if (rm_is_reg) memcpy(vals, regs_.xmm[(int)rm_addr & 15], 16);
                else break; // memory form doesn't exist
                uint32_t mask = 0;
                for (int i = 0; i < 4; i++) if (vals[i] >> 31) mask |= (1 << i);
                RegByIndex(reg) = mask;
                break;
            }

            // VPXOR (66 prefix, opcode 0xEF)
            case 0xEF: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t src_b[16], vvvv_b[16];
                memcpy(vvvv_b, regs_.xmm[vex_vvvv & 15], 16);
                if (rm_is_reg) memcpy(src_b, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src_b, mem_base_ + off, 16); else memset(src_b, 0, 16); }
                for (int i = 0; i < 16; i++) regs_.xmm[reg][i] = vvvv_b[i] ^ src_b[i];
                break;
            }

            // VPADDQ (66 prefix)
            case 0xD4: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint64_t src[2], vvvv_q[2];
                memcpy(vvvv_q, regs_.xmm[vex_vvvv & 15], 16);
                if (rm_is_reg) memcpy(src, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
                vvvv_q[0] += src[0];
                vvvv_q[1] += src[1];
                memcpy(regs_.xmm[reg], vvvv_q, 16);
                break;
            }

            // VPSUBQ (66 prefix)
            case 0xFB: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint64_t src[2], vvvv_q[2];
                memcpy(vvvv_q, regs_.xmm[vex_vvvv & 15], 16);
                if (rm_is_reg) memcpy(src, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
                vvvv_q[0] -= src[0];
                vvvv_q[1] -= src[1];
                memcpy(regs_.xmm[reg], vvvv_q, 16);
                break;
            }

            // VPSRLQ (66 prefix) — shift right logical qword
            case 0xD3: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint64_t shift_amount;
                if (rm_is_reg) memcpy(&shift_amount, regs_.xmm[(int)rm_addr & 15], 8);
                else shift_amount = ReadMemory64(rm_addr);
                uint64_t data[2];
                memcpy(data, regs_.xmm[vex_vvvv & 15], 16);
                if (shift_amount < 64) { data[0] >>= shift_amount; data[1] >>= shift_amount; }
                else { data[0] = 0; data[1] = 0; }
                memcpy(regs_.xmm[reg], data, 16);
                break;
            }

            // VPINSRW
            case 0xC4: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t imm = read_imm8();
                uint16_t val;
                if (rm_is_reg) val = (uint16_t)RegByIndex((int)rm_addr);
                else val = ReadMemory16(rm_addr);
                memcpy(regs_.xmm[reg], regs_.xmm[vex_vvvv & 15], 16);
                int slot = imm & 7;
                memcpy(regs_.xmm[reg] + slot * 2, &val, 2);
                break;
            }

            // VPUNPCKLQDQ (66 prefix)
            case 0x6C: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint64_t src_lo, vvvv_lo;
                memcpy(&vvvv_lo, regs_.xmm[vex_vvvv & 15], 8);
                if (rm_is_reg) memcpy(&src_lo, regs_.xmm[(int)rm_addr & 15], 8);
                else src_lo = ReadMemory64(rm_addr);
                memcpy(regs_.xmm[reg], &vvvv_lo, 8);
                memcpy(regs_.xmm[reg] + 8, &src_lo, 8);
                break;
            }

            // VPEXTRW
            case 0xC5: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t imm = read_imm8();
                if (rm_is_reg) {
                    uint16_t val;
                    memcpy(&val, regs_.xmm[(int)rm_addr & 15] + (imm & 7) * 2, 2);
                    RegByIndex(reg) = val;
                }
                break;
            }

            // VPSHUFD
            case 0x70: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t imm = read_imm8();
                uint32_t src_d[4];
                if (rm_is_reg) memcpy(src_d, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src_d, mem_base_ + off, 16); else memset(src_d, 0, 16); }
                uint32_t dst_d[4];
                dst_d[0] = src_d[(imm >> 0) & 3];
                dst_d[1] = src_d[(imm >> 2) & 3];
                dst_d[2] = src_d[(imm >> 4) & 3];
                dst_d[3] = src_d[(imm >> 6) & 3];
                memcpy(regs_.xmm[reg], dst_d, 16);
                break;
            }

            // VPCMPEQB/W/D (0x74, 0x75, 0x76)
            case 0x74: case 0x75: case 0x76: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t src[16], vvvv_b[16];
                memcpy(vvvv_b, regs_.xmm[vex_vvvv & 15], 16);
                if (rm_is_reg) memcpy(src, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
                if (vex_opcode == 0x74) { // byte
                    for (int i = 0; i < 16; i++) regs_.xmm[reg][i] = (vvvv_b[i] == src[i]) ? 0xFF : 0;
                } else if (vex_opcode == 0x75) { // word
                    for (int i = 0; i < 8; i++) { uint16_t a, b; memcpy(&a, vvvv_b + i*2, 2); memcpy(&b, src + i*2, 2); uint16_t r = (a == b) ? 0xFFFF : 0; memcpy(regs_.xmm[reg] + i*2, &r, 2); }
                } else { // dword
                    for (int i = 0; i < 4; i++) { uint32_t a, b; memcpy(&a, vvvv_b + i*4, 4); memcpy(&b, src + i*4, 4); uint32_t r = (a == b) ? 0xFFFFFFFF : 0; memcpy(regs_.xmm[reg] + i*4, &r, 4); }
                }
                break;
            }

            // VPCMPGTB/W/D (0x64, 0x65, 0x66)
            case 0x64: case 0x65: case 0x66: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t src[16], vvvv_b[16];
                memcpy(vvvv_b, regs_.xmm[vex_vvvv & 15], 16);
                if (rm_is_reg) memcpy(src, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
                if (vex_opcode == 0x64) { for (int i = 0; i < 16; i++) regs_.xmm[reg][i] = ((int8_t)vvvv_b[i] > (int8_t)src[i]) ? 0xFF : 0; }
                else if (vex_opcode == 0x65) { for (int i = 0; i < 8; i++) { int16_t a, b; memcpy(&a, vvvv_b + i*2, 2); memcpy(&b, src + i*2, 2); uint16_t r = (a > b) ? 0xFFFF : 0; memcpy(regs_.xmm[reg] + i*2, &r, 2); } }
                else { for (int i = 0; i < 4; i++) { int32_t a, b; memcpy(&a, vvvv_b + i*4, 4); memcpy(&b, src + i*4, 4); uint32_t r = (a > b) ? 0xFFFFFFFF : 0; memcpy(regs_.xmm[reg] + i*4, &r, 4); } }
                break;
            }

            // VPAND (0xDB)
            case 0xDB: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t src[16], vvvv_b[16];
                memcpy(vvvv_b, regs_.xmm[vex_vvvv & 15], 16);
                if (rm_is_reg) memcpy(src, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
                for (int i = 0; i < 16; i++) regs_.xmm[reg][i] = vvvv_b[i] & src[i];
                break;
            }

            // VPANDN (0xDF)
            case 0xDF: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t src[16], vvvv_b[16];
                memcpy(vvvv_b, regs_.xmm[vex_vvvv & 15], 16);
                if (rm_is_reg) memcpy(src, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
                for (int i = 0; i < 16; i++) regs_.xmm[reg][i] = (~vvvv_b[i]) & src[i];
                break;
            }

            // VPOR (0xEB)
            case 0xEB: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t src[16], vvvv_b[16];
                memcpy(vvvv_b, regs_.xmm[vex_vvvv & 15], 16);
                if (rm_is_reg) memcpy(src, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
                for (int i = 0; i < 16; i++) regs_.xmm[reg][i] = vvvv_b[i] | src[i];
                break;
            }

            // VPADDD (0xFE), VPSUBD (0xFA), VPADDB (0xFC), VPADDW (0xFD)
            case 0xFC: case 0xFD: case 0xFE: case 0xFA: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t src[16], vvvv_b[16];
                memcpy(vvvv_b, regs_.xmm[vex_vvvv & 15], 16);
                if (rm_is_reg) memcpy(src, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
                if (vex_opcode == 0xFC) { for (int i = 0; i < 16; i++) regs_.xmm[reg][i] = vvvv_b[i] + src[i]; }
                else if (vex_opcode == 0xFD) { for (int i = 0; i < 8; i++) { uint16_t a, b; memcpy(&a, vvvv_b+i*2, 2); memcpy(&b, src+i*2, 2); a += b; memcpy(regs_.xmm[reg]+i*2, &a, 2); } }
                else if (vex_opcode == 0xFE) { for (int i = 0; i < 4; i++) { uint32_t a, b; memcpy(&a, vvvv_b+i*4, 4); memcpy(&b, src+i*4, 4); a += b; memcpy(regs_.xmm[reg]+i*4, &a, 4); } }
                else { for (int i = 0; i < 4; i++) { uint32_t a, b; memcpy(&a, vvvv_b+i*4, 4); memcpy(&b, src+i*4, 4); a -= b; memcpy(regs_.xmm[reg]+i*4, &a, 4); } }
                break;
            }

            // VPSUBB (0xF8), VPSUBW (0xF9)
            case 0xF8: case 0xF9: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t src[16], vvvv_b[16];
                memcpy(vvvv_b, regs_.xmm[vex_vvvv & 15], 16);
                if (rm_is_reg) memcpy(src, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
                if (vex_opcode == 0xF8) { for (int i = 0; i < 16; i++) regs_.xmm[reg][i] = vvvv_b[i] - src[i]; }
                else { for (int i = 0; i < 8; i++) { uint16_t a, b; memcpy(&a, vvvv_b+i*2, 2); memcpy(&b, src+i*2, 2); a -= b; memcpy(regs_.xmm[reg]+i*2, &a, 2); } }
                break;
            }

            // VCVTDQ2PS (0x5B, no prefix) / VCVTPS2DQ (0x5B, 66 prefix)
            case 0x5B: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t src_raw[16];
                if (rm_is_reg) memcpy(src_raw, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src_raw, mem_base_ + off, 16); else memset(src_raw, 0, 16); }
                if (vex_pp == 1) { // 66: VCVTPS2DQ
                    for (int i = 0; i < 4; i++) { float f; memcpy(&f, src_raw + i*4, 4); int32_t r = (int32_t)f; memcpy(regs_.xmm[reg] + i*4, &r, 4); }
                } else { // no prefix: VCVTDQ2PS
                    for (int i = 0; i < 4; i++) { int32_t d; memcpy(&d, src_raw + i*4, 4); float f = (float)d; memcpy(regs_.xmm[reg] + i*4, &f, 4); }
                }
                break;
            }

            // VCVTPS2PD (0x5A no prefix) / VCVTSS2SD (0x5A F3) / VCVTSD2SS (0x5A F2)
            case 0x5A: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                if (vex_pp == 2) { // F3: VCVTSS2SD
                    float f; if (rm_is_reg) memcpy(&f, regs_.xmm[(int)rm_addr & 15], 4); else { uint32_t v = ReadMemory32(rm_addr); memcpy(&f, &v, 4); }
                    double d = (double)f; memcpy(regs_.xmm[reg], regs_.xmm[vex_vvvv & 15], 16); memcpy(regs_.xmm[reg], &d, 8);
                } else if (vex_pp == 3) { // F2: VCVTSD2SS
                    double d; if (rm_is_reg) memcpy(&d, regs_.xmm[(int)rm_addr & 15], 8); else { uint64_t v = ReadMemory64(rm_addr); memcpy(&d, &v, 8); }
                    float f = (float)d; memcpy(regs_.xmm[reg], regs_.xmm[vex_vvvv & 15], 16); memcpy(regs_.xmm[reg], &f, 4);
                } else { // VCVTPS2PD
                    uint8_t src_raw[16]; if (rm_is_reg) memcpy(src_raw, regs_.xmm[(int)rm_addr & 15], 16);
                    else { uint64_t off = rm_addr - mem_base_addr_; if (off + 8 <= mem_size_) memcpy(src_raw, mem_base_ + off, 8); else memset(src_raw, 0, 8); }
                    for (int i = 0; i < 2; i++) { float f; memcpy(&f, src_raw + i*4, 4); double d = (double)f; memcpy(regs_.xmm[reg] + i*8, &d, 8); }
                }
                break;
            }

            // VUNPCKLPS (0x14), VUNPCKHPS (0x15)
            case 0x14: case 0x15: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint32_t src[4], vvvv_d[4];
                memcpy(vvvv_d, regs_.xmm[vex_vvvv & 15], 16);
                if (rm_is_reg) memcpy(src, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
                uint32_t dst[4];
                if (vex_opcode == 0x14) { dst[0]=vvvv_d[0]; dst[1]=src[0]; dst[2]=vvvv_d[1]; dst[3]=src[1]; }
                else                    { dst[0]=vvvv_d[2]; dst[1]=src[2]; dst[2]=vvvv_d[3]; dst[3]=src[3]; }
                memcpy(regs_.xmm[reg], dst, 16);
                break;
            }

            // VMOVHPS/VMOVLHPS (0x16), VMOVLPS/VMOVHLPS (0x12)
            case 0x16: case 0x12: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                memcpy(regs_.xmm[reg], regs_.xmm[vex_vvvv & 15], 16);
                if (vex_opcode == 0x16) {
                    if (rm_is_reg) memcpy(regs_.xmm[reg] + 8, regs_.xmm[(int)rm_addr & 15], 8); // VMOVLHPS
                    else { uint64_t off = rm_addr - mem_base_addr_; if (off + 8 <= mem_size_) memcpy(regs_.xmm[reg] + 8, mem_base_ + off, 8); } // VMOVHPS load
                } else {
                    if (rm_is_reg) memcpy(regs_.xmm[reg], regs_.xmm[(int)rm_addr & 15] + 8, 8); // VMOVHLPS
                    else { uint64_t off = rm_addr - mem_base_addr_; if (off + 8 <= mem_size_) memcpy(regs_.xmm[reg], mem_base_ + off, 8); } // VMOVLPS load
                }
                break;
            }

            // VMOVHPS store (0x17), VMOVLPS store (0x13)
            case 0x17: case 0x13: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                if (!rm_is_reg) {
                    uint64_t off = rm_addr - mem_base_addr_;
                    if (off + 8 <= mem_size_) {
                        if (vex_opcode == 0x17) memcpy(mem_base_ + off, regs_.xmm[reg] + 8, 8);
                        else memcpy(mem_base_ + off, regs_.xmm[reg], 8);
                    }
                }
                break;
            }

            // VPSRLD/VPSLLD/VPSRAD imm (0x72) — group 13
            case 0x72: {
                uint8_t modrm = read_modrm();
                int grp = reg_field(modrm) & 7; // /2=VPSRLD, /4=VPSRAD, /6=VPSLLD
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t imm = read_imm8();
                uint32_t data[4];
                if (rm_is_reg) memcpy(data, regs_.xmm[(int)rm_addr & 15], 16);
                else memcpy(data, regs_.xmm[vex_vvvv & 15], 16);
                int dest = vex_vvvv & 15;
                if (rm_is_reg) dest = (int)rm_addr & 15;
                memcpy(data, regs_.xmm[rm_is_reg ? ((int)rm_addr & 15) : (vex_vvvv & 15)], 16);
                for (int i = 0; i < 4; i++) {
                    if (grp == 2) data[i] = (imm < 32) ? data[i] >> imm : 0; // VPSRLD
                    else if (grp == 4) data[i] = (imm < 32) ? (uint32_t)((int32_t)data[i] >> imm) : (uint32_t)((int32_t)data[i] >> 31); // VPSRAD
                    else if (grp == 6) data[i] = (imm < 32) ? data[i] << imm : 0; // VPSLLD
                }
                // VEX encoding: dest is vvvv for imm forms
                memcpy(regs_.xmm[vex_vvvv & 15], data, 16);
                break;
            }

            // VPSRLW/VPSRAW/VPSLLW imm (0x71) — group 12
            case 0x71: {
                uint8_t modrm = read_modrm();
                int grp = reg_field(modrm) & 7;
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t imm = read_imm8();
                uint16_t data[8];
                memcpy(data, regs_.xmm[rm_is_reg ? ((int)rm_addr & 15) : (vex_vvvv & 15)], 16);
                for (int i = 0; i < 8; i++) {
                    if (grp == 2) data[i] = (imm < 16) ? data[i] >> imm : 0;
                    else if (grp == 4) data[i] = (imm < 16) ? (uint16_t)((int16_t)data[i] >> imm) : (uint16_t)((int16_t)data[i] >> 15);
                    else if (grp == 6) data[i] = (imm < 16) ? data[i] << imm : 0;
                }
                memcpy(regs_.xmm[vex_vvvv & 15], data, 16);
                break;
            }

            // VPSRLQ/VPSLLQ imm (0x73) — group 14
            case 0x73: {
                uint8_t modrm = read_modrm();
                int grp = reg_field(modrm) & 7;
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t imm = read_imm8();
                uint64_t data[2];
                memcpy(data, regs_.xmm[rm_is_reg ? ((int)rm_addr & 15) : (vex_vvvv & 15)], 16);
                for (int i = 0; i < 2; i++) {
                    if (grp == 2) data[i] = (imm < 64) ? data[i] >> imm : 0;
                    else if (grp == 6) data[i] = (imm < 64) ? data[i] << imm : 0;
                    else if (grp == 3) { // VPSRLDQ (byte shift right)
                        uint8_t tmp[16]; memcpy(tmp, data, 16);
                        memset(regs_.xmm[vex_vvvv & 15], 0, 16);
                        for (int j = 0; j + imm < 16; j++) regs_.xmm[vex_vvvv & 15][j] = tmp[j + imm];
                        return X86Exception::None;
                    } else if (grp == 7) { // VPSLLDQ (byte shift left)
                        uint8_t tmp[16]; memcpy(tmp, data, 16);
                        memset(regs_.xmm[vex_vvvv & 15], 0, 16);
                        for (int j = imm; j < 16; j++) regs_.xmm[vex_vvvv & 15][j] = tmp[j - imm];
                        return X86Exception::None;
                    }
                }
                memcpy(regs_.xmm[vex_vvvv & 15], data, 16);
                break;
            }

            // VPSLLD reg (0xF2), VPSRLD reg (0xD2), VPSRAD reg (0xE2)
            case 0xF2: case 0xD2: case 0xE2: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint64_t shift;
                if (rm_is_reg) memcpy(&shift, regs_.xmm[(int)rm_addr & 15], 8);
                else shift = ReadMemory64(rm_addr);
                uint32_t data[4]; memcpy(data, regs_.xmm[vex_vvvv & 15], 16);
                for (int i = 0; i < 4; i++) {
                    if (vex_opcode == 0xD2) data[i] = (shift < 32) ? data[i] >> shift : 0;
                    else if (vex_opcode == 0xE2) data[i] = (shift < 32) ? (uint32_t)((int32_t)data[i] >> shift) : (uint32_t)((int32_t)data[i] >> 31);
                    else data[i] = (shift < 32) ? data[i] << shift : 0;
                }
                memcpy(regs_.xmm[reg], data, 16);
                break;
            }

            // VCOMISS/VCOMISD (0x2F)
            case 0x2F: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                if (vex_pp == 1) { // 66: VCOMISD
                    double a, b; memcpy(&a, regs_.xmm[reg], 8);
                    if (rm_is_reg) memcpy(&b, regs_.xmm[(int)rm_addr & 15], 8); else { uint64_t v = ReadMemory64(rm_addr); memcpy(&b, &v, 8); }
                    SetZF(a == b); SetCF(a < b); SetPF(std::isunordered(a, b));
                } else { // VCOMISS
                    float a, b; memcpy(&a, regs_.xmm[reg], 4);
                    if (rm_is_reg) memcpy(&b, regs_.xmm[(int)rm_addr & 15], 4); else { uint32_t v = ReadMemory32(rm_addr); memcpy(&b, &v, 4); }
                    SetZF(a == b); SetCF(a < b); SetPF(std::isunordered(a, b));
                }
                SetOF(false); SetSF(false); SetAF(false);
                break;
            }

            // VRSQRTPS/SS (0x52), VRCPPS/SS (0x53)
            case 0x52: case 0x53: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                if (vex_pp == 2) { // F3: scalar
                    float f; if (rm_is_reg) memcpy(&f, regs_.xmm[(int)rm_addr & 15], 4); else { uint32_t v = ReadMemory32(rm_addr); memcpy(&f, &v, 4); }
                    memcpy(regs_.xmm[reg], regs_.xmm[vex_vvvv & 15], 16);
                    float r = (vex_opcode == 0x52) ? (1.0f / sqrtf(f)) : (1.0f / f);
                    memcpy(regs_.xmm[reg], &r, 4);
                } else { // packed
                    float src_f[4];
                    if (rm_is_reg) memcpy(src_f, regs_.xmm[(int)rm_addr & 15], 16);
                    else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src_f, mem_base_ + off, 16); else memset(src_f, 0, 16); }
                    for (int i = 0; i < 4; i++) src_f[i] = (vex_opcode == 0x52) ? (1.0f / sqrtf(src_f[i])) : (1.0f / src_f[i]);
                    memcpy(regs_.xmm[reg], src_f, 16);
                }
                break;
            }

            // VPUNPCKLDQ (0x62), VPUNPCKHDQ (0x6A)
            case 0x62: case 0x6A: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint32_t src[4], vvvv_d[4];
                memcpy(vvvv_d, regs_.xmm[vex_vvvv & 15], 16);
                if (rm_is_reg) memcpy(src, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
                uint32_t dst[4];
                if (vex_opcode == 0x62) { dst[0]=vvvv_d[0]; dst[1]=src[0]; dst[2]=vvvv_d[1]; dst[3]=src[1]; }
                else                    { dst[0]=vvvv_d[2]; dst[1]=src[2]; dst[2]=vvvv_d[3]; dst[3]=src[3]; }
                memcpy(regs_.xmm[reg], dst, 16);
                break;
            }

            // VPUNPCKHQDQ (0x6D)
            case 0x6D: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint64_t src_hi, vvvv_hi;
                memcpy(&vvvv_hi, regs_.xmm[vex_vvvv & 15] + 8, 8);
                if (rm_is_reg) memcpy(&src_hi, regs_.xmm[(int)rm_addr & 15] + 8, 8);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(&src_hi, mem_base_ + off + 8, 8); else src_hi = 0; }
                memcpy(regs_.xmm[reg], &vvvv_hi, 8);
                memcpy(regs_.xmm[reg] + 8, &src_hi, 8);
                break;
            }

            // VPUNPCKLBW (0x60), VPUNPCKHBW (0x68), VPUNPCKLWD (0x61), VPUNPCKHWD (0x69)
            case 0x60: case 0x61: case 0x68: case 0x69: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t src[16], vvvv_b2[16];
                memcpy(vvvv_b2, regs_.xmm[vex_vvvv & 15], 16);
                if (rm_is_reg) memcpy(src, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
                uint8_t dst[16];
                bool high = (vex_opcode == 0x68 || vex_opcode == 0x69);
                if (vex_opcode == 0x60 || vex_opcode == 0x68) { // byte interleave
                    int base2 = high ? 8 : 0;
                    for (int i = 0; i < 8; i++) { dst[i*2] = vvvv_b2[base2+i]; dst[i*2+1] = src[base2+i]; }
                } else { // word interleave
                    int base2 = high ? 4 : 0;
                    for (int i = 0; i < 4; i++) { memcpy(dst+i*4, vvvv_b2+(base2+i)*2, 2); memcpy(dst+i*4+2, src+(base2+i)*2, 2); }
                }
                memcpy(regs_.xmm[reg], dst, 16);
                break;
            }

            // VPMOVMSKB (0xD7)
            case 0xD7: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                int rm = rm_field(modrm);
                uint32_t mask = 0;
                for (int i = 0; i < 16; i++) if (regs_.xmm[rm][i] & 0x80) mask |= (1 << i);
                RegByIndex(reg) = mask;
                break;
            }

            // VPACKSSDW (0x6B), VPACKSSWB (0x63), VPACKUSWB (0x67), VPACKUSDW (requires 0F38)
            case 0x6B: case 0x63: case 0x67: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t src[16], vvvv_b2[16];
                memcpy(vvvv_b2, regs_.xmm[vex_vvvv & 15], 16);
                if (rm_is_reg) memcpy(src, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
                uint8_t dst[16];
                if (vex_opcode == 0x6B) { // PACKSSDW
                    for (int i = 0; i < 4; i++) { int32_t v; memcpy(&v, vvvv_b2+i*4, 4); int16_t r = v > 32767 ? 32767 : (v < -32768 ? -32768 : (int16_t)v); memcpy(dst+i*2, &r, 2); }
                    for (int i = 0; i < 4; i++) { int32_t v; memcpy(&v, src+i*4, 4); int16_t r = v > 32767 ? 32767 : (v < -32768 ? -32768 : (int16_t)v); memcpy(dst+8+i*2, &r, 2); }
                } else if (vex_opcode == 0x63) { // PACKSSWB
                    for (int i = 0; i < 8; i++) { int16_t v; memcpy(&v, vvvv_b2+i*2, 2); dst[i] = v > 127 ? 127 : (v < -128 ? -128 : (int8_t)v); }
                    for (int i = 0; i < 8; i++) { int16_t v; memcpy(&v, src+i*2, 2); dst[8+i] = v > 127 ? 127 : (v < -128 ? -128 : (int8_t)v); }
                } else { // PACKUSWB
                    for (int i = 0; i < 8; i++) { int16_t v; memcpy(&v, vvvv_b2+i*2, 2); dst[i] = v > 255 ? 255 : (v < 0 ? 0 : (uint8_t)v); }
                    for (int i = 0; i < 8; i++) { int16_t v; memcpy(&v, src+i*2, 2); dst[8+i] = v > 255 ? 255 : (v < 0 ? 0 : (uint8_t)v); }
                }
                memcpy(regs_.xmm[reg], dst, 16);
                break;
            }

            // VPMULLW (0xD5), VPMULHW (0xE5), VPMULHUW (0xE4)
            case 0xD5: case 0xE5: case 0xE4: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint16_t src_w[8], vvvv_w[8];
                memcpy(vvvv_w, regs_.xmm[vex_vvvv & 15], 16);
                if (rm_is_reg) memcpy(src_w, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src_w, mem_base_ + off, 16); else memset(src_w, 0, 16); }
                uint16_t dst_w[8];
                for (int i = 0; i < 8; i++) {
                    if (vex_opcode == 0xD5) dst_w[i] = (uint16_t)((uint32_t)vvvv_w[i] * (uint32_t)src_w[i]); // low
                    else if (vex_opcode == 0xE5) dst_w[i] = (uint16_t)(((int32_t)(int16_t)vvvv_w[i] * (int32_t)(int16_t)src_w[i]) >> 16); // signed high
                    else dst_w[i] = (uint16_t)(((uint32_t)vvvv_w[i] * (uint32_t)src_w[i]) >> 16); // unsigned high
                }
                memcpy(regs_.xmm[reg], dst_w, 16);
                break;
            }

            // VPMADDWD (0xF5)
            case 0xF5: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                int16_t src_w[8], vvvv_w[8];
                memcpy(vvvv_w, regs_.xmm[vex_vvvv & 15], 16);
                if (rm_is_reg) memcpy(src_w, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src_w, mem_base_ + off, 16); else memset(src_w, 0, 16); }
                int32_t dst_d[4];
                for (int i = 0; i < 4; i++) dst_d[i] = (int32_t)vvvv_w[i*2] * (int32_t)src_w[i*2] + (int32_t)vvvv_w[i*2+1] * (int32_t)src_w[i*2+1];
                memcpy(regs_.xmm[reg], dst_d, 16);
                break;
            }

            // VPMINSD (0x39→0F38), VPMAXSD (0x3D→0F38) handled in 0F38 map
            // VPMINUB (0xDA), VPMAXUB (0xDE), VPMINUW/VPMINSW (0xEA/0xEE→0F38)
            case 0xDA: case 0xDE: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t src[16], vvvv_b2[16];
                memcpy(vvvv_b2, regs_.xmm[vex_vvvv & 15], 16);
                if (rm_is_reg) memcpy(src, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
                for (int i = 0; i < 16; i++)
                    regs_.xmm[reg][i] = (vex_opcode == 0xDA) ? std::min(vvvv_b2[i], src[i]) : std::max(vvvv_b2[i], src[i]);
                break;
            }

            // VPSLLW reg (0xF1), VPSRLW reg (0xD1), VPSRAW reg (0xE1)
            case 0xF1: case 0xD1: case 0xE1: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint64_t shift;
                if (rm_is_reg) memcpy(&shift, regs_.xmm[(int)rm_addr & 15], 8);
                else shift = ReadMemory64(rm_addr);
                uint16_t data[8]; memcpy(data, regs_.xmm[vex_vvvv & 15], 16);
                for (int i = 0; i < 8; i++) {
                    if (vex_opcode == 0xD1) data[i] = (shift < 16) ? data[i] >> shift : 0;
                    else if (vex_opcode == 0xE1) data[i] = (shift < 16) ? (uint16_t)((int16_t)data[i] >> shift) : (uint16_t)((int16_t)data[i] >> 15);
                    else data[i] = (shift < 16) ? data[i] << shift : 0;
                }
                memcpy(regs_.xmm[reg], data, 16);
                break;
            }

            // VPSLLQ reg (0xF3)
            case 0xF3: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint64_t shift;
                if (rm_is_reg) memcpy(&shift, regs_.xmm[(int)rm_addr & 15], 8);
                else shift = ReadMemory64(rm_addr);
                uint64_t data[2]; memcpy(data, regs_.xmm[vex_vvvv & 15], 16);
                if (shift < 64) { data[0] <<= shift; data[1] <<= shift; }
                else { data[0] = 0; data[1] = 0; }
                memcpy(regs_.xmm[reg], data, 16);
                break;
            }

            // VMOVNTDQ (0xE7) — non-temporal store (treat as regular store)
            case 0xE7: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                if (!rm_is_reg) {
                    uint64_t off = rm_addr - mem_base_addr_;
                    if (off + 16 <= mem_size_) memcpy(mem_base_ + off, regs_.xmm[reg], 16);
                }
                break;
            }

            // NOP (multi-byte) — VEX-encoded NOPs
            case 0x1F: {
                uint8_t modrm = read_modrm();
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                break;
            }

            default: {
                static int vex_unk_count = 0;
                if (vex_unk_count < 20)
                    fprintf(stderr, "[x86] Unimpl VEX 0F opcode: pp=%d 0x%02X at RIP=0x%llX\n",
                            vex_pp, vex_opcode, (unsigned long long)(regs_.rip - 1));
                vex_unk_count++;
                // Skip modrm/sib/disp/imm bytes so RIP lands on next instruction
                if (vex_opcode < 0xC0) {
                    uint8_t modrm = read_modrm();
                    uint64_t rm_addr; bool rm_is_reg;
                    decode_rm(modrm, &rm_addr, &rm_is_reg);
                }
                // Check if instruction has an immediate byte (common for VEX 0F)
                if (vex_opcode == 0x70 || vex_opcode == 0x71 || vex_opcode == 0x72 ||
                    vex_opcode == 0x73 || vex_opcode == 0xC2 || vex_opcode == 0xC4 ||
                    vex_opcode == 0xC5 || vex_opcode == 0xC6)
                    regs_.rip++; // skip imm8
                return X86Exception::None; // skip gracefully
            }
            }
        } else if (vex_mmmmm == 2) {
            // 0F 38 opcode map
            switch (vex_opcode) {
            // VBROADCASTSS
            case 0x18: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint32_t val;
                if (rm_is_reg) memcpy(&val, regs_.xmm[(int)rm_addr & 15], 4);
                else val = ReadMemory32(rm_addr);
                for (int i = 0; i < 4; i++) memcpy(regs_.xmm[reg] + i * 4, &val, 4);
                break;
            }
            // VBROADCASTSD
            case 0x19: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint64_t val;
                if (rm_is_reg) memcpy(&val, regs_.xmm[(int)rm_addr & 15], 8);
                else val = ReadMemory64(rm_addr);
                memcpy(regs_.xmm[reg], &val, 8);
                memcpy(regs_.xmm[reg] + 8, &val, 8);
                break;
            }
            // VPBROADCASTD / VPBROADCASTQ
            case 0x58: case 0x59: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                if (vex_opcode == 0x58) {
                    uint32_t val;
                    if (rm_is_reg) memcpy(&val, regs_.xmm[(int)rm_addr & 15], 4);
                    else val = ReadMemory32(rm_addr);
                    for (int i = 0; i < 4; i++) memcpy(regs_.xmm[reg] + i * 4, &val, 4);
                } else {
                    uint64_t val;
                    if (rm_is_reg) memcpy(&val, regs_.xmm[(int)rm_addr & 15], 8);
                    else val = ReadMemory64(rm_addr);
                    memcpy(regs_.xmm[reg], &val, 8);
                    memcpy(regs_.xmm[reg] + 8, &val, 8);
                }
                break;
            }

            // VPSHUFB (0x00)
            case 0x00: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t src[16], vvvv_b2[16];
                memcpy(vvvv_b2, regs_.xmm[vex_vvvv & 15], 16);
                if (rm_is_reg) memcpy(src, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
                uint8_t dst[16];
                for (int i = 0; i < 16; i++) dst[i] = (src[i] & 0x80) ? 0 : vvvv_b2[src[i] & 15];
                memcpy(regs_.xmm[reg], dst, 16);
                break;
            }

            // VPMULLD (0x40)
            case 0x40: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint32_t src[4], vvvv_d[4];
                memcpy(vvvv_d, regs_.xmm[vex_vvvv & 15], 16);
                if (rm_is_reg) memcpy(src, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
                for (int i = 0; i < 4; i++) vvvv_d[i] *= src[i];
                memcpy(regs_.xmm[reg], vvvv_d, 16);
                break;
            }

            // VPCMPEQQ (0x29)
            case 0x29: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint64_t src[2], vvvv_q[2];
                memcpy(vvvv_q, regs_.xmm[vex_vvvv & 15], 16);
                if (rm_is_reg) memcpy(src, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
                for (int i = 0; i < 2; i++) vvvv_q[i] = (vvvv_q[i] == src[i]) ? ~0ULL : 0;
                memcpy(regs_.xmm[reg], vvvv_q, 16);
                break;
            }

            // VPMINSB/D (0x38/0x39), VPMAXSB/D (0x3C/0x3D), VPMINUW/UD (0x3A/0x3B), VPMAXUW/UD (0x3E/0x3F)
            case 0x38: case 0x39: case 0x3A: case 0x3B: case 0x3C: case 0x3D: case 0x3E: case 0x3F: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t src[16], vvvv_b2[16];
                memcpy(vvvv_b2, regs_.xmm[vex_vvvv & 15], 16);
                if (rm_is_reg) memcpy(src, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
                if (vex_opcode == 0x38) { for (int i = 0; i < 16; i++) regs_.xmm[reg][i] = ((int8_t)vvvv_b2[i] < (int8_t)src[i]) ? vvvv_b2[i] : src[i]; }
                else if (vex_opcode == 0x39) { for (int i = 0; i < 4; i++) { int32_t a,b; memcpy(&a, vvvv_b2+i*4, 4); memcpy(&b, src+i*4, 4); int32_t r = a < b ? a : b; memcpy(regs_.xmm[reg]+i*4, &r, 4); } }
                else if (vex_opcode == 0x3A) { for (int i = 0; i < 8; i++) { uint16_t a,b; memcpy(&a, vvvv_b2+i*2, 2); memcpy(&b, src+i*2, 2); uint16_t r = a < b ? a : b; memcpy(regs_.xmm[reg]+i*2, &r, 2); } }
                else if (vex_opcode == 0x3B) { for (int i = 0; i < 4; i++) { uint32_t a,b; memcpy(&a, vvvv_b2+i*4, 4); memcpy(&b, src+i*4, 4); uint32_t r = a < b ? a : b; memcpy(regs_.xmm[reg]+i*4, &r, 4); } }
                else if (vex_opcode == 0x3C) { for (int i = 0; i < 16; i++) regs_.xmm[reg][i] = ((int8_t)vvvv_b2[i] > (int8_t)src[i]) ? vvvv_b2[i] : src[i]; }
                else if (vex_opcode == 0x3D) { for (int i = 0; i < 4; i++) { int32_t a,b; memcpy(&a, vvvv_b2+i*4, 4); memcpy(&b, src+i*4, 4); int32_t r = a > b ? a : b; memcpy(regs_.xmm[reg]+i*4, &r, 4); } }
                else if (vex_opcode == 0x3E) { for (int i = 0; i < 8; i++) { uint16_t a,b; memcpy(&a, vvvv_b2+i*2, 2); memcpy(&b, src+i*2, 2); uint16_t r = a > b ? a : b; memcpy(regs_.xmm[reg]+i*2, &r, 2); } }
                else { for (int i = 0; i < 4; i++) { uint32_t a,b; memcpy(&a, vvvv_b2+i*4, 4); memcpy(&b, src+i*4, 4); uint32_t r = a > b ? a : b; memcpy(regs_.xmm[reg]+i*4, &r, 4); } }
                break;
            }

            // VPMOVZXBW (0x30), VPMOVZXBD (0x31), VPMOVZXBQ (0x32)
            // VPMOVZXWD (0x33), VPMOVZXWQ (0x34), VPMOVZXDQ (0x35)
            case 0x30: case 0x31: case 0x32: case 0x33: case 0x34: case 0x35: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t src[16];
                if (rm_is_reg) memcpy(src, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
                memset(regs_.xmm[reg], 0, 16);
                if (vex_opcode == 0x30) { for (int i = 0; i < 8; i++) { uint16_t v = src[i]; memcpy(regs_.xmm[reg]+i*2, &v, 2); } }
                else if (vex_opcode == 0x31) { for (int i = 0; i < 4; i++) { uint32_t v = src[i]; memcpy(regs_.xmm[reg]+i*4, &v, 4); } }
                else if (vex_opcode == 0x32) { for (int i = 0; i < 2; i++) { uint64_t v = src[i]; memcpy(regs_.xmm[reg]+i*8, &v, 8); } }
                else if (vex_opcode == 0x33) { for (int i = 0; i < 4; i++) { uint16_t s; memcpy(&s, src+i*2, 2); uint32_t v = s; memcpy(regs_.xmm[reg]+i*4, &v, 4); } }
                else if (vex_opcode == 0x34) { for (int i = 0; i < 2; i++) { uint16_t s; memcpy(&s, src+i*2, 2); uint64_t v = s; memcpy(regs_.xmm[reg]+i*8, &v, 8); } }
                else { for (int i = 0; i < 2; i++) { uint32_t s; memcpy(&s, src+i*4, 4); uint64_t v = s; memcpy(regs_.xmm[reg]+i*8, &v, 8); } }
                break;
            }

            // VPMOVSXBW (0x20), VPMOVSXBD (0x21), VPMOVSXBQ (0x22)
            // VPMOVSXWD (0x23), VPMOVSXWQ (0x24), VPMOVSXDQ (0x25)
            case 0x20: case 0x21: case 0x22: case 0x23: case 0x24: case 0x25: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t src[16];
                if (rm_is_reg) memcpy(src, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
                memset(regs_.xmm[reg], 0, 16);
                if (vex_opcode == 0x20) { for (int i = 0; i < 8; i++) { int16_t v = (int8_t)src[i]; memcpy(regs_.xmm[reg]+i*2, &v, 2); } }
                else if (vex_opcode == 0x21) { for (int i = 0; i < 4; i++) { int32_t v = (int8_t)src[i]; memcpy(regs_.xmm[reg]+i*4, &v, 4); } }
                else if (vex_opcode == 0x22) { for (int i = 0; i < 2; i++) { int64_t v = (int8_t)src[i]; memcpy(regs_.xmm[reg]+i*8, &v, 8); } }
                else if (vex_opcode == 0x23) { for (int i = 0; i < 4; i++) { int16_t s; memcpy(&s, src+i*2, 2); int32_t v = s; memcpy(regs_.xmm[reg]+i*4, &v, 4); } }
                else if (vex_opcode == 0x24) { for (int i = 0; i < 2; i++) { int16_t s; memcpy(&s, src+i*2, 2); int64_t v = s; memcpy(regs_.xmm[reg]+i*8, &v, 8); } }
                else { for (int i = 0; i < 2; i++) { int32_t s; memcpy(&s, src+i*4, 4); int64_t v = s; memcpy(regs_.xmm[reg]+i*8, &v, 8); } }
                break;
            }

            // VTESTPS (0x0E), VTESTPD (0x0F) — set ZF/CF from AND/ANDN
            case 0x0E: case 0x0F: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t src[16], dst[16];
                memcpy(dst, regs_.xmm[reg], 16);
                if (rm_is_reg) memcpy(src, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
                bool all_and_zero = true, all_andn_zero = true;
                for (int i = 0; i < 16; i++) { if (dst[i] & src[i]) all_and_zero = false; if ((~dst[i]) & src[i]) all_andn_zero = false; }
                SetZF(all_and_zero); SetCF(all_andn_zero);
                SetOF(false); SetSF(false); SetAF(false); SetPF(false);
                break;
            }

            // VPTEST (0x17)
            case 0x17: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t src[16], dst[16];
                memcpy(dst, regs_.xmm[reg], 16);
                if (rm_is_reg) memcpy(src, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
                bool all_and_zero = true, all_andn_zero = true;
                for (int i = 0; i < 16; i++) { if (dst[i] & src[i]) all_and_zero = false; if ((~dst[i]) & src[i]) all_andn_zero = false; }
                SetZF(all_and_zero); SetCF(all_andn_zero);
                SetOF(false); SetSF(false); SetAF(false); SetPF(false);
                break;
            }

            default: {
                static int vex38_unk = 0;
                if (vex38_unk < 10)
                    fprintf(stderr, "[x86] Unimpl VEX 0F38 opcode: 0x%02X at RIP=0x%llX\n",
                            vex_opcode, (unsigned long long)(regs_.rip - 1));
                vex38_unk++;
                if (vex_opcode < 0xC0) {
                    uint8_t modrm = read_modrm();
                    uint64_t rm_addr; bool rm_is_reg;
                    decode_rm(modrm, &rm_addr, &rm_is_reg);
                }
                return X86Exception::None; // skip gracefully
            }
            }
        } else if (vex_mmmmm == 3) {
            // 0F 3A opcode map
            switch (vex_opcode) {
            // VPERMILPS
            case 0x04: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t imm = read_imm8();
                float src_f[4], dst_f[4];
                if (rm_is_reg) memcpy(src_f, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src_f, mem_base_ + off, 16); else memset(src_f, 0, 16); }
                dst_f[0] = src_f[(imm >> 0) & 3];
                dst_f[1] = src_f[(imm >> 2) & 3];
                dst_f[2] = src_f[(imm >> 4) & 3];
                dst_f[3] = src_f[(imm >> 6) & 3];
                memcpy(regs_.xmm[reg], dst_f, 16);
                break;
            }
            // VPERMILPD
            case 0x05: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t imm = read_imm8();
                double src_d[2], dst_d[2];
                if (rm_is_reg) memcpy(src_d, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src_d, mem_base_ + off, 16); else memset(src_d, 0, 16); }
                dst_d[0] = src_d[(imm >> 0) & 1];
                dst_d[1] = src_d[(imm >> 1) & 1];
                memcpy(regs_.xmm[reg], dst_d, 16);
                break;
            }
            // VINSERTPS
            case 0x21: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t imm = read_imm8();
                float dst_f[4];
                memcpy(dst_f, regs_.xmm[vex_vvvv & 15], 16);
                float src_val;
                if (rm_is_reg) {
                    float src_f[4];
                    memcpy(src_f, regs_.xmm[(int)rm_addr & 15], 16);
                    src_val = src_f[(imm >> 6) & 3];
                } else {
                    uint32_t tmp = ReadMemory32(rm_addr);
                    memcpy(&src_val, &tmp, 4);
                }
                dst_f[(imm >> 4) & 3] = src_val;
                for (int i = 0; i < 4; i++) if (imm & (1 << i)) dst_f[i] = 0.0f;
                memcpy(regs_.xmm[reg], dst_f, 16);
                break;
            }

            // VBLENDPS (0x0C)
            case 0x0C: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t imm = read_imm8();
                uint32_t src[4], vvvv_d[4];
                memcpy(vvvv_d, regs_.xmm[vex_vvvv & 15], 16);
                if (rm_is_reg) memcpy(src, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
                for (int i = 0; i < 4; i++)
                    memcpy(regs_.xmm[reg] + i*4, (imm & (1 << i)) ? (uint8_t*)&src[i] : (uint8_t*)&vvvv_d[i], 4);
                break;
            }

            // VBLENDVPS (0x4A)
            case 0x4A: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t imm = read_imm8();
                int mask_reg = (imm >> 4) & 15;
                uint32_t src[4], vvvv_d[4], mask[4];
                memcpy(vvvv_d, regs_.xmm[vex_vvvv & 15], 16);
                memcpy(mask, regs_.xmm[mask_reg], 16);
                if (rm_is_reg) memcpy(src, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
                for (int i = 0; i < 4; i++)
                    memcpy(regs_.xmm[reg] + i*4, (mask[i] & 0x80000000) ? (uint8_t*)&src[i] : (uint8_t*)&vvvv_d[i], 4);
                break;
            }

            // VBLENDVPD (0x4B)
            case 0x4B: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t imm = read_imm8();
                int mask_reg = (imm >> 4) & 15;
                uint64_t src[2], vvvv_q[2], mask[2];
                memcpy(vvvv_q, regs_.xmm[vex_vvvv & 15], 16);
                memcpy(mask, regs_.xmm[mask_reg], 16);
                if (rm_is_reg) memcpy(src, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
                for (int i = 0; i < 2; i++)
                    memcpy(regs_.xmm[reg] + i*8, (mask[i] >> 63) ? (uint8_t*)&src[i] : (uint8_t*)&vvvv_q[i], 8);
                break;
            }

            // VPBLENDVB (0x4C)
            case 0x4C: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t imm = read_imm8();
                int mask_reg = (imm >> 4) & 15;
                uint8_t src[16], vvvv_b2[16];
                memcpy(vvvv_b2, regs_.xmm[vex_vvvv & 15], 16);
                if (rm_is_reg) memcpy(src, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
                for (int i = 0; i < 16; i++)
                    regs_.xmm[reg][i] = (regs_.xmm[mask_reg][i] & 0x80) ? src[i] : vvvv_b2[i];
                break;
            }

            // VPEXTRD/Q (0x16)
            case 0x16: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t imm = read_imm8();
                if (rex_w) { // VPEXTRQ
                    uint64_t val; memcpy(&val, regs_.xmm[reg] + (imm & 1) * 8, 8);
                    if (rm_is_reg) RegByIndex((int)rm_addr) = val;
                    else WriteMemory64(rm_addr, val);
                } else { // VPEXTRD
                    uint32_t val; memcpy(&val, regs_.xmm[reg] + (imm & 3) * 4, 4);
                    if (rm_is_reg) RegByIndex((int)rm_addr) = val;
                    else WriteMemory32(rm_addr, val);
                }
                break;
            }

            // VPEXTRB (0x14)
            case 0x14: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t imm = read_imm8();
                uint8_t val = regs_.xmm[reg][imm & 15];
                if (rm_is_reg) RegByIndex((int)rm_addr) = val;
                else WriteMemory8(rm_addr, val);
                break;
            }

            // VPINSRD/Q (0x22)
            case 0x22: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t imm = read_imm8();
                memcpy(regs_.xmm[reg], regs_.xmm[vex_vvvv & 15], 16);
                if (rex_w) { // VPINSRQ
                    uint64_t val = rm_is_reg ? RegByIndex((int)rm_addr) : ReadMemory64(rm_addr);
                    memcpy(regs_.xmm[reg] + (imm & 1) * 8, &val, 8);
                } else { // VPINSRD
                    uint32_t val = rm_is_reg ? (uint32_t)RegByIndex((int)rm_addr) : ReadMemory32(rm_addr);
                    memcpy(regs_.xmm[reg] + (imm & 3) * 4, &val, 4);
                }
                break;
            }

            // VPINSRB (0x20)
            case 0x20: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t imm = read_imm8();
                memcpy(regs_.xmm[reg], regs_.xmm[vex_vvvv & 15], 16);
                uint8_t val = rm_is_reg ? (uint8_t)RegByIndex((int)rm_addr) : ReadMemory8(rm_addr);
                regs_.xmm[reg][imm & 15] = val;
                break;
            }

            // VEXTRACTPS (0x17)
            case 0x17: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t imm = read_imm8();
                uint32_t val; memcpy(&val, regs_.xmm[reg] + (imm & 3) * 4, 4);
                if (rm_is_reg) RegByIndex((int)rm_addr) = val;
                else WriteMemory32(rm_addr, val);
                break;
            }

            // VROUNDPS (0x08), VROUNDPD (0x09), VROUNDSS (0x0A), VROUNDSD (0x0B)
            case 0x08: case 0x09: case 0x0A: case 0x0B: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t imm = read_imm8();
                int mode = imm & 3;
                if (vex_opcode == 0x08) { // VROUNDPS
                    float src_f[4];
                    if (rm_is_reg) memcpy(src_f, regs_.xmm[(int)rm_addr & 15], 16);
                    else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src_f, mem_base_ + off, 16); else memset(src_f, 0, 16); }
                    for (int i = 0; i < 4; i++) {
                        if (mode == 0) src_f[i] = roundf(src_f[i]); // nearest
                        else if (mode == 1) src_f[i] = floorf(src_f[i]);
                        else if (mode == 2) src_f[i] = ceilf(src_f[i]);
                        else src_f[i] = truncf(src_f[i]);
                    }
                    memcpy(regs_.xmm[reg], src_f, 16);
                } else if (vex_opcode == 0x0A) { // VROUNDSS
                    float f;
                    if (rm_is_reg) memcpy(&f, regs_.xmm[(int)rm_addr & 15], 4); else { uint32_t v = ReadMemory32(rm_addr); memcpy(&f, &v, 4); }
                    if (mode == 0) f = roundf(f); else if (mode == 1) f = floorf(f); else if (mode == 2) f = ceilf(f); else f = truncf(f);
                    memcpy(regs_.xmm[reg], regs_.xmm[vex_vvvv & 15], 16);
                    memcpy(regs_.xmm[reg], &f, 4);
                } else if (vex_opcode == 0x09) { // VROUNDPD
                    double src_d[2];
                    if (rm_is_reg) memcpy(src_d, regs_.xmm[(int)rm_addr & 15], 16);
                    else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src_d, mem_base_ + off, 16); else memset(src_d, 0, 16); }
                    for (int i = 0; i < 2; i++) {
                        if (mode == 0) src_d[i] = round(src_d[i]); else if (mode == 1) src_d[i] = floor(src_d[i]);
                        else if (mode == 2) src_d[i] = ceil(src_d[i]); else src_d[i] = trunc(src_d[i]);
                    }
                    memcpy(regs_.xmm[reg], src_d, 16);
                } else { // VROUNDSD
                    double d;
                    if (rm_is_reg) memcpy(&d, regs_.xmm[(int)rm_addr & 15], 8); else { uint64_t v = ReadMemory64(rm_addr); memcpy(&d, &v, 8); }
                    if (mode == 0) d = round(d); else if (mode == 1) d = floor(d); else if (mode == 2) d = ceil(d); else d = trunc(d);
                    memcpy(regs_.xmm[reg], regs_.xmm[vex_vvvv & 15], 16);
                    memcpy(regs_.xmm[reg], &d, 8);
                }
                break;
            }

            // VDPPS (0x40)
            case 0x40: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t imm = read_imm8();
                float src_f[4], vvvv_f[4];
                memcpy(vvvv_f, regs_.xmm[vex_vvvv & 15], 16);
                if (rm_is_reg) memcpy(src_f, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src_f, mem_base_ + off, 16); else memset(src_f, 0, 16); }
                float dp = 0;
                for (int i = 0; i < 4; i++) if (imm & (1 << (i + 4))) dp += vvvv_f[i] * src_f[i];
                float dst_f[4] = {0, 0, 0, 0};
                for (int i = 0; i < 4; i++) if (imm & (1 << i)) dst_f[i] = dp;
                memcpy(regs_.xmm[reg], dst_f, 16);
                break;
            }

            // VEXTRACTI128 (0x39) — extract high 128 bits (for 256-bit, we only have low 128)
            case 0x39: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t imm = read_imm8();
                // We only support 128-bit lanes; imm=0 returns low, imm=1 returns zeros (no high lane)
                if (imm & 1) {
                    if (rm_is_reg) memset(regs_.xmm[(int)rm_addr & 15], 0, 16);
                    else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memset(mem_base_ + off, 0, 16); }
                } else {
                    if (rm_is_reg) memcpy(regs_.xmm[(int)rm_addr & 15], regs_.xmm[reg], 16);
                    else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(mem_base_ + off, regs_.xmm[reg], 16); }
                }
                break;
            }

            // VPBLENDW (0x0E)
            case 0x0E: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t imm = read_imm8();
                uint16_t src_w[8], vvvv_w[8];
                memcpy(vvvv_w, regs_.xmm[vex_vvvv & 15], 16);
                if (rm_is_reg) memcpy(src_w, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src_w, mem_base_ + off, 16); else memset(src_w, 0, 16); }
                uint16_t dst_w[8];
                for (int i = 0; i < 8; i++) dst_w[i] = (imm & (1 << i)) ? src_w[i] : vvvv_w[i];
                memcpy(regs_.xmm[reg], dst_w, 16);
                break;
            }

            // VPALIGNR (0x0F)
            case 0x0F: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t imm = read_imm8();
                uint8_t src[16], vvvv_b2[16];
                memcpy(vvvv_b2, regs_.xmm[vex_vvvv & 15], 16);
                if (rm_is_reg) memcpy(src, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (off + 16 <= mem_size_) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
                uint8_t concat[32];
                memcpy(concat, src, 16);
                memcpy(concat + 16, vvvv_b2, 16);
                if (imm >= 32) memset(regs_.xmm[reg], 0, 16);
                else { for (int i = 0; i < 16; i++) regs_.xmm[reg][i] = (imm + i < 32) ? concat[imm + i] : 0; }
                break;
            }

            default: {
                static int vex3a_unk = 0;
                if (vex3a_unk < 10)
                    fprintf(stderr, "[x86] Unimpl VEX 0F3A opcode: 0x%02X at RIP=0x%llX\n",
                            vex_opcode, (unsigned long long)(regs_.rip - 1));
                vex3a_unk++;
                // All 0F3A opcodes have modrm + imm8
                {
                    uint8_t modrm = read_modrm();
                    uint64_t rm_addr; bool rm_is_reg;
                    decode_rm(modrm, &rm_addr, &rm_is_reg);
                    read_imm8();
                }
                return X86Exception::None; // skip gracefully
            }
            }
        } else {
            fprintf(stderr, "[x86] Unknown VEX map: %d at RIP=0x%llX\n",
                    vex_mmmmm, (unsigned long long)(regs_.rip - 1));
        }
        (void)vec_len;
        break;
    }

    default:
        fprintf(stderr, "[x86] Unimplemented opcode: 0x%02X at RIP=0x%llX\n", opcode,
                (unsigned long long)(regs_.rip - 1));
        return X86Exception::InvalidOpcode;
    }

    return X86Exception::None;
}

} // namespace KytyWeb
