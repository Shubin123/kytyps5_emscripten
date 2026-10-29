#include "x86_interpreter.h"
#include <cstring>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <limits>

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

// Sign-extends the low from_bits of val (anything above them is ignored).
static inline uint64_t sign_extend(uint64_t val, int from_bits) {
    if (from_bits >= 64) return val;
    const uint64_t mask = 1ULL << (from_bits - 1);
    val &= (mask << 1) - 1;
    return (val ^ mask) - mask;
}

static inline uint64_t truncate(uint64_t val, int bits) {
    if (bits >= 64) return val;
    return val & ((1ULL << bits) - 1);
}

// x86 float->int conversions return the "integer indefinite" value for NaN/out-of-range inputs
// (a C++ cast is undefined there, and traps or saturates in WebAssembly).
static int32_t CvtToI32(double v, bool trunc) {
    v = trunc ? std::trunc(v) : std::nearbyint(v);
    return (v >= -2147483648.0 && v <= 2147483647.0) ? (int32_t)v : INT32_MIN;
}
static int64_t CvtToI64(double v, bool trunc) {
    v = trunc ? std::trunc(v) : std::nearbyint(v);
    return (v >= -9223372036854775808.0 && v < 9223372036854775808.0) ? (int64_t)v : INT64_MIN;
}

static float HalfToFloat(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16, exp = (h >> 10) & 0x1F, mant = h & 0x3FF;
    if (exp == 0) {
        float f = (float)mant * (1.0f / 16777216.0f);
        return sign ? -f : f;
    }
    // Infinity keeps its zero mantissa; a NaN comes out quiet, as VCVTPH2PS converts it.
    uint32_t bits = exp == 31 ? (sign | 0x7F800000 | (mant << 13) | (mant ? 0x00400000 : 0))
                              : (sign | ((exp + 112) << 23) | (mant << 13));
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static uint16_t FloatToHalf(float f) {  // round to nearest even
    uint32_t x;
    memcpy(&x, &f, 4);
    uint16_t sign = (x >> 16) & 0x8000;
    uint32_t a = x & 0x7FFFFFFF;
    if (a >= 0x7F800000) return sign | 0x7C00 | (a > 0x7F800000 ? 0x200 : 0);
    if (a >= 0x477FF000) return sign | 0x7C00;
    if (a < 0x38800000) {
        float af;
        memcpy(&af, &a, 4);
        return sign | (uint16_t)std::nearbyint(af * 16777216.0f);
    }
    uint32_t h = (((a >> 23) - 112) << 10) | ((a & 0x7FFFFF) >> 13), rem = a & 0x1FFF;
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) h++;
    return sign | (uint16_t)h;
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

// The address must hold a RET (0xC3): Step() only looks handlers up on RET instructions.
void X86Interpreter::RegisterHLEFunction(uint64_t address, HLEHandler handler) {
    hle_handlers_[address] = std::move(handler);
}

void X86Interpreter::RegisterSyscallHandler(HLEHandler handler) {
    syscall_handler_ = std::move(handler);
}

// Out-of-range accesses read as 0 and drop writes, but are recorded so Step() reports a page fault
// (on hardware they would crash; silently continuing hides the real failure).
bool X86Interpreter::Mapped(uint64_t addr, uint64_t n) const {
    if (addr >= mem_base_addr_ && n <= mem_size_ && addr - mem_base_addr_ <= mem_size_ - n) return true;
    if (!fault_) { fault_ = true; fault_addr_ = addr; }
    return false;
}

template <class T> static T LoadLE(const uint8_t* p) { T v; memcpy(&v, p, sizeof(T)); return v; }

// --- AES-NI: one round of AES on a 16-byte column-major state (byte r + 4c is row r, column c) ---
namespace {
struct AesTables {
    uint8_t sbox[256], inv[256];
    AesTables() {
        auto mul = [](uint8_t a, uint8_t b) {
            uint8_t p = 0;
            for (int i = 0; i < 8; i++) {
                if (b & 1) p ^= a;
                const bool hi = a & 0x80;
                a <<= 1;
                if (hi) a ^= 0x1B;
                b >>= 1;
            }
            return p;
        };
        for (int x = 0; x < 256; x++) {
            uint8_t invx = 0;
            for (int y = 1; y < 256 && x; y++) if (mul((uint8_t)x, (uint8_t)y) == 1) { invx = (uint8_t)y; break; }
            uint8_t b = invx, r = invx;
            for (int i = 0; i < 4; i++) { b = (uint8_t)((b << 1) | (b >> 7)); r ^= b; }
            sbox[x] = r ^ 0x63;
        }
        for (int x = 0; x < 256; x++) inv[sbox[x]] = (uint8_t)x;
    }
};
const AesTables& Aes() { static const AesTables t; return t; }
uint8_t GfMul(uint8_t a, uint8_t b) {
    uint8_t p = 0;
    for (; b; b >>= 1) {
        if (b & 1) p ^= a;
        a = (uint8_t)((a << 1) ^ ((a & 0x80) ? 0x1B : 0));
    }
    return p;
}
void AesMixColumns(uint8_t* s, bool inverse) {
    static const uint8_t fwd[4] = {2, 3, 1, 1}, bwd[4] = {14, 11, 13, 9};
    const uint8_t* m = inverse ? bwd : fwd;
    for (int c = 0; c < 4; c++) {
        uint8_t col[4], out[4];
        memcpy(col, s + 4 * c, 4);
        for (int r = 0; r < 4; r++) {
            out[r] = 0;
            for (int k = 0; k < 4; k++) out[r] ^= GfMul(col[k], m[(k - r + 4) % 4]);
        }
        memcpy(s + 4 * c, out, 4);
    }
}
// op: 0xDC AESENC, 0xDD AESENCLAST, 0xDE AESDEC, 0xDF AESDECLAST.
void AesRound(uint8_t op, const uint8_t* state, const uint8_t* key, uint8_t* out) {
    const bool dec = op >= 0xDE, last = op == 0xDD || op == 0xDF;
    uint8_t t[16];
    for (int c = 0; c < 4; c++)
        for (int r = 0; r < 4; r++) {
            const int from = dec ? (c - r + 4) % 4 : (c + r) % 4;  // (inverse) ShiftRows
            const uint8_t v = state[r + 4 * from];
            t[r + 4 * c] = dec ? Aes().inv[v] : Aes().sbox[v];
        }
    if (!last) AesMixColumns(t, dec);
    for (int i = 0; i < 16; i++) out[i] = t[i] ^ key[i];
}
}  // namespace

// --- SSE/AVX floating point with x86 results ---
// NaNs: an operand NaN propagates quieted (the first source wins); invalid operations produce the negative
// "default NaN". Hosts (wasm in particular) don't guarantee either, so NaN cases are decided here.
namespace {
template <class T> struct FpInfo;
template <> struct FpInfo<float> {
    using Bits = uint32_t;
    static constexpr Bits kQuiet = 0x00400000u, kDefaultNaN = 0xFFC00000u;
    static constexpr int kSize = 4;
};
template <> struct FpInfo<double> {
    using Bits = uint64_t;
    static constexpr Bits kQuiet = 0x0008000000000000ull, kDefaultNaN = 0xFFF8000000000000ull;
    static constexpr int kSize = 8;
};
template <class T> T FromBits(typename FpInfo<T>::Bits b) { T v; memcpy(&v, &b, sizeof(v)); return v; }
template <class T> typename FpInfo<T>::Bits ToBits(T v) { typename FpInfo<T>::Bits b; memcpy(&b, &v, sizeof(b)); return b; }

// op: the 0F opcode (0x51 sqrt, 0x58 add, 0x59 mul, 0x5C sub, 0x5D min, 0x5E div, 0x5F max).
template <class T> typename FpInfo<T>::Bits FpArith(uint8_t op, typename FpInfo<T>::Bits a, typename FpInfo<T>::Bits b) {
    using I = FpInfo<T>;
    const T x = FromBits<T>(a), y = FromBits<T>(b);
    if (op == 0x5D) return x < y ? a : b;  // MIN/MAX return the second source for NaNs and equal values
    if (op == 0x5F) return x > y ? a : b;
    if (op == 0x51) {
        if (y != y) return b | I::kQuiet;
        const T r = std::sqrt(y);
        return r != r ? I::kDefaultNaN : ToBits<T>(r);
    }
    if (x != x) return a | I::kQuiet;
    if (y != y) return b | I::kQuiet;
    T r;
    switch (op) {
        case 0x58: r = x + y; break;
        case 0x59: r = x * y; break;
        case 0x5C: r = x - y; break;
        default: r = x / y; break;
    }
    return r != r ? I::kDefaultNaN : ToBits<T>(r);
}

// CMPPS/CMPSD predicates 0-31 (the upper 16 repeat the lower ones with the other signaling behavior).
template <class T> bool FpCompare(int pred, T x, T y) {
    const bool unord = x != x || y != y;
    switch (pred & 15) {
        case 0: return !unord && x == y;
        case 1: return !unord && x < y;
        case 2: return !unord && x <= y;
        case 3: return unord;
        case 4: return unord || x != y;
        case 5: return unord || !(x < y);
        case 6: return unord || !(x <= y);
        case 7: return !unord;
        case 8: return unord || x == y;
        case 9: return unord || x < y;
        case 10: return unord || x <= y;
        case 11: return false;
        case 12: return !unord && x != y;
        case 13: return !unord && x >= y;
        case 14: return !unord && x > y;
        default: return true;
    }
}

// Float to integer conversion; out-of-range values and NaNs give the "integer indefinite" (only the sign bit).
template <class I, class T> I FpToInt(T v, bool truncate) {
    if (v != v) return std::numeric_limits<I>::min();
    const T r = truncate ? std::trunc(v) : std::nearbyint(v);
    if (r < (T)std::numeric_limits<I>::min() || r >= -(T)std::numeric_limits<I>::min())
        return std::numeric_limits<I>::min();
    return (I)r;
}

// Conversions between precisions keep a NaN's sign and payload (quieted), as x86 does.
uint64_t FloatToDoubleBits(uint32_t f) {
    const float v = FromBits<float>(f);
    if (v != v) return ((uint64_t)(f >> 31) << 63) | 0x7FF8000000000000ull | ((uint64_t)(f & 0x3FFFFF) << 29);
    return ToBits<double>((double)v);
}
uint32_t DoubleToFloatBits(uint64_t d) {
    const double v = FromBits<double>(d);
    if (v != v) return ((uint32_t)(d >> 63) << 31) | 0x7FC00000u | (uint32_t)((d >> 29) & 0x3FFFFF);
    return ToBits<float>((float)v);
}
}  // namespace

uint8_t X86Interpreter::ReadMemory8(uint64_t addr) const {
    return Mapped(addr, 1) ? mem_base_[addr - mem_base_addr_] : 0;
}

uint16_t X86Interpreter::ReadMemory16(uint64_t addr) const {
    return Mapped(addr, 2) ? LoadLE<uint16_t>(mem_base_ + (addr - mem_base_addr_)) : 0;
}

uint32_t X86Interpreter::ReadMemory32(uint64_t addr) const {
    return Mapped(addr, 4) ? LoadLE<uint32_t>(mem_base_ + (addr - mem_base_addr_)) : 0;
}

uint64_t X86Interpreter::ReadMemory64(uint64_t addr) const {
    return Mapped(addr, 8) ? LoadLE<uint64_t>(mem_base_ + (addr - mem_base_addr_)) : 0;
}

void X86Interpreter::WriteMemory8(uint64_t addr, uint8_t val) {
    if (Mapped(addr, 1)) mem_base_[addr - mem_base_addr_] = val;
}

void X86Interpreter::WriteMemory16(uint64_t addr, uint16_t val) {
    if (Mapped(addr, 2)) memcpy(mem_base_ + (addr - mem_base_addr_), &val, 2);
}

void X86Interpreter::WriteMemory32(uint64_t addr, uint32_t val) {
    if (Mapped(addr, 4)) memcpy(mem_base_ + (addr - mem_base_addr_), &val, 4);
}

void X86Interpreter::WriteMemory64(uint64_t addr, uint64_t val) {
    if (Mapped(addr, 8)) memcpy(mem_base_ + (addr - mem_base_addr_), &val, 8);
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
    if (idx >= 16 && idx < 20) {  // AH, CH, DH, BH: bits 15:8 of RAX/RCX/RDX/RBX, written back after the step
        const int k = idx - 16;
        if (!(high8_used_ & (1 << k))) {
            high8_[k] = (RegByIndex(k) >> 8) & 0xFF;
            high8_used_ |= 1 << k;
        }
        return high8_[k];
    }
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

// carry is ADC's carry-in (result = a + b + carry); SBB's borrow-in works the same way below.
void X86Interpreter::UpdateFlagsAdd(uint64_t a, uint64_t b, uint64_t result, int bits, int carry) {
    uint64_t r = truncate(result, bits);
    uint64_t ta = truncate(a, bits);
    uint64_t tb = truncate(b, bits);
    SetZF(r == 0);
    SetSF((r >> (bits - 1)) & 1);
    SetPF(parity_table[r & 0xFF]);
    SetCF((((unsigned __int128)ta + tb + (unsigned)carry) >> bits) != 0);
    SetAF(((ta ^ tb ^ r) >> 4) & 1);
    uint64_t sign_mask = 1ULL << (bits - 1);
    SetOF(((~(ta ^ tb)) & (ta ^ r) & sign_mask) != 0);
}

void X86Interpreter::UpdateFlagsSub(uint64_t a, uint64_t b, uint64_t result, int bits, int borrow) {
    uint64_t r = truncate(result, bits);
    uint64_t ta = truncate(a, bits);
    uint64_t tb = truncate(b, bits);
    SetZF(r == 0);
    SetSF((r >> (bits - 1)) & 1);
    SetPF(parity_table[r & 0xFF]);
    SetCF((unsigned __int128)ta < (unsigned __int128)tb + (unsigned)borrow);
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

    if (!watch_.empty()) {
        auto w = watch_.find(regs_.rip);
        if (w != watch_.end() && w->second++ < 8) {
            printf("[watch] hit 0x%llx at instruction %llu, [rsp]=0x%llx rax=0x%llx rcx=0x%llx rdx=0x%llx "
                   "rsi=0x%llx rdi=0x%llx [rdi]=",
                   (unsigned long long)regs_.rip, (unsigned long long)insn_count_,
                   (unsigned long long)ReadMemory64(regs_.rsp), (unsigned long long)regs_.rax,
                   (unsigned long long)regs_.rcx, (unsigned long long)regs_.rdx, (unsigned long long)regs_.rsi,
                   (unsigned long long)regs_.rdi);
            for (int i = 0; i < 4; i++) {
                uint64_t a = regs_.rdi + 8 * i;
                bool ok = a >= mem_base_addr_ && a - mem_base_addr_ + 8 <= mem_size_;
                printf(" %llx", ok ? (unsigned long long)LoadLE<uint64_t>(mem_base_ + (a - mem_base_addr_)) : 0ULL);
            }
            printf(" [rsi]=");
            for (int i = 0; i < 8; i++) {
                uint64_t a = regs_.rsi + 8 * i;
                bool ok = a >= mem_base_addr_ && a - mem_base_addr_ + 8 <= mem_size_;
                printf(" %llx", ok ? (unsigned long long)LoadLE<uint64_t>(mem_base_ + (a - mem_base_addr_)) : 0ULL);
            }
            printf("\n");
        }
    }
    for (auto& w : mem_watch_) {
        const uint64_t v = ReadMemory64(w.addr);
        if (v != w.value && w.hits++ < 16) {
            const std::string name = address_namer_ ? address_namer_(prev_rip_) : std::string();
            // After a call's RET, [rsp-8] is the return address and rdi/rsi/rdx usually still hold its arguments.
            printf("[watch] 0x%llx changed 0x%llx -> 0x%llx after the instruction at 0x%llx%s%s (instruction %llu, "
                   "rsp=0x%llx [rsp]=0x%llx [rsp-8]=0x%llx rax=0x%llx rdi=0x%llx rsi=0x%llx rdx=0x%llx)\n",
                   (unsigned long long)w.addr, (unsigned long long)w.value, (unsigned long long)v,
                   (unsigned long long)prev_rip_, name.empty() ? "" : " = ", name.c_str(),
                   (unsigned long long)insn_count_, (unsigned long long)regs_.rsp,
                   (unsigned long long)ReadMemory64(regs_.rsp), (unsigned long long)ReadMemory64(regs_.rsp - 8),
                   (unsigned long long)regs_.rax, (unsigned long long)regs_.rdi, (unsigned long long)regs_.rsi,
                   (unsigned long long)regs_.rdx);
            printf("[watch]   return addresses (frame pointers):");
            uint64_t fp = regs_.rbp;
            for (int i = 0; i < 12 && fp && (fp & 7) == 0 && fp - mem_base_addr_ + 16 <= mem_size_; i++) {
                printf(" 0x%llx", (unsigned long long)LoadLE<uint64_t>(mem_base_ + (fp + 8 - mem_base_addr_)));
                fp = LoadLE<uint64_t>(mem_base_ + (fp - mem_base_addr_));
            }
            printf("\n");
        }
        w.value = v;
    }
    prev_rip_ = regs_.rip;
    const uint64_t start_rip = regs_.rip;
    // HLE entry points are RET instructions, so only those need the (hashing) handler lookup.
    const uint64_t rip_off = regs_.rip - mem_base_addr_;
    const bool maybe_hle = rip_off < mem_size_ && mem_base_[rip_off] == 0xC3;
    auto it = maybe_hle ? hle_handlers_.find(regs_.rip) : hle_handlers_.end();
    if (it != hle_handlers_.end()) {
        it->second(regs_, regs_.rip);
        insn_count_++;
        if (yield_ && yield_retry_) {
            yield_ = yield_retry_ = false;
            return fault_ ? (fault_rip_ = start_rip, X86Exception::PageFault) : X86Exception::Yield;
        }
        DecodeAndExecute();
        if (high8_used_) FlushHigh8();
        insn_count_++;
        if (fault_) { fault_rip_ = start_rip; return X86Exception::PageFault; }
        if (yield_) { yield_ = false; return X86Exception::Yield; }
        return X86Exception::HLECall;
    }

    X86Exception ex = DecodeAndExecute();
    if (high8_used_) FlushHigh8();
    insn_count_++;
    if (fault_) { fault_rip_ = start_rip; return X86Exception::PageFault; }
    if (yield_ && ex == X86Exception::None) { yield_ = false; return X86Exception::Yield; }
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
    const uint64_t seg_base = seg_override == 0x64 ? regs_.fs_base
                            : seg_override == 0x65 ? regs_.gs_base : 0;

    int op_size = rex_w ? 64 : (has_op_size ? 16 : 32);
    int op_bytes = op_size / 8;

    uint8_t opcode = ReadMemory8(regs_.rip++);

    // Byte-register operands: without a REX prefix, register numbers 4-7 in them mean AH, CH, DH and BH,
    // which RegByIndex serves as virtual registers 16-19. Bit 0: the r/m operand is a byte, bit 1: reg is.
    int byte_ops = 0;
    if (!rex) {
        if (opcode < 0x40 && ((opcode & 7) == 0 || (opcode & 7) == 2)) byte_ops = 3;
        else if (opcode == 0x84 || opcode == 0x86 || opcode == 0x88 || opcode == 0x8A) byte_ops = 3;
        else if (opcode == 0x80 || opcode == 0xC0 || opcode == 0xC6 || opcode == 0xD0 || opcode == 0xD2 ||
                 opcode == 0xF6 || opcode == 0xFE) byte_ops = 1;
    }

    // Opcode map (0 = one-byte, 1 = 0F, 2 = 0F38, 3 = 0F3A) and opcode, for sizing immediates.
    int cur_map = 0;
    uint8_t cur_op = opcode;
    // VEX fields. Legacy SSE encodings reuse the VEX handlers with the destination as first source.
    int vex_R = 1, vex_X = 1, vex_B = 1, vex_W = 0, vex_L = 0, vex_pp = 0, vex_vvvv = 0, vex_mmmmm = 1;
    uint8_t vex_opcode = 0;
    bool legacy_simd = false;
    // Added to memory operands while executing the upper 128-bit lane of a 256-bit instruction.
    uint64_t lane_mem_offset = 0;

    // RIP-relative addresses are relative to the end of the instruction, including any immediate.
    auto imm_bytes_after_modrm = [&](uint8_t modrm) -> int {
        const int reg = (modrm >> 3) & 7;
        const int z = op_size == 16 ? 2 : 4;
        if (cur_map == 0) {
            switch (cur_op) {
                case 0x80: case 0x82: case 0x83: case 0xC0: case 0xC1: case 0xC6: case 0x6B: return 1;
                case 0x81: case 0xC7: case 0x69: return z;
                case 0xF6: return reg < 2 ? 1 : 0;
                case 0xF7: return reg < 2 ? z : 0;
                default: return 0;
            }
        }
        if (cur_map == 1) {
            switch (cur_op) {
                case 0x70: case 0x71: case 0x72: case 0x73: case 0xA4: case 0xAC: case 0xBA:
                case 0xC2: case 0xC4: case 0xC5: case 0xC6: return 1;
                default: return 0;
            }
        }
        return cur_map == 3 ? 1 : 0;
    };

    // Helper lambdas for reading ModRM operands
    auto read_modrm = [&]() -> uint8_t {
        return ReadMemory8(regs_.rip++);
    };

    auto reg_field = [&](uint8_t modrm) -> int {
        const int idx = ((modrm >> 3) & 7) | (rex_r << 3);
        return (byte_ops & 2) && idx >= 4 && idx < 8 ? idx + 12 : idx;
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
            const int idx = rm_raw | (rex_b << 3);
            *addr_out = (byte_ops & 1) && idx >= 4 && idx < 8 ? idx + 12 : idx;
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
            if (index != 4) {  // 100b means "no index" only without REX.X; with it, it's r12
                ea += RegByIndex(index) << scale;
            }
        } else if (rm_raw == 5 && mod == 0) {
            int32_t d32 = fetch_disp32();
            ea = regs_.rip + (int64_t)d32 + imm_bytes_after_modrm(modrm);
            *addr_out = ea + seg_base + lane_mem_offset;
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
        *addr_out = ea + seg_base + lane_mem_offset;
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
            case 2: { // ADC
                const int c = GetCF();
                result = a + b + (uint64_t)c;
                UpdateFlagsAdd(a, b, result, bits, c);
                break;
            }
            case 3: { // SBB
                const int c = GetCF();
                result = a - b - (uint64_t)c;
                UpdateFlagsSub(a, b, result, bits, c);
                break;
            }
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

    // Two- and three-operand IMUL: the truncated product goes to reg; CF = OF = it didn't fit.
    auto imul_store = [&](int reg, __int128 wide) {
        const uint64_t result = truncate((uint64_t)wide, op_size);
        uint64_t& d = RegByIndex(reg);
        if (op_size == 32) d = (uint32_t)result;
        else if (op_size == 16) d = (d & ~0xFFFFull) | result;
        else d = result;
        const bool overflow = (__int128)(int64_t)sign_extend(result, op_size) != wide;
        SetCF(overflow);
        SetOF(overflow);
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

    // Group 2 (ROL ROR RCL RCR SHL SHR SAL SAR) on r/m. A count of 0 (after masking) changes nothing.
    // OF is only defined for single-bit shifts; it is computed the same way for larger counts.
    auto shift_group = [&](uint8_t modrm, uint64_t rm_addr, bool rm_is_reg, int bits, uint8_t raw_count) {
        const int sub = (modrm >> 3) & 7;
        unsigned count = raw_count & (bits == 64 ? 0x3F : 0x1F);
        if (!count) {
            if (bits == 32 && rm_is_reg) RegByIndex((int)rm_addr) &= 0xFFFFFFFFull;  // a 32-bit register write still zero-extends
            return;
        }
        const uint64_t mask = bits == 64 ? ~0ull : (1ull << bits) - 1;
        const uint64_t val = read_from_rm(rm_addr, rm_is_reg, bits) & mask;
        const uint64_t msb = 1ull << (bits - 1);
        uint64_t r = val;
        switch (sub) {
            case 0: case 1: {  // ROL, ROR
                const unsigned c = count % bits;
                if (c) r = sub == 0 ? ((val << c) | (val >> (bits - c))) & mask : ((val >> c) | (val << (bits - c))) & mask;
                if (sub == 0) { SetCF(r & 1); SetOF(((r & msb) != 0) != GetCF()); }
                else { SetCF((r & msb) != 0); SetOF(((r ^ (r << 1)) & msb) != 0); }
                write_to_rm(rm_addr, rm_is_reg, r, bits);
                return;
            }
            case 2: case 3: {  // RCL, RCR: rotate through CF
                unsigned c = bits <= 16 ? count % (bits + 1) : count;
                bool cf = GetCF();
                if (sub == 3) SetOF(((val & msb) != 0) != cf);
                for (; c; c--) {
                    if (sub == 2) { const bool out = (r & msb) != 0; r = ((r << 1) | cf) & mask; cf = out; }
                    else { const bool out = r & 1; r = (r >> 1) | (cf ? msb : 0); cf = out; }
                }
                SetCF(cf);
                if (sub == 2) SetOF(((r & msb) != 0) != cf);
                write_to_rm(rm_addr, rm_is_reg, r, bits);
                return;
            }
            case 4: case 6:  // SHL/SAL
                SetCF(count <= (unsigned)bits ? (val >> (bits - count)) & 1 : 0);
                r = count < (unsigned)bits ? (val << count) & mask : 0;
                SetOF(((r & msb) != 0) != GetCF());
                break;
            case 5:  // SHR
                SetCF(count <= (unsigned)bits ? (val >> (count - 1)) & 1 : 0);
                r = count < (unsigned)bits ? val >> count : 0;
                SetOF((val & msb) != 0);
                break;
            case 7: {  // SAR
                const int64_t sv = (int64_t)sign_extend(val, bits);
                const unsigned c = count < (unsigned)bits ? count : bits - 1;
                SetCF(count < (unsigned)bits ? (sv >> (count - 1)) & 1 : (sv < 0));
                r = (uint64_t)(sv >> c) & mask;
                SetOF(false);
                break;
            }
        }
        UpdateFlags(r, bits);
        SetAF(false);
        write_to_rm(rm_addr, rm_is_reg, r, bits);
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
        int64_t imm = op_size == 16 ? (int64_t)(int16_t)read_imm16() : (int64_t)(int32_t)read_imm32();
        imul_store(reg, (__int128)rm_val * (__int128)imm);
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
        imul_store(reg, (__int128)rm_val * (__int128)imm);
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
        uint64_t& dst = RegByIndex(reg);
        if (op_size == 32) dst = (uint32_t)val;
        else if (op_size == 16) dst = (dst & ~0xFFFFull) | (val & 0xFFFF);  // 16-bit writes keep bits 63:16
        else dst = val;
        break;
    }

    // 0x8D: LEA r, m
    case 0x8D: {
        uint8_t modrm = read_modrm();
        uint64_t rm_addr; bool rm_is_reg;
        decode_rm(modrm, &rm_addr, &rm_is_reg);
        int reg = reg_field(modrm);
        uint64_t& dst = RegByIndex(reg);
        if (op_size == 32) dst = (uint32_t)rm_addr;
        else if (op_size == 16) dst = (dst & ~0xFFFFull) | (rm_addr & 0xFFFF);
        else dst = rm_addr;
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
        } else if (has_rep) {
            yield_ = true;  // PAUSE: the thread is spin-waiting, so let another thread run
            yield_retry_ = false;
        }
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
        uint64_t addr = ((rex_w || !has_addr_size) ? read_imm64() : read_imm32()) + seg_base;
        regs_.rax = (regs_.rax & ~0xFFULL) | ReadMemory8(addr);
        break;
    }

    // 0xA1: MOV rAX, moffs
    case 0xA1: {
        uint64_t addr = ((rex_w || !has_addr_size) ? read_imm64() : read_imm32()) + seg_base;
        if (op_size == 64) regs_.rax = ReadMemory64(addr);
        else if (op_size == 16) regs_.rax = (regs_.rax & ~0xFFFFULL) | ReadMemory16(addr);
        else regs_.rax = ReadMemory32(addr);
        break;
    }

    // 0xA2: MOV moffs8, AL
    case 0xA2: {
        uint64_t addr = ((rex_w || !has_addr_size) ? read_imm64() : read_imm32()) + seg_base;
        WriteMemory8(addr, (uint8_t)regs_.rax);
        break;
    }

    // 0xA3: MOV moffs, rAX
    case 0xA3: {
        uint64_t addr = ((rex_w || !has_addr_size) ? read_imm64() : read_imm32()) + seg_base;
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

    // 0xC0 / 0xC1: shift/rotate r/m8 / r/m, imm8
    case 0xC0: case 0xC1: {
        uint8_t modrm = read_modrm();
        uint64_t rm_addr; bool rm_is_reg;
        decode_rm(modrm, &rm_addr, &rm_is_reg);
        const int bits = opcode == 0xC0 ? 8 : op_size;
        const uint8_t count = read_imm8();
        shift_group(modrm, rm_addr, rm_is_reg, bits, count);
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

    // 0xD0-0xD3: shift/rotate r/m8 or r/m by 1 or CL
    case 0xD0: case 0xD1: case 0xD2: case 0xD3: {
        uint8_t modrm = read_modrm();
        uint64_t rm_addr; bool rm_is_reg;
        decode_rm(modrm, &rm_addr, &rm_is_reg);
        const int bits = (opcode & 1) ? op_size : 8;
        shift_group(modrm, rm_addr, rm_is_reg, bits, opcode >= 0xD2 ? (uint8_t)regs_.rcx : 1);
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
                if (ax / (uint8_t)val > 0xFF) return X86Exception::DivideByZero;
                regs_.rax = (regs_.rax & ~0xFFFFULL) | ((ax % (uint8_t)val) << 8) | (ax / (uint8_t)val);
                break;
            }
            case 7: { // IDIV
                if (val == 0) return X86Exception::DivideByZero;
                int16_t ax = (int16_t)(uint16_t)regs_.rax;
                int8_t divisor = (int8_t)(uint8_t)val;
                if (ax / divisor > INT8_MAX || ax / divisor < INT8_MIN) return X86Exception::DivideByZero;
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
                // A quotient too big for the destination raises #DE, like division by zero.
                if (op_size == 32) {
                    uint64_t dividend = ((uint64_t)(uint32_t)regs_.rdx << 32) | (uint32_t)regs_.rax;
                    if (dividend / (uint32_t)val > 0xFFFFFFFFull) return X86Exception::DivideByZero;
                    regs_.rax = (uint32_t)(dividend / (uint32_t)val);
                    regs_.rdx = (uint32_t)(dividend % (uint32_t)val);
                } else if (op_size == 64) {
                    unsigned __int128 dividend = ((unsigned __int128)regs_.rdx << 64) | regs_.rax;
                    if ((dividend / val) >> 64) return X86Exception::DivideByZero;
                    regs_.rax = (uint64_t)(dividend / val);
                    regs_.rdx = (uint64_t)(dividend % val);
                } else {
                    uint32_t dividend = ((uint32_t)(uint16_t)regs_.rdx << 16) | (uint16_t)regs_.rax;
                    if (dividend / (uint16_t)val > 0xFFFF) return X86Exception::DivideByZero;
                    regs_.rax = (regs_.rax & ~0xFFFFULL) | (uint16_t)(dividend / (uint16_t)val);
                    regs_.rdx = (regs_.rdx & ~0xFFFFULL) | (uint16_t)(dividend % (uint16_t)val);
                }
                break;
            }
            case 7: { // IDIV
                if (val == 0) return X86Exception::DivideByZero;
                if (op_size == 32) {
                    int64_t dividend = (int64_t)(((uint64_t)(uint32_t)regs_.rdx << 32) | (uint32_t)regs_.rax);
                    int32_t divisor = (int32_t)(uint32_t)val;
                    const int64_t q = dividend == INT64_MIN && divisor == -1 ? 0x100000000ll : dividend / divisor;
                    if (q > INT32_MAX || q < INT32_MIN) return X86Exception::DivideByZero;
                    regs_.rax = (uint32_t)q;
                    regs_.rdx = (uint32_t)(dividend - q * divisor);
                } else if (op_size == 64) {
                    __int128 dividend = (__int128)(((unsigned __int128)regs_.rdx << 64) | regs_.rax);
                    int64_t divisor = (int64_t)val;
                    const __int128 q = dividend / divisor;
                    if (q > INT64_MAX || q < INT64_MIN) return X86Exception::DivideByZero;
                    regs_.rax = (uint64_t)q;
                    regs_.rdx = (uint64_t)(dividend % divisor);
                } else {
                    int32_t dividend = (int32_t)(((uint32_t)(uint16_t)regs_.rdx << 16) | (uint16_t)regs_.rax);
                    int16_t divisor = (int16_t)(uint16_t)val;
                    const int32_t q = dividend / divisor;
                    if (q > INT16_MAX || q < INT16_MIN) return X86Exception::DivideByZero;
                    regs_.rax = (regs_.rax & ~0xFFFFULL) | (uint16_t)q;
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
        byte_ops = rex ? 0 : (op2 >= 0x90 && op2 <= 0x9F) || op2 == 0xB6 || op2 == 0xBE ? 1
                 : op2 == 0xB0 || op2 == 0xC0 ? 3 : 0;
        cur_map = 1;
        cur_op = op2;
        auto set_reg_bits = [&](int idx, uint64_t v, int bits) {
            uint64_t& r = RegByIndex(idx);
            if (bits == 64) r = v;
            else if (bits == 32) r = (uint32_t)v;
            else { uint64_t m = (1ULL << bits) - 1; r = (r & ~m) | (v & m); }
        };
        switch (op2) {

        // 0x0F 0x05: SYSCALL
        case 0x05: {
            if (syscall_handler_) {
                syscall_handler_(regs_, regs_.rax);
                return X86Exception::Syscall;
            }
            return X86Exception::InvalidOpcode;
        }

        // 0x0F 0x10 / 0x11: MOVUPS/MOVUPD load and store; F3 MOVSS and F2 MOVSD move one scalar. A scalar load
        // from memory zeroes the rest of the register; between registers only the low element moves.
        case 0x10: case 0x11: {
            uint8_t modrm = read_modrm();
            int reg = reg_field(modrm);
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            const size_t n = has_rep ? 4 : has_repne ? 8 : 16;
            const bool load = op2 == 0x10;
            if (rm_is_reg) {
                if (load) memcpy(regs_.xmm[reg], regs_.xmm[(int)rm_addr & 15], n);
                else memcpy(regs_.xmm[(int)rm_addr & 15], regs_.xmm[reg], n);
            } else if (load) {
                uint8_t v[16] = {};
                if (Mapped(rm_addr, n)) memcpy(v, mem_base_ + (rm_addr - mem_base_addr_), n);
                memcpy(regs_.xmm[reg], v, 16);
            } else if (Mapped(rm_addr, n)) {
                memcpy(mem_base_ + (rm_addr - mem_base_addr_), regs_.xmm[reg], n);
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
            else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(regs_.xmm[reg], mem_base_ + off, 16); }
            break;
        }

        // 0x0F 0x29: MOVAPS xmm/m128, xmm
        case 0x29: {
            uint8_t modrm = read_modrm();
            int reg = reg_field(modrm);
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            if (rm_is_reg) memcpy(regs_.xmm[(int)rm_addr], regs_.xmm[reg], 16);
            else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(mem_base_ + off, regs_.xmm[reg], 16); }
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
            uint64_t& dst = RegByIndex(reg);
            if (eval_cc(op2 - 0x40)) {
                if (op_size == 32) dst = (uint32_t)val;
                else if (op_size == 16) dst = (dst & ~0xFFFFull) | (val & 0xFFFF);
                else dst = val;
            } else if (op_size == 32) {
                dst = (uint32_t)dst;  // a 32-bit destination is zero-extended even when nothing moves
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
            else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
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
            else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(regs_.xmm[reg], mem_base_ + off, 16); }
            break;
        }

        // 0x0F 0x7F: MOVDQA/MOVDQU xmm/m128, xmm
        case 0x7F: {
            uint8_t modrm = read_modrm();
            int reg = reg_field(modrm);
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            if (rm_is_reg) memcpy(regs_.xmm[(int)rm_addr], regs_.xmm[reg], 16);
            else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(mem_base_ + off, regs_.xmm[reg], 16); }
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
            imul_store(reg, (__int128)a * (__int128)b);
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

        // 0x0F 0xBC: BSF r, r/m  (F3: TZCNT)
        case 0xBC: {
            uint8_t modrm = read_modrm();
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            int reg = reg_field(modrm);
            uint64_t val = read_from_rm(rm_addr, rm_is_reg, op_size);
            val = truncate(val, op_size);
            if (has_rep) {
                uint64_t cnt = val ? (uint64_t)__builtin_ctzll(val) : (uint64_t)op_size;
                set_reg_bits(reg, cnt, op_size);
                SetCF(val == 0);
                SetZF(cnt == 0);
                break;
            }
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

        // 0x0F 0xBD: BSR r, r/m  (F3: LZCNT)
        case 0xBD: {
            uint8_t modrm = read_modrm();
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            int reg = reg_field(modrm);
            uint64_t val = read_from_rm(rm_addr, rm_is_reg, op_size);
            val = truncate(val, op_size);
            if (has_rep) {
                uint64_t cnt = val ? (uint64_t)(__builtin_clzll(val) - (64 - op_size)) : (uint64_t)op_size;
                set_reg_bits(reg, cnt, op_size);
                SetCF(val == 0);
                SetZF(cnt == 0);
                break;
            }
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
            else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
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

        // Prefetch and hint NOPs (incl. ENDBR64)
        case 0x0D: case 0x18: case 0x19: case 0x1A: case 0x1B: case 0x1C: case 0x1D: case 0x1E: {
            uint8_t modrm = read_modrm();
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            break;
        }

        // SHLD (A4 imm8, A5 CL) / SHRD (AC imm8, AD CL)
        case 0xA4: case 0xA5: case 0xAC: case 0xAD: {
            uint8_t modrm = read_modrm();
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            unsigned cnt = (op2 & 1) ? (unsigned)(regs_.rcx & 0xFF) : read_imm8();
            cnt &= op_size == 64 ? 63 : 31;
            if (cnt && cnt < (unsigned)op_size) {
                uint64_t dst = read_from_rm(rm_addr, rm_is_reg, op_size);
                uint64_t src = truncate(RegByIndex(reg_field(modrm)), op_size);
                uint64_t res;
                if (op2 <= 0xA5) {
                    res = truncate((dst << cnt) | (src >> (op_size - cnt)), op_size);
                    SetCF((dst >> (op_size - cnt)) & 1);
                } else {
                    res = truncate((dst >> cnt) | (src << (op_size - cnt)), op_size);
                    SetCF((dst >> (cnt - 1)) & 1);
                }
                write_to_rm(rm_addr, rm_is_reg, res, op_size);
                UpdateFlags(res, op_size);
            }
            break;
        }

        // Group 15: fences, LDMXCSR/STMXCSR, CLFLUSH
        case 0xAE: {
            uint8_t modrm = read_modrm();
            int sub = (modrm >> 3) & 7;
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            if (rm_is_reg) {
                if (sub < 5) return X86Exception::InvalidOpcode;  // RDFSBASE etc.
            } else if (sub == 3) {
                WriteMemory32(rm_addr, 0x1F80);                    // STMXCSR: default MXCSR
            } else if (sub != 2 && sub != 7) {
                fprintf(stderr, "[x86] Unimplemented 0F AE /%d at RIP=0x%llX\n", sub, (unsigned long long)regs_.rip);
                return X86Exception::InvalidOpcode;
            }
            break;
        }

        // CMPXCHG r/m8, r8 (B0) / r/m, r (B1)
        case 0xB0: case 0xB1: {
            int bits = op2 == 0xB0 ? 8 : op_size;
            uint8_t modrm = read_modrm();
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            uint64_t dst = read_from_rm(rm_addr, rm_is_reg, bits);
            uint64_t acc = truncate(regs_.rax, bits);
            UpdateFlagsSub(acc, dst, acc - dst, bits);
            if (acc == dst) write_to_rm(rm_addr, rm_is_reg, truncate(RegByIndex(reg_field(modrm)), bits), bits);
            else set_reg_bits(0, dst, bits);
            break;
        }

        // POPCNT (F3 0F B8)
        case 0xB8: {
            if (!has_rep) return X86Exception::InvalidOpcode;
            uint8_t modrm = read_modrm();
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            uint64_t val = read_from_rm(rm_addr, rm_is_reg, op_size);
            set_reg_bits(reg_field(modrm), (uint64_t)__builtin_popcountll(val), op_size);
            regs_.rflags &= ~0x8D5ULL;  // clear OF SF ZF AF PF CF
            SetZF(val == 0);
            break;
        }

        // BTC r/m, r
        case 0xBB: {
            uint8_t modrm = read_modrm();
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            uint64_t val = read_from_rm(rm_addr, rm_is_reg, op_size);
            int bit = truncate(RegByIndex(reg_field(modrm)), op_size) % op_size;
            SetCF((val >> bit) & 1);
            write_to_rm(rm_addr, rm_is_reg, truncate(val ^ (1ULL << bit), op_size), op_size);
            break;
        }

        // XADD r/m8, r8 (C0) / r/m, r (C1)
        case 0xC0: case 0xC1: {
            int bits = op2 == 0xC0 ? 8 : op_size;
            uint8_t modrm = read_modrm();
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            int reg = reg_field(modrm);
            uint64_t dst = read_from_rm(rm_addr, rm_is_reg, bits);
            uint64_t src = truncate(RegByIndex(reg), bits);
            uint64_t sum = truncate(dst + src, bits);
            UpdateFlagsAdd(dst, src, dst + src, bits);
            set_reg_bits(reg, dst, bits);
            write_to_rm(rm_addr, rm_is_reg, sum, bits);
            break;
        }

        // MOVNTI m, r
        case 0xC3: {
            uint8_t modrm = read_modrm();
            uint64_t rm_addr; bool rm_is_reg;
            decode_rm(modrm, &rm_addr, &rm_is_reg);
            write_to_rm(rm_addr, rm_is_reg, truncate(RegByIndex(reg_field(modrm)), op_size), op_size);
            break;
        }

        default: {
            // SSE encodings are the VEX forms with the destination as the first source operand.
            const bool sse = (op2 >= 0x10 && op2 <= 0x17) || (op2 >= 0x28 && op2 <= 0x2F) ||
                             (op2 >= 0x50 && op2 <= 0x7F) || (op2 >= 0xC2 && op2 <= 0xC6) ||
                             op2 >= 0xD0 || op2 == 0x38 || op2 == 0x3A;
            const int pp = has_op_size ? 1 : has_rep ? 2 : has_repne ? 3 : 0;
            const bool mmx = pp == 0 && ((op2 >= 0x60 && op2 <= 0x7F) || op2 >= 0xD0);
            if (!sse || mmx) {
                fprintf(stderr, "[x86] Unimplemented 0F opcode: 0x%02X at RIP=0x%llX\n", op2,
                        (unsigned long long)(regs_.rip - 2));
                return X86Exception::InvalidOpcode;
            }
            vex_mmmmm = 1;
            vex_opcode = op2;
            if (op2 == 0x38 || op2 == 0x3A) {
                vex_mmmmm = op2 == 0x38 ? 2 : 3;
                vex_opcode = ReadMemory8(regs_.rip++);
                if (op2 == 0x38 && (vex_opcode == 0xF0 || vex_opcode == 0xF1)) {
                    uint8_t modrm = read_modrm();
                    uint64_t a; bool is_reg;
                    decode_rm(modrm, &a, &is_reg);
                    const int reg = reg_field(modrm);
                    if (has_repne) {  // CRC32: CRC-32C, reflected polynomial 0x82F63B78
                        const int bits = vex_opcode == 0xF0 ? 8 : (rex_w ? 64 : has_op_size ? 16 : 32);
                        const uint64_t src = read_from_rm(a, is_reg, bits);
                        uint32_t crc = (uint32_t)RegByIndex(reg);
                        for (int i = 0; i < bits / 8; i++) {
                            crc ^= (uint8_t)(src >> (8 * i));
                            for (int k = 0; k < 8; k++) crc = (crc >> 1) ^ (0x82F63B78u & (0u - (crc & 1u)));
                        }
                        RegByIndex(reg) = crc;
                    } else {  // MOVBE: F0 loads, F1 stores, byte-swapped
                        if (is_reg) return X86Exception::InvalidOpcode;
                        auto bswap = [&](uint64_t v) -> uint64_t {
                            return op_size == 64 ? __builtin_bswap64(v)
                                 : op_size == 32 ? __builtin_bswap32((uint32_t)v) : __builtin_bswap16((uint16_t)v);
                        };
                        if (vex_opcode == 0xF0) set_reg_bits(reg, bswap(read_from_rm(a, false, op_size)), op_size);
                        else write_to_rm(a, false, bswap(truncate(RegByIndex(reg), op_size)), op_size);
                    }
                    break;
                }
                if (vex_opcode >= 0xF0) {  // ADCX/ADOX and friends are not SIMD
                    fprintf(stderr, "[x86] Unimplemented 0F %02X %02X at RIP=0x%llX\n", op2, vex_opcode,
                            (unsigned long long)(regs_.rip - 3));
                    return X86Exception::InvalidOpcode;
                }
            }
            // Immediate-count shifts (0F 71-73) take their destination from ModRM.rm, everything else from ModRM.reg.
            const uint8_t modrm_peek = ReadMemory8(regs_.rip);
            const bool rm_dest = vex_mmmmm == 1 && op2 >= 0x71 && op2 <= 0x73;
            vex_vvvv = rm_dest ? ((modrm_peek & 7) | (rex_b << 3)) : (((modrm_peek >> 3) & 7) | (rex_r << 3));
            vex_pp = pp;
            vex_W = rex_w;
            vex_L = 0;
            legacy_simd = true;
            goto simd_dispatch;
        }
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

    simd_dispatch:
        cur_map = vex_mmmmm;
        cur_op = vex_opcode;
        // VEX.pp maps to legacy prefix: 0=none, 1=66, 2=F3, 3=F2
        bool vex_66 = (vex_pp == 1);
        bool vex_f3 = (vex_pp == 2);
        bool vex_f2 = (vex_pp == 3);
        (void)vex_66; (void)vex_f2;
        int vec_len = vex_L ? 32 : 16; // YMM (256) or XMM (128)

        const uint64_t operand_rip = regs_.rip;
        const uint8_t op_modrm = ReadMemory8(operand_rip);
        const int op_reg = ((op_modrm >> 3) & 7) | (rex_r << 3);
        const int op_rm = (op_modrm & 7) | (rex_b << 3);
        const bool op_rm_is_reg = (op_modrm >> 6) == 3;

        // Forms that move data across 128-bit lanes (or change width) are handled whole here.
        if (!legacy_simd) {
            auto mem_in = [&](uint64_t a, uint8_t* out, size_t n) {
                uint64_t off = a - mem_base_addr_;
                if (Mapped(a, n)) memcpy(out, mem_base_ + off, n);
                else memset(out, 0, n);
            };
            auto mem_out = [&](uint64_t a, const uint8_t* in, size_t n) {
                uint64_t off = a - mem_base_addr_;
                if (Mapped(a, n)) memcpy(mem_base_ + off, in, n);
            };
            auto ymm_get = [&](int r, uint8_t* out) {
                memcpy(out, regs_.xmm[r & 15], 16);
                memcpy(out + 16, regs_.ymm_hi[r & 15], 16);
            };
            auto ymm_set = [&](int r, const uint8_t* in, bool wide) {
                memcpy(regs_.xmm[r & 15], in, 16);
                if (wide) memcpy(regs_.ymm_hi[r & 15], in + 16, 16);
                else memset(regs_.ymm_hi[r & 15], 0, 16);
            };
            const int width = vex_L ? 32 : 16;
            const uint8_t op = vex_opcode;
            bool handled = true;
            if (vex_mmmmm == 1 && op == 0x77) {  // VZEROUPPER / VZEROALL
                for (int i = 0; i < 16; i++) {
                    memset(regs_.ymm_hi[i], 0, 16);
                    if (vex_L) memset(regs_.xmm[i], 0, 16);
                }
            } else if (vex_mmmmm == 2 && (op == 0x18 || op == 0x19 || op == 0x1A || op == 0x58 ||
                                          op == 0x59 || op == 0x5A || op == 0x78 || op == 0x79)) {
                // VBROADCASTSS/SD/F128, VPBROADCASTB/W/D/Q/I128
                uint8_t modrm = read_modrm(); uint64_t a; bool is_reg; decode_rm(modrm, &a, &is_reg);
                int esz = op == 0x78 ? 1 : op == 0x79 ? 2 : (op == 0x18 || op == 0x58) ? 4
                        : (op == 0x19 || op == 0x59) ? 8 : 16;
                uint8_t src[16] = {}, out[32] = {};
                if (is_reg) memcpy(src, regs_.xmm[(int)a & 15], 16); else mem_in(a, src, esz);
                for (int i = 0; i < width; i += esz) memcpy(out + i, src, esz);
                ymm_set(reg_field(modrm), out, vex_L);
            } else if (vex_mmmmm == 2 && vex_pp == 1 && (op == 0x8C || op == 0x8E || (op >= 0x2C && op <= 0x2F))) {
                // VPMASKMOVD/Q, VMASKMOVPS/PD: elements whose mask (vvvv) sign bit is set are loaded (others
                // zeroed) or stored (others untouched); masked-off elements never touch memory.
                uint8_t modrm = read_modrm(); uint64_t a; bool is_reg; decode_rm(modrm, &a, &is_reg);
                if (is_reg) return X86Exception::InvalidOpcode;
                const int esz = (op == 0x8C || op == 0x8E) ? (vex_W ? 8 : 4) : (op & 1) ? 8 : 4;
                const bool store = op == 0x8E || op == 0x2E || op == 0x2F;
                uint8_t mask[32] = {}, data[32] = {};
                ymm_get(vex_vvvv, mask);
                if (store) ymm_get(reg_field(modrm), data);
                for (int i = 0; i < width; i += esz) {
                    if (!(mask[i + esz - 1] & 0x80)) continue;
                    if (store) mem_out(a + i, data + i, esz);
                    else mem_in(a + i, data + i, esz);
                }
                if (!store) ymm_set(reg_field(modrm), data, vex_L);
            } else if (vex_mmmmm == 2 && (op == 0x0E || op == 0x0F || op == 0x17)) {
                // VTESTPS/VTESTPD test sign bits; VPTEST tests every bit
                uint8_t modrm = read_modrm(); uint64_t a; bool is_reg; decode_rm(modrm, &a, &is_reg);
                uint8_t d[32] = {}, s[32] = {};
                ymm_get(reg_field(modrm), d);
                if (is_reg) ymm_get((int)a, s); else mem_in(a, s, width);
                const int esz = op == 0x0E ? 4 : op == 0x0F ? 8 : 1;
                const uint8_t mask = op == 0x17 ? 0xFF : 0x80;
                bool zf = true, cf = true;
                for (int i = 0; i < width; i += esz) {
                    int b = i + esz - 1;
                    if (d[b] & s[b] & mask) zf = false;
                    if (~d[b] & s[b] & mask) cf = false;
                }
                SetZF(zf); SetCF(cf); SetOF(false); SetSF(false); SetAF(false); SetPF(false);
            } else if (vex_mmmmm == 2 && ((op >= 0x20 && op <= 0x25) || (op >= 0x30 && op <= 0x35))) {
                // VPMOVSX / VPMOVZX
                static const int kSrc[6] = {1, 1, 1, 2, 2, 4}, kDst[6] = {2, 4, 8, 4, 8, 8};
                const int k = op & 0xF;
                uint8_t modrm = read_modrm(); uint64_t a; bool is_reg; decode_rm(modrm, &a, &is_reg);
                const int count = width / kDst[k];
                uint8_t src[16] = {}, out[32] = {};
                if (is_reg) memcpy(src, regs_.xmm[(int)a & 15], 16); else mem_in(a, src, count * kSrc[k]);
                for (int i = 0; i < count; i++) {
                    uint64_t v = 0;
                    memcpy(&v, src + i * kSrc[k], kSrc[k]);
                    if (op < 0x30) { int sh = 64 - 8 * kSrc[k]; v = (uint64_t)((int64_t)(v << sh) >> sh); }
                    memcpy(out + i * kDst[k], &v, kDst[k]);
                }
                ymm_set(reg_field(modrm), out, vex_L);
            } else if (vex_mmmmm == 1 && vex_L && (op == 0x50 || op == 0xD7)) {
                // VMOVMSKPS/PD ymm, VPMOVMSKB ymm
                uint8_t modrm = read_modrm(); uint64_t a; bool is_reg; decode_rm(modrm, &a, &is_reg);
                uint8_t s[32];
                ymm_get((int)a, s);
                const int esz = op == 0xD7 ? 1 : vex_pp == 1 ? 8 : 4;
                uint32_t m = 0;
                for (int i = 0; i < 32 / esz; i++) m |= (uint32_t)(s[i * esz + esz - 1] >> 7) << i;
                RegByIndex(reg_field(modrm)) = m;
            } else if (vex_mmmmm == 1 && vex_L && op == 0x5A && vex_pp <= 1) {
                uint8_t modrm = read_modrm(); uint64_t a; bool is_reg; decode_rm(modrm, &a, &is_reg);
                uint8_t out[32] = {};
                if (vex_pp == 0) {  // VCVTPS2PD ymm, xmm/m128
                    float f[4];
                    if (is_reg) memcpy(f, regs_.xmm[(int)a & 15], 16); else mem_in(a, (uint8_t*)f, 16);
                    for (int i = 0; i < 4; i++) { double d = f[i]; memcpy(out + i * 8, &d, 8); }
                    ymm_set(reg_field(modrm), out, true);
                } else {            // VCVTPD2PS xmm, ymm/m256
                    double d[4];
                    if (is_reg) ymm_get((int)a, (uint8_t*)d); else mem_in(a, (uint8_t*)d, 32);
                    for (int i = 0; i < 4; i++) { float f = (float)d[i]; memcpy(out + i * 4, &f, 4); }
                    ymm_set(reg_field(modrm), out, false);
                }
            } else if (vex_mmmmm == 3 && vex_L && (op == 0x18 || op == 0x38)) {
                // VINSERTF128 / VINSERTI128
                uint8_t modrm = read_modrm(); uint64_t a; bool is_reg; decode_rm(modrm, &a, &is_reg);
                uint8_t imm = read_imm8();
                uint8_t v[32], s[16];
                ymm_get(vex_vvvv, v);
                if (is_reg) memcpy(s, regs_.xmm[(int)a & 15], 16); else mem_in(a, s, 16);
                memcpy(v + (imm & 1) * 16, s, 16);
                ymm_set(reg_field(modrm), v, true);
            } else if (vex_mmmmm == 3 && vex_L && (op == 0x19 || op == 0x39)) {
                // VEXTRACTF128 / VEXTRACTI128
                uint8_t modrm = read_modrm(); uint64_t a; bool is_reg; decode_rm(modrm, &a, &is_reg);
                uint8_t imm = read_imm8();
                uint8_t v[32], out[32] = {};
                ymm_get(reg_field(modrm), v);
                if (is_reg) { memcpy(out, v + (imm & 1) * 16, 16); ymm_set((int)a, out, false); }
                else mem_out(a, v + (imm & 1) * 16, 16);
            } else if (vex_mmmmm == 3 && vex_L && (op == 0x06 || op == 0x46)) {
                // VPERM2F128 / VPERM2I128
                uint8_t modrm = read_modrm(); uint64_t a; bool is_reg; decode_rm(modrm, &a, &is_reg);
                uint8_t imm = read_imm8();
                uint8_t s1[32], s2[32], out[32];
                ymm_get(vex_vvvv, s1);
                if (is_reg) ymm_get((int)a, s2); else mem_in(a, s2, 32);
                for (int lane = 0; lane < 2; lane++) {
                    int sel = (imm >> (lane * 4)) & 0xF;
                    if (sel & 8) memset(out + lane * 16, 0, 16);
                    else memcpy(out + lane * 16, ((sel & 2) ? s2 : s1) + (sel & 1) * 16, 16);
                }
                ymm_set(reg_field(modrm), out, true);
            } else if (vex_mmmmm == 3 && vex_L && (op == 0x00 || op == 0x01)) {
                // VPERMQ / VPERMPD
                uint8_t modrm = read_modrm(); uint64_t a; bool is_reg; decode_rm(modrm, &a, &is_reg);
                uint8_t imm = read_imm8();
                uint64_t s[4], out[4];
                if (is_reg) ymm_get((int)a, (uint8_t*)s); else mem_in(a, (uint8_t*)s, 32);
                for (int i = 0; i < 4; i++) out[i] = s[(imm >> (2 * i)) & 3];
                ymm_set(reg_field(modrm), (const uint8_t*)out, true);
            } else if (vex_mmmmm == 2 && vex_L && (op == 0x36 || op == 0x16)) {
                // VPERMD / VPERMPS
                uint8_t modrm = read_modrm(); uint64_t a; bool is_reg; decode_rm(modrm, &a, &is_reg);
                uint32_t idx[8], s[8], out[8];
                ymm_get(vex_vvvv, (uint8_t*)idx);
                if (is_reg) ymm_get((int)a, (uint8_t*)s); else mem_in(a, (uint8_t*)s, 32);
                for (int i = 0; i < 8; i++) out[i] = s[idx[i] & 7];
                ymm_set(reg_field(modrm), (const uint8_t*)out, true);
            } else {
                handled = false;
            }
            if (handled) break;
        }

        // At 256 bits, element-wise and in-lane ops run twice: the second pass sees each register's
        // upper half and memory operands 16 bytes further on. Scalar, GPR and flag forms ignore VEX.L.
        auto lane_wise_256 = [&]() -> int {  // 1: two lanes, 0: one pass, -1: unsupported
            const uint8_t op = vex_opcode;
            if (vex_mmmmm == 1) {
                if ((vex_pp >= 2) && (op == 0x10 || op == 0x11 || op == 0x2A || op == 0x2C || op == 0x2D ||
                                      (op >= 0x51 && op <= 0x5F && op != 0x5B) || op == 0xC2))
                    return 0;  // scalar (F3 0F 5B, CVTTPS2DQ, is packed)
                if (op == 0x2E || op == 0x2F || op == 0x6E || op == 0x7E || op == 0xC4 || op == 0xC5 || op == 0xD6 ||
                    op == 0xAE)
                    return 0;
                if (op == 0x5A || op == 0xE6) return 0;  // width-changing conversions handle both halves at once
                return 1;
            }
            if (vex_mmmmm == 2) return op >= 0xF0 ? 0 : op == 0x13 ? -1 : 1;
            if ((op >= 0x14 && op <= 0x17) || (op >= 0x20 && op <= 0x22) || op == 0xF0) return 0;
            if (op == 0x1D) return -1;
            return 1;
        };
        int lanes = 1;
        if (!legacy_simd && vex_L) {
            int lw = lane_wise_256();
            if (lw < 0) {
                fprintf(stderr, "[x86] Unimplemented 256-bit form: map %d opcode 0x%02X at RIP=0x%llX\n",
                        vex_mmmmm, vex_opcode, (unsigned long long)(operand_rip - 1));
                return X86Exception::InvalidOpcode;
            }
            lanes = lw ? 2 : 1;
        }
        // Shifts by an XMM count use the low 64 bits of the count in both lanes.
        const bool count_operand = vex_mmmmm == 1 &&
            (vex_opcode == 0xD1 || vex_opcode == 0xD2 || vex_opcode == 0xD3 || vex_opcode == 0xE1 ||
             vex_opcode == 0xE2 || vex_opcode == 0xF1 || vex_opcode == 0xF2 || vex_opcode == 0xF3);
        uint8_t count_lo[16], count_hi[16];
        if (lanes == 2 && count_operand && op_rm_is_reg) memcpy(count_lo, regs_.xmm[op_rm], 16);
        auto swap_lanes = [&]() {
            uint8_t tmp[16];
            for (int i = 0; i < 16; i++) {
                memcpy(tmp, regs_.xmm[i], 16);
                memcpy(regs_.xmm[i], regs_.ymm_hi[i], 16);
                memcpy(regs_.ymm_hi[i], tmp, 16);
            }
        };

        auto ld = [&](uint64_t a, bool is_reg, void* out, size_t n) {
            if (is_reg) memcpy(out, regs_.xmm[(int)a & 15], n);
            else if (Mapped(a, n)) memcpy(out, mem_base_ + (a - mem_base_addr_), n);
            else memset(out, 0, n);
        };
        // dst = op(vvvv, r/m) on 16-byte vectors
        auto vex_binop = [&](auto fn) {
            uint8_t modrm = read_modrm();
            int reg = reg_field(modrm);
            uint64_t a; bool is_reg;
            decode_rm(modrm, &a, &is_reg);
            uint8_t x[16], y[16], out[16];
            memcpy(x, regs_.xmm[vex_vvvv & 15], 16);
            ld(a, is_reg, y, 16);
            fn(out, x, y);
            memcpy(regs_.xmm[reg], out, 16);
        };
        auto i16 = [](const uint8_t* p, int i) { int16_t v; memcpy(&v, p + 2 * i, 2); return v; };
        auto u16 = [](const uint8_t* p, int i) { uint16_t v; memcpy(&v, p + 2 * i, 2); return v; };
        auto put16 = [](uint8_t* p, int i, int v) { uint16_t s = (uint16_t)v; memcpy(p + 2 * i, &s, 2); };
        auto sat_s16 = [](int v) { return v > 32767 ? 32767 : v < -32768 ? -32768 : v; };
        auto sat_u16 = [](int v) { return v > 65535 ? 65535 : v < 0 ? 0 : v; };

        // SSE/AVX floating point by prefix (none: ps, 66: pd, F3: ss, F2: sd): arithmetic, compares, ordered
        // compares into flags, conversions, unpacks, sign masks and duplicating moves. Returns false for other
        // forms of these opcodes, which the switch below handles.
        bool full_width_written = false;  // a single pass already wrote all 256 bits of the destination
        auto fp_family = [&]() -> bool {
            const uint8_t op = vex_opcode;
            const int pp = vex_pp;
            const bool scalar = pp >= 2;
            const bool dbl = pp == 1 || pp == 3;
            const bool arith = op == 0x51 || op == 0x58 || op == 0x59 || (op >= 0x5C && op <= 0x5F);
            const bool cmp = op == 0xC2;
            if (arith || cmp) {
                uint8_t modrm = read_modrm();
                const int reg = reg_field(modrm);
                uint64_t a; bool is_reg;
                decode_rm(modrm, &a, &is_reg);
                const int pred = cmp ? (read_imm8() & (legacy_simd ? 7 : 31)) : 0;
                uint8_t x[16], y[16] = {}, out[16];
                memcpy(x, regs_.xmm[vex_vvvv & 15], 16);
                ld(a, is_reg, y, scalar ? (dbl ? 8 : 4) : 16);
                if (op == 0x51 && !scalar) memset(x, 0, 16);
                memcpy(out, x, 16);
                const int n = scalar ? 1 : dbl ? 2 : 4;
                for (int i = 0; i < n; i++) {
                    if (dbl) {
                        uint64_t xa, yb, r;
                        memcpy(&xa, x + 8 * i, 8); memcpy(&yb, y + 8 * i, 8);
                        r = cmp ? (FpCompare(pred, FromBits<double>(xa), FromBits<double>(yb)) ? ~0ull : 0)
                                : FpArith<double>(op, xa, yb);
                        memcpy(out + 8 * i, &r, 8);
                    } else {
                        uint32_t xa, yb, r;
                        memcpy(&xa, x + 4 * i, 4); memcpy(&yb, y + 4 * i, 4);
                        r = cmp ? (FpCompare(pred, FromBits<float>(xa), FromBits<float>(yb)) ? ~0u : 0)
                                : FpArith<float>(op, xa, yb);
                        memcpy(out + 4 * i, &r, 4);
                    }
                }
                memcpy(regs_.xmm[reg], out, 16);
                return true;
            }
            if ((op == 0x2E || op == 0x2F) && pp <= 1) {  // (V)UCOMISS/SD, (V)COMISS/SD
                uint8_t modrm = read_modrm();
                const int reg = reg_field(modrm);
                uint64_t a; bool is_reg;
                decode_rm(modrm, &a, &is_reg);
                uint8_t y[8] = {};
                ld(a, is_reg, y, pp ? 8 : 4);
                double xv, yv;
                if (pp) { memcpy(&xv, regs_.xmm[reg], 8); memcpy(&yv, y, 8); }
                else { float xf, yf; memcpy(&xf, regs_.xmm[reg], 4); memcpy(&yf, y, 4); xv = xf; yv = yf; }
                const bool unord = xv != xv || yv != yv;
                SetZF(unord || xv == yv); SetPF(unord); SetCF(unord || xv < yv);
                SetOF(false); SetSF(false); SetAF(false);
                return true;
            }
            if ((op == 0x2A || op == 0x2C || op == 0x2D) && scalar) {
                uint8_t modrm = read_modrm();
                const int reg = reg_field(modrm);
                uint64_t a; bool is_reg;
                decode_rm(modrm, &a, &is_reg);
                const bool wide = legacy_simd ? rex_w : vex_W;
                if (op == 0x2A) {  // CVTSI2SS/SD: integer source, the rest of the destination from src1
                    int64_t v = is_reg ? (int64_t)RegByIndex((int)a) : 0;
                    if (!is_reg) v = wide ? (int64_t)ReadMemory64(a) : (int64_t)(int32_t)ReadMemory32(a);
                    else if (!wide) v = (int32_t)v;
                    uint8_t out[16];
                    memcpy(out, regs_.xmm[vex_vvvv & 15], 16);
                    if (dbl) { const double d = (double)v; memcpy(out, &d, 8); }
                    else { const float f = (float)v; memcpy(out, &f, 4); }
                    memcpy(regs_.xmm[reg], out, 16);
                    return true;
                }
                uint8_t y[8] = {};
                ld(a, is_reg, y, dbl ? 8 : 4);
                double v;
                if (dbl) memcpy(&v, y, 8); else { float f; memcpy(&f, y, 4); v = f; }
                if (wide) RegByIndex(reg) = (uint64_t)FpToInt<int64_t>(v, op == 0x2C);
                else RegByIndex(reg) = (uint32_t)FpToInt<int32_t>(v, op == 0x2C);
                return true;
            }
            if (op == 0x5A || op == 0x5B || op == 0xE6) {  // conversions
                if (op == 0xE6 && pp == 0) return false;
                uint8_t modrm = read_modrm();
                const int reg = reg_field(modrm);
                uint64_t a; bool is_reg;
                decode_rm(modrm, &a, &is_reg);
                const bool wide = vex_L && !legacy_simd && op != 0x5B;
                uint8_t src[32] = {}, out[32] = {};
                auto load_src = [&](size_t n) {
                    if (is_reg) { memcpy(src, regs_.xmm[(int)a & 15], 16); memcpy(src + 16, regs_.ymm_hi[(int)a & 15], 16); }
                    else ld(a, false, src, n);
                };
                auto store = [&]() {
                    memcpy(regs_.xmm[reg], out, 16);
                    if (wide) { memcpy(regs_.ymm_hi[reg], out + 16, 16); full_width_written = true; }
                };
                if (op == 0x5A && pp >= 2) {  // CVTSS2SD / CVTSD2SS: low element, the rest from src1
                    load_src(pp == 2 ? 4 : 8);
                    memcpy(out, regs_.xmm[vex_vvvv & 15], 16);
                    if (pp == 2) { uint32_t f; memcpy(&f, src, 4); const uint64_t d = FloatToDoubleBits(f); memcpy(out, &d, 8); }
                    else { uint64_t d; memcpy(&d, src, 8); const uint32_t f = DoubleToFloatBits(d); memcpy(out, &f, 4); }
                    memcpy(regs_.xmm[reg], out, 16);
                    return true;
                }
                const int count = wide ? 4 : 2;  // elements for the width-changing forms
                if (op == 0x5A && pp == 0) {  // CVTPS2PD
                    load_src(4 * count);
                    for (int i = 0; i < count; i++) { uint32_t f; memcpy(&f, src + 4 * i, 4); const uint64_t d = FloatToDoubleBits(f); memcpy(out + 8 * i, &d, 8); }
                } else if (op == 0x5A) {  // CVTPD2PS
                    load_src(8 * count);
                    for (int i = 0; i < count; i++) { uint64_t d; memcpy(&d, src + 8 * i, 8); const uint32_t f = DoubleToFloatBits(d); memcpy(out + 4 * i, &f, 4); }
                } else if (op == 0xE6 && pp == 2) {  // CVTDQ2PD
                    load_src(4 * count);
                    for (int i = 0; i < count; i++) { int32_t v; memcpy(&v, src + 4 * i, 4); const double d = v; memcpy(out + 8 * i, &d, 8); }
                } else if (op == 0xE6) {  // CVTTPD2DQ (66) / CVTPD2DQ (F2)
                    load_src(8 * count);
                    for (int i = 0; i < count; i++) { double d; memcpy(&d, src + 8 * i, 8); const int32_t v = FpToInt<int32_t>(d, pp == 1); memcpy(out + 4 * i, &v, 4); }
                } else {  // 0x5B: CVTDQ2PS (none), CVTPS2DQ (66), CVTTPS2DQ (F3); same width, one 128-bit lane per pass
                    if (pp == 3) return false;
                    ld(a, is_reg, src, 16);
                    for (int i = 0; i < 4; i++) {
                        if (pp == 0) { int32_t v; memcpy(&v, src + 4 * i, 4); const float f = (float)v; memcpy(out + 4 * i, &f, 4); }
                        else { float f; memcpy(&f, src + 4 * i, 4); const int32_t v = FpToInt<int32_t>(f, pp == 2); memcpy(out + 4 * i, &v, 4); }
                    }
                    memcpy(regs_.xmm[reg], out, 16);
                    return true;
                }
                store();
                return true;
            }
            if ((op == 0x14 || op == 0x15) && pp <= 1) {  // UNPCKL/HPS, UNPCKL/HPD (per 128-bit lane)
                uint8_t modrm = read_modrm();
                const int reg = reg_field(modrm);
                uint64_t a; bool is_reg;
                decode_rm(modrm, &a, &is_reg);
                uint8_t x[16], y[16], out[16];
                memcpy(x, regs_.xmm[vex_vvvv & 15], 16);
                ld(a, is_reg, y, 16);
                const int hi = op == 0x15 ? 8 : 0;
                if (pp == 1) { memcpy(out, x + hi, 8); memcpy(out + 8, y + hi, 8); }
                else {
                    memcpy(out, x + hi, 4); memcpy(out + 4, y + hi, 4);
                    memcpy(out + 8, x + hi + 4, 4); memcpy(out + 12, y + hi + 4, 4);
                }
                memcpy(regs_.xmm[reg], out, 16);
                return true;
            }
            if ((op == 0x12 && pp >= 2) || (op == 0x16 && pp == 2)) {  // MOVDDUP, MOVSLDUP, MOVSHDUP
                uint8_t modrm = read_modrm();
                const int reg = reg_field(modrm);
                uint64_t a; bool is_reg;
                decode_rm(modrm, &a, &is_reg);
                uint8_t y[16] = {}, out[16];
                ld(a, is_reg, y, pp == 3 ? 8 : 16);
                if (pp == 3) { memcpy(out, y, 8); memcpy(out + 8, y, 8); }
                else {
                    const int k = op == 0x16 ? 4 : 0;
                    memcpy(out, y + k, 4); memcpy(out + 4, y + k, 4);
                    memcpy(out + 8, y + 8 + k, 4); memcpy(out + 12, y + 8 + k, 4);
                }
                memcpy(regs_.xmm[reg], out, 16);
                return true;
            }
            if (op == 0x50 && pp == 1 && !vex_L) {  // MOVMSKPD
                uint8_t modrm = read_modrm();
                const int reg = reg_field(modrm);
                uint64_t a; bool is_reg;
                decode_rm(modrm, &a, &is_reg);
                if (!is_reg) return false;
                RegByIndex(reg) = (uint32_t)((regs_.xmm[(int)a & 15][7] >> 7) | ((regs_.xmm[(int)a & 15][15] >> 7) << 1));
                return true;
            }
            return false;
        };

        for (int lane = 0; lane < lanes; lane++) {
        if (lane == 1) {
            swap_lanes();
            regs_.rip = operand_rip;
            lane_mem_offset = count_operand ? 0 : 16;
            if (count_operand && op_rm_is_reg) {
                memcpy(count_hi, regs_.xmm[op_rm], 16);
                memcpy(regs_.xmm[op_rm], count_lo, 16);
            }
        }

        if (vex_mmmmm == 1) {
            // 0F opcode map (most AVX instructions)
            if (fp_family()) continue;
            switch (vex_opcode) {

            // VMOVUPS / VMOVSS / VMOVSD / VMOVUPD
            case 0x10: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);

                if (vex_f3 || vex_f2) {
                    // VMOVSS/VMOVSD xmm, xmm, xmm (scalar from rm, the rest from vvvv) or xmm, m32/m64
                    const size_t n = vex_f3 ? 4 : 8;
                    uint8_t v[16] = {};
                    if (rm_is_reg) {
                        memcpy(v, regs_.xmm[vex_vvvv & 15], 16);
                        memcpy(v, regs_.xmm[(int)rm_addr & 15], n);
                    } else if (Mapped(rm_addr, n)) {
                        memcpy(v, mem_base_ + (rm_addr - mem_base_addr_), n);
                    }
                    memcpy(regs_.xmm[reg], v, 16);
                } else {
                    // VMOVUPS xmm/ymm, xmm/m128/m256
                    if (rm_is_reg) {
                        memcpy(regs_.xmm[reg], regs_.xmm[(int)rm_addr & 15], 16);
                    } else {
                        uint64_t off = rm_addr - mem_base_addr_;
                        if (Mapped(rm_addr, 16)) memcpy(regs_.xmm[reg], mem_base_ + off, 16);
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

                if (vex_f3 || vex_f2) {
                    // VMOVSS/VMOVSD m32/m64, xmm, or xmm(rm), xmm(vvvv), xmm(reg): scalar from reg, the rest from vvvv
                    const size_t n = vex_f3 ? 4 : 8;
                    if (rm_is_reg) {
                        uint8_t v[16];
                        memcpy(v, regs_.xmm[vex_vvvv & 15], 16);
                        memcpy(v, regs_.xmm[reg], n);
                        memcpy(regs_.xmm[(int)rm_addr & 15], v, 16);
                    } else if (Mapped(rm_addr, n)) {
                        memcpy(mem_base_ + (rm_addr - mem_base_addr_), regs_.xmm[reg], n);
                    }
                } else {
                    // VMOVUPS xmm/m128, xmm
                    if (rm_is_reg) {
                        memcpy(regs_.xmm[(int)rm_addr & 15], regs_.xmm[reg], 16);
                    } else {
                        uint64_t off = rm_addr - mem_base_addr_;
                        if (Mapped(rm_addr, 16)) memcpy(mem_base_ + off, regs_.xmm[reg], 16);
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
                    if (Mapped(rm_addr, 16)) memcpy(regs_.xmm[reg], mem_base_ + off, 16);
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
                    if (Mapped(rm_addr, 16)) memcpy(mem_base_ + off, regs_.xmm[reg], 16);
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
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
                    else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
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
                    else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src_f, mem_base_ + off, 16); else memset(src_f, 0, 16); }
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
                    else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src_f, mem_base_ + off, 16); else memset(src_f, 0, 16); }
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
                    else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src_f, mem_base_ + off, 16); else memset(src_f, 0, 16); }
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src_b, mem_base_ + off, 16); else memset(src_b, 0, 16); }
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src_b, mem_base_ + off, 16); else memset(src_b, 0, 16); }
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src_b, mem_base_ + off, 16); else memset(src_b, 0, 16); }
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

            // F3: VMOVQ xmm, xmm/m64 (load, upper bits zeroed). 66: VMOVD / VMOVQ r/m32/64, xmm.
            case 0x7E: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                if (vex_pp == 2) {
                    uint64_t val;
                    if (rm_is_reg) memcpy(&val, regs_.xmm[(int)rm_addr & 15], 8);
                    else val = ReadMemory64(rm_addr);
                    memset(regs_.xmm[reg], 0, 16);
                    memcpy(regs_.xmm[reg], &val, 8);
                } else if (rex_w) {
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

            // 66: VMOVQ xmm/m64, xmm (a register destination gets its upper bits zeroed)
            case 0xD6: {
                if (vex_pp != 1) return X86Exception::InvalidOpcode;
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint64_t val;
                memcpy(&val, regs_.xmm[reg], 8);
                if (rm_is_reg) {
                    memset(regs_.xmm[(int)rm_addr & 15], 0, 16);
                    memcpy(regs_.xmm[(int)rm_addr & 15], &val, 8);
                } else {
                    WriteMemory64(rm_addr, val);
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(regs_.xmm[reg], mem_base_ + off, 16); }
                break;
            }
            case 0x7F: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm) | (rex_r << 3);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                if (rm_is_reg) memcpy(regs_.xmm[(int)rm_addr & 15], regs_.xmm[reg], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(mem_base_ + off, regs_.xmm[reg], 16); }
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
                    else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src_f, mem_base_ + off, 16); else memset(src_f, 0, 16); }
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
                    else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src_f, mem_base_ + off, 16); else memset(src_f, 0, 16); }
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
                    else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src_f, mem_base_ + off, 16); else memset(src_f, 0, 16); }
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
                if (vex_pp == 1) {  // VSHUFPD
                    double x[2], y[2], r[2];
                    memcpy(x, regs_.xmm[vex_vvvv & 15], 16);
                    ld(rm_addr, rm_is_reg, y, 16);
                    r[0] = x[(imm >> (2 * lane)) & 1];
                    r[1] = y[(imm >> (2 * lane + 1)) & 1];
                    memcpy(regs_.xmm[reg], r, 16);
                    break;
                }
                float src_f[4], vvvv_f[4], dst_f[4];
                memcpy(vvvv_f, regs_.xmm[vex_vvvv & 15], 16);
                if (rm_is_reg) memcpy(src_f, regs_.xmm[(int)rm_addr & 15], 16);
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src_f, mem_base_ + off, 16); else memset(src_f, 0, 16); }
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
                    else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src_f, mem_base_ + off, 16); else memset(src_f, 0, 16); }
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src_b, mem_base_ + off, 16); else memset(src_b, 0, 16); }
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src_d, mem_base_ + off, 16); else memset(src_d, 0, 16); }
                uint32_t dst_d[4];
                if (vex_pp == 1) {  // PSHUFD
                    for (int i = 0; i < 4; i++) dst_d[i] = src_d[(imm >> (2 * i)) & 3];
                } else {  // PSHUFLW (F2) / PSHUFHW (F3): shuffle the words of one half, copy the other
                    uint16_t w[8], o[8];
                    memcpy(w, src_d, 16);
                    memcpy(o, w, 16);
                    const int base = vex_pp == 3 ? 0 : 4;
                    for (int i = 0; i < 4; i++) o[base + i] = w[base + ((imm >> (2 * i)) & 3)];
                    memcpy(dst_d, o, 16);
                }
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src_raw, mem_base_ + off, 16); else memset(src_raw, 0, 16); }
                if (vex_pp == 1 || vex_pp == 2) { // 66: VCVTPS2DQ (round), F3: VCVTTPS2DQ (truncate)
                    for (int i = 0; i < 4; i++) { float f; memcpy(&f, src_raw + i*4, 4); int32_t r = CvtToI32(f, vex_pp == 2); memcpy(regs_.xmm[reg] + i*4, &r, 4); }
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
                    else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 8)) memcpy(src_raw, mem_base_ + off, 8); else memset(src_raw, 0, 8); }
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
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
                    else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 8)) memcpy(regs_.xmm[reg] + 8, mem_base_ + off, 8); } // VMOVHPS load
                } else {
                    if (rm_is_reg) memcpy(regs_.xmm[reg], regs_.xmm[(int)rm_addr & 15] + 8, 8); // VMOVHLPS
                    else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 8)) memcpy(regs_.xmm[reg], mem_base_ + off, 8); } // VMOVLPS load
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
                    if (Mapped(rm_addr, 8)) {
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
                uint8_t bytes[16], out[16] = {};
                memcpy(bytes, regs_.xmm[rm_is_reg ? ((int)rm_addr & 15) : (vex_vvvv & 15)], 16);
                if (grp == 3) {         // VPSRLDQ (byte shift right)
                    for (int j = 0; j + imm < 16; j++) out[j] = bytes[j + imm];
                } else if (grp == 7) {  // VPSLLDQ (byte shift left)
                    for (int j = imm; j < 16; j++) out[j] = bytes[j - imm];
                } else {
                    uint64_t data[2];
                    memcpy(data, bytes, 16);
                    for (int i = 0; i < 2; i++) {
                        if (grp == 2) data[i] = (imm < 64) ? data[i] >> imm : 0;
                        else if (grp == 6) data[i] = (imm < 64) ? data[i] << imm : 0;
                    }
                    memcpy(out, data, 16);
                }
                memcpy(regs_.xmm[vex_vvvv & 15], out, 16);
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
                    else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src_f, mem_base_ + off, 16); else memset(src_f, 0, 16); }
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(&src_hi, mem_base_ + off + 8, 8); else src_hi = 0; }
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src_w, mem_base_ + off, 16); else memset(src_w, 0, 16); }
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src_w, mem_base_ + off, 16); else memset(src_w, 0, 16); }
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
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
                    if (Mapped(rm_addr, 16)) memcpy(mem_base_ + off, regs_.xmm[reg], 16);
                }
                break;
            }

            case 0xF4:  // VPMULUDQ
                vex_binop([&](uint8_t* o, const uint8_t* x, const uint8_t* y) {
                    for (int i = 0; i < 2; i++) {
                        uint32_t a, b; memcpy(&a, x + 8 * i, 4); memcpy(&b, y + 8 * i, 4);
                        uint64_t p = (uint64_t)a * b; memcpy(o + 8 * i, &p, 8);
                    }
                });
                break;
            case 0xF6:  // VPSADBW
                vex_binop([&](uint8_t* o, const uint8_t* x, const uint8_t* y) {
                    memset(o, 0, 16);
                    for (int h = 0; h < 2; h++) {
                        int s = 0;
                        for (int i = 0; i < 8; i++) s += std::abs((int)x[h * 8 + i] - (int)y[h * 8 + i]);
                        put16(o, h * 4, s);
                    }
                });
                break;
            case 0xE0:  // VPAVGB
                vex_binop([&](uint8_t* o, const uint8_t* x, const uint8_t* y) {
                    for (int i = 0; i < 16; i++) o[i] = (uint8_t)((x[i] + y[i] + 1) >> 1);
                });
                break;
            case 0xE3:  // VPAVGW
                vex_binop([&](uint8_t* o, const uint8_t* x, const uint8_t* y) {
                    for (int i = 0; i < 8; i++) put16(o, i, (u16(x, i) + u16(y, i) + 1) >> 1);
                });
                break;
            case 0xD8: case 0xD9: case 0xDC: case 0xDD: case 0xE8: case 0xE9: case 0xEC: case 0xED: {
                // saturating add/sub: D8/D9 PSUBUS B/W, DC/DD PADDUS B/W, E8/E9 PSUBS B/W, EC/ED PADDS B/W
                const uint8_t op = vex_opcode;
                const bool word = op & 1, add = (op & 0x04) != 0, is_signed = op >= 0xE8;
                vex_binop([&](uint8_t* o, const uint8_t* x, const uint8_t* y) {
                    if (word) {
                        for (int i = 0; i < 8; i++) {
                            int a = is_signed ? i16(x, i) : u16(x, i), b = is_signed ? i16(y, i) : u16(y, i);
                            int r = add ? a + b : a - b;
                            put16(o, i, is_signed ? sat_s16(r) : sat_u16(r));
                        }
                    } else {
                        for (int i = 0; i < 16; i++) {
                            int a = is_signed ? (int8_t)x[i] : x[i], b = is_signed ? (int8_t)y[i] : y[i];
                            int r = add ? a + b : a - b;
                            o[i] = (uint8_t)(is_signed ? std::clamp(r, -128, 127) : std::clamp(r, 0, 255));
                        }
                    }
                });
                break;
            }
            case 0xEA: case 0xEE: {  // VPMINSW / VPMAXSW
                const bool mx = vex_opcode == 0xEE;
                vex_binop([&](uint8_t* o, const uint8_t* x, const uint8_t* y) {
                    for (int i = 0; i < 8; i++) put16(o, i, mx ? std::max(i16(x, i), i16(y, i)) : std::min(i16(x, i), i16(y, i)));
                });
                break;
            }
            case 0x7C: case 0x7D: {  // VHADDPS/VHSUBPS (F2), VHADDPD/VHSUBPD (66)
                const bool sub = vex_opcode == 0x7D;
                if (vex_pp == 3) {
                    vex_binop([&](uint8_t* o, const uint8_t* x, const uint8_t* y) {
                        float a[4], b[4], r[4];
                        memcpy(a, x, 16); memcpy(b, y, 16);
                        r[0] = sub ? a[0] - a[1] : a[0] + a[1]; r[1] = sub ? a[2] - a[3] : a[2] + a[3];
                        r[2] = sub ? b[0] - b[1] : b[0] + b[1]; r[3] = sub ? b[2] - b[3] : b[2] + b[3];
                        memcpy(o, r, 16);
                    });
                } else {
                    vex_binop([&](uint8_t* o, const uint8_t* x, const uint8_t* y) {
                        double a[2], b[2], r[2];
                        memcpy(a, x, 16); memcpy(b, y, 16);
                        r[0] = sub ? a[0] - a[1] : a[0] + a[1]; r[1] = sub ? b[0] - b[1] : b[0] + b[1];
                        memcpy(o, r, 16);
                    });
                }
                break;
            }
            case 0xD0:  // VADDSUBPS (F2) / VADDSUBPD (66)
                if (vex_pp == 3) {
                    vex_binop([&](uint8_t* o, const uint8_t* x, const uint8_t* y) {
                        float a[4], b[4];
                        memcpy(a, x, 16); memcpy(b, y, 16);
                        for (int i = 0; i < 4; i++) a[i] = (i & 1) ? a[i] + b[i] : a[i] - b[i];
                        memcpy(o, a, 16);
                    });
                } else {
                    vex_binop([&](uint8_t* o, const uint8_t* x, const uint8_t* y) {
                        double a[2], b[2];
                        memcpy(a, x, 16); memcpy(b, y, 16);
                        a[0] -= b[0]; a[1] += b[1];
                        memcpy(o, a, 16);
                    });
                }
                break;
            case 0xE6: {  // 66: CVTTPD2DQ, F2: CVTPD2DQ, F3: CVTDQ2PD
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm);
                uint64_t a; bool is_reg;
                decode_rm(modrm, &a, &is_reg);
                uint8_t out[16] = {};
                if (vex_pp == 2) {
                    int32_t s[2]; ld(a, is_reg, s, 8);
                    double d[2] = {(double)s[0], (double)s[1]};
                    memcpy(out, d, 16);
                } else if (vex_pp == 1 || vex_pp == 3) {
                    double s[2]; ld(a, is_reg, s, 16);
                    int32_t r[2] = {CvtToI32(s[0], vex_pp == 1), CvtToI32(s[1], vex_pp == 1)};
                    memcpy(out, r, 8);
                } else {
                    return X86Exception::InvalidOpcode;
                }
                memcpy(regs_.xmm[reg], out, 16);
                break;
            }
            case 0x2D: {  // VCVTSS2SI (F3) / VCVTSD2SI (F2), rounding
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm);
                uint64_t a; bool is_reg;
                decode_rm(modrm, &a, &is_reg);
                double v;
                if (vex_pp == 2) { float f; ld(a, is_reg, &f, 4); v = f; }
                else if (vex_pp == 3) { ld(a, is_reg, &v, 8); }
                else return X86Exception::InvalidOpcode;
                RegByIndex(reg) = vex_W ? (uint64_t)CvtToI64(v, false) : (uint64_t)(uint32_t)CvtToI32(v, false);
                break;
            }
            case 0x2B: {  // VMOVNTPS / VMOVNTPD (store)
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm);
                uint64_t a; bool is_reg;
                decode_rm(modrm, &a, &is_reg);
                if (!is_reg && Mapped(a, 16)) memcpy(mem_base_ + (a - mem_base_addr_), regs_.xmm[reg], 16);
                break;
            }
            case 0xAE: {  // VLDMXCSR (/2), VSTMXCSR (/3)
                uint8_t modrm = read_modrm();
                int sub = (modrm >> 3) & 7;
                uint64_t a; bool is_reg;
                decode_rm(modrm, &a, &is_reg);
                if (is_reg || (sub != 2 && sub != 3)) return X86Exception::InvalidOpcode;
                if (sub == 3) WriteMemory32(a, 0x1F80);
                break;
            }

            // NOP (multi-byte) — VEX-encoded NOPs
            case 0x1F: {
                uint8_t modrm = read_modrm();
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                break;
            }

            case 0xF0: {  // (V)LDDQU: an unaligned load
                if (vex_pp != 3) return X86Exception::InvalidOpcode;
                uint8_t modrm = read_modrm();
                uint64_t a; bool is_reg;
                decode_rm(modrm, &a, &is_reg);
                if (is_reg) return X86Exception::InvalidOpcode;
                ld(a, false, regs_.xmm[reg_field(modrm)], 16);
                break;
            }

            default:
                fprintf(stderr, "[x86] Unimplemented %s 0F opcode: pp=%d 0x%02X at RIP=0x%llX\n",
                        legacy_simd ? "SSE" : "VEX", vex_pp, vex_opcode, (unsigned long long)(regs_.rip - 1));
                return X86Exception::InvalidOpcode;
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
                bool all_and_zero = true, all_andn_zero = true;
                for (int i = 0; i < 16; i++) { if (dst[i] & src[i]) all_and_zero = false; if ((~dst[i]) & src[i]) all_andn_zero = false; }
                SetZF(all_and_zero); SetCF(all_andn_zero);
                SetOF(false); SetSF(false); SetAF(false); SetPF(false);
                break;
            }

            // FMA3: 9x = 132, Ax = 213, Bx = 231 operand order; low nibble picks the operation.
            case 0x96: case 0x97: case 0x98: case 0x99: case 0x9A: case 0x9B: case 0x9C: case 0x9D: case 0x9E: case 0x9F:
            case 0xA6: case 0xA7: case 0xA8: case 0xA9: case 0xAA: case 0xAB: case 0xAC: case 0xAD: case 0xAE: case 0xAF:
            case 0xB6: case 0xB7: case 0xB8: case 0xB9: case 0xBA: case 0xBB: case 0xBC: case 0xBD: case 0xBE: case 0xBF: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm);
                uint64_t a; bool is_reg;
                decode_rm(modrm, &a, &is_reg);
                const int kind = vex_opcode & 0xF, order = vex_opcode >> 4;
                const bool scalar = kind >= 8 && (kind & 1);
                const bool dbl = vex_W;
                const int esz = dbl ? 8 : 4, n = scalar ? 1 : 16 / esz;
                uint8_t d[16], v[16], m[16] = {};
                memcpy(d, regs_.xmm[reg], 16);
                memcpy(v, regs_.xmm[vex_vvvv & 15], 16);
                ld(a, is_reg, m, scalar ? esz : 16);
                uint8_t out[16];
                memcpy(out, d, 16);
                for (int i = 0; i < n; i++) {
                    // A NaN source wins in operand order (dst, vvvv, r/m), quieted and unaffected by negation.
                    const uint8_t* srcs[3] = {d, v, m};
                    bool nan_out = false;
                    for (int k = 0; k < 3 && !nan_out; k++) {
                        if (dbl) {
                            uint64_t b; memcpy(&b, srcs[k] + 8 * i, 8);
                            if ((b & 0x7FF0000000000000ull) == 0x7FF0000000000000ull && (b & 0xFFFFFFFFFFFFFull)) {
                                b |= 0x0008000000000000ull; memcpy(out + 8 * i, &b, 8); nan_out = true;
                            }
                        } else {
                            uint32_t b; memcpy(&b, srcs[k] + 4 * i, 4);
                            if ((b & 0x7F800000u) == 0x7F800000u && (b & 0x7FFFFFu)) {
                                b |= 0x00400000u; memcpy(out + 4 * i, &b, 4); nan_out = true;
                            }
                        }
                    }
                    if (nan_out) continue;
                    double x1, x2, x3;  // dst, vvvv, r/m
                    if (dbl) { memcpy(&x1, d + 8 * i, 8); memcpy(&x2, v + 8 * i, 8); memcpy(&x3, m + 8 * i, 8); }
                    else { float f1, f2, f3; memcpy(&f1, d + 4 * i, 4); memcpy(&f2, v + 4 * i, 4); memcpy(&f3, m + 4 * i, 4); x1 = f1; x2 = f2; x3 = f3; }
                    double p, q, c;  // result = +/-(p*q) +/- c
                    if (order == 9) { p = x1; q = x3; c = x2; }
                    else if (order == 0xA) { p = x2; q = x1; c = x3; }
                    else { p = x2; q = x3; c = x1; }
                    bool neg_prod = false, neg_c = false;
                    switch (kind) {
                        case 6: neg_c = !(i & 1); break;           // fmaddsub: even lanes subtract
                        case 7: neg_c = (i & 1); break;            // fmsubadd: odd lanes subtract
                        case 8: case 9: break;                     // fmadd
                        case 0xA: case 0xB: neg_c = true; break;   // fmsub
                        case 0xC: case 0xD: neg_prod = true; break;// fnmadd
                        default: neg_prod = true; neg_c = true;    // fnmsub
                    }
                    if (neg_prod) p = -p;
                    if (neg_c) c = -c;
                    if (dbl) {
                        double r = std::fma(p, q, c);
                        uint64_t b = r != r ? 0xFFF8000000000000ull : ToBits<double>(r);  // invalid: default NaN
                        memcpy(out + 8 * i, &b, 8);
                    } else {
                        float r = std::fmaf((float)p, (float)q, (float)c);
                        uint32_t b = r != r ? 0xFFC00000u : ToBits<float>(r);
                        memcpy(out + 4 * i, &b, 4);
                    }
                }
                memcpy(regs_.xmm[reg], out, 16);
                break;
            }
            case 0x45: case 0x46: case 0x47: {  // VPSRLVD/Q, VPSRAVD, VPSLLVD/Q
                const uint8_t op = vex_opcode;
                const bool q = vex_W && op != 0x46;
                vex_binop([&](uint8_t* o, const uint8_t* x, const uint8_t* y) {
                    if (q) {
                        for (int i = 0; i < 2; i++) {
                            uint64_t a, c; memcpy(&a, x + 8 * i, 8); memcpy(&c, y + 8 * i, 8);
                            uint64_t r = c > 63 ? 0 : (op == 0x47 ? a << c : a >> c);
                            memcpy(o + 8 * i, &r, 8);
                        }
                    } else {
                        for (int i = 0; i < 4; i++) {
                            uint32_t a, c; memcpy(&a, x + 4 * i, 4); memcpy(&c, y + 4 * i, 4);
                            uint32_t r;
                            if (op == 0x46) r = (uint32_t)((int32_t)a >> (c > 31 ? 31 : c));
                            else r = c > 31 ? 0 : (op == 0x47 ? a << c : a >> c);
                            memcpy(o + 4 * i, &r, 4);
                        }
                    }
                });
                break;
            }
            case 0x37:  // VPCMPGTQ
                vex_binop([&](uint8_t* o, const uint8_t* x, const uint8_t* y) {
                    for (int i = 0; i < 2; i++) {
                        int64_t a, b; memcpy(&a, x + 8 * i, 8); memcpy(&b, y + 8 * i, 8);
                        uint64_t r = a > b ? ~0ULL : 0; memcpy(o + 8 * i, &r, 8);
                    }
                });
                break;
            case 0x28:  // VPMULDQ
                vex_binop([&](uint8_t* o, const uint8_t* x, const uint8_t* y) {
                    for (int i = 0; i < 2; i++) {
                        int32_t a, b; memcpy(&a, x + 8 * i, 4); memcpy(&b, y + 8 * i, 4);
                        int64_t p = (int64_t)a * b; memcpy(o + 8 * i, &p, 8);
                    }
                });
                break;
            case 0x2B:  // VPACKUSDW
                vex_binop([&](uint8_t* o, const uint8_t* x, const uint8_t* y) {
                    for (int i = 0; i < 4; i++) {
                        int32_t a, b; memcpy(&a, x + 4 * i, 4); memcpy(&b, y + 4 * i, 4);
                        put16(o, i, std::clamp(a, 0, 65535)); put16(o, 4 + i, std::clamp(b, 0, 65535));
                    }
                });
                break;
            case 0x1C: case 0x1D: case 0x1E: {  // VPABSB/W/D
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm);
                uint64_t a; bool is_reg;
                decode_rm(modrm, &a, &is_reg);
                uint8_t s[16], o[16];
                ld(a, is_reg, s, 16);
                if (vex_opcode == 0x1C) for (int i = 0; i < 16; i++) o[i] = (uint8_t)std::abs((int)(int8_t)s[i]);
                else if (vex_opcode == 0x1D) for (int i = 0; i < 8; i++) put16(o, i, std::abs((int)i16(s, i)));
                else for (int i = 0; i < 4; i++) { int32_t v; memcpy(&v, s + 4 * i, 4); uint32_t r = v < 0 ? 0u - (uint32_t)v : (uint32_t)v; memcpy(o + 4 * i, &r, 4); }
                memcpy(regs_.xmm[reg], o, 16);
                break;
            }
            case 0x0C: case 0x0D: {  // VPERMILPS / VPERMILPD with a vector control
                const bool pd = vex_opcode == 0x0D;
                vex_binop([&](uint8_t* o, const uint8_t* x, const uint8_t* y) {
                    if (pd) for (int i = 0; i < 2; i++) { uint64_t c; memcpy(&c, y + 8 * i, 8); memcpy(o + 8 * i, x + 8 * ((c >> 1) & 1), 8); }
                    else for (int i = 0; i < 4; i++) { uint32_t c; memcpy(&c, y + 4 * i, 4); memcpy(o + 4 * i, x + 4 * (c & 3), 4); }
                });
                break;
            }
            case 0x04:  // VPMADDUBSW
                vex_binop([&](uint8_t* o, const uint8_t* x, const uint8_t* y) {
                    for (int i = 0; i < 8; i++)
                        put16(o, i, sat_s16(x[2 * i] * (int8_t)y[2 * i] + x[2 * i + 1] * (int8_t)y[2 * i + 1]));
                });
                break;
            case 0x01: case 0x02: case 0x05: case 0x06: {  // VPHADDW/D, VPHSUBW/D
                const bool sub = vex_opcode >= 0x05, dword = (vex_opcode & 3) == 2;
                vex_binop([&](uint8_t* o, const uint8_t* x, const uint8_t* y) {
                    if (dword) {
                        int32_t a[4], b[4], r[4];
                        memcpy(a, x, 16); memcpy(b, y, 16);
                        for (int i = 0; i < 2; i++) {
                            r[i] = (int32_t)(sub ? (uint32_t)a[2 * i] - (uint32_t)a[2 * i + 1] : (uint32_t)a[2 * i] + (uint32_t)a[2 * i + 1]);
                            r[2 + i] = (int32_t)(sub ? (uint32_t)b[2 * i] - (uint32_t)b[2 * i + 1] : (uint32_t)b[2 * i] + (uint32_t)b[2 * i + 1]);
                        }
                        memcpy(o, r, 16);
                    } else {
                        for (int i = 0; i < 4; i++) {
                            put16(o, i, sub ? i16(x, 2 * i) - i16(x, 2 * i + 1) : i16(x, 2 * i) + i16(x, 2 * i + 1));
                            put16(o, 4 + i, sub ? i16(y, 2 * i) - i16(y, 2 * i + 1) : i16(y, 2 * i) + i16(y, 2 * i + 1));
                        }
                    }
                });
                break;
            }
            case 0x0B:  // VPMULHRSW
                vex_binop([&](uint8_t* o, const uint8_t* x, const uint8_t* y) {
                    for (int i = 0; i < 8; i++) put16(o, i, (((int32_t)i16(x, i) * i16(y, i) >> 14) + 1) >> 1);
                });
                break;
            case 0x13: {  // VCVTPH2PS
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm);
                uint64_t a; bool is_reg;
                decode_rm(modrm, &a, &is_reg);
                uint16_t h[4]; float f[4];
                ld(a, is_reg, h, 8);
                for (int i = 0; i < 4; i++) f[i] = HalfToFloat(h[i]);
                memcpy(regs_.xmm[reg], f, 16);
                break;
            }

            // BMI1/BMI2 general-purpose register forms
            case 0xF2: case 0xF3: case 0xF5: case 0xF6: case 0xF7: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                const int bits = vex_W ? 64 : 32;
                const uint64_t mask = vex_W ? ~0ULL : 0xFFFFFFFFULL;
                uint64_t src = rm_is_reg ? RegByIndex((int)rm_addr) : (vex_W ? ReadMemory64(rm_addr) : ReadMemory32(rm_addr));
                src &= mask;
                uint64_t v = RegByIndex(vex_vvvv) & mask;
                auto put = [&](int idx, uint64_t val) { RegByIndex(idx) = val & mask; };
                if (vex_opcode == 0xF7) {
                    unsigned cnt = (unsigned)(v & (bits - 1));
                    if (vex_pp == 1) put(reg, src << cnt);                                     // SHLX
                    else if (vex_pp == 3) put(reg, src >> cnt);                                // SHRX
                    else if (vex_pp == 2) put(reg, vex_W ? (uint64_t)((int64_t)src >> cnt)    // SARX
                                                         : (uint64_t)(uint32_t)((int32_t)src >> cnt));
                    else {                                                                     // BEXTR
                        unsigned start = v & 0xFF, len = (v >> 8) & 0xFF;
                        uint64_t r = start >= (unsigned)bits ? 0 : src >> start;
                        if (len < (unsigned)bits) r &= (1ULL << len) - 1;
                        put(reg, r);
                        SetZF((r & mask) == 0); SetCF(false); SetOF(false);
                    }
                } else if (vex_opcode == 0xF2 && vex_pp == 0) {                               // ANDN
                    uint64_t r = ~v & src & mask;
                    put(reg, r);
                    UpdateFlags(r, bits); SetCF(false); SetOF(false);
                } else if (vex_opcode == 0xF3 && vex_pp == 0) {                               // BLSR/BLSMSK/BLSI
                    int op = (modrm >> 3) & 7;
                    uint64_t r = op == 1 ? (src & (src - 1)) : op == 2 ? (src ^ (src - 1)) : (src & (0 - src));
                    put(vex_vvvv, r);
                    UpdateFlags(r & mask, bits); SetCF(op == 3 ? src != 0 : src == 0); SetOF(false);
                } else if (vex_opcode == 0xF5 && vex_pp == 0) {                               // BZHI
                    unsigned n = v & 0xFF;
                    uint64_t r = n < (unsigned)bits ? src & ((1ULL << n) - 1) : src;
                    put(reg, r);
                    UpdateFlags(r, bits); SetCF(n > (unsigned)bits - 1); SetOF(false);
                } else if (vex_opcode == 0xF5) {                                              // PDEP (F2) / PEXT (F3)
                    uint64_t r = 0;
                    if (vex_pp == 3) { for (uint64_t m = src, k = 1; m; m &= m - 1, k <<= 1) if (v & k) r |= m & (0 - m); }
                    else { uint64_t k = 1; for (uint64_t m = src; m; m &= m - 1, k <<= 1) if (v & m & (0 - m)) r |= k; }
                    put(reg, r);
                } else if (vex_opcode == 0xF6 && vex_pp == 3) {                               // MULX
                    uint64_t d = regs_.rdx & mask;
                    if (vex_W) {
                        unsigned __int128 p = (unsigned __int128)d * src;
                        put(vex_vvvv, (uint64_t)p);
                        put(reg, (uint64_t)(p >> 64));
                    } else {
                        uint64_t p = d * src;
                        put(vex_vvvv, p & 0xFFFFFFFFULL);
                        put(reg, p >> 32);
                    }
                } else {
                    return X86Exception::InvalidOpcode;
                }
                break;
            }

            case 0x41: {  // (V)PHMINPOSUW: the smallest unsigned word and its index
                uint8_t modrm = read_modrm();
                uint64_t a; bool is_reg;
                decode_rm(modrm, &a, &is_reg);
                uint16_t w[8];
                ld(a, is_reg, w, 16);
                int best = 0;
                for (int i = 1; i < 8; i++) if (w[i] < w[best]) best = i;
                uint16_t out[8] = {w[best], (uint16_t)best};
                memcpy(regs_.xmm[reg_field(modrm)], out, 16);
                break;
            }

            case 0xDB: case 0xDC: case 0xDD: case 0xDE: case 0xDF: {  // AESIMC, AESENC(LAST), AESDEC(LAST)
                if (vex_pp != 1) return X86Exception::InvalidOpcode;
                uint8_t modrm = read_modrm();
                uint64_t a; bool is_reg;
                decode_rm(modrm, &a, &is_reg);
                uint8_t key[16], out[16];
                ld(a, is_reg, key, 16);
                if (vex_opcode == 0xDB) {
                    memcpy(out, key, 16);
                    AesMixColumns(out, true);
                } else {
                    uint8_t state[16];
                    memcpy(state, regs_.xmm[vex_vvvv & 15], 16);
                    AesRound(vex_opcode, state, key, out);
                }
                memcpy(regs_.xmm[reg_field(modrm)], out, 16);
                break;
            }

            default:
                fprintf(stderr, "[x86] Unimplemented %s 0F38 opcode: pp=%d 0x%02X at RIP=0x%llX\n",
                        legacy_simd ? "SSE" : "VEX", vex_pp, vex_opcode, (unsigned long long)(regs_.rip - 1));
                return X86Exception::InvalidOpcode;
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src_f, mem_base_ + off, 16); else memset(src_f, 0, 16); }
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src_d, mem_base_ + off, 16); else memset(src_d, 0, 16); }
                dst_d[0] = src_d[(imm >> (2 * lane)) & 1];
                dst_d[1] = src_d[(imm >> (2 * lane + 1)) & 1];
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
                for (int i = 0; i < 4; i++)
                    memcpy(regs_.xmm[reg] + i*4, ((imm >> (i + 4 * lane)) & 1) ? (uint8_t*)&src[i] : (uint8_t*)&vvvv_d[i], 4);
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
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
                    else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src_f, mem_base_ + off, 16); else memset(src_f, 0, 16); }
                    for (int i = 0; i < 4; i++) {
                        if (mode == 0) src_f[i] = nearbyintf(src_f[i]); // nearest, ties to even
                        else if (mode == 1) src_f[i] = floorf(src_f[i]);
                        else if (mode == 2) src_f[i] = ceilf(src_f[i]);
                        else src_f[i] = truncf(src_f[i]);
                    }
                    memcpy(regs_.xmm[reg], src_f, 16);
                } else if (vex_opcode == 0x0A) { // VROUNDSS
                    float f;
                    if (rm_is_reg) memcpy(&f, regs_.xmm[(int)rm_addr & 15], 4); else { uint32_t v = ReadMemory32(rm_addr); memcpy(&f, &v, 4); }
                    if (mode == 0) f = nearbyintf(f); else if (mode == 1) f = floorf(f); else if (mode == 2) f = ceilf(f); else f = truncf(f);
                    memcpy(regs_.xmm[reg], regs_.xmm[vex_vvvv & 15], 16);
                    memcpy(regs_.xmm[reg], &f, 4);
                } else if (vex_opcode == 0x09) { // VROUNDPD
                    double src_d[2];
                    if (rm_is_reg) memcpy(src_d, regs_.xmm[(int)rm_addr & 15], 16);
                    else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src_d, mem_base_ + off, 16); else memset(src_d, 0, 16); }
                    for (int i = 0; i < 2; i++) {
                        if (mode == 0) src_d[i] = nearbyint(src_d[i]); else if (mode == 1) src_d[i] = floor(src_d[i]);
                        else if (mode == 2) src_d[i] = ceil(src_d[i]); else src_d[i] = trunc(src_d[i]);
                    }
                    memcpy(regs_.xmm[reg], src_d, 16);
                } else { // VROUNDSD
                    double d;
                    if (rm_is_reg) memcpy(&d, regs_.xmm[(int)rm_addr & 15], 8); else { uint64_t v = ReadMemory64(rm_addr); memcpy(&d, &v, 8); }
                    if (mode == 0) d = nearbyint(d); else if (mode == 1) d = floor(d); else if (mode == 2) d = ceil(d); else d = trunc(d);
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src_f, mem_base_ + off, 16); else memset(src_f, 0, 16); }
                // Products of the selected elements, summed pairwise as the hardware does (x86 NaN rules).
                uint32_t p[4];
                for (int i = 0; i < 4; i++)
                    p[i] = (imm & (1 << (i + 4))) ? FpArith<float>(0x59, ToBits<float>(vvvv_f[i]), ToBits<float>(src_f[i])) : 0;
                const float dp = FromBits<float>(FpArith<float>(0x58, FpArith<float>(0x58, p[0], p[1]), FpArith<float>(0x58, p[2], p[3])));
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
                    else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memset(mem_base_ + off, 0, 16); }
                } else {
                    if (rm_is_reg) memcpy(regs_.xmm[(int)rm_addr & 15], regs_.xmm[reg], 16);
                    else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(mem_base_ + off, regs_.xmm[reg], 16); }
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src_w, mem_base_ + off, 16); else memset(src_w, 0, 16); }
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
                else { uint64_t off = rm_addr - mem_base_addr_; if (Mapped(rm_addr, 16)) memcpy(src, mem_base_ + off, 16); else memset(src, 0, 16); }
                uint8_t concat[32];
                memcpy(concat, src, 16);
                memcpy(concat + 16, vvvv_b2, 16);
                if (imm >= 32) memset(regs_.xmm[reg], 0, 16);
                else { for (int i = 0; i < 16; i++) regs_.xmm[reg][i] = (imm + i < 32) ? concat[imm + i] : 0; }
                break;
            }

            case 0x02: case 0x0D: {  // VPBLENDD (dwords) / VBLENDPD (qwords) by imm8 mask
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm);
                uint64_t a; bool is_reg;
                decode_rm(modrm, &a, &is_reg);
                uint8_t imm = read_imm8();
                uint8_t x[16], y[16];
                memcpy(x, regs_.xmm[vex_vvvv & 15], 16);
                ld(a, is_reg, y, 16);
                const int esz = vex_opcode == 0x02 ? 4 : 8;
                // The upper lane of a 256-bit blend uses the next mask bits.
                const int shift = lane ? 16 / esz : 0;
                for (int i = 0; i < 16 / esz; i++)
                    memcpy(regs_.xmm[reg] + i * esz, ((imm >> (i + shift)) & 1) ? y + i * esz : x + i * esz, esz);
                break;
            }
            case 0x1D: {  // VCVTPS2PH xmm/m64, xmm, imm8 (round to nearest even)
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm);
                uint64_t a; bool is_reg;
                decode_rm(modrm, &a, &is_reg);
                read_imm8();
                float f[4]; uint16_t h[4];
                memcpy(f, regs_.xmm[reg], 16);
                for (int i = 0; i < 4; i++) h[i] = FloatToHalf(f[i]);
                if (is_reg) { memset(regs_.xmm[(int)a & 15], 0, 16); memcpy(regs_.xmm[(int)a & 15], h, 8); }
                else if (Mapped(a, 8)) memcpy(mem_base_ + (a - mem_base_addr_), h, 8);
                break;
            }
            case 0x41: {  // VDPPD
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm);
                uint64_t a; bool is_reg;
                decode_rm(modrm, &a, &is_reg);
                uint8_t imm = read_imm8();
                double x[2], y[2], out[2] = {0, 0}, dp = 0;
                memcpy(x, regs_.xmm[vex_vvvv & 15], 16);
                ld(a, is_reg, y, 16);
                for (int i = 0; i < 2; i++) if (imm & (0x10 << i)) dp += x[i] * y[i];
                for (int i = 0; i < 2; i++) if (imm & (1 << i)) out[i] = dp;
                memcpy(regs_.xmm[reg], out, 16);
                break;
            }
            case 0x44: {  // VPCLMULQDQ
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm);
                uint64_t a; bool is_reg;
                decode_rm(modrm, &a, &is_reg);
                uint8_t imm = read_imm8();
                uint64_t x[2], y[2];
                memcpy(x, regs_.xmm[vex_vvvv & 15], 16);
                ld(a, is_reg, y, 16);
                uint64_t p = x[imm & 1], q = y[(imm >> 4) & 1], lo = 0, hi = 0;
                for (int i = 0; i < 64; i++)
                    if ((q >> i) & 1) { lo ^= p << i; if (i) hi ^= p >> (64 - i); }
                uint64_t out[2] = {lo, hi};
                memcpy(regs_.xmm[reg], out, 16);
                break;
            }

            // RORX (F2 prefix)
            case 0xF0: {
                uint8_t modrm = read_modrm();
                int reg = reg_field(modrm);
                uint64_t rm_addr; bool rm_is_reg;
                decode_rm(modrm, &rm_addr, &rm_is_reg);
                uint8_t imm = read_imm8();
                if (vex_W) {
                    uint64_t src = rm_is_reg ? RegByIndex((int)rm_addr) : ReadMemory64(rm_addr);
                    unsigned c = imm & 63;
                    RegByIndex(reg) = c ? (src >> c) | (src << (64 - c)) : src;
                } else {
                    uint32_t src = rm_is_reg ? (uint32_t)RegByIndex((int)rm_addr) : ReadMemory32(rm_addr);
                    unsigned c = imm & 31;
                    RegByIndex(reg) = c ? (uint32_t)((src >> c) | (src << (32 - c))) : src;
                }
                break;
            }

            case 0x15: {  // (V)PEXTRW r/m16, xmm, imm8 (a register destination is zero-extended)
                if (vex_pp != 1) return X86Exception::InvalidOpcode;
                uint8_t modrm = read_modrm();
                uint64_t a; bool is_reg;
                decode_rm(modrm, &a, &is_reg);
                const uint8_t imm = read_imm8();
                uint16_t w;
                memcpy(&w, regs_.xmm[reg_field(modrm)] + 2 * (imm & 7), 2);
                if (is_reg) RegByIndex((int)a) = w;
                else WriteMemory16(a, w);
                break;
            }

            case 0xDF: {  // AESKEYGENASSIST
                if (vex_pp != 1) return X86Exception::InvalidOpcode;
                uint8_t modrm = read_modrm();
                uint64_t a; bool is_reg;
                decode_rm(modrm, &a, &is_reg);
                const uint32_t rcon = read_imm8();
                uint32_t src[4], out[4];
                ld(a, is_reg, src, 16);
                auto sub_word = [](uint32_t x) {
                    uint32_t r = 0;
                    for (int i = 0; i < 4; i++) r |= (uint32_t)Aes().sbox[(x >> (8 * i)) & 0xFF] << (8 * i);
                    return r;
                };
                auto rot_word = [](uint32_t x) { return (x >> 8) | (x << 24); };
                out[0] = sub_word(src[1]);
                out[1] = rot_word(sub_word(src[1])) ^ rcon;
                out[2] = sub_word(src[3]);
                out[3] = rot_word(sub_word(src[3])) ^ rcon;
                memcpy(regs_.xmm[reg_field(modrm)], out, 16);
                break;
            }

            default:
                fprintf(stderr, "[x86] Unimplemented %s 0F3A opcode: pp=%d 0x%02X at RIP=0x%llX\n",
                        legacy_simd ? "SSE" : "VEX", vex_pp, vex_opcode, (unsigned long long)(regs_.rip - 1));
                return X86Exception::InvalidOpcode;
            }
        } else {
            fprintf(stderr, "[x86] Unknown VEX map: %d at RIP=0x%llX\n",
                    vex_mmmmm, (unsigned long long)(regs_.rip - 1));
            return X86Exception::InvalidOpcode;
        }
        }  // lanes

        if (lanes == 2) {
            if (count_operand && op_rm_is_reg && op_rm != op_reg) memcpy(regs_.xmm[op_rm], count_hi, 16);
            swap_lanes();
            lane_mem_offset = 0;
        } else if (!legacy_simd) {
            // VEX.128 (and scalar) writes zero bits 255:128 of the destination register.
            int dest = op_reg;
            if (vex_mmmmm == 1) {
                switch (vex_opcode) {
                    case 0x11: case 0x29: case 0x7F: case 0xD6: dest = op_rm_is_reg ? op_rm : -1; break;
                    case 0x13: case 0x17: case 0x2B: case 0xE7: case 0x2E: case 0x2F: case 0x50:
                    case 0xD7: case 0xC5: case 0x2C: case 0x2D: case 0xAE: dest = -1; break;
                    case 0x7E: dest = vex_pp == 2 ? op_reg : -1; break;
                    case 0x71: case 0x72: case 0x73: dest = vex_vvvv; break;
                    default: break;
                }
            } else if (vex_mmmmm == 2) {
                if (vex_opcode >= 0xF0) dest = -1;
            } else if ((vex_opcode >= 0x14 && vex_opcode <= 0x17) || vex_opcode == 0xF0) {
                dest = -1;
            } else if (vex_opcode == 0x1D) {
                dest = op_rm_is_reg ? op_rm : -1;
            }
            if (dest >= 0 && !full_width_written) memset(regs_.ymm_hi[dest & 15], 0, 16);
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
