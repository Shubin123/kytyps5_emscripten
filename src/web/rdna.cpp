// RDNA (gfx10) shader interpreter: see rdna.h. Instruction encodings follow AMD's RDNA ISA; opcode numbers come
// from rdna_ops.h, generated from LLVM's assembler by tools/rdna/gen_ops.py.

#include "rdna.h"
#include "rdna_ops.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace KytyWeb::Rdna {
namespace {

inline float F(uint32_t b) { float f; memcpy(&f, &b, 4); return f; }
inline uint32_t U(float f) { uint32_t b; memcpy(&b, &f, 4); return b; }
inline double D(uint64_t b) { double d; memcpy(&d, &b, 8); return d; }
inline uint64_t U64(double d) { uint64_t b; memcpy(&b, &d, 8); return b; }
inline int32_t Sext(uint32_t v, int bits) { return (int32_t)(v << (32 - bits)) >> (32 - bits); }

// Scalar operand numbers.
constexpr uint32_t kVccLo = 106, kM0 = 124, kNull = 125, kLiteral = 255;
constexpr uint32_t kSdwa = 249, kDpp16 = 250, kDpp8 = 233, kDpp8Fi = 234;

// VALU opcodes in the VOP3 numbering: VOPC 0x000-0x0ff, VOP2 0x100 + op, VOP1 0x180 + op.
constexpr unsigned V2(unsigned op) { return 0x100 + op; }
constexpr unsigned V1(unsigned op) { return 0x180 + op; }

uint32_t InlineFloat(uint32_t op) {
    static const float k[] = {0.5f, -0.5f, 1.0f, -1.0f, 2.0f, -2.0f, 4.0f, -4.0f, 0.15915494f};
    return U(k[op - 240]);
}
uint64_t InlineDouble(uint32_t op) {
    static const double k[] = {0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 0.15915494309189535};
    return U64(k[op - 240]);
}
uint16_t InlineHalf(uint32_t op) {
    static const uint16_t k[] = {0x3800, 0xb800, 0x3c00, 0xbc00, 0x4000, 0xc000, 0x4400, 0xc400, 0x3118};
    return k[op - 240];
}

}  // namespace

float HalfToFloat(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16, exp = (h >> 10) & 0x1f, mant = h & 0x3ff;
    if (exp == 0) {
        const float f = (float)mant * (1.0f / 16777216.0f);
        return sign ? -f : f;
    }
    const uint32_t bits = exp == 31 ? (sign | 0x7f800000 | (mant << 13) | (mant ? 0x400000 : 0))
                                    : (sign | ((exp + 112) << 23) | (mant << 13));
    return F(bits);
}

uint16_t FloatToHalf(float f) {
    const uint32_t x = U(f);
    const uint32_t sign = (x >> 16) & 0x8000;
    const int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffff;
    if (((x >> 23) & 0xff) == 0xff) return (uint16_t)(sign | 0x7c00 | (mant ? 0x200 : 0));
    if (exp >= 31) return (uint16_t)(sign | 0x7c00);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000;
        const int shift = 14 - exp;
        uint32_t h = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1))) h++;
        return (uint16_t)(sign | h);
    }
    uint32_t h = ((uint32_t)exp << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) h++;
    return (uint16_t)(sign | h);
}

namespace {

// Round toward zero, as v_cvt_pkrtz_f16_f32 converts.
uint16_t FloatToHalfRtz(float f) {
    const uint32_t x = U(f);
    const uint32_t sign = (x >> 16) & 0x8000;
    const int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    const uint32_t mant = x & 0x7fffff;
    if (((x >> 23) & 0xff) == 0xff) return (uint16_t)(sign | 0x7c00 | (mant ? 0x200 : 0));
    if (exp >= 31) return (uint16_t)(sign | 0x7bff);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        return (uint16_t)(sign | ((mant | 0x800000) >> (14 - exp)));
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

// --- buffer data formats (gfx10's unified format numbers) ---

struct FormatInfo {
    uint8_t comps;    // number of components
    uint8_t bits[4];  // bits per component, from the lowest
    uint8_t type;     // 0 unorm, 1 snorm, 2 uscaled, 3 sscaled, 4 uint, 5 sint, 6 float
};
bool LookupFormat(uint32_t fmt, FormatInfo* f) {
    auto set = [&](int comps, int b0, int b1, int b2, int b3, int type) {
        f->comps = (uint8_t)comps;
        f->bits[0] = (uint8_t)b0; f->bits[1] = (uint8_t)b1; f->bits[2] = (uint8_t)b2; f->bits[3] = (uint8_t)b3;
        f->type = (uint8_t)type;
        return true;
    };
    if (fmt >= 1 && fmt <= 6) return set(1, 8, 0, 0, 0, (int)fmt - 1);
    if (fmt >= 7 && fmt <= 12) return set(1, 16, 0, 0, 0, (int)fmt - 7);
    if (fmt == 13) return set(1, 16, 0, 0, 0, 6);
    if (fmt >= 14 && fmt <= 19) return set(2, 8, 8, 0, 0, (int)fmt - 14);
    if (fmt == 20) return set(1, 32, 0, 0, 0, 4);
    if (fmt == 21) return set(1, 32, 0, 0, 0, 5);
    if (fmt == 22) return set(1, 32, 0, 0, 0, 6);
    if (fmt >= 23 && fmt <= 28) return set(2, 16, 16, 0, 0, (int)fmt - 23);
    if (fmt == 29) return set(2, 16, 16, 0, 0, 6);
    // Names list components from the most significant: 11_11_10 has x in its low 10 bits.
    if (fmt >= 30 && fmt <= 36) return set(3, 10, 11, 11, 0, fmt == 36 ? 6 : (int)fmt - 30);
    if (fmt >= 37 && fmt <= 43) return set(3, 11, 11, 10, 0, fmt == 43 ? 6 : (int)fmt - 37);
    if (fmt >= 44 && fmt <= 49) return set(4, 10, 10, 10, 2, (int)fmt - 44);
    if (fmt >= 50 && fmt <= 55) return set(4, 2, 10, 10, 10, (int)fmt - 50);
    if (fmt >= 56 && fmt <= 61) return set(4, 8, 8, 8, 8, (int)fmt - 56);
    if (fmt == 62) return set(2, 32, 32, 0, 0, 4);
    if (fmt == 63) return set(2, 32, 32, 0, 0, 5);
    if (fmt == 64) return set(2, 32, 32, 0, 0, 6);
    if (fmt >= 65 && fmt <= 70) return set(4, 16, 16, 16, 16, (int)fmt - 65);
    if (fmt == 71) return set(4, 16, 16, 16, 16, 6);
    if (fmt == 72) return set(3, 32, 32, 32, 0, 4);
    if (fmt == 73) return set(3, 32, 32, 32, 0, 5);
    if (fmt == 74) return set(3, 32, 32, 32, 0, 6);
    if (fmt == 75) return set(4, 32, 32, 32, 32, 4);
    if (fmt == 76) return set(4, 32, 32, 32, 32, 5);
    if (fmt == 77) return set(4, 32, 32, 32, 32, 6);
    return false;
}

// The unsigned small floats of the 11_11_10 and 10_11_11 formats.
float UnsignedSmallFloat(uint32_t v, int bits) {
    const int mbits = bits == 11 ? 6 : 5;
    const uint32_t exp = v >> mbits, mant = v & ((1u << mbits) - 1);
    if (exp == 0) return (float)mant / (float)(1 << mbits) * (1.0f / 16384.0f);
    if (exp == 31) return mant ? NAN : INFINITY;
    return std::ldexp(1.0f + (float)mant / (float)(1 << mbits), (int)exp - 15);
}

int FormatBytes(const FormatInfo& f) {
    int total = 0;
    for (int c = 0; c < f.comps; c++) total += f.bits[c];
    return total / 8;
}

// Reads one formatted element and expands it to 4 dwords (floats, or integers for the integer types).
void DecodeFormatted(const uint8_t* p, const FormatInfo& f, uint32_t out[4]) {
    uint64_t raw[2] = {0, 0};
    memcpy(raw, p, (size_t)FormatBytes(f));
    int shift = 0;
    for (int c = 0; c < 4; c++) {
        if (c >= f.comps) {
            out[c] = c == 3 ? (f.type == 4 || f.type == 5 ? 1u : U(1.0f)) : 0u;
            continue;
        }
        const int bits = f.bits[c];
        const uint64_t word = raw[shift / 64] >> (shift % 64);
        const uint32_t v = bits == 32 ? (uint32_t)word : (uint32_t)(word & ((1ull << bits) - 1));
        shift += bits;
        const int32_t sv = bits == 32 ? (int32_t)v : Sext(v, bits);
        switch (f.type) {
            case 0: out[c] = U((float)((double)v / (double)((1ull << bits) - 1))); break;
            case 1: out[c] = U(std::max(-1.0f, (float)((double)sv / (double)((1ull << (bits - 1)) - 1)))); break;
            case 2: out[c] = U((float)v); break;
            case 3: out[c] = U((float)sv); break;
            case 4: out[c] = v; break;
            case 5: out[c] = (uint32_t)sv; break;
            default:
                out[c] = bits == 32 ? v : bits == 16 ? U(HalfToFloat((uint16_t)v)) : U(UnsignedSmallFloat(v, bits));
                break;
        }
    }
}

void EncodeFormatted(uint8_t* p, const FormatInfo& f, const uint32_t in[4]) {
    uint64_t raw[2] = {0, 0};
    int shift = 0;
    for (int c = 0; c < f.comps; c++) {
        const int bits = f.bits[c];
        const float x = F(in[c]);
        uint64_t v;
        switch (f.type) {
            case 0: v = (uint64_t)std::lround(std::clamp(x, 0.0f, 1.0f) * (float)((1ull << bits) - 1)); break;
            case 1:
                v = (uint64_t)(int64_t)std::lround(std::clamp(x, -1.0f, 1.0f) * (float)((1ull << (bits - 1)) - 1));
                break;
            case 2: case 3: v = (uint64_t)(int64_t)x; break;
            case 4: case 5: v = in[c]; break;
            default: v = bits == 32 ? in[c] : bits == 16 ? FloatToHalf(x) : 0; break;
        }
        v &= bits == 32 ? 0xffffffffull : (1ull << bits) - 1;
        raw[shift / 64] |= v << (shift % 64);
        shift += bits;
    }
    memcpy(p, raw, (size_t)FormatBytes(f));
}

// Destination swizzle of buffer descriptors and images (0 zero, 1 one, 4-7 x/y/z/w).
void ApplyDstSel(uint32_t sel_bits, bool integer, uint32_t v[4]) {
    uint32_t in[4];
    memcpy(in, v, sizeof(in));
    for (int c = 0; c < 4; c++) {
        const uint32_t sel = (sel_bits >> (3 * c)) & 7;
        v[c] = sel >= 4 ? in[sel - 4] : sel == 1 ? (integer ? 1u : U(1.0f)) : 0u;
    }
}

// --- float helpers ---

float Fract(float x) {
    const float r = x - std::floor(x);
    return r >= 1.0f ? std::nextafter(1.0f, 0.0f) : r;
}
double FractD(double x) {
    const double r = x - std::floor(x);
    return r >= 1.0 ? std::nextafter(1.0, 0.0) : r;
}
// Float to integer conversions saturate and turn NaN into 0.
int32_t CvtI32(double x) {
    if (std::isnan(x)) return 0;
    if (x >= 2147483647.0) return INT32_MAX;
    if (x <= -2147483648.0) return INT32_MIN;
    return (int32_t)x;
}
uint32_t CvtU32(double x) {
    if (std::isnan(x) || x <= 0) return 0;
    if (x >= 4294967295.0) return UINT32_MAX;
    return (uint32_t)x;
}
float Med3(float a, float b, float c) {
    if (std::isnan(a) || std::isnan(b) || std::isnan(c)) return std::fmin(a, std::fmin(b, c));
    return std::max(std::min(a, b), std::min(std::max(a, b), c));
}
template <typename T> T Med3I(T a, T b, T c) { return std::max(std::min(a, b), std::min(std::max(a, b), c)); }

// IEEE comparison by VOPC condition code (f lt eq le gt lg ge o u nge nlg ngt nle neq nlt tru).
template <typename T> bool FCompare(unsigned code, T a, T b) {
    switch (code & 15) {
        case 0: return false;
        case 1: return a < b;
        case 2: return a == b;
        case 3: return a <= b;
        case 4: return a > b;
        case 5: return a < b || a > b;
        case 6: return a >= b;
        case 7: return !std::isnan(a) && !std::isnan(b);
        case 8: return std::isnan(a) || std::isnan(b);
        case 9: return !(a >= b);
        case 10: return !(a < b || a > b);
        case 11: return !(a > b);
        case 12: return !(a <= b);
        case 13: return !(a == b);
        case 14: return !(a < b);
        default: return true;
    }
}
// Integer comparison by code (f lt eq le gt ne ge t).
template <typename T> bool ICompare(unsigned code, T a, T b) {
    switch (code & 7) {
        case 0: return false;
        case 1: return a < b;
        case 2: return a == b;
        case 3: return a <= b;
        case 4: return a > b;
        case 5: return a != b;
        case 6: return a >= b;
        default: return true;
    }
}
// v_cmp_class: mask bits 0 sNaN, 1 qNaN, 2 -inf, 3 -normal, 4 -denorm, 5 -0, 6 +0, 7 +denorm, 8 +normal, 9 +inf.
bool FClass(double x, bool is_snan, uint32_t mask) {
    int bit;
    if (std::isnan(x)) bit = is_snan ? 0 : 1;
    else if (std::isinf(x)) bit = x < 0 ? 2 : 9;
    else if (x == 0) bit = std::signbit(x) ? 5 : 6;
    else if (std::fpclassify(x) == FP_SUBNORMAL) bit = x < 0 ? 4 : 7;
    else bit = x < 0 ? 3 : 8;
    return (mask >> bit) & 1;
}
bool ClassF32(uint32_t b, uint32_t mask) {
    const bool nan = (b & 0x7f800000) == 0x7f800000 && (b & 0x7fffff);
    const float f = F(b);
    if (!nan && (b & 0x7f800000) == 0 && (b & 0x7fffff)) return (mask >> ((b >> 31) ? 4 : 7)) & 1;
    return FClass(f, nan && !(b & 0x400000), mask);
}
bool ClassF64(uint64_t b, uint32_t mask) {
    const bool nan = std::isnan(D(b));
    return FClass(D(b), nan && !(b & (1ull << 51)), mask);
}
bool ClassF16(uint16_t h, uint32_t mask) {
    const uint32_t exp = (h >> 10) & 0x1f, mant = h & 0x3ff;
    const bool neg = h >> 15;
    int bit;
    if (exp == 31) bit = mant ? ((mant & 0x200) ? 1 : 0) : (neg ? 2 : 9);
    else if (exp == 0) bit = mant ? (neg ? 4 : 7) : (neg ? 5 : 6);
    else bit = neg ? 3 : 8;
    return (mask >> bit) & 1;
}

uint32_t Popcount(uint64_t v) { return (uint32_t)__builtin_popcountll(v); }
uint32_t BitReverse32(uint32_t v) {
    v = ((v >> 1) & 0x55555555u) | ((v & 0x55555555u) << 1);
    v = ((v >> 2) & 0x33333333u) | ((v & 0x33333333u) << 2);
    v = ((v >> 4) & 0x0f0f0f0fu) | ((v & 0x0f0f0f0fu) << 4);
    v = ((v >> 8) & 0x00ff00ffu) | ((v & 0x00ff00ffu) << 8);
    return (v >> 16) | (v << 16);
}
uint64_t Wqm(uint64_t v) {
    uint64_t r = 0;
    for (int q = 0; q < 64; q += 4)
        if ((v >> q) & 0xf) r |= 0xfull << q;
    return r;
}

// One decoded vector ALU instruction, in the VOP3 opcode numbering whatever its encoding.
struct VInst {
    unsigned op = 0;
    unsigned vdst = 0;
    unsigned src[3] = {0, 0, 0};  // 9-bit operand numbers (256+ = VGPR)
    unsigned abs = 0, neg = 0;    // per-source input modifiers
    unsigned neg_hi = 0, opsel = 0, opsel_hi = 7;  // VOP3P / 16-bit operand halves
    bool clamp = false;
    unsigned omod = 0;
    unsigned sdst = kVccLo;  // VOPC, carry-out and VOP3b destination
    bool vop3 = false;
    bool sdwa = false;
    uint32_t sdwa_word = 0;
    int dpp = 0;  // 16 or 8
    uint32_t dpp_word = 0;
};

enum class Status { kNext, kEnd, kBarrier, kError };

class Interp {
public:
    Interp(Wave& w, Env& env) : w_(w), env_(env) {}

    Result Run(uint64_t max) {
        Result r;
        while (r.instructions < max) {
            if (!Fetch()) {
                r.ok = false;
                r.error = "instruction fetch failed at " + Hex(w_.pc);
                return r;
            }
            r.instructions++;
            const uint64_t pc = w_.pc;
            jumped_ = false;
            lit_used_ = false;
            base_ = 1;
            if (env_.trace) Snapshot();
            const Status st = Step();
            if (env_.trace) TraceStep(pc);
            if (st == Status::kError) {
                r.ok = false;
                r.error = error_ + " at " + Hex(pc) + " [" + Words() + "]";
                return r;
            }
            if (!jumped_) w_.pc = pc + 4ull * (base_ + (lit_used_ ? 1 : 0));
            if (st == Status::kEnd) return r;
            if (st == Status::kBarrier) {
                r.barrier = true;
                return r;
            }
        }
        r.ok = false;
        r.error = "instruction limit reached at " + Hex(w_.pc);
        return r;
    }

private:
    Wave& w_;
    Env& env_;
    uint32_t ins_[16] = {};
    uint32_t snap_s_[128], snap_v_[256];
    bool snap_scc_ = false;
    void Snapshot() {
        memcpy(snap_s_, w_.s, sizeof(snap_s_));
        for (int i = 0; i < 256; i++) snap_v_[i] = w_.v[i][env_.trace_lane];
        snap_scc_ = w_.scc;
    }
    void TraceStep(uint64_t pc) {
        std::string line = "[rdna] " + Hex(pc) + " " + Words() + " :";
        char buf[64];
        for (int i = 0; i < 128; i++)
            if (w_.s[i] != snap_s_[i]) { snprintf(buf, sizeof(buf), " s%d=%08x", i, w_.s[i]); line += buf; }
        for (int i = 0; i < 256; i++)
            if (w_.v[i][env_.trace_lane] != snap_v_[i]) {
                const uint32_t b = w_.v[i][env_.trace_lane];
                float f;
                memcpy(&f, &b, 4);
                snprintf(buf, sizeof(buf), " v%d=%08x(%g)", i, b, f);
                line += buf;
            }
        if (w_.scc != snap_scc_) line += w_.scc ? " scc=1" : " scc=0";
        printf("%s\n", line.c_str());
    }
    int base_ = 1;           // dwords before a literal would follow
    bool lit_used_ = false;  // an operand used the literal
    bool jumped_ = false;
    std::string error_;

    static std::string Hex(uint64_t v) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%#llx", (unsigned long long)v);
        return buf;
    }
    std::string Words() const {
        std::string s;
        for (int i = 0; i < 3; i++) {
            char buf[16];
            snprintf(buf, sizeof(buf), "%s%08x", i ? " " : "", ins_[i]);
            s += buf;
        }
        return s;
    }
    Status Fail(const char* what) {
        error_ = what;
        return Status::kError;
    }
    Status Unsupported(const char* fmt, unsigned op) {
        char buf[64];
        snprintf(buf, sizeof(buf), "unsupported %s opcode %#x", fmt, op);
        error_ = buf;
        return Status::kError;
    }

    bool Fetch() {
        const uint8_t* p = env_.Read(w_.pc, sizeof(ins_));
        if (p) {
            memcpy(ins_, p, sizeof(ins_));
            return true;
        }
        memset(ins_, 0, sizeof(ins_));
        for (uint64_t n = 12; n >= 4; n -= 4) {
            if ((p = env_.Read(w_.pc, n)) != nullptr) {
                memcpy(ins_, p, n);
                return true;
            }
        }
        return false;
    }

    uint64_t Exec() const { return w_.Exec(); }

    // --- scalar operands ---

    uint32_t S(uint32_t op) {
        if (op < 128) return w_.s[op];
        if (op <= 192) return op - 128;
        if (op <= 208) return (uint32_t)(192 - (int)op);
        if (op >= 240 && op <= 248) return InlineFloat(op);
        switch (op) {
            case 251: return w_.Vcc() == 0;
            case 252: return w_.Exec() == 0;
            case 253: return w_.scc;
            case kLiteral: lit_used_ = true; return ins_[base_];
            default: return 0;
        }
    }
    uint64_t S64(uint32_t op, bool fp64 = false) {
        if (op < 127) return w_.s[op] | (uint64_t)w_.s[op + 1] << 32;
        if (op >= 128 && op <= 208) return (uint64_t)(int64_t)(int32_t)S(op);
        if (op >= 240 && op <= 248) return InlineDouble(op);
        if (op == kLiteral) {
            lit_used_ = true;
            return fp64 ? (uint64_t)ins_[base_] << 32 : ins_[base_];
        }
        return S(op);
    }
    // 16-bit operand: inline float constants are halves.
    uint32_t S16(uint32_t op) { return op >= 240 && op <= 248 ? InlineHalf(op) : S(op); }
    void SetS(uint32_t op, uint32_t v) {
        if (op < 128 && op != kNull) w_.s[op] = v;
    }
    void SetS64(uint32_t op, uint64_t v) {
        SetS(op, (uint32_t)v);
        SetS(op + 1, (uint32_t)(v >> 32));
    }

    Status Step() {
        const uint32_t w0 = ins_[0];
        if ((w0 >> 30) == 2) {
            const uint32_t top = w0 >> 23;
            if (top == 0x17d) return Sop1(w0);
            if (top == 0x17e) return Sopc(w0);
            if (top == 0x17f) return Sopp(w0);
            if ((w0 >> 28) == 0xb) return Sopk(w0);
            return Sop2(w0);
        }
        if ((w0 >> 31) == 0) {
            VInst in;
            const uint32_t src0 = w0 & 0x1ff;
            if ((w0 >> 25) == 0x3f) {  // VOP1
                in.op = V1((w0 >> 9) & 0xff);
                in.vdst = (w0 >> 17) & 0xff;
                in.src[0] = src0;
            } else if ((w0 >> 25) == 0x3e) {  // VOPC
                in.op = (w0 >> 17) & 0xff;
                in.src[0] = src0;
                in.src[1] = 256 + ((w0 >> 9) & 0xff);
            } else {  // VOP2
                const unsigned op = (w0 >> 25) & 0x3f;
                in.op = V2(op);
                in.vdst = (w0 >> 17) & 0xff;
                in.src[0] = src0;
                in.src[1] = 256 + ((w0 >> 9) & 0xff);
                switch (op) {
                    case Op::V_MADMK_F32: case Op::V_FMAMK_F32: case Op::V_FMAMK_F16:  // D = S0 * K + S1
                        in.op = op == Op::V_MADMK_F32 ? Op::V_MAD_F32 : op == Op::V_FMAMK_F32 ? Op::V_FMA_F32 : Op::V_FMA_F16;
                        in.src[2] = in.src[1];
                        in.src[1] = kLiteral;
                        break;
                    case Op::V_MADAK_F32: case Op::V_FMAAK_F32: case Op::V_FMAAK_F16:  // D = S0 * S1 + K
                        in.op = op == Op::V_MADAK_F32 ? Op::V_MAD_F32 : op == Op::V_FMAAK_F32 ? Op::V_FMA_F32 : Op::V_FMA_F16;
                        in.src[2] = kLiteral;
                        break;
                    case Op::V_MAC_F32: case Op::V_MAC_LEGACY_F32: case Op::V_FMAC_F32: case Op::V_FMAC_F16:
                    case Op::V_PK_FMAC_F16:
                        in.src[2] = 256 + in.vdst;
                        break;
                    case Op::V_CNDMASK_B32: case Op::V_ADD_CO_CI_U32: case Op::V_SUB_CO_CI_U32:
                    case Op::V_SUBREV_CO_CI_U32:
                        in.src[2] = kVccLo;
                        break;
                    default: break;
                }
            }
            if (src0 == kSdwa) {
                in.sdwa = true;
                in.sdwa_word = ins_[1];
                base_ = 2;
                const uint32_t sw = in.sdwa_word;
                in.src[0] = (sw & 0xff) + (((sw >> 23) & 1) ? 0 : 256);
                if (in.op >= 0x100 && in.op < 0x180 && ((sw >> 31) & 1)) in.src[1] = (w0 >> 9) & 0xff;
                if (in.op < 0x100 && ((sw >> 31) & 1)) in.src[1] = (w0 >> 9) & 0xff;
                in.neg = ((sw >> 20) & 1) | (((sw >> 28) & 1) << 1);
                in.abs = ((sw >> 21) & 1) | (((sw >> 29) & 1) << 1);
                if (in.op < 0x100) {
                    if ((sw >> 15) & 1) in.sdst = (sw >> 8) & 0x7f;
                } else {
                    in.clamp = (sw >> 13) & 1;
                    in.omod = (sw >> 14) & 3;
                }
            } else if (src0 == kDpp16 || src0 == kDpp8 || src0 == kDpp8Fi) {
                in.dpp = src0 == kDpp16 ? 16 : 8;
                in.dpp_word = ins_[1];
                base_ = 2;
                in.src[0] = 256 + (in.dpp_word & 0xff);
                if (in.dpp == 16) {
                    in.neg = ((in.dpp_word >> 20) & 1) | (((in.dpp_word >> 22) & 1) << 1);
                    in.abs = ((in.dpp_word >> 21) & 1) | (((in.dpp_word >> 23) & 1) << 1);
                }
            }
            // VOP2 forms with a constant K always carry the literal.
            if (in.src[1] == kLiteral || in.src[2] == kLiteral) lit_used_ = true;
            return Valu(in);
        }
        switch (w0 >> 26) {
            case 0x32: return Vintrp(w0);
            case 0x33: return Vop3p(w0, ins_[1]);
            case 0x35: return Vop3(w0, ins_[1]);
            case 0x36: return Ds(w0, ins_[1]);
            case 0x37: return Flat(w0, ins_[1]);
            case 0x38: return Mubuf(w0, ins_[1], false);
            case 0x3a: return Mubuf(w0, ins_[1], true);
            case 0x3c: return Mimg(w0, ins_[1]);
            case 0x3d: return Smem(w0, ins_[1]);
            case 0x3e: return Exp(w0, ins_[1]);
            default: return Fail("unknown instruction encoding");
        }
    }

    // --- scalar ALU ---

    Status Sop2(uint32_t w0) {
        const uint32_t op = (w0 >> 23) & 0x7f, sdst = (w0 >> 16) & 0x7f, s0 = w0 & 0xff, s1 = (w0 >> 8) & 0xff;
        bool& scc = w_.scc;
        switch (op) {
            case Op::S_CSELECT_B64: SetS64(sdst, scc ? S64(s0) : S64(s1)); return Status::kNext;
            case Op::S_AND_B64: case Op::S_OR_B64: case Op::S_XOR_B64: case Op::S_ANDN2_B64: case Op::S_ORN2_B64:
            case Op::S_NAND_B64: case Op::S_NOR_B64: case Op::S_XNOR_B64: {
                const uint64_t a = S64(s0), b = S64(s1);
                uint64_t r;
                switch (op) {
                    case Op::S_AND_B64: r = a & b; break;
                    case Op::S_OR_B64: r = a | b; break;
                    case Op::S_XOR_B64: r = a ^ b; break;
                    case Op::S_ANDN2_B64: r = a & ~b; break;
                    case Op::S_ORN2_B64: r = a | ~b; break;
                    case Op::S_NAND_B64: r = ~(a & b); break;
                    case Op::S_NOR_B64: r = ~(a | b); break;
                    default: r = ~(a ^ b); break;
                }
                SetS64(sdst, r);
                scc = r != 0;
                return Status::kNext;
            }
            case Op::S_LSHL_B64: case Op::S_LSHR_B64: case Op::S_ASHR_I64: {
                const uint64_t a = S64(s0);
                const uint32_t n = S(s1) & 63;
                const uint64_t r = op == Op::S_LSHL_B64 ? a << n : op == Op::S_LSHR_B64 ? a >> n
                                                                                      : (uint64_t)((int64_t)a >> n);
                SetS64(sdst, r);
                scc = r != 0;
                return Status::kNext;
            }
            case Op::S_BFM_B64: {
                const uint32_t width = S(s0) & 63, offset = S(s1) & 63;
                SetS64(sdst, (width == 0 ? 0 : (~0ull >> (64 - width))) << offset);
                return Status::kNext;
            }
            case Op::S_BFE_U64: case Op::S_BFE_I64: {
                const uint64_t a = S64(s0);
                const uint32_t b = S(s1), offset = b & 63, width = std::min((b >> 16) & 0x7f, 64u);
                uint64_t r = width == 0 ? 0 : (a >> offset) & (width >= 64 ? ~0ull : (1ull << width) - 1);
                if (op == Op::S_BFE_I64 && width > 0 && width < 64) r = (uint64_t)((int64_t)(r << (64 - width)) >> (64 - width));
                SetS64(sdst, r);
                scc = r != 0;
                return Status::kNext;
            }
            default: break;
        }
        const uint32_t a = S(s0), b = S(s1);
        uint32_t r = 0;
        switch (op) {
            case Op::S_ADD_U32: { const uint64_t t = (uint64_t)a + b; r = (uint32_t)t; scc = t >> 32; break; }
            case Op::S_SUB_U32: r = a - b; scc = b > a; break;
            case Op::S_ADD_I32: r = a + b; scc = ((~(a ^ b) & (a ^ r)) >> 31) != 0; break;
            case Op::S_SUB_I32: r = a - b; scc = (((a ^ b) & (a ^ r)) >> 31) != 0; break;
            case Op::S_ADDC_U32: { const uint64_t t = (uint64_t)a + b + scc; r = (uint32_t)t; scc = t >> 32; break; }
            case Op::S_SUBB_U32: { const uint64_t t = (uint64_t)a - b - scc; r = (uint32_t)t; scc = (t >> 32) != 0; break; }
            case Op::S_MIN_I32: scc = (int32_t)a < (int32_t)b; r = scc ? a : b; break;
            case Op::S_MIN_U32: scc = a < b; r = scc ? a : b; break;
            case Op::S_MAX_I32: scc = (int32_t)a > (int32_t)b; r = scc ? a : b; break;
            case Op::S_MAX_U32: scc = a > b; r = scc ? a : b; break;
            case Op::S_CSELECT_B32: SetS(sdst, scc ? a : b); return Status::kNext;
            case Op::S_AND_B32: r = a & b; scc = r != 0; break;
            case Op::S_OR_B32: r = a | b; scc = r != 0; break;
            case Op::S_XOR_B32: r = a ^ b; scc = r != 0; break;
            case Op::S_ANDN2_B32: r = a & ~b; scc = r != 0; break;
            case Op::S_ORN2_B32: r = a | ~b; scc = r != 0; break;
            case Op::S_NAND_B32: r = ~(a & b); scc = r != 0; break;
            case Op::S_NOR_B32: r = ~(a | b); scc = r != 0; break;
            case Op::S_XNOR_B32: r = ~(a ^ b); scc = r != 0; break;
            case Op::S_LSHL_B32: r = a << (b & 31); scc = r != 0; break;
            case Op::S_LSHR_B32: r = a >> (b & 31); scc = r != 0; break;
            case Op::S_ASHR_I32: r = (uint32_t)((int32_t)a >> (b & 31)); scc = r != 0; break;
            case Op::S_BFM_B32: SetS(sdst, (uint32_t)(((1ull << (a & 31)) - 1) << (b & 31))); return Status::kNext;
            case Op::S_MUL_I32: SetS(sdst, a * b); return Status::kNext;
            case Op::S_BFE_U32: case Op::S_BFE_I32: {
                const uint32_t offset = b & 31, width = std::min((b >> 16) & 0x7f, 32u);
                r = width == 0 ? 0 : (uint32_t)(((uint64_t)a >> offset) & ((1ull << width) - 1));
                if (op == Op::S_BFE_I32 && width > 0 && width < 32) r = (uint32_t)Sext(r, (int)width);
                scc = r != 0;
                break;
            }
            case Op::S_ABSDIFF_I32: {
                const int64_t d = (int64_t)(int32_t)a - (int32_t)b;
                r = (uint32_t)(d < 0 ? -d : d);
                scc = r != 0;
                break;
            }
            case Op::S_LSHL1_ADD_U32: case Op::S_LSHL2_ADD_U32: case Op::S_LSHL3_ADD_U32: case Op::S_LSHL4_ADD_U32: {
                const uint64_t t = ((uint64_t)a << (op - Op::S_LSHL1_ADD_U32 + 1)) + b;
                r = (uint32_t)t;
                scc = (t >> 32) != 0;
                break;
            }
            case Op::S_PACK_LL_B32_B16: SetS(sdst, (a & 0xffff) | (b << 16)); return Status::kNext;
            case Op::S_PACK_LH_B32_B16: SetS(sdst, (a & 0xffff) | (b & 0xffff0000)); return Status::kNext;
            case Op::S_PACK_HH_B32_B16: SetS(sdst, (a >> 16) | (b & 0xffff0000)); return Status::kNext;
            case Op::S_MUL_HI_U32: SetS(sdst, (uint32_t)(((uint64_t)a * b) >> 32)); return Status::kNext;
            case Op::S_MUL_HI_I32:
                SetS(sdst, (uint32_t)(((int64_t)(int32_t)a * (int32_t)b) >> 32));
                return Status::kNext;
            default: return Unsupported("SOP2", op);
        }
        SetS(sdst, r);
        return Status::kNext;
    }

    Status Sopk(uint32_t w0) {
        const uint32_t op = (w0 >> 23) & 0x1f, sdst = (w0 >> 16) & 0x7f, imm = w0 & 0xffff;
        const uint32_t simm = (uint32_t)Sext(imm, 16);
        bool& scc = w_.scc;
        const uint32_t d = S(sdst);
        switch (op) {
            case Op::S_MOVK_I32: SetS(sdst, simm); break;
            case Op::S_CMOVK_I32: if (scc) SetS(sdst, simm); break;
            case Op::S_CMPK_EQ_I32: case Op::S_CMPK_LG_I32: case Op::S_CMPK_GT_I32: case Op::S_CMPK_GE_I32:
            case Op::S_CMPK_LT_I32: case Op::S_CMPK_LE_I32: {
                static const unsigned kCode[] = {2, 5, 4, 6, 1, 3};
                scc = ICompare(kCode[op - Op::S_CMPK_EQ_I32], (int32_t)d, (int32_t)simm);
                break;
            }
            case Op::S_CMPK_EQ_U32: case Op::S_CMPK_LG_U32: case Op::S_CMPK_GT_U32: case Op::S_CMPK_GE_U32:
            case Op::S_CMPK_LT_U32: case Op::S_CMPK_LE_U32: {
                static const unsigned kCode[] = {2, 5, 4, 6, 1, 3};
                scc = ICompare(kCode[op - Op::S_CMPK_EQ_U32], d, imm);
                break;
            }
            case Op::S_ADDK_I32: {
                const uint32_t r = d + simm;
                scc = ((~(d ^ simm) & (d ^ r)) >> 31) != 0;
                SetS(sdst, r);
                break;
            }
            case Op::S_MULK_I32: SetS(sdst, d * simm); break;
            case Op::S_GETREG_B32: SetS(sdst, 0); break;
            case Op::S_SETREG_B32: break;
            case Op::S_SETREG_IMM32_B32: lit_used_ = true; break;
            case Op::S_CALL_B64:
                SetS64(sdst, w_.pc + 4);
                w_.pc = w_.pc + 4 + (int64_t)(int32_t)simm * 4;
                jumped_ = true;
                break;
            case Op::S_WAITCNT_VSCNT: case Op::S_WAITCNT_VMCNT: case Op::S_WAITCNT_EXPCNT:
            case Op::S_WAITCNT_LGKMCNT: case Op::S_VERSION:
                break;
            default: return Unsupported("SOPK", op);
        }
        return Status::kNext;
    }

    Status Sop1(uint32_t w0) {
        const uint32_t op = (w0 >> 8) & 0xff, sdst = (w0 >> 16) & 0x7f, s0 = w0 & 0xff;
        bool& scc = w_.scc;
        auto saveexec = [&](uint64_t (*f)(uint64_t, uint64_t), bool write_dst_new) {
            const uint64_t s = S64(s0), exec = w_.Exec();
            const uint64_t r = f(s, exec);
            if (!write_dst_new) SetS64(sdst, exec);
            w_.SetExec(r);
            if (write_dst_new) SetS64(sdst, r);
            scc = r != 0;
        };
        switch (op) {
            case Op::S_MOV_B32: SetS(sdst, S(s0)); break;
            case Op::S_MOV_B64: SetS64(sdst, S64(s0)); break;
            case Op::S_CMOV_B32: { const uint32_t v = S(s0); if (scc) SetS(sdst, v); break; }
            case Op::S_CMOV_B64: { const uint64_t v = S64(s0); if (scc) SetS64(sdst, v); break; }
            case Op::S_NOT_B32: { const uint32_t r = ~S(s0); SetS(sdst, r); scc = r != 0; break; }
            case Op::S_NOT_B64: { const uint64_t r = ~S64(s0); SetS64(sdst, r); scc = r != 0; break; }
            case Op::S_WQM_B32: { const uint32_t r = (uint32_t)Wqm(S(s0)); SetS(sdst, r); scc = r != 0; break; }
            case Op::S_WQM_B64: { const uint64_t r = Wqm(S64(s0)); SetS64(sdst, r); scc = r != 0; break; }
            case Op::S_BREV_B32: SetS(sdst, BitReverse32(S(s0))); break;
            case Op::S_BREV_B64: {
                const uint64_t v = S64(s0);
                SetS64(sdst, (uint64_t)BitReverse32((uint32_t)v) << 32 | BitReverse32((uint32_t)(v >> 32)));
                break;
            }
            case Op::S_BCNT0_I32_B32: { const uint32_t r = 32 - Popcount(S(s0)); SetS(sdst, r); scc = r != 0; break; }
            case Op::S_BCNT0_I32_B64: { const uint32_t r = 64 - Popcount(S64(s0)); SetS(sdst, r); scc = r != 0; break; }
            case Op::S_BCNT1_I32_B32: { const uint32_t r = Popcount(S(s0)); SetS(sdst, r); scc = r != 0; break; }
            case Op::S_BCNT1_I32_B64: { const uint32_t r = Popcount(S64(s0)); SetS(sdst, r); scc = r != 0; break; }
            case Op::S_FF0_I32_B32: { const uint32_t v = ~S(s0); SetS(sdst, v ? __builtin_ctz(v) : ~0u); break; }
            case Op::S_FF0_I32_B64: { const uint64_t v = ~S64(s0); SetS(sdst, v ? __builtin_ctzll(v) : ~0u); break; }
            case Op::S_FF1_I32_B32: { const uint32_t v = S(s0); SetS(sdst, v ? __builtin_ctz(v) : ~0u); break; }
            case Op::S_FF1_I32_B64: { const uint64_t v = S64(s0); SetS(sdst, v ? __builtin_ctzll(v) : ~0u); break; }
            case Op::S_FLBIT_I32_B32: { const uint32_t v = S(s0); SetS(sdst, v ? __builtin_clz(v) : ~0u); break; }
            case Op::S_FLBIT_I32_B64: { const uint64_t v = S64(s0); SetS(sdst, v ? __builtin_clzll(v) : ~0u); break; }
            case Op::S_FLBIT_I32: {
                const uint32_t v = S(s0), x = (int32_t)v < 0 ? ~v : v;
                SetS(sdst, x ? __builtin_clz(x) : ~0u);
                break;
            }
            case Op::S_FLBIT_I32_I64: {
                const uint64_t v = S64(s0), x = (int64_t)v < 0 ? ~v : v;
                SetS(sdst, x ? __builtin_clzll(x) : ~0u);
                break;
            }
            case Op::S_SEXT_I32_I8: SetS(sdst, (uint32_t)Sext(S(s0), 8)); break;
            case Op::S_SEXT_I32_I16: SetS(sdst, (uint32_t)Sext(S(s0), 16)); break;
            case Op::S_BITSET0_B32: SetS(sdst, S(sdst) & ~(1u << (S(s0) & 31))); break;
            case Op::S_BITSET1_B32: SetS(sdst, S(sdst) | (1u << (S(s0) & 31))); break;
            case Op::S_BITSET0_B64: SetS64(sdst, S64(sdst) & ~(1ull << (S(s0) & 63))); break;
            case Op::S_BITSET1_B64: SetS64(sdst, S64(sdst) | (1ull << (S(s0) & 63))); break;
            case Op::S_GETPC_B64: SetS64(sdst, w_.pc + 4); break;
            case Op::S_SETPC_B64: w_.pc = S64(s0); jumped_ = true; break;
            case Op::S_SWAPPC_B64: {
                const uint64_t target = S64(s0);
                SetS64(sdst, w_.pc + 4);
                w_.pc = target;
                jumped_ = true;
                break;
            }
            case Op::S_AND_SAVEEXEC_B64: saveexec([](uint64_t s, uint64_t e) { return s & e; }, false); break;
            case Op::S_OR_SAVEEXEC_B64: saveexec([](uint64_t s, uint64_t e) { return s | e; }, false); break;
            case Op::S_XOR_SAVEEXEC_B64: saveexec([](uint64_t s, uint64_t e) { return s ^ e; }, false); break;
            case Op::S_ANDN2_SAVEEXEC_B64: saveexec([](uint64_t s, uint64_t e) { return s & ~e; }, false); break;
            case Op::S_ORN2_SAVEEXEC_B64: saveexec([](uint64_t s, uint64_t e) { return s | ~e; }, false); break;
            case Op::S_NAND_SAVEEXEC_B64: saveexec([](uint64_t s, uint64_t e) { return ~(s & e); }, false); break;
            case Op::S_NOR_SAVEEXEC_B64: saveexec([](uint64_t s, uint64_t e) { return ~(s | e); }, false); break;
            case Op::S_XNOR_SAVEEXEC_B64: saveexec([](uint64_t s, uint64_t e) { return ~(s ^ e); }, false); break;
            case Op::S_ANDN1_SAVEEXEC_B64: saveexec([](uint64_t s, uint64_t e) { return ~s & e; }, false); break;
            case Op::S_ORN1_SAVEEXEC_B64: saveexec([](uint64_t s, uint64_t e) { return ~s | e; }, false); break;
            case Op::S_ANDN1_WREXEC_B64: saveexec([](uint64_t s, uint64_t e) { return ~s & e; }, true); break;
            case Op::S_ANDN2_WREXEC_B64: saveexec([](uint64_t s, uint64_t e) { return s & ~e; }, true); break;
            case Op::S_QUADMASK_B32: case Op::S_QUADMASK_B64: {
                const uint64_t v = op == Op::S_QUADMASK_B32 ? S(s0) : S64(s0);
                uint64_t r = 0;
                for (int q = 0; q < 16; q++)
                    if ((v >> (4 * q)) & 0xf) r |= 1ull << q;
                if (op == Op::S_QUADMASK_B32) SetS(sdst, (uint32_t)r); else SetS64(sdst, r);
                scc = r != 0;
                break;
            }
            case Op::S_MOVRELS_B32: SetS(sdst, S((s0 + w_.s[kM0]) & 0x7f)); break;
            case Op::S_MOVRELS_B64: SetS64(sdst, S64((s0 + w_.s[kM0]) & 0x7f)); break;
            case Op::S_MOVRELD_B32: SetS((sdst + w_.s[kM0]) & 0x7f, S(s0)); break;
            case Op::S_MOVRELD_B64: SetS64((sdst + w_.s[kM0]) & 0x7f, S64(s0)); break;
            case Op::S_ABS_I32: {
                const int32_t v = (int32_t)S(s0);
                const uint32_t r = v < 0 ? (uint32_t)0 - (uint32_t)v : (uint32_t)v;
                SetS(sdst, r);
                scc = r != 0;
                break;
            }
            case Op::S_BITREPLICATE_B64_B32: {
                const uint32_t v = S(s0);
                uint64_t r = 0;
                for (int i = 0; i < 32; i++)
                    if ((v >> i) & 1) r |= 3ull << (2 * i);
                SetS64(sdst, r);
                break;
            }
            default: return Unsupported("SOP1", op);
        }
        return Status::kNext;
    }

    Status Sopc(uint32_t w0) {
        const uint32_t op = (w0 >> 16) & 0x7f, s0 = w0 & 0xff, s1 = (w0 >> 8) & 0xff;
        bool& scc = w_.scc;
        if (op >= Op::S_CMP_EQ_I32 && op <= Op::S_CMP_LE_I32) {
            static const unsigned kCode[] = {2, 5, 4, 6, 1, 3};
            scc = ICompare(kCode[op - Op::S_CMP_EQ_I32], (int32_t)S(s0), (int32_t)S(s1));
            return Status::kNext;
        }
        if (op >= Op::S_CMP_EQ_U32 && op <= Op::S_CMP_LE_U32) {
            static const unsigned kCode[] = {2, 5, 4, 6, 1, 3};
            scc = ICompare(kCode[op - Op::S_CMP_EQ_U32], S(s0), S(s1));
            return Status::kNext;
        }
        switch (op) {
            case Op::S_BITCMP0_B32: scc = ((S(s0) >> (S(s1) & 31)) & 1) == 0; break;
            case Op::S_BITCMP1_B32: scc = ((S(s0) >> (S(s1) & 31)) & 1) != 0; break;
            case Op::S_BITCMP0_B64: scc = ((S64(s0) >> (S(s1) & 63)) & 1) == 0; break;
            case Op::S_BITCMP1_B64: scc = ((S64(s0) >> (S(s1) & 63)) & 1) != 0; break;
            case Op::S_CMP_EQ_U64: scc = S64(s0) == S64(s1); break;
            case Op::S_CMP_LG_U64: scc = S64(s0) != S64(s1); break;
            default: return Unsupported("SOPC", op);
        }
        return Status::kNext;
    }

    Status Sopp(uint32_t w0) {
        const uint32_t op = (w0 >> 16) & 0x7f, imm = w0 & 0xffff;
        auto branch = [&](bool taken) {
            if (taken) {
                w_.pc = w_.pc + 4 + (int64_t)Sext(imm, 16) * 4;
                jumped_ = true;
            }
        };
        switch (op) {
            case Op::S_ENDPGM: case Op::S_ENDPGM_SAVED: case Op::S_ENDPGM_ORDERED_PS_DONE: case Op::S_CODE_END:
                return Status::kEnd;
            case Op::S_BRANCH: branch(true); break;
            case Op::S_CBRANCH_SCC0: branch(!w_.scc); break;
            case Op::S_CBRANCH_SCC1: branch(w_.scc); break;
            case Op::S_CBRANCH_VCCZ: branch(w_.Vcc() == 0); break;
            case Op::S_CBRANCH_VCCNZ: branch(w_.Vcc() != 0); break;
            case Op::S_CBRANCH_EXECZ: branch(w_.Exec() == 0); break;
            case Op::S_CBRANCH_EXECNZ: branch(w_.Exec() != 0); break;
            case Op::S_BARRIER: return Status::kBarrier;
            case Op::S_SENDMSG: case Op::S_SENDMSGHALT: env_.SendMsg(imm, w_.s[kM0]); break;
            case Op::S_NOP: case Op::S_WAKEUP: case Op::S_SETKILL: case Op::S_WAITCNT: case Op::S_SETHALT:
            case Op::S_SLEEP: case Op::S_SETPRIO: case Op::S_TRAP: case Op::S_ICACHE_INV: case Op::S_INCPERFLEVEL:
            case Op::S_DECPERFLEVEL: case Op::S_TTRACEDATA: case Op::S_INST_PREFETCH: case Op::S_CLAUSE:
            case Op::S_ROUND_MODE: case Op::S_DENORM_MODE: case Op::S_TTRACEDATA_IMM:
                break;
            default: return Unsupported("SOPP", op);
        }
        return Status::kNext;
    }

    // --- scalar memory ---

    Status Smem(uint32_t w0, uint32_t w1) {
        base_ = 2;
        const uint32_t op = (w0 >> 18) & 0xff, sdata = (w0 >> 6) & 0x7f, sbase = (w0 & 0x3f) * 2;
        const uint32_t soff = (w1 >> 25) & 0x7f;
        int64_t offset = Sext(w1 & 0x1fffff, 21);
        if (soff != kNull) offset += S(soff);
        int count;
        bool buffer = false;
        switch (op) {
            case Op::S_LOAD_DWORD: count = 1; break;
            case Op::S_LOAD_DWORDX2: count = 2; break;
            case Op::S_LOAD_DWORDX4: count = 4; break;
            case Op::S_LOAD_DWORDX8: count = 8; break;
            case Op::S_LOAD_DWORDX16: count = 16; break;
            case Op::S_BUFFER_LOAD_DWORD: count = 1; buffer = true; break;
            case Op::S_BUFFER_LOAD_DWORDX2: count = 2; buffer = true; break;
            case Op::S_BUFFER_LOAD_DWORDX4: count = 4; buffer = true; break;
            case Op::S_BUFFER_LOAD_DWORDX8: count = 8; buffer = true; break;
            case Op::S_BUFFER_LOAD_DWORDX16: count = 16; buffer = true; break;
            case Op::S_MEMTIME: case Op::S_MEMREALTIME: SetS64(sdata, ++clock_ * 64); return Status::kNext;
            case Op::S_DCACHE_INV: case Op::S_GL1_INV: return Status::kNext;
            default: return Unsupported("SMEM", op);
        }
        uint64_t base, range = ~0ull;
        if (buffer) {
            base = w_.s[sbase] | (uint64_t)(w_.s[sbase + 1] & 0xffff) << 32;
            const uint32_t stride = (w_.s[sbase + 1] >> 16) & 0x3fff, records = w_.s[sbase + 2];
            range = stride ? (uint64_t)records * stride : records;
        } else {
            base = (w_.s[sbase] | (uint64_t)w_.s[sbase + 1] << 32) & 0xffffffffffffull;
        }
        const uint64_t first = (uint64_t)offset & ~3ull;
        const uint8_t* p = env_.Read(base + first, 4ull * count);
        for (int i = 0; i < count; i++) {
            uint32_t v = 0;
            const uint64_t at = first + 4ull * i;
            if (at + 4 <= range) {
                if (p) memcpy(&v, p + 4 * i, 4);
                else if (const uint8_t* q = env_.Read(base + at, 4)) memcpy(&v, q, 4);
            }
            SetS(sdata + i, v);
        }
        return Status::kNext;
    }
    uint64_t clock_ = 0;

    // --- vector ALU ---

    // Source rows: VGPR rows are used in place; scalar and constant sources are broadcast into tmp_.
    uint32_t tmp_[6][kLanes];

    const uint32_t* Row(unsigned operand, int slot, bool half = false) {
        if (operand >= 256) return w_.v[operand - 256];
        const uint32_t v = half ? S16(operand) : S(operand);
        uint32_t* t = tmp_[slot];
        for (int l = 0; l < kLanes; l++) t[l] = v;
        return t;
    }
    // 64-bit sources as two rows (lo, hi).
    void Row64(unsigned operand, int slot, const uint32_t** lo, const uint32_t** hi, bool fp64) {
        if (operand >= 256) {
            *lo = w_.v[operand - 256];
            *hi = w_.v[(operand - 256 + 1) & 0xff];
            return;
        }
        const uint64_t v = S64(operand, fp64);
        uint32_t* a = tmp_[slot];
        uint32_t* b = tmp_[slot + 3];
        for (int l = 0; l < kLanes; l++) {
            a[l] = (uint32_t)v;
            b[l] = (uint32_t)(v >> 32);
        }
        *lo = a;
        *hi = b;
    }

    static bool IsF64Op(unsigned op) {
        switch (op) {
            case Op::V_FMA_F64: case Op::V_DIV_FIXUP_F64: case Op::V_ADD_F64: case Op::V_MUL_F64: case Op::V_MIN_F64:
            case Op::V_MAX_F64: case Op::V_LDEXP_F64: case Op::V_DIV_SCALE_F64: case Op::V_DIV_FMAS_F64:
            case Op::V_TRIG_PREOP_F64:
            case V1(Op::V_CVT_I32_F64): case V1(Op::V_CVT_F32_F64): case V1(Op::V_CVT_U32_F64):
            case V1(Op::V_TRUNC_F64): case V1(Op::V_CEIL_F64): case V1(Op::V_RNDNE_F64): case V1(Op::V_FLOOR_F64):
            case V1(Op::V_RCP_F64): case V1(Op::V_RSQ_F64): case V1(Op::V_SQRT_F64): case V1(Op::V_FREXP_EXP_I32_F64):
            case V1(Op::V_FREXP_MANT_F64): case V1(Op::V_FRACT_F64):
                return true;
            default:
                return false;
        }
    }

    // SDWA operand selection (BYTE_0-3, WORD_0-1, DWORD) with optional sign extension.
    static uint32_t SdwaSelect(uint32_t v, unsigned sel, bool sext) {
        switch (sel) {
            case 0: case 1: case 2: case 3: {
                const uint32_t b = (v >> (8 * sel)) & 0xff;
                return sext ? (uint32_t)Sext(b, 8) : b;
            }
            case 4: case 5: {
                const uint32_t h = (v >> (16 * (sel - 4))) & 0xffff;
                return sext ? (uint32_t)Sext(h, 16) : h;
            }
            default: return v;
        }
    }

    // DPP16 source lane for lane l, or -1 when it reads nothing.
    static int DppLane(uint32_t ctrl, int l) {
        const int row = l & ~15, in = l & 15;
        if (ctrl <= 0xff) return (l & ~3) | ((ctrl >> (2 * (l & 3))) & 3);
        if (ctrl >= 0x101 && ctrl <= 0x10f) { const int s = in + (int)(ctrl & 15); return s < 16 ? row + s : -1; }
        if (ctrl >= 0x111 && ctrl <= 0x11f) { const int s = in - (int)(ctrl & 15); return s >= 0 ? row + s : -1; }
        if (ctrl >= 0x121 && ctrl <= 0x12f) return row + ((in - (int)(ctrl & 15)) & 15);
        if (ctrl == 0x140) return row + (15 - in);
        if (ctrl == 0x141) return row + ((in & 8) | (7 - (in & 7)));
        if (ctrl >= 0x150 && ctrl <= 0x15f) return row + (int)(ctrl & 15);
        if (ctrl >= 0x160 && ctrl <= 0x16f) return row + (in ^ (int)(ctrl & 15));
        return -1;
    }

    Status Vop3(uint32_t w0, uint32_t w1) {
        base_ = 2;
        VInst in;
        in.vop3 = true;
        in.op = (w0 >> 16) & 0x3ff;
        in.vdst = w0 & 0xff;
        in.src[0] = w1 & 0x1ff;
        in.src[1] = (w1 >> 9) & 0x1ff;
        in.src[2] = (w1 >> 18) & 0x1ff;
        in.omod = (w1 >> 27) & 3;
        in.neg = (w1 >> 29) & 7;
        in.clamp = (w0 >> 15) & 1;
        switch (in.op) {
            case Op::V_ADD_CO_U32: case Op::V_SUB_CO_U32: case Op::V_SUBREV_CO_U32: case V2(Op::V_ADD_CO_CI_U32):
            case V2(Op::V_SUB_CO_CI_U32): case V2(Op::V_SUBREV_CO_CI_U32): case Op::V_DIV_SCALE_F32:
            case Op::V_DIV_SCALE_F64: case Op::V_MAD_U64_U32: case Op::V_MAD_I64_I32:
                in.sdst = (w0 >> 8) & 0x7f;
                break;
            default:
                in.abs = (w0 >> 8) & 7;
                in.opsel = (w0 >> 11) & 0xf;
                if (in.op < 0x100) in.sdst = in.vdst;
                break;
        }
        switch (in.op) {
            case Op::V_INTERP_P1_F32_E64: case Op::V_INTERP_P2_F32_E64: case Op::V_INTERP_MOV_F32_E64: {
                const unsigned attr = in.src[0] & 0x3f, chan = (in.src[0] >> 6) & 3;
                return Interp1(in.op - Op::V_INTERP_P1_F32_E64, in.vdst, in.src[1], attr, chan, in);
            }
            default: break;
        }
        return Valu(in);
    }

    Status Valu(VInst& in);
    Status Vop3p(uint32_t w0, uint32_t w1);

    Status Vintrp(uint32_t w0) {
        const unsigned vsrc = w0 & 0xff, chan = (w0 >> 8) & 3, attr = (w0 >> 10) & 0x3f, op = (w0 >> 16) & 3;
        const unsigned vdst = (w0 >> 18) & 0xff;
        VInst in;
        return Interp1(op, vdst, op == 2 ? vsrc : 256 + vsrc, attr, chan, in);
    }

    // v_interp_p1 (D = P0 + P10 * I), p2 (D += P20 * J) and mov (D = P0, P10 or P20).
    Status Interp1(unsigned op, unsigned vdst, unsigned src, unsigned attr, unsigned chan, const VInst& mods) {
        float p[3];
        env_.Attribute((int)attr, (int)chan, p);
        const uint64_t exec = w_.Exec();
        uint32_t* d = w_.v[vdst];
        const float p10 = p[1] - p[0], p20 = p[2] - p[0];
        if (op == 2) {
            const float v = (src & 3) == 2 ? p[0] : (src & 3) == 0 ? p10 : p20;
            for (int l = 0; l < kLanes; l++)
                if ((exec >> l) & 1) d[l] = U(v);
            return Status::kNext;
        }
        const uint32_t* s = w_.v[(src - 256) & 0xff];
        for (int l = 0; l < kLanes; l++) {
            if (!((exec >> l) & 1)) continue;
            float ij = F(s[l]);
            if (mods.abs & 2) ij = std::fabs(ij);
            if (mods.neg & 2) ij = -ij;
            float r = op == 0 ? p[0] + p10 * ij : F(d[l]) + p20 * ij;
            if (mods.clamp) r = std::clamp(r, 0.0f, 1.0f);
            d[l] = U(r);
        }
        return Status::kNext;
    }

    Status Ds(uint32_t w0, uint32_t w1);
    Status Flat(uint32_t w0, uint32_t w1);
    Status Mubuf(uint32_t w0, uint32_t w1, bool typed);
    Status Mimg(uint32_t w0, uint32_t w1);
    Status Exp(uint32_t w0, uint32_t w1);
};

Status Interp::Valu(VInst& in) {
    const uint64_t exec = w_.Exec();
    const unsigned op = in.op;
    const bool f64 = IsF64Op(op);
    const bool cmp = op < 0x100;

    // Gather 32-bit sources (3 rows). 64-bit operands are fetched per op below.
    const uint32_t* r[3] = {nullptr, nullptr, nullptr};
    const bool half_ops = (op >= V2(Op::V_ADD_F16) && op <= V2(Op::V_LDEXP_F16)) || op == Op::V_FMA_F16 ||
                          op == Op::V_DIV_FIXUP_F16 || (op >= 0xc8 && op <= 0xcf) || (op >= 0xd8 && op <= 0xdf) ||
                          (op >= 0xe8 && op <= 0xef) || (op >= 0xf8 && op <= 0xff) || op == Op::V_CMP_CLASS_F16 ||
                          op == Op::V_CMPX_CLASS_F16 || (op >= V1(Op::V_CVT_F16_U16) && op <= V1(Op::V_CVT_NORM_U16_F16)) ||
                          op == V1(Op::V_CVT_F32_F16);
    if (!f64) {
        for (int i = 0; i < 3; i++) r[i] = Row(in.src[i], i, half_ops);
    }
    uint32_t sel_rows[2][kLanes];
    if (in.sdwa) {
        const uint32_t sw = in.sdwa_word;
        const unsigned sel[2] = {(sw >> 16) & 7, (sw >> 24) & 7};
        const bool sext[2] = {((sw >> 19) & 1) != 0, ((sw >> 27) & 1) != 0};
        for (int i = 0; i < 2; i++) {
            if (sel[i] == 6) continue;
            for (int l = 0; l < kLanes; l++) sel_rows[i][l] = SdwaSelect(r[i][l], sel[i], sext[i]);
            r[i] = sel_rows[i];
        }
    }
    uint64_t write_mask = exec;
    if (in.dpp) {
        // A lane whose source lane doesn't exist or is inactive reads 0 with bound_ctrl, otherwise isn't
        // written; fetch-inactive (fi) reads inactive lanes anyway.
        const uint32_t* src = r[0];
        const bool fi = in.dpp == 16 ? ((in.dpp_word >> 18) & 1) != 0 : ins_[0] % 512 == kDpp8Fi;
        const bool bound_zero = in.dpp == 16 && ((in.dpp_word >> 19) & 1);
        for (int l = 0; l < kLanes; l++) {
            int from;
            if (in.dpp == 8) {
                from = (l & ~7) | (int)((in.dpp_word >> (8 + 3 * (l & 7))) & 7);
            } else {
                from = DppLane((in.dpp_word >> 8) & 0x1ff, l);
                const unsigned row_mask = (in.dpp_word >> 28) & 0xf, bank_mask = (in.dpp_word >> 24) & 0xf;
                if (!((row_mask >> (l >> 4)) & 1) || !((bank_mask >> ((l >> 2) & 3)) & 1)) write_mask &= ~(1ull << l);
            }
            if (from >= 0 && (fi || ((exec >> from) & 1))) {
                sel_rows[0][l] = src[from];
            } else {
                sel_rows[0][l] = 0;
                if (!bound_zero) write_mask &= ~(1ull << l);
            }
        }
        r[0] = sel_rows[0];
    }

    auto fin = [&](int i, int l) -> float {
        uint32_t b = r[i][l];
        if ((in.abs >> i) & 1) b &= 0x7fffffffu;
        if ((in.neg >> i) & 1) b ^= 0x80000000u;
        return F(b);
    };
    // 16-bit inputs: opsel picks the high half in VOP3.
    auto hin = [&](int i, int l) -> uint16_t {
        uint32_t b = r[i][l];
        if ((in.opsel >> i) & 1) b >>= 16;
        uint16_t h = (uint16_t)b;
        if ((in.abs >> i) & 1) h &= 0x7fff;
        if ((in.neg >> i) & 1) h ^= 0x8000;
        return h;
    };
    auto hf = [&](int i, int l) { return HalfToFloat(hin(i, l)); };

    uint32_t res[kLanes];
    uint32_t res_hi[kLanes];
    enum class Kind { kV32, kV64, kMask, kNone, kHalf } kind = Kind::kV32;
    bool fop = false;  // float result: output modifiers apply
    uint64_t mask = 0, carry = 0;
    bool cmpx = false;

#define LANES for (int l = 0; l < kLanes; l++) if ((exec >> l) & 1)

    if (cmp) {
        kind = Kind::kMask;
        const unsigned code = op & 15;
        if (op <= 0x3f) {
            cmpx = (op & 0x10) != 0;
            if (op < 0x20) {
                LANES if (FCompare(code, fin(0, l), fin(1, l))) mask |= 1ull << l;
            } else {
                const uint32_t *alo, *ahi, *blo, *bhi;
                Row64(in.src[0], 0, &alo, &ahi, true);
                Row64(in.src[1], 1, &blo, &bhi, true);
                LANES {
                    double a = D(alo[l] | (uint64_t)ahi[l] << 32), b = D(blo[l] | (uint64_t)bhi[l] << 32);
                    if (in.abs & 1) a = std::fabs(a);
                    if (in.neg & 1) a = -a;
                    if (in.abs & 2) b = std::fabs(b);
                    if (in.neg & 2) b = -b;
                    if (FCompare(code, a, b)) mask |= 1ull << l;
                }
            }
        } else if ((op >= 0xc8 && op <= 0xcf) || (op >= 0xd8 && op <= 0xdf) || (op >= 0xe8 && op <= 0xef) ||
                   (op >= 0xf8 && op <= 0xff)) {
            cmpx = (op & 0x10) != 0;
            const unsigned c16 = (op & 7) | (op >= 0xe8 ? 8 : 0);
            LANES if (FCompare(c16, hf(0, l), hf(1, l))) mask |= 1ull << l;
        } else {
            const unsigned group = op & 0xf0, low = op & 0xf;
            cmpx = (group & 0x10) != 0;
            if (low == 8) {  // class
                const uint32_t* mrow = r[1];
                if (group == 0x80 || group == 0x90) {
                    LANES if (ClassF32(r[0][l] & ((in.abs & 1) ? 0x7fffffffu : ~0u), mrow[l])) mask |= 1ull << l;
                } else if (group == 0xa0 || group == 0xb0) {
                    const uint32_t *alo, *ahi;
                    Row64(in.src[0], 0, &alo, &ahi, true);
                    LANES if (ClassF64(alo[l] | (uint64_t)ahi[l] << 32, mrow[l])) mask |= 1ull << l;
                }
            } else if (low == 0xf && (group == 0x80 || group == 0x90)) {
                LANES if (ClassF16(hin(0, l), r[1][l])) mask |= 1ull << l;
            } else if (low < 8) {
                const unsigned c = low;
                switch (group) {
                    case 0x80: case 0x90: LANES if (ICompare(c, (int32_t)r[0][l], (int32_t)r[1][l])) mask |= 1ull << l; break;
                    case 0xc0: case 0xd0: LANES if (ICompare(c, r[0][l], r[1][l])) mask |= 1ull << l; break;
                    case 0xa0: case 0xb0: case 0xe0: case 0xf0: {
                        const uint32_t *alo, *ahi, *blo, *bhi;
                        Row64(in.src[0], 0, &alo, &ahi, false);
                        Row64(in.src[1], 1, &blo, &bhi, false);
                        const bool sgn = group == 0xa0 || group == 0xb0;
                        LANES {
                            const uint64_t a = alo[l] | (uint64_t)ahi[l] << 32, b = blo[l] | (uint64_t)bhi[l] << 32;
                            if (sgn ? ICompare(c, (int64_t)a, (int64_t)b) : ICompare(c, a, b)) mask |= 1ull << l;
                        }
                        break;
                    }
                    default: return Unsupported("VOPC", op);
                }
            } else {
                // 16-bit integer compares: 0x89-0x8e signed (codes 1-6), 0xa9-0xae unsigned.
                const unsigned c = low - 8;
                if (group == 0x80 || group == 0x90) {
                    LANES if (ICompare(c, (int16_t)hin(0, l), (int16_t)hin(1, l))) mask |= 1ull << l;
                } else if (group == 0xa0 || group == 0xb0) {
                    LANES if (ICompare(c, hin(0, l), hin(1, l))) mask |= 1ull << l;
                } else {
                    return Unsupported("VOPC", op);
                }
            }
        }
    } else if (f64) {
        kind = Kind::kV64;
        fop = true;
        const uint32_t *lo[3], *hi[3];
        const int nsrc = op >= 0x180 ? 1 : 3;
        for (int i = 0; i < nsrc; i++) {
            const bool is_fp = !(op == Op::V_LDEXP_F64 && i == 1) && !(op == Op::V_TRIG_PREOP_F64 && i == 1);
            if (is_fp) Row64(in.src[i], i, &lo[i], &hi[i], true);
            else { lo[i] = Row(in.src[i], i); hi[i] = lo[i]; }
        }
        auto din = [&](int i, int l) {
            double d = D(lo[i][l] | (uint64_t)hi[i][l] << 32);
            if ((in.abs >> i) & 1) d = std::fabs(d);
            if ((in.neg >> i) & 1) d = -d;
            return d;
        };
        auto put = [&](int l, double d) {
            const uint64_t b = U64(d);
            res[l] = (uint32_t)b;
            res_hi[l] = (uint32_t)(b >> 32);
        };
        switch (op) {
            case Op::V_FMA_F64: case Op::V_DIV_FMAS_F64: LANES put(l, std::fma(din(0, l), din(1, l), din(2, l))); break;
            case Op::V_DIV_FIXUP_F64: LANES put(l, din(2, l) / din(1, l)); break;
            case Op::V_ADD_F64: LANES put(l, din(0, l) + din(1, l)); break;
            case Op::V_MUL_F64: LANES put(l, din(0, l) * din(1, l)); break;
            case Op::V_MIN_F64: LANES put(l, std::fmin(din(0, l), din(1, l))); break;
            case Op::V_MAX_F64: LANES put(l, std::fmax(din(0, l), din(1, l))); break;
            case Op::V_LDEXP_F64: LANES put(l, std::ldexp(din(0, l), (int32_t)lo[1][l])); break;
            case Op::V_DIV_SCALE_F64: LANES put(l, din(0, l)); break;
            case V1(Op::V_CVT_I32_F64): kind = Kind::kV32; fop = false; LANES res[l] = (uint32_t)CvtI32(din(0, l)); break;
            case V1(Op::V_CVT_U32_F64): kind = Kind::kV32; fop = false; LANES res[l] = CvtU32(din(0, l)); break;
            case V1(Op::V_CVT_F32_F64): kind = Kind::kV32; LANES res[l] = U((float)din(0, l)); break;
            case V1(Op::V_TRUNC_F64): LANES put(l, std::trunc(din(0, l))); break;
            case V1(Op::V_CEIL_F64): LANES put(l, std::ceil(din(0, l))); break;
            case V1(Op::V_RNDNE_F64): LANES put(l, std::nearbyint(din(0, l))); break;
            case V1(Op::V_FLOOR_F64): LANES put(l, std::floor(din(0, l))); break;
            case V1(Op::V_RCP_F64): LANES put(l, 1.0 / din(0, l)); break;
            case V1(Op::V_RSQ_F64): LANES put(l, 1.0 / std::sqrt(din(0, l))); break;
            case V1(Op::V_SQRT_F64): LANES put(l, std::sqrt(din(0, l))); break;
            case V1(Op::V_FRACT_F64): LANES put(l, FractD(din(0, l))); break;
            case V1(Op::V_FREXP_MANT_F64): LANES { int e; put(l, std::frexp(din(0, l), &e)); } break;
            case V1(Op::V_FREXP_EXP_I32_F64):
                kind = Kind::kV32;
                fop = false;
                LANES { int e = 0; const double d = din(0, l); if (std::isfinite(d)) std::frexp(d, &e); res[l] = (uint32_t)e; }
                break;
            default: return Unsupported("VALU f64", op);
        }
    } else {
        switch (op) {
            // --- f32 arithmetic ---
            case V2(Op::V_ADD_F32): fop = true; LANES res[l] = U(fin(0, l) + fin(1, l)); break;
            case V2(Op::V_SUB_F32): fop = true; LANES res[l] = U(fin(0, l) - fin(1, l)); break;
            case V2(Op::V_SUBREV_F32): fop = true; LANES res[l] = U(fin(1, l) - fin(0, l)); break;
            case V2(Op::V_MUL_F32): fop = true; LANES res[l] = U(fin(0, l) * fin(1, l)); break;
            case V2(Op::V_MUL_LEGACY_F32): fop = true;
                LANES { const float a = fin(0, l), b = fin(1, l); res[l] = U(a == 0 || b == 0 ? 0.0f : a * b); }
                break;
            case V2(Op::V_MAC_LEGACY_F32): case Op::V_MAD_LEGACY_F32: fop = true;
                LANES { const float a = fin(0, l), b = fin(1, l); res[l] = U((a == 0 || b == 0 ? 0.0f : a * b) + fin(2, l)); }
                break;
            case V2(Op::V_MAC_F32): case Op::V_MAD_F32: fop = true; LANES res[l] = U(fin(0, l) * fin(1, l) + fin(2, l)); break;
            case V2(Op::V_FMAC_F32): case Op::V_FMA_F32: fop = true;
                LANES res[l] = U(std::fma(fin(0, l), fin(1, l), fin(2, l)));
                break;
            case V2(Op::V_MIN_F32): fop = true; LANES res[l] = U(std::fmin(fin(0, l), fin(1, l))); break;
            case V2(Op::V_MAX_F32): fop = true; LANES res[l] = U(std::fmax(fin(0, l), fin(1, l))); break;
            case Op::V_MIN3_F32: fop = true; LANES res[l] = U(std::fmin(fin(0, l), std::fmin(fin(1, l), fin(2, l)))); break;
            case Op::V_MAX3_F32: fop = true; LANES res[l] = U(std::fmax(fin(0, l), std::fmax(fin(1, l), fin(2, l)))); break;
            case Op::V_MED3_F32: fop = true; LANES res[l] = U(Med3(fin(0, l), fin(1, l), fin(2, l))); break;
            case Op::V_LDEXP_F32: fop = true; LANES res[l] = U(std::ldexp(fin(0, l), (int32_t)r[1][l])); break;
            case Op::V_DIV_FIXUP_F32: fop = true; LANES res[l] = U(fin(2, l) / fin(1, l)); break;
            case Op::V_DIV_FMAS_F32: fop = true; LANES res[l] = U(std::fma(fin(0, l), fin(1, l), fin(2, l))); break;
            case Op::V_DIV_SCALE_F32: fop = true; LANES res[l] = U(fin(0, l)); break;
            case Op::V_CUBEID_F32: case Op::V_CUBESC_F32: case Op::V_CUBETC_F32: case Op::V_CUBEMA_F32:
                fop = true;
                LANES {
                    const float x = fin(0, l), y = fin(1, l), z = fin(2, l);
                    const float ax = std::fabs(x), ay = std::fabs(y), az = std::fabs(z);
                    float id, sc, tc, ma;
                    if (az >= ax && az >= ay) { id = z < 0 ? 5.0f : 4.0f; sc = z < 0 ? -x : x; tc = -y; ma = 2 * z; }
                    else if (ay >= ax) { id = y < 0 ? 3.0f : 2.0f; sc = x; tc = y < 0 ? -z : z; ma = 2 * y; }
                    else { id = x < 0 ? 1.0f : 0.0f; sc = x < 0 ? z : -z; tc = -y; ma = 2 * x; }
                    const float v = op == Op::V_CUBEID_F32 ? id : op == Op::V_CUBESC_F32 ? sc
                                  : op == Op::V_CUBETC_F32 ? tc : ma;
                    res[l] = U(v);
                }
                break;
            case V2(Op::V_CVT_PKRTZ_F16_F32):
                LANES res[l] = FloatToHalfRtz(fin(0, l)) | (uint32_t)FloatToHalfRtz(fin(1, l)) << 16;
                break;
            case Op::V_CVT_PKNORM_I16_F32:
                LANES {
                    auto n = [](float f) { return (uint32_t)(uint16_t)(int16_t)std::lround(std::clamp(f, -1.0f, 1.0f) * 32767.0f); };
                    res[l] = n(fin(0, l)) | n(fin(1, l)) << 16;
                }
                break;
            case Op::V_CVT_PKNORM_U16_F32:
                LANES {
                    auto n = [](float f) { return (uint32_t)std::lround(std::clamp(f, 0.0f, 1.0f) * 65535.0f); };
                    res[l] = n(fin(0, l)) | n(fin(1, l)) << 16;
                }
                break;
            case Op::V_CVT_PK_U16_U32: LANES res[l] = std::min(r[0][l], 0xffffu) | std::min(r[1][l], 0xffffu) << 16; break;
            case Op::V_CVT_PK_I16_I32:
                LANES {
                    auto n = [](uint32_t v) { return (uint32_t)(uint16_t)std::clamp((int32_t)v, -32768, 32767); };
                    res[l] = n(r[0][l]) | n(r[1][l]) << 16;
                }
                break;
            case Op::V_CVT_PK_U8_F32:
                LANES {
                    const uint32_t byte = r[1][l] & 3;
                    const uint32_t v = CvtU32(std::nearbyint(fin(0, l))) & 0xff;
                    res[l] = (r[2][l] & ~(0xffu << (8 * byte))) | v << (8 * byte);
                }
                break;
            case V2(Op::V_CNDMASK_B32): {
                const uint32_t *mlo, *mhi;
                if (in.vop3) Row64(in.src[2], 2, &mlo, &mhi, false);
                const uint64_t vcc = in.vop3 ? 0 : w_.Vcc();
                LANES {
                    const bool c = in.vop3 ? (((l < 32 ? mlo[l] : mhi[l]) >> (l & 31)) & 1) : ((vcc >> l) & 1);
                    uint32_t a = r[0][l], b = r[1][l];
                    if (in.abs & 1) a &= 0x7fffffff;
                    if (in.neg & 1) a ^= 0x80000000;
                    if (in.abs & 2) b &= 0x7fffffff;
                    if (in.neg & 2) b ^= 0x80000000;
                    res[l] = c ? b : a;
                }
                break;
            }
            // --- integer ---
            case V2(Op::V_MUL_I32_I24): LANES res[l] = (uint32_t)(Sext(r[0][l], 24) * Sext(r[1][l], 24)); break;
            case V2(Op::V_MUL_HI_I32_I24):
                LANES res[l] = (uint32_t)(((int64_t)Sext(r[0][l], 24) * Sext(r[1][l], 24)) >> 32);
                break;
            case V2(Op::V_MUL_U32_U24): LANES res[l] = (r[0][l] & 0xffffff) * (r[1][l] & 0xffffff); break;
            case V2(Op::V_MUL_HI_U32_U24):
                LANES res[l] = (uint32_t)(((uint64_t)(r[0][l] & 0xffffff) * (r[1][l] & 0xffffff)) >> 32);
                break;
            case V2(Op::V_MIN_I32): LANES res[l] = (uint32_t)std::min((int32_t)r[0][l], (int32_t)r[1][l]); break;
            case V2(Op::V_MAX_I32): LANES res[l] = (uint32_t)std::max((int32_t)r[0][l], (int32_t)r[1][l]); break;
            case V2(Op::V_MIN_U32): LANES res[l] = std::min(r[0][l], r[1][l]); break;
            case V2(Op::V_MAX_U32): LANES res[l] = std::max(r[0][l], r[1][l]); break;
            case V2(Op::V_LSHRREV_B32): LANES res[l] = r[1][l] >> (r[0][l] & 31); break;
            case V2(Op::V_ASHRREV_I32): LANES res[l] = (uint32_t)((int32_t)r[1][l] >> (r[0][l] & 31)); break;
            case V2(Op::V_LSHLREV_B32): LANES res[l] = r[1][l] << (r[0][l] & 31); break;
            case V2(Op::V_AND_B32): LANES res[l] = r[0][l] & r[1][l]; break;
            case V2(Op::V_OR_B32): LANES res[l] = r[0][l] | r[1][l]; break;
            case V2(Op::V_XOR_B32): LANES res[l] = r[0][l] ^ r[1][l]; break;
            case V2(Op::V_XNOR_B32): LANES res[l] = ~(r[0][l] ^ r[1][l]); break;
            case V2(Op::V_ADD_NC_U32): case Op::V_ADD_NC_I32:
                LANES {
                    const uint64_t t = (uint64_t)r[0][l] + r[1][l];
                    res[l] = in.clamp && (t >> 32) ? 0xffffffffu : (uint32_t)t;
                }
                break;
            case V2(Op::V_SUB_NC_U32): case Op::V_SUB_NC_I32:
                LANES res[l] = in.clamp && r[1][l] > r[0][l] ? 0 : r[0][l] - r[1][l];
                break;
            case V2(Op::V_SUBREV_NC_U32): LANES res[l] = in.clamp && r[0][l] > r[1][l] ? 0 : r[1][l] - r[0][l]; break;
            case Op::V_ADD_CO_U32:
                LANES { const uint64_t t = (uint64_t)r[0][l] + r[1][l]; res[l] = (uint32_t)t; if (t >> 32) carry |= 1ull << l; }
                break;
            case Op::V_SUB_CO_U32:
                LANES { res[l] = r[0][l] - r[1][l]; if (r[1][l] > r[0][l]) carry |= 1ull << l; }
                break;
            case Op::V_SUBREV_CO_U32:
                LANES { res[l] = r[1][l] - r[0][l]; if (r[0][l] > r[1][l]) carry |= 1ull << l; }
                break;
            case V2(Op::V_ADD_CO_CI_U32): case V2(Op::V_SUB_CO_CI_U32): case V2(Op::V_SUBREV_CO_CI_U32): {
                const uint32_t *clo, *chi;
                Row64(in.src[2], 2, &clo, &chi, false);
                LANES {
                    const uint32_t ci = ((l < 32 ? clo[l] : chi[l]) >> (l & 31)) & 1;
                    uint32_t a = r[0][l], b = r[1][l];
                    if (op == V2(Op::V_SUBREV_CO_CI_U32)) std::swap(a, b);
                    if (op == V2(Op::V_ADD_CO_CI_U32)) {
                        const uint64_t t = (uint64_t)a + b + ci;
                        res[l] = (uint32_t)t;
                        if (t >> 32) carry |= 1ull << l;
                    } else {
                        const uint64_t t = (uint64_t)a - b - ci;
                        res[l] = (uint32_t)t;
                        if (t >> 32) carry |= 1ull << l;
                    }
                }
                break;
            }
            case Op::V_MAD_I32_I24: LANES res[l] = (uint32_t)(Sext(r[0][l], 24) * Sext(r[1][l], 24)) + r[2][l]; break;
            case Op::V_MAD_U32_U24: LANES res[l] = (r[0][l] & 0xffffff) * (r[1][l] & 0xffffff) + r[2][l]; break;
            case Op::V_BFE_U32:
                LANES {
                    const uint32_t off = r[1][l] & 31, width = r[2][l] & 31;
                    res[l] = width == 0 ? 0 : (r[0][l] >> off) & ((1u << width) - 1);
                }
                break;
            case Op::V_BFE_I32:
                LANES {
                    const uint32_t off = r[1][l] & 31, width = r[2][l] & 31;
                    res[l] = width == 0 ? 0 : (uint32_t)Sext((r[0][l] >> off) & ((1u << width) - 1), (int)width);
                }
                break;
            case Op::V_BFI_B32: LANES res[l] = (r[0][l] & r[1][l]) | (~r[0][l] & r[2][l]); break;
            case Op::V_ALIGNBIT_B32:
                LANES res[l] = (uint32_t)((((uint64_t)r[0][l] << 32) | r[1][l]) >> (r[2][l] & 31));
                break;
            case Op::V_ALIGNBYTE_B32:
                LANES res[l] = (uint32_t)((((uint64_t)r[0][l] << 32) | r[1][l]) >> (8 * (r[2][l] & 3)));
                break;
            case Op::V_MIN3_I32:
                LANES res[l] = (uint32_t)std::min({(int32_t)r[0][l], (int32_t)r[1][l], (int32_t)r[2][l]});
                break;
            case Op::V_MAX3_I32:
                LANES res[l] = (uint32_t)std::max({(int32_t)r[0][l], (int32_t)r[1][l], (int32_t)r[2][l]});
                break;
            case Op::V_MED3_I32: LANES res[l] = (uint32_t)Med3I((int32_t)r[0][l], (int32_t)r[1][l], (int32_t)r[2][l]); break;
            case Op::V_MIN3_U32: LANES res[l] = std::min({r[0][l], r[1][l], r[2][l]}); break;
            case Op::V_MAX3_U32: LANES res[l] = std::max({r[0][l], r[1][l], r[2][l]}); break;
            case Op::V_MED3_U32: LANES res[l] = Med3I(r[0][l], r[1][l], r[2][l]); break;
            case Op::V_SAD_U32:
                LANES res[l] = (r[0][l] > r[1][l] ? r[0][l] - r[1][l] : r[1][l] - r[0][l]) + r[2][l];
                break;
            case Op::V_SAD_U8: case Op::V_MSAD_U8:
                LANES {
                    uint32_t s = r[2][l];
                    for (int b = 0; b < 4; b++) {
                        const int x = (r[0][l] >> (8 * b)) & 0xff, y = (r[1][l] >> (8 * b)) & 0xff;
                        if (op == Op::V_MSAD_U8 && y == 0) continue;
                        s += (uint32_t)std::abs(x - y);
                    }
                    res[l] = s;
                }
                break;
            case Op::V_LERP_U8:
                LANES {
                    uint32_t v = 0;
                    for (int b = 0; b < 4; b++) {
                        const uint32_t x = (r[0][l] >> (8 * b)) & 0xff, y = (r[1][l] >> (8 * b)) & 0xff;
                        v |= ((x + y + ((r[2][l] >> (8 * b)) & 1)) >> 1) << (8 * b);
                    }
                    res[l] = v;
                }
                break;
            case Op::V_MUL_LO_U32: LANES res[l] = r[0][l] * r[1][l]; break;
            case Op::V_MUL_HI_U32: LANES res[l] = (uint32_t)(((uint64_t)r[0][l] * r[1][l]) >> 32); break;
            case Op::V_MUL_HI_I32:
                LANES res[l] = (uint32_t)(((int64_t)(int32_t)r[0][l] * (int32_t)r[1][l]) >> 32);
                break;
            case Op::V_MAD_U64_U32: case Op::V_MAD_I64_I32: {
                const uint32_t *clo, *chi;
                Row64(in.src[2], 2, &clo, &chi, false);
                kind = Kind::kV64;
                LANES {
                    const uint64_t c = clo[l] | (uint64_t)chi[l] << 32;
                    const uint64_t p = op == Op::V_MAD_U64_U32 ? (uint64_t)r[0][l] * r[1][l]
                                                               : (uint64_t)((int64_t)(int32_t)r[0][l] * (int32_t)r[1][l]);
                    const uint64_t t = p + c;
                    if (t < c) carry |= 1ull << l;
                    res[l] = (uint32_t)t;
                    res_hi[l] = (uint32_t)(t >> 32);
                }
                break;
            }
            case Op::V_XOR3_B32: LANES res[l] = r[0][l] ^ r[1][l] ^ r[2][l]; break;
            case Op::V_LSHL_ADD_U32: LANES res[l] = (r[0][l] << (r[1][l] & 31)) + r[2][l]; break;
            case Op::V_ADD_LSHL_U32: LANES res[l] = (r[0][l] + r[1][l]) << (r[2][l] & 31); break;
            case Op::V_ADD3_U32: LANES res[l] = r[0][l] + r[1][l] + r[2][l]; break;
            case Op::V_LSHL_OR_B32: LANES res[l] = (r[0][l] << (r[1][l] & 31)) | r[2][l]; break;
            case Op::V_AND_OR_B32: LANES res[l] = (r[0][l] & r[1][l]) | r[2][l]; break;
            case Op::V_OR3_B32: LANES res[l] = r[0][l] | r[1][l] | r[2][l]; break;
            case Op::V_MAD_U32_U16: LANES res[l] = (uint32_t)hin(0, l) * hin(1, l) + r[2][l]; break;
            case Op::V_MAD_I32_I16: LANES res[l] = (uint32_t)((int32_t)(int16_t)hin(0, l) * (int16_t)hin(1, l)) + r[2][l]; break;
            case Op::V_LSHLREV_B64: case Op::V_LSHRREV_B64: case Op::V_ASHRREV_I64: {
                const uint32_t *lo, *hi;
                Row64(in.src[1], 1, &lo, &hi, false);
                kind = Kind::kV64;
                LANES {
                    const uint64_t v = lo[l] | (uint64_t)hi[l] << 32;
                    const uint32_t n = r[0][l] & 63;
                    const uint64_t t = op == Op::V_LSHLREV_B64 ? v << n : op == Op::V_LSHRREV_B64 ? v >> n
                                                                                               : (uint64_t)((int64_t)v >> n);
                    res[l] = (uint32_t)t;
                    res_hi[l] = (uint32_t)(t >> 32);
                }
                break;
            }
            case Op::V_BFM_B32: LANES res[l] = (uint32_t)(((1ull << (r[0][l] & 31)) - 1) << (r[1][l] & 31)); break;
            case Op::V_BCNT_U32_B32: LANES res[l] = Popcount(r[0][l]) + r[1][l]; break;
            case Op::V_MBCNT_LO_U32_B32:
                LANES res[l] = Popcount(r[0][l] & (l >= 32 ? 0xffffffffu : (1u << l) - 1)) + r[1][l];
                break;
            case Op::V_MBCNT_HI_U32_B32:
                LANES res[l] = Popcount(r[0][l] & (l < 32 ? 0u : l == 32 ? 0u : (uint32_t)((1ull << (l - 32)) - 1))) + r[1][l];
                break;
            case Op::V_READLANE_B32:
                SetS(in.vdst, r[0][S(in.src[1]) & 63]);
                kind = Kind::kNone;
                break;
            case Op::V_WRITELANE_B32:
                w_.v[in.vdst][S(in.src[1]) & 63] = S(in.src[0]);
                kind = Kind::kNone;
                break;
            case V1(Op::V_READFIRSTLANE_B32): {
                const int first = exec ? __builtin_ctzll(exec) : 0;
                SetS(in.vdst, r[0][first]);
                kind = Kind::kNone;
                break;
            }
            case Op::V_PERMLANE16_B32: case Op::V_PERMLANEX16_B32: {
                const uint64_t sel = S(in.src[1]) | (uint64_t)S(in.src[2]) << 32;
                const uint32_t* src = r[0];
                LANES {
                    const int row = l & ~15;
                    const int lane = (int)((sel >> (4 * (l & 15))) & 15);
                    const int from = op == Op::V_PERMLANE16_B32 ? row + lane : ((row ^ 16) & 63) + lane;
                    res[l] = src[from];
                }
                break;
            }
            // --- VOP1 ---
            case V1(Op::V_NOP): case V1(Op::V_PIPEFLUSH): case V1(Op::V_CLREXCP): kind = Kind::kNone; break;
            case V1(Op::V_MOV_B32):
                LANES {
                    uint32_t b = r[0][l];
                    if (in.abs & 1) b &= 0x7fffffff;
                    if (in.neg & 1) b ^= 0x80000000;
                    res[l] = b;
                }
                break;
            case V1(Op::V_CVT_F32_I32): fop = true; LANES res[l] = U((float)(int32_t)r[0][l]); break;
            case V1(Op::V_CVT_F32_U32): fop = true; LANES res[l] = U((float)r[0][l]); break;
            case V1(Op::V_CVT_U32_F32): LANES res[l] = CvtU32(fin(0, l)); break;
            case V1(Op::V_CVT_I32_F32): LANES res[l] = (uint32_t)CvtI32(fin(0, l)); break;
            case V1(Op::V_CVT_F16_F32): LANES res[l] = FloatToHalf(fin(0, l)); break;
            case V1(Op::V_CVT_F32_F16): fop = true; LANES res[l] = U(hf(0, l)); break;
            case V1(Op::V_CVT_RPI_I32_F32): LANES res[l] = (uint32_t)CvtI32(std::floor(fin(0, l) + 0.5f)); break;
            case V1(Op::V_CVT_FLR_I32_F32): LANES res[l] = (uint32_t)CvtI32(std::floor(fin(0, l))); break;
            case V1(Op::V_CVT_OFF_F32_I4): fop = true; LANES res[l] = U((float)Sext(r[0][l] & 15, 4) / 16.0f); break;
            case V1(Op::V_CVT_F64_I32): case V1(Op::V_CVT_F64_U32): case V1(Op::V_CVT_F64_F32): {
                kind = Kind::kV64;
                LANES {
                    const double d = op == V1(Op::V_CVT_F64_I32) ? (double)(int32_t)r[0][l]
                                   : op == V1(Op::V_CVT_F64_U32) ? (double)r[0][l] : (double)fin(0, l);
                    const uint64_t b = U64(d);
                    res[l] = (uint32_t)b;
                    res_hi[l] = (uint32_t)(b >> 32);
                }
                break;
            }
            case V1(Op::V_CVT_F32_UBYTE0): case V1(Op::V_CVT_F32_UBYTE1): case V1(Op::V_CVT_F32_UBYTE2):
            case V1(Op::V_CVT_F32_UBYTE3): {
                const int b = (int)(op - V1(Op::V_CVT_F32_UBYTE0));
                fop = true;
                LANES res[l] = U((float)((r[0][l] >> (8 * b)) & 0xff));
                break;
            }
            case V1(Op::V_FRACT_F32): fop = true; LANES res[l] = U(Fract(fin(0, l))); break;
            case V1(Op::V_TRUNC_F32): fop = true; LANES res[l] = U(std::trunc(fin(0, l))); break;
            case V1(Op::V_CEIL_F32): fop = true; LANES res[l] = U(std::ceil(fin(0, l))); break;
            case V1(Op::V_RNDNE_F32): fop = true; LANES res[l] = U(std::nearbyint(fin(0, l))); break;
            case V1(Op::V_FLOOR_F32): fop = true; LANES res[l] = U(std::floor(fin(0, l))); break;
            case V1(Op::V_EXP_F32): fop = true; LANES res[l] = U(std::exp2(fin(0, l))); break;
            case V1(Op::V_LOG_F32): fop = true; LANES res[l] = U(std::log2(fin(0, l))); break;
            case V1(Op::V_RCP_F32): case V1(Op::V_RCP_IFLAG_F32): fop = true; LANES res[l] = U(1.0f / fin(0, l)); break;
            case V1(Op::V_RSQ_F32): fop = true; LANES res[l] = U(1.0f / std::sqrt(fin(0, l))); break;
            case V1(Op::V_SQRT_F32): fop = true; LANES res[l] = U(std::sqrt(fin(0, l))); break;
            case V1(Op::V_SIN_F32): fop = true; LANES res[l] = U(std::sin(fin(0, l) * 6.2831853f)); break;
            case V1(Op::V_COS_F32): fop = true; LANES res[l] = U(std::cos(fin(0, l) * 6.2831853f)); break;
            case V1(Op::V_NOT_B32): LANES res[l] = ~r[0][l]; break;
            case V1(Op::V_BFREV_B32): LANES res[l] = BitReverse32(r[0][l]); break;
            case V1(Op::V_FFBH_U32): LANES res[l] = r[0][l] ? (uint32_t)__builtin_clz(r[0][l]) : ~0u; break;
            case V1(Op::V_FFBL_B32): LANES res[l] = r[0][l] ? (uint32_t)__builtin_ctz(r[0][l]) : ~0u; break;
            case V1(Op::V_FFBH_I32):
                LANES { const uint32_t v = r[0][l], x = (int32_t)v < 0 ? ~v : v; res[l] = x ? (uint32_t)__builtin_clz(x) : ~0u; }
                break;
            case V1(Op::V_FREXP_EXP_I32_F32):
                LANES { int e = 0; const float f = fin(0, l); if (std::isfinite(f)) std::frexp(f, &e); res[l] = (uint32_t)e; }
                break;
            case V1(Op::V_FREXP_MANT_F32): fop = true;
                LANES { int e; const float f = fin(0, l); res[l] = U(std::isfinite(f) ? std::frexp(f, &e) : f); }
                break;
            case V1(Op::V_MOVRELS_B32): case V1(Op::V_MOVRELD_B32): case V1(Op::V_MOVRELSD_B32): {
                const uint32_t m0 = w_.s[kM0];
                const unsigned srcv = in.src[0] >= 256 ? in.src[0] - 256 : 0;
                const unsigned from = op == V1(Op::V_MOVRELD_B32) ? srcv : (srcv + m0) & 0xff;
                const unsigned to = op == V1(Op::V_MOVRELS_B32) ? in.vdst : (in.vdst + m0) & 0xff;
                LANES w_.v[to][l] = in.src[0] >= 256 ? w_.v[from][l] : r[0][l];
                kind = Kind::kNone;
                break;
            }
            case V1(Op::V_SWAP_B32): {
                const unsigned a = in.vdst, b = in.src[0] - 256;
                LANES std::swap(w_.v[a][l], w_.v[b & 0xff][l]);
                kind = Kind::kNone;
                break;
            }
            // --- 16-bit float and integer ---
            case V1(Op::V_CVT_F16_U16): LANES res[l] = FloatToHalf((float)hin(0, l)); kind = Kind::kHalf; break;
            case V1(Op::V_CVT_F16_I16): LANES res[l] = FloatToHalf((float)(int16_t)hin(0, l)); kind = Kind::kHalf; break;
            case V1(Op::V_CVT_U16_F16): LANES res[l] = std::min(CvtU32(hf(0, l)), 0xffffu); kind = Kind::kHalf; break;
            case V1(Op::V_CVT_I16_F16):
                LANES res[l] = (uint16_t)std::clamp(CvtI32(hf(0, l)), -32768, 32767);
                kind = Kind::kHalf;
                break;
            case V1(Op::V_RCP_F16): LANES res[l] = FloatToHalf(1.0f / hf(0, l)); kind = Kind::kHalf; break;
            case V1(Op::V_SQRT_F16): LANES res[l] = FloatToHalf(std::sqrt(hf(0, l))); kind = Kind::kHalf; break;
            case V1(Op::V_RSQ_F16): LANES res[l] = FloatToHalf(1.0f / std::sqrt(hf(0, l))); kind = Kind::kHalf; break;
            case V1(Op::V_LOG_F16): LANES res[l] = FloatToHalf(std::log2(hf(0, l))); kind = Kind::kHalf; break;
            case V1(Op::V_EXP_F16): LANES res[l] = FloatToHalf(std::exp2(hf(0, l))); kind = Kind::kHalf; break;
            case V1(Op::V_FLOOR_F16): LANES res[l] = FloatToHalf(std::floor(hf(0, l))); kind = Kind::kHalf; break;
            case V1(Op::V_CEIL_F16): LANES res[l] = FloatToHalf(std::ceil(hf(0, l))); kind = Kind::kHalf; break;
            case V1(Op::V_TRUNC_F16): LANES res[l] = FloatToHalf(std::trunc(hf(0, l))); kind = Kind::kHalf; break;
            case V1(Op::V_RNDNE_F16): LANES res[l] = FloatToHalf(std::nearbyint(hf(0, l))); kind = Kind::kHalf; break;
            case V1(Op::V_FRACT_F16): LANES res[l] = FloatToHalf(Fract(hf(0, l))); kind = Kind::kHalf; break;
            case V1(Op::V_SIN_F16): LANES res[l] = FloatToHalf(std::sin(hf(0, l) * 6.2831853f)); kind = Kind::kHalf; break;
            case V1(Op::V_COS_F16): LANES res[l] = FloatToHalf(std::cos(hf(0, l) * 6.2831853f)); kind = Kind::kHalf; break;
            case V1(Op::V_CVT_NORM_I16_F16):
                LANES res[l] = (uint16_t)(int16_t)std::lround(std::clamp(hf(0, l), -1.0f, 1.0f) * 32767.0f);
                kind = Kind::kHalf;
                break;
            case V1(Op::V_CVT_NORM_U16_F16):
                LANES res[l] = (uint16_t)std::lround(std::clamp(hf(0, l), 0.0f, 1.0f) * 65535.0f);
                kind = Kind::kHalf;
                break;
            case V1(Op::V_SAT_PK_U8_I16):
                LANES {
                    auto s = [](uint32_t h) { return (uint32_t)std::clamp((int)(int16_t)h, 0, 255); };
                    res[l] = s(r[0][l] & 0xffff) | s(r[0][l] >> 16) << 8;
                }
                kind = Kind::kHalf;
                break;
            case V2(Op::V_ADD_F16): LANES res[l] = FloatToHalf(hf(0, l) + hf(1, l)); kind = Kind::kHalf; break;
            case V2(Op::V_SUB_F16): LANES res[l] = FloatToHalf(hf(0, l) - hf(1, l)); kind = Kind::kHalf; break;
            case V2(Op::V_SUBREV_F16): LANES res[l] = FloatToHalf(hf(1, l) - hf(0, l)); kind = Kind::kHalf; break;
            case V2(Op::V_MUL_F16): LANES res[l] = FloatToHalf(hf(0, l) * hf(1, l)); kind = Kind::kHalf; break;
            case V2(Op::V_FMAC_F16): case Op::V_FMA_F16:
                LANES res[l] = FloatToHalf(std::fma(hf(0, l), hf(1, l), hf(2, l)));
                kind = Kind::kHalf;
                break;
            case V2(Op::V_MAX_F16): LANES res[l] = FloatToHalf(std::fmax(hf(0, l), hf(1, l))); kind = Kind::kHalf; break;
            case V2(Op::V_MIN_F16): LANES res[l] = FloatToHalf(std::fmin(hf(0, l), hf(1, l))); kind = Kind::kHalf; break;
            case V2(Op::V_LDEXP_F16):
                LANES res[l] = FloatToHalf(std::ldexp(hf(0, l), (int16_t)r[1][l]));
                kind = Kind::kHalf;
                break;
            case Op::V_DIV_FIXUP_F16: LANES res[l] = FloatToHalf(hf(2, l) / hf(1, l)); kind = Kind::kHalf; break;
            case V2(Op::V_PK_FMAC_F16):
                LANES {
                    const uint32_t a = r[0][l], b = r[1][l], c = r[2][l];
                    const uint16_t lo = FloatToHalf(std::fma(HalfToFloat((uint16_t)a), HalfToFloat((uint16_t)b), HalfToFloat((uint16_t)c)));
                    const uint16_t hi = FloatToHalf(std::fma(HalfToFloat((uint16_t)(a >> 16)), HalfToFloat((uint16_t)(b >> 16)),
                                                             HalfToFloat((uint16_t)(c >> 16))));
                    res[l] = lo | (uint32_t)hi << 16;
                }
                break;
            case Op::V_ADD_NC_U16: LANES res[l] = (uint16_t)(hin(0, l) + hin(1, l)); kind = Kind::kHalf; break;
            case Op::V_SUB_NC_U16: LANES res[l] = (uint16_t)(hin(0, l) - hin(1, l)); kind = Kind::kHalf; break;
            case Op::V_ADD_NC_I16: LANES res[l] = (uint16_t)(hin(0, l) + hin(1, l)); kind = Kind::kHalf; break;
            case Op::V_SUB_NC_I16: LANES res[l] = (uint16_t)(hin(0, l) - hin(1, l)); kind = Kind::kHalf; break;
            case Op::V_MUL_LO_U16: LANES res[l] = (uint16_t)(hin(0, l) * hin(1, l)); kind = Kind::kHalf; break;
            case Op::V_LSHLREV_B16: LANES res[l] = (uint16_t)(hin(1, l) << (hin(0, l) & 15)); kind = Kind::kHalf; break;
            case Op::V_LSHRREV_B16: LANES res[l] = (uint16_t)(hin(1, l) >> (hin(0, l) & 15)); kind = Kind::kHalf; break;
            case Op::V_ASHRREV_I16: LANES res[l] = (uint16_t)((int16_t)hin(1, l) >> (hin(0, l) & 15)); kind = Kind::kHalf; break;
            case Op::V_MAX_U16: LANES res[l] = std::max(hin(0, l), hin(1, l)); kind = Kind::kHalf; break;
            case Op::V_MIN_U16: LANES res[l] = std::min(hin(0, l), hin(1, l)); kind = Kind::kHalf; break;
            case Op::V_MAX_I16: LANES res[l] = (uint16_t)std::max((int16_t)hin(0, l), (int16_t)hin(1, l)); kind = Kind::kHalf; break;
            case Op::V_MIN_I16: LANES res[l] = (uint16_t)std::min((int16_t)hin(0, l), (int16_t)hin(1, l)); kind = Kind::kHalf; break;
            case Op::V_MAD_U16: LANES res[l] = (uint16_t)(hin(0, l) * hin(1, l) + hin(2, l)); kind = Kind::kHalf; break;
            case Op::V_MAD_I16:
                LANES res[l] = (uint16_t)((int16_t)hin(0, l) * (int16_t)hin(1, l) + (int16_t)hin(2, l));
                kind = Kind::kHalf;
                break;
            case Op::V_PACK_B32_F16: LANES res[l] = hin(0, l) | (uint32_t)hin(1, l) << 16; break;
            case Op::V_CVT_PKNORM_I16_F16:
                LANES {
                    auto n = [](float f) { return (uint32_t)(uint16_t)(int16_t)std::lround(std::clamp(f, -1.0f, 1.0f) * 32767.0f); };
                    res[l] = n(hf(0, l)) | n(hf(1, l)) << 16;
                }
                break;
            case Op::V_CVT_PKNORM_U16_F16:
                LANES {
                    auto n = [](float f) { return (uint32_t)std::lround(std::clamp(f, 0.0f, 1.0f) * 65535.0f); };
                    res[l] = n(hf(0, l)) | n(hf(1, l)) << 16;
                }
                break;
            case Op::V_INTERP_P1LL_F16: case Op::V_INTERP_P1LV_F16: case Op::V_INTERP_P2_F16: {
                const unsigned attr = in.src[0] & 0x3f, chan = (in.src[0] >> 6) & 3;
                float p[3];
                env_.Attribute((int)attr, (int)chan, p);
                const uint32_t* ij = Row(in.src[1], 1);
                if (op == Op::V_INTERP_P2_F16) {
                    const uint32_t* acc = Row(in.src[2], 2);
                    LANES res[l] = FloatToHalf(F(acc[l]) + (p[2] - p[0]) * F(ij[l]));
                    kind = Kind::kHalf;
                } else {
                    LANES res[l] = U(p[0] + (p[1] - p[0]) * F(ij[l]));
                    fop = true;
                }
                break;
            }
            default: return Unsupported("VALU", op);
        }
    }
#undef LANES

    // --- write back ---
    switch (kind) {
        case Kind::kNone: break;
        case Kind::kMask:
            mask &= exec;
            if (cmpx) w_.SetExec(mask);
            else SetS64(in.sdst, mask);
            break;
        case Kind::kV64: {
            uint32_t* lo = w_.v[in.vdst];
            uint32_t* hi = w_.v[(in.vdst + 1) & 0xff];
            for (int l = 0; l < kLanes; l++) {
                if (!((write_mask >> l) & 1)) continue;
                uint64_t b = res[l] | (uint64_t)res_hi[l] << 32;
                if (fop && (in.omod || in.clamp)) {
                    double d = D(b);
                    if (in.omod == 1) d *= 2; else if (in.omod == 2) d *= 4; else if (in.omod == 3) d *= 0.5;
                    if (in.clamp) d = std::isnan(d) ? 0.0 : std::clamp(d, 0.0, 1.0);
                    b = U64(d);
                }
                lo[l] = (uint32_t)b;
                hi[l] = (uint32_t)(b >> 32);
            }
            break;
        }
        case Kind::kHalf: {
            // 16-bit results go to the low half (zeroing the high half), or with opsel[3] to the high half.
            uint32_t* d = w_.v[in.vdst];
            const bool high = (in.opsel >> 3) & 1;
            for (int l = 0; l < kLanes; l++) {
                if (!((write_mask >> l) & 1)) continue;
                const uint32_t h = res[l] & 0xffff;
                d[l] = high ? (d[l] & 0xffff) | h << 16 : h;
            }
            break;
        }
        case Kind::kV32: {
            uint32_t* d = w_.v[in.vdst];
            const bool mods = fop && (in.omod || in.clamp);
            unsigned dst_sel = 6, dst_unused = 0;
            if (in.sdwa) {
                dst_sel = (in.sdwa_word >> 8) & 7;
                dst_unused = (in.sdwa_word >> 11) & 3;
            }
            for (int l = 0; l < kLanes; l++) {
                if (!((write_mask >> l) & 1)) continue;
                uint32_t v = res[l];
                if (mods) {
                    float f = F(v);
                    if (in.omod == 1) f *= 2; else if (in.omod == 2) f *= 4; else if (in.omod == 3) f *= 0.5f;
                    if (in.clamp) f = std::isnan(f) ? 0.0f : std::clamp(f, 0.0f, 1.0f);
                    v = U(f);
                }
                if (dst_sel != 6) {
                    // dst_unused: 0 zero the other bits, 1 sign-extend into the bits above, 2 keep them.
                    const unsigned width = dst_sel < 4 ? 8 : 16, shift = dst_sel < 4 ? 8 * dst_sel : 16 * (dst_sel - 4);
                    const uint32_t m = ((1u << width) - 1) << shift;
                    const uint32_t part = (v << shift) & m;
                    if (dst_unused == 2) v = (d[l] & ~m) | part;
                    else if (dst_unused == 1 && ((v >> (width - 1)) & 1) && shift + width < 32)
                        v = part | ~((1u << (shift + width)) - 1);
                    else v = part;
                }
                d[l] = v;
            }
            break;
        }
    }
    switch (op) {
        case Op::V_ADD_CO_U32: case Op::V_SUB_CO_U32: case Op::V_SUBREV_CO_U32: case V2(Op::V_ADD_CO_CI_U32):
        case V2(Op::V_SUB_CO_CI_U32): case V2(Op::V_SUBREV_CO_CI_U32): case Op::V_MAD_U64_U32: case Op::V_MAD_I64_I32:
            SetS64(in.sdst, carry & exec);
            break;
        case Op::V_DIV_SCALE_F32: case Op::V_DIV_SCALE_F64:
            SetS64(in.sdst, 0);
            break;
        default: break;
    }
    return Status::kNext;
}

}  // namespace

namespace {

Status Interp::Vop3p(uint32_t w0, uint32_t w1) {
    base_ = 2;
    const unsigned op = (w0 >> 16) & 0x7f, vdst = w0 & 0xff;
    const unsigned src[3] = {w1 & 0x1ff, (w1 >> 9) & 0x1ff, (w1 >> 18) & 0x1ff};
    const unsigned neg_lo = (w1 >> 29) & 7, neg_hi = (w0 >> 8) & 7, opsel = (w0 >> 11) & 7;
    const unsigned opsel_hi = ((w1 >> 27) & 3) | (((w0 >> 14) & 1) << 2);
    const bool clamp = (w0 >> 15) & 1;
    const uint64_t exec = w_.Exec();
    const uint32_t* r[3];
    bool constant[3];
    for (int i = 0; i < 3; i++) {
        constant[i] = src[i] < 256 && src[i] != kLiteral && !(src[i] < 128);
        r[i] = Row(src[i], i, true);
    }
    uint32_t* d = w_.v[vdst];

    if (op == Op::V_FMA_MIX_F32 || op == Op::V_FMA_MIXLO_F16 || op == Op::V_FMA_MIXHI_F16) {
        // Each source is f32, or (opsel_hi) the f16 half chosen by opsel.
        auto in = [&](int i, int l) {
            float f;
            if ((opsel_hi >> i) & 1) {
                const uint32_t b = r[i][l];
                f = HalfToFloat((uint16_t)(((opsel >> i) & 1) && !constant[i] ? b >> 16 : b));
            } else {
                f = F(src[i] >= 240 && src[i] <= 248 ? InlineFloat(src[i]) : r[i][l]);
            }
            if ((neg_hi >> i) & 1) f = std::fabs(f);
            if ((neg_lo >> i) & 1) f = -f;
            return f;
        };
        for (int l = 0; l < kLanes; l++) {
            if (!((exec >> l) & 1)) continue;
            float v = std::fma(in(0, l), in(1, l), in(2, l));
            if (clamp) v = std::isnan(v) ? 0.0f : std::clamp(v, 0.0f, 1.0f);
            if (op == Op::V_FMA_MIX_F32) d[l] = U(v);
            else if (op == Op::V_FMA_MIXLO_F16) d[l] = (d[l] & 0xffff0000u) | FloatToHalf(v);
            else d[l] = (d[l] & 0xffffu) | (uint32_t)FloatToHalf(v) << 16;
        }
        return Status::kNext;
    }

    // Packed 16-bit: the low result uses the halves opsel picks, the high result the halves opsel_hi picks.
    auto half = [&](int i, int l, bool high_result) -> uint16_t {
        const uint32_t b = r[i][l];
        const bool hi = constant[i] ? false : high_result ? ((opsel_hi >> i) & 1) : ((opsel >> i) & 1);
        return (uint16_t)(hi ? b >> 16 : b);
    };
    auto fh = [&](int i, int l, bool high_result) {
        float f = HalfToFloat(half(i, l, high_result));
        if (((high_result ? neg_hi : neg_lo) >> i) & 1) f = -f;
        return f;
    };
    auto ih = [&](int i, int l, bool high_result) { return half(i, l, high_result); };
    for (int l = 0; l < kLanes; l++) {
        if (!((exec >> l) & 1)) continue;
        uint16_t out[2];
        for (int h = 0; h < 2; h++) {
            const bool hr = h == 1;
            switch (op) {
                case Op::V_PK_FMA_F16: out[h] = FloatToHalf(std::fma(fh(0, l, hr), fh(1, l, hr), fh(2, l, hr))); break;
                case Op::V_PK_ADD_F16: out[h] = FloatToHalf(fh(0, l, hr) + fh(1, l, hr)); break;
                case Op::V_PK_MUL_F16: out[h] = FloatToHalf(fh(0, l, hr) * fh(1, l, hr)); break;
                case Op::V_PK_MIN_F16: out[h] = FloatToHalf(std::fmin(fh(0, l, hr), fh(1, l, hr))); break;
                case Op::V_PK_MAX_F16: out[h] = FloatToHalf(std::fmax(fh(0, l, hr), fh(1, l, hr))); break;
                case Op::V_PK_MAD_I16:
                    out[h] = (uint16_t)((int16_t)ih(0, l, hr) * (int16_t)ih(1, l, hr) + (int16_t)ih(2, l, hr));
                    break;
                case Op::V_PK_MAD_U16: out[h] = (uint16_t)(ih(0, l, hr) * ih(1, l, hr) + ih(2, l, hr)); break;
                case Op::V_PK_MUL_LO_U16: out[h] = (uint16_t)(ih(0, l, hr) * ih(1, l, hr)); break;
                case Op::V_PK_ADD_I16: case Op::V_PK_ADD_U16: out[h] = (uint16_t)(ih(0, l, hr) + ih(1, l, hr)); break;
                case Op::V_PK_SUB_I16: case Op::V_PK_SUB_U16: out[h] = (uint16_t)(ih(0, l, hr) - ih(1, l, hr)); break;
                case Op::V_PK_LSHLREV_B16: out[h] = (uint16_t)(ih(1, l, hr) << (ih(0, l, hr) & 15)); break;
                case Op::V_PK_LSHRREV_B16: out[h] = (uint16_t)(ih(1, l, hr) >> (ih(0, l, hr) & 15)); break;
                case Op::V_PK_ASHRREV_I16: out[h] = (uint16_t)((int16_t)ih(1, l, hr) >> (ih(0, l, hr) & 15)); break;
                case Op::V_PK_MAX_I16: out[h] = (uint16_t)std::max((int16_t)ih(0, l, hr), (int16_t)ih(1, l, hr)); break;
                case Op::V_PK_MIN_I16: out[h] = (uint16_t)std::min((int16_t)ih(0, l, hr), (int16_t)ih(1, l, hr)); break;
                case Op::V_PK_MAX_U16: out[h] = std::max(ih(0, l, hr), ih(1, l, hr)); break;
                case Op::V_PK_MIN_U16: out[h] = std::min(ih(0, l, hr), ih(1, l, hr)); break;
                default: return Unsupported("VOP3P", op);
            }
            if (clamp && (op == Op::V_PK_FMA_F16 || op == Op::V_PK_ADD_F16 || op == Op::V_PK_MUL_F16 ||
                          op == Op::V_PK_MIN_F16 || op == Op::V_PK_MAX_F16)) {
                const float f = HalfToFloat(out[h]);
                out[h] = FloatToHalf(std::isnan(f) ? 0.0f : std::clamp(f, 0.0f, 1.0f));
            }
        }
        d[l] = out[0] | (uint32_t)out[1] << 16;
    }
    return Status::kNext;
}

// --- LDS ---

Status Interp::Ds(uint32_t w0, uint32_t w1) {
    base_ = 2;
    const unsigned op = (w0 >> 18) & 0xff, off0 = w0 & 0xff, off1 = (w0 >> 8) & 0xff, offset = w0 & 0xffff;
    const unsigned addr = w1 & 0xff, data0 = (w1 >> 8) & 0xff, data1 = (w1 >> 16) & 0xff, vdst = (w1 >> 24) & 0xff;
    if ((w0 >> 17) & 1) return Fail("GDS access");
    const uint64_t exec = w_.Exec();
    uint8_t* lds = w_.lds;
    const uint32_t size = w_.lds_bytes;
    auto at = [&](uint32_t a, uint32_t n) -> uint8_t* { return lds && a + n <= size && a + n >= a ? lds + a : nullptr; };
    auto load = [&](uint32_t a, uint32_t n) -> uint64_t {
        uint64_t v = 0;
        if (uint8_t* p = at(a, n)) memcpy(&v, p, n);
        return v;
    };
    auto store = [&](uint32_t a, uint32_t n, uint64_t v) {
        if (uint8_t* p = at(a, n)) memcpy(p, &v, n);
    };
    const uint32_t* va = w_.v[addr];
    auto V = [&](unsigned reg, int l) { return w_.v[reg & 0xff][l]; };
#define LANES for (int l = 0; l < kLanes; l++) if ((exec >> l) & 1)
    switch (op) {
        case Op::DS_WRITE_B32: LANES store(va[l] + offset, 4, V(data0, l)); break;
        case Op::DS_WRITE_B8: LANES store(va[l] + offset, 1, V(data0, l)); break;
        case Op::DS_WRITE_B16: LANES store(va[l] + offset, 2, V(data0, l)); break;
        case Op::DS_WRITE_B8_D16_HI: LANES store(va[l] + offset, 1, V(data0, l) >> 16); break;
        case Op::DS_WRITE_B16_D16_HI: LANES store(va[l] + offset, 2, V(data0, l) >> 16); break;
        case Op::DS_WRITE_B64: LANES store(va[l] + offset, 8, V(data0, l) | (uint64_t)V(data0 + 1, l) << 32); break;
        case Op::DS_WRITE_B96: case Op::DS_WRITE_B128: {
            const int n = op == Op::DS_WRITE_B96 ? 3 : 4;
            LANES for (int i = 0; i < n; i++) store(va[l] + offset + 4 * i, 4, V(data0 + i, l));
            break;
        }
        case Op::DS_WRITE2_B32: case Op::DS_WRITE2ST64_B32: {
            const uint32_t scale = op == Op::DS_WRITE2_B32 ? 4 : 256;
            LANES {
                store(va[l] + off0 * scale, 4, V(data0, l));
                store(va[l] + off1 * scale, 4, V(data1, l));
            }
            break;
        }
        case Op::DS_WRITE2_B64: case Op::DS_WRITE2ST64_B64: {
            const uint32_t scale = op == Op::DS_WRITE2_B64 ? 8 : 512;
            LANES {
                store(va[l] + off0 * scale, 8, V(data0, l) | (uint64_t)V(data0 + 1, l) << 32);
                store(va[l] + off1 * scale, 8, V(data1, l) | (uint64_t)V(data1 + 1, l) << 32);
            }
            break;
        }
        case Op::DS_READ_B32: LANES w_.v[vdst][l] = (uint32_t)load(va[l] + offset, 4); break;
        case Op::DS_READ_U8: LANES w_.v[vdst][l] = (uint32_t)load(va[l] + offset, 1); break;
        case Op::DS_READ_I8: LANES w_.v[vdst][l] = (uint32_t)Sext((uint32_t)load(va[l] + offset, 1), 8); break;
        case Op::DS_READ_U16: LANES w_.v[vdst][l] = (uint32_t)load(va[l] + offset, 2); break;
        case Op::DS_READ_I16: LANES w_.v[vdst][l] = (uint32_t)Sext((uint32_t)load(va[l] + offset, 2), 16); break;
        case Op::DS_READ_U8_D16: LANES w_.v[vdst][l] = (w_.v[vdst][l] & 0xffff0000u) | (uint32_t)load(va[l] + offset, 1); break;
        case Op::DS_READ_U16_D16: LANES w_.v[vdst][l] = (w_.v[vdst][l] & 0xffff0000u) | (uint32_t)load(va[l] + offset, 2); break;
        case Op::DS_READ_U8_D16_HI:
            LANES w_.v[vdst][l] = (w_.v[vdst][l] & 0xffffu) | (uint32_t)load(va[l] + offset, 1) << 16;
            break;
        case Op::DS_READ_U16_D16_HI:
            LANES w_.v[vdst][l] = (w_.v[vdst][l] & 0xffffu) | (uint32_t)load(va[l] + offset, 2) << 16;
            break;
        case Op::DS_READ_B64:
            LANES {
                const uint64_t v = load(va[l] + offset, 8);
                w_.v[vdst][l] = (uint32_t)v;
                w_.v[(vdst + 1) & 0xff][l] = (uint32_t)(v >> 32);
            }
            break;
        case Op::DS_READ_B96: case Op::DS_READ_B128: {
            const int n = op == Op::DS_READ_B96 ? 3 : 4;
            LANES for (int i = 0; i < n; i++) w_.v[(vdst + i) & 0xff][l] = (uint32_t)load(va[l] + offset + 4 * i, 4);
            break;
        }
        case Op::DS_READ2_B32: case Op::DS_READ2ST64_B32: {
            const uint32_t scale = op == Op::DS_READ2_B32 ? 4 : 256;
            LANES {
                const uint32_t a = (uint32_t)load(va[l] + off0 * scale, 4), b = (uint32_t)load(va[l] + off1 * scale, 4);
                w_.v[vdst][l] = a;
                w_.v[(vdst + 1) & 0xff][l] = b;
            }
            break;
        }
        case Op::DS_READ2_B64: case Op::DS_READ2ST64_B64: {
            const uint32_t scale = op == Op::DS_READ2_B64 ? 8 : 512;
            LANES {
                const uint64_t a = load(va[l] + off0 * scale, 8), b = load(va[l] + off1 * scale, 8);
                w_.v[vdst][l] = (uint32_t)a;
                w_.v[(vdst + 1) & 0xff][l] = (uint32_t)(a >> 32);
                w_.v[(vdst + 2) & 0xff][l] = (uint32_t)b;
                w_.v[(vdst + 3) & 0xff][l] = (uint32_t)(b >> 32);
            }
            break;
        }
        case Op::DS_SWIZZLE_B32: {
            uint32_t out[kLanes];
            const uint32_t* src = w_.v[data0];
            for (int l = 0; l < kLanes; l++) {
                int from;
                if (offset & 0x8000) {
                    from = (l & ~3) | (int)((offset >> (2 * (l & 3))) & 3);
                } else {
                    const int and_mask = offset & 0x1f, or_mask = (offset >> 5) & 0x1f, xor_mask = (offset >> 10) & 0x1f;
                    from = (l & 32) | ((((l & 31) & and_mask) | or_mask) ^ xor_mask);
                }
                out[l] = ((exec >> from) & 1) ? src[from] : 0;
            }
            LANES w_.v[vdst][l] = out[l];
            break;
        }
        case Op::DS_PERMUTE_B32: case Op::DS_BPERMUTE_B32: {
            uint32_t out[kLanes] = {};
            const uint32_t* src = w_.v[data0];
            LANES {
                const int other = (int)(((va[l] + offset) >> 2) & 63);
                if (op == Op::DS_BPERMUTE_B32) out[l] = ((exec >> other) & 1) ? src[other] : 0;
                else out[other] = src[l];
            }
            LANES w_.v[vdst][l] = out[l];
            break;
        }
        case Op::DS_APPEND: case Op::DS_CONSUME: {
            const uint32_t a = (w_.s[kM0] & 0xffff) + offset;
            const uint32_t old = (uint32_t)load(a, 4), n = Popcount(exec);
            store(a, 4, op == Op::DS_APPEND ? old + n : old - n);
            LANES w_.v[vdst][l] = old;
            break;
        }
        default: {
            // 32-bit atomics, with (_rtn) and without a returned pre-op value.
            static const struct { unsigned op, rtn; char kind; } kAtomics[] = {
                {Op::DS_ADD_U32, 0, '+'}, {Op::DS_ADD_RTN_U32, 1, '+'}, {Op::DS_SUB_U32, 0, '-'},
                {Op::DS_SUB_RTN_U32, 1, '-'}, {Op::DS_RSUB_U32, 0, 'r'}, {Op::DS_INC_U32, 0, 'i'},
                {Op::DS_DEC_U32, 0, 'd'}, {Op::DS_MIN_I32, 0, 'm'}, {Op::DS_MIN_RTN_I32, 1, 'm'},
                {Op::DS_MAX_I32, 0, 'M'}, {Op::DS_MAX_RTN_I32, 1, 'M'}, {Op::DS_MIN_U32, 0, 'n'},
                {Op::DS_MIN_RTN_U32, 1, 'n'}, {Op::DS_MAX_U32, 0, 'N'}, {Op::DS_MAX_RTN_U32, 1, 'N'},
                {Op::DS_AND_B32, 0, '&'}, {Op::DS_AND_RTN_B32, 1, '&'}, {Op::DS_OR_B32, 0, '|'},
                {Op::DS_OR_RTN_B32, 1, '|'}, {Op::DS_XOR_B32, 0, '^'}, {Op::DS_XOR_RTN_B32, 1, '^'},
                {Op::DS_MSKOR_B32, 0, 'k'}, {Op::DS_WRXCHG_RTN_B32, 1, '='}, {Op::DS_CMPST_B32, 0, 'c'},
                {Op::DS_CMPST_RTN_B32, 1, 'c'}, {Op::DS_ADD_F32, 0, 'f'}, {Op::DS_ADD_RTN_F32, 1, 'f'},
                {Op::DS_MIN_F32, 0, 'x'}, {Op::DS_MAX_F32, 0, 'X'},
            };
            for (const auto& a : kAtomics) {
                if (a.op != op) continue;
                LANES {
                    const uint32_t ad = va[l] + offset;
                    const uint32_t old = (uint32_t)load(ad, 4), x = V(data0, l), y = V(data1, l);
                    uint32_t v = old;
                    switch (a.kind) {
                        case '+': v = old + x; break;
                        case '-': v = old - x; break;
                        case 'r': v = x - old; break;
                        case 'i': v = old >= x ? 0 : old + 1; break;
                        case 'd': v = (old == 0 || old > x) ? x : old - 1; break;
                        case 'm': v = (uint32_t)std::min((int32_t)old, (int32_t)x); break;
                        case 'M': v = (uint32_t)std::max((int32_t)old, (int32_t)x); break;
                        case 'n': v = std::min(old, x); break;
                        case 'N': v = std::max(old, x); break;
                        case '&': v = old & x; break;
                        case '|': v = old | x; break;
                        case '^': v = old ^ x; break;
                        case 'k': v = (old & ~x) | y; break;
                        case '=': v = x; break;
                        case 'c': v = old == x ? y : old; break;
                        case 'f': v = U(F(old) + F(x)); break;
                        case 'x': v = U(std::fmin(F(old), F(x))); break;
                        case 'X': v = U(std::fmax(F(old), F(x))); break;
                    }
                    store(ad, 4, v);
                    if (a.rtn) w_.v[vdst][l] = old;
                }
                return Status::kNext;
            }
            return Unsupported("DS", op);
        }
    }
#undef LANES
    return Status::kNext;
}

// Global memory atomics shared by buffer and flat instructions; returns the value before the operation.
bool AtomicOp(unsigned kind, uint8_t* p, const uint32_t* data, uint32_t* old_out, bool wide) {
    if (!p) return false;
    if (wide) return false;
    uint32_t old;
    memcpy(&old, p, 4);
    const uint32_t x = data[0];
    uint32_t v = old;
    switch (kind) {
        case 0: v = x; break;                                      // swap
        case 1: v = old == data[1] ? x : old; break;               // cmpswap: data[0] src, data[1] cmp
        case 2: v = old + x; break;
        case 3: v = old - x; break;
        case 4: v = (uint32_t)std::min((int32_t)old, (int32_t)x); break;
        case 5: v = std::min(old, x); break;
        case 6: v = (uint32_t)std::max((int32_t)old, (int32_t)x); break;
        case 7: v = std::max(old, x); break;
        case 8: v = old & x; break;
        case 9: v = old | x; break;
        case 10: v = old ^ x; break;
        case 11: v = old >= x ? 0 : old + 1; break;
        case 12: v = (old == 0 || old > x) ? x : old - 1; break;
        default: return false;
    }
    memcpy(p, &v, 4);
    *old_out = old;
    return true;
}

// --- buffers ---

Status Interp::Mubuf(uint32_t w0, uint32_t w1, bool typed) {
    base_ = 2;
    const uint32_t inst_offset = w0 & 0xfff;
    const bool offen = (w0 >> 12) & 1, idxen = (w0 >> 13) & 1, glc = (w0 >> 14) & 1;
    unsigned op, fmt = 0;
    if (typed) {
        op = ((w0 >> 16) & 7) | (((w1 >> 21) & 1) << 3);
        fmt = (w0 >> 19) & 0x7f;
    } else {
        op = ((w0 >> 18) & 0x7f) | (((w0 >> 25) & 1) << 7);
        if ((w0 >> 16) & 1) return Fail("buffer load to LDS");
    }
    const unsigned vaddr = w1 & 0xff, vdata = (w1 >> 8) & 0xff, srsrc = ((w1 >> 16) & 0x1f) * 4;
    const uint32_t soffset = S((w1 >> 24) & 0xff);
    const uint32_t* vs = &w_.s[srsrc];
    const uint64_t base = vs[0] | (uint64_t)(vs[1] & 0xffff) << 32;
    const uint32_t stride = (vs[1] >> 16) & 0x3fff, records = vs[2];
    const uint32_t dst_sel = vs[3] & 0xfff, oob = (vs[3] >> 28) & 3;
    const bool add_tid = (vs[3] >> 23) & 1;
    const uint64_t exec = w_.Exec();

    // Classify: format load/store (n components), raw load/store (bytes), or atomic.
    enum { kLoadFmt, kStoreFmt, kLoad, kStore, kAtomic, kNop } kind;
    int n = 0, bytes = 0;
    bool sign = false, d16 = false;
    unsigned atomic = 0;
    if (typed) {
        if (op <= Op::TBUFFER_LOAD_FORMAT_XYZW) { kind = kLoadFmt; n = (int)op + 1; }
        else if (op <= Op::TBUFFER_STORE_FORMAT_XYZW) { kind = kStoreFmt; n = (int)(op - Op::TBUFFER_STORE_FORMAT_X) + 1; }
        else return Unsupported("MTBUF", op);
    } else {
        fmt = (vs[3] >> 12) & 0x7f;
        switch (op) {
            case Op::BUFFER_LOAD_FORMAT_X: case Op::BUFFER_LOAD_FORMAT_XY: case Op::BUFFER_LOAD_FORMAT_XYZ:
            case Op::BUFFER_LOAD_FORMAT_XYZW:
                kind = kLoadFmt; n = (int)(op - Op::BUFFER_LOAD_FORMAT_X) + 1; break;
            case Op::BUFFER_STORE_FORMAT_X: case Op::BUFFER_STORE_FORMAT_XY: case Op::BUFFER_STORE_FORMAT_XYZ:
            case Op::BUFFER_STORE_FORMAT_XYZW:
                kind = kStoreFmt; n = (int)(op - Op::BUFFER_STORE_FORMAT_X) + 1; break;
            case Op::BUFFER_LOAD_FORMAT_D16_X: case Op::BUFFER_LOAD_FORMAT_D16_XY: case Op::BUFFER_LOAD_FORMAT_D16_XYZ:
            case Op::BUFFER_LOAD_FORMAT_D16_XYZW:
                kind = kLoadFmt; d16 = true; n = (int)(op - Op::BUFFER_LOAD_FORMAT_D16_X) + 1; break;
            case Op::BUFFER_STORE_FORMAT_D16_X: case Op::BUFFER_STORE_FORMAT_D16_XY:
            case Op::BUFFER_STORE_FORMAT_D16_XYZ: case Op::BUFFER_STORE_FORMAT_D16_XYZW:
                kind = kStoreFmt; d16 = true; n = (int)(op - Op::BUFFER_STORE_FORMAT_D16_X) + 1; break;
            case Op::BUFFER_LOAD_UBYTE: kind = kLoad; bytes = 1; break;
            case Op::BUFFER_LOAD_SBYTE: kind = kLoad; bytes = 1; sign = true; break;
            case Op::BUFFER_LOAD_USHORT: kind = kLoad; bytes = 2; break;
            case Op::BUFFER_LOAD_SSHORT: kind = kLoad; bytes = 2; sign = true; break;
            case Op::BUFFER_LOAD_DWORD: kind = kLoad; bytes = 4; break;
            case Op::BUFFER_LOAD_DWORDX2: kind = kLoad; bytes = 8; break;
            case Op::BUFFER_LOAD_DWORDX3: kind = kLoad; bytes = 12; break;
            case Op::BUFFER_LOAD_DWORDX4: kind = kLoad; bytes = 16; break;
            case Op::BUFFER_STORE_BYTE: kind = kStore; bytes = 1; break;
            case Op::BUFFER_STORE_SHORT: kind = kStore; bytes = 2; break;
            case Op::BUFFER_STORE_DWORD: kind = kStore; bytes = 4; break;
            case Op::BUFFER_STORE_DWORDX2: kind = kStore; bytes = 8; break;
            case Op::BUFFER_STORE_DWORDX3: kind = kStore; bytes = 12; break;
            case Op::BUFFER_STORE_DWORDX4: kind = kStore; bytes = 16; break;
            case Op::BUFFER_GL0_INV: case Op::BUFFER_GL1_INV: kind = kNop; break;
            case Op::BUFFER_ATOMIC_SWAP: kind = kAtomic; atomic = 0; break;
            case Op::BUFFER_ATOMIC_CMPSWAP: kind = kAtomic; atomic = 1; break;
            case Op::BUFFER_ATOMIC_ADD: kind = kAtomic; atomic = 2; break;
            case Op::BUFFER_ATOMIC_SUB: kind = kAtomic; atomic = 3; break;
            case Op::BUFFER_ATOMIC_SMIN: kind = kAtomic; atomic = 4; break;
            case Op::BUFFER_ATOMIC_UMIN: kind = kAtomic; atomic = 5; break;
            case Op::BUFFER_ATOMIC_SMAX: kind = kAtomic; atomic = 6; break;
            case Op::BUFFER_ATOMIC_UMAX: kind = kAtomic; atomic = 7; break;
            case Op::BUFFER_ATOMIC_AND: kind = kAtomic; atomic = 8; break;
            case Op::BUFFER_ATOMIC_OR: kind = kAtomic; atomic = 9; break;
            case Op::BUFFER_ATOMIC_XOR: kind = kAtomic; atomic = 10; break;
            case Op::BUFFER_ATOMIC_INC: kind = kAtomic; atomic = 11; break;
            case Op::BUFFER_ATOMIC_DEC: kind = kAtomic; atomic = 12; break;
            default: return Unsupported("MUBUF", op);
        }
    }
    if (kind == kNop) return Status::kNext;
    FormatInfo fi{};
    const bool fmt_ok = (kind == kLoadFmt || kind == kStoreFmt) ? LookupFormat(fmt, &fi) : true;
    const int elem = kind == kLoadFmt || kind == kStoreFmt ? (fmt_ok ? FormatBytes(fi) : 0) : kind == kAtomic ? 4 : bytes;

    for (int l = 0; l < kLanes; l++) {
        if (!((exec >> l) & 1)) continue;
        uint32_t index = idxen ? w_.v[vaddr][l] : 0;
        if (add_tid) index += w_.first_thread + (uint32_t)l;
        const uint32_t voff = offen ? w_.v[(vaddr + (idxen ? 1 : 0)) & 0xff][l] : 0;
        const uint64_t off = (uint64_t)voff + inst_offset;
        bool out = false;
        switch (oob) {
            case 0: out = index >= records || (stride && off >= stride && idxen); break;
            case 1: out = index >= records; break;
            case 2: out = false; break;
            default: out = (uint64_t)index * stride + off + (uint64_t)elem > records; break;
        }
        if (stride == 0 && !idxen && oob != 2) out = off + (uint64_t)elem > records;
        const uint64_t addr = base + soffset + (uint64_t)index * stride + off;
        switch (kind) {
            case kLoadFmt: {
                uint32_t v[4] = {0, 0, 0, 0};
                const uint8_t* p = (!out && fmt_ok) ? env_.Read(addr, (uint64_t)elem) : nullptr;
                if (p) {
                    DecodeFormatted(p, fi, v);
                    ApplyDstSel(dst_sel, fi.type == 4 || fi.type == 5, v);
                }
                if (d16) {
                    for (int c = 0; c < n; c += 2) {
                        const uint32_t lo = fi.type >= 4 && fi.type <= 5 ? (v[c] & 0xffff) : FloatToHalf(F(v[c]));
                        const uint32_t hi = c + 1 < n ? (fi.type >= 4 && fi.type <= 5 ? (v[c + 1] & 0xffff) : FloatToHalf(F(v[c + 1]))) : 0;
                        w_.v[(vdata + c / 2) & 0xff][l] = lo | hi << 16;
                    }
                } else {
                    for (int c = 0; c < n; c++) w_.v[(vdata + c) & 0xff][l] = v[c];
                }
                break;
            }
            case kStoreFmt: {
                if (out || !fmt_ok) break;
                uint32_t v[4] = {0, 0, 0, U(1.0f)};
                for (int c = 0; c < n; c++) {
                    if (d16) {
                        const uint16_t h = (uint16_t)(w_.v[(vdata + c / 2) & 0xff][l] >> (16 * (c & 1)));
                        v[c] = fi.type >= 4 && fi.type <= 5 ? h : U(HalfToFloat(h));
                    } else {
                        v[c] = w_.v[(vdata + c) & 0xff][l];
                    }
                }
                if (uint8_t* p = env_.Write(addr, (uint64_t)elem)) EncodeFormatted(p, fi, v);
                break;
            }
            case kLoad: {
                uint32_t v[4] = {0, 0, 0, 0};
                if (!out) {
                    if (const uint8_t* p = env_.Read(addr, (uint64_t)bytes)) memcpy(v, p, (size_t)bytes);
                }
                if (bytes < 4) v[0] = sign ? (uint32_t)Sext(v[0], 8 * bytes) : v[0];
                for (int c = 0; c < std::max(1, bytes / 4); c++) w_.v[(vdata + c) & 0xff][l] = v[c];
                break;
            }
            case kStore: {
                if (out) break;
                uint32_t v[4];
                for (int c = 0; c < 4; c++) v[c] = w_.v[(vdata + c) & 0xff][l];
                if (uint8_t* p = env_.Write(addr, (uint64_t)bytes)) memcpy(p, v, (size_t)bytes);
                break;
            }
            case kAtomic: {
                if (out) break;
                const uint32_t data[2] = {w_.v[vdata][l], w_.v[(vdata + 1) & 0xff][l]};
                uint32_t old = 0;
                AtomicOp(atomic, env_.Write(addr, 4), data, &old, false);
                if (glc) w_.v[vdata][l] = old;
                break;
            }
            default: break;
        }
    }
    return Status::kNext;
}

// --- flat / global memory ---

Status Interp::Flat(uint32_t w0, uint32_t w1) {
    base_ = 2;
    const unsigned seg = (w0 >> 14) & 3, op = (w0 >> 18) & 0x7f;
    const int32_t offset = seg == 0 ? (int32_t)(w0 & 0x7ff) : Sext(w0 & 0xfff, 12);
    const bool glc = (w0 >> 16) & 1;
    const unsigned addr = w1 & 0xff, data = (w1 >> 8) & 0xff, saddr = (w1 >> 16) & 0x7f, vdst = (w1 >> 24) & 0xff;
    if (seg == 1) return Fail("scratch memory");
    const bool use_saddr = seg == 2 && saddr != kNull && saddr != 0x7f;
    const uint64_t sbase = use_saddr ? (w_.s[saddr] | (uint64_t)w_.s[saddr + 1] << 32) : 0;
    const uint64_t exec = w_.Exec();
    int bytes = 0;
    bool load = false, sign = false, atomic = false;
    unsigned akind = 0;
    switch (op) {
        case Op::GLOBAL_LOAD_UBYTE: load = true; bytes = 1; break;
        case Op::GLOBAL_LOAD_SBYTE: load = true; bytes = 1; sign = true; break;
        case Op::GLOBAL_LOAD_USHORT: load = true; bytes = 2; break;
        case Op::GLOBAL_LOAD_SSHORT: load = true; bytes = 2; sign = true; break;
        case Op::GLOBAL_LOAD_DWORD: load = true; bytes = 4; break;
        case Op::GLOBAL_LOAD_DWORDX2: load = true; bytes = 8; break;
        case Op::GLOBAL_LOAD_DWORDX3: load = true; bytes = 12; break;
        case Op::GLOBAL_LOAD_DWORDX4: load = true; bytes = 16; break;
        case Op::GLOBAL_STORE_BYTE: bytes = 1; break;
        case Op::GLOBAL_STORE_SHORT: bytes = 2; break;
        case Op::GLOBAL_STORE_DWORD: bytes = 4; break;
        case Op::GLOBAL_STORE_DWORDX2: bytes = 8; break;
        case Op::GLOBAL_STORE_DWORDX3: bytes = 12; break;
        case Op::GLOBAL_STORE_DWORDX4: bytes = 16; break;
        case Op::GLOBAL_ATOMIC_SWAP: atomic = true; akind = 0; break;
        case Op::GLOBAL_ATOMIC_CMPSWAP: atomic = true; akind = 1; break;
        case Op::GLOBAL_ATOMIC_ADD: atomic = true; akind = 2; break;
        case Op::GLOBAL_ATOMIC_SUB: atomic = true; akind = 3; break;
        case Op::GLOBAL_ATOMIC_SMIN: atomic = true; akind = 4; break;
        case Op::GLOBAL_ATOMIC_UMIN: atomic = true; akind = 5; break;
        case Op::GLOBAL_ATOMIC_SMAX: atomic = true; akind = 6; break;
        case Op::GLOBAL_ATOMIC_UMAX: atomic = true; akind = 7; break;
        case Op::GLOBAL_ATOMIC_AND: atomic = true; akind = 8; break;
        case Op::GLOBAL_ATOMIC_OR: atomic = true; akind = 9; break;
        case Op::GLOBAL_ATOMIC_XOR: atomic = true; akind = 10; break;
        case Op::GLOBAL_ATOMIC_INC: atomic = true; akind = 11; break;
        case Op::GLOBAL_ATOMIC_DEC: atomic = true; akind = 12; break;
        default: return Unsupported("FLAT", op);
    }
    for (int l = 0; l < kLanes; l++) {
        if (!((exec >> l) & 1)) continue;
        const uint64_t a = (use_saddr ? sbase + w_.v[addr][l] : (w_.v[addr][l] | (uint64_t)w_.v[(addr + 1) & 0xff][l] << 32)) +
                           (int64_t)offset;
        if (atomic) {
            const uint32_t d[2] = {w_.v[data][l], w_.v[(data + 1) & 0xff][l]};
            uint32_t old = 0;
            AtomicOp(akind, env_.Write(a, 4), d, &old, false);
            if (glc) w_.v[vdst][l] = old;
        } else if (load) {
            uint32_t v[4] = {0, 0, 0, 0};
            if (const uint8_t* p = env_.Read(a, (uint64_t)bytes)) memcpy(v, p, (size_t)bytes);
            if (bytes < 4 && sign) v[0] = (uint32_t)Sext(v[0], 8 * bytes);
            for (int c = 0; c < std::max(1, bytes / 4); c++) w_.v[(vdst + c) & 0xff][l] = v[c];
        } else {
            uint32_t v[4];
            for (int c = 0; c < 4; c++) v[c] = w_.v[(data + c) & 0xff][l];
            if (uint8_t* p = env_.Write(a, (uint64_t)bytes)) memcpy(p, v, (size_t)bytes);
        }
    }
    return Status::kNext;
}

// --- images ---

Status Interp::Mimg(uint32_t w0, uint32_t w1) {
    const unsigned op = ((w0 >> 18) & 0x7f) | ((w0 & 1) << 7);
    const unsigned nsa = (w0 >> 1) & 3, dim = (w0 >> 3) & 7, dmask = (w0 >> 8) & 0xf;
    const bool glc = (w0 >> 13) & 1, r128 = (w0 >> 15) & 1;
    const unsigned vaddr = w1 & 0xff, vdata = (w1 >> 8) & 0xff, srsrc = ((w1 >> 16) & 0x1f) * 4;
    const unsigned ssamp = ((w1 >> 21) & 0x1f) * 4;
    const bool a16 = (w1 >> 30) & 1, d16 = (w1 >> 31) & 1;
    base_ = 2 + (int)nsa;
    if (a16) return Fail("16-bit image addresses");
    auto areg = [&](int i) -> unsigned {
        if (i == 0 || nsa == 0) return (vaddr + (unsigned)i) & 0xff;
        return (ins_[2 + (i - 1) / 4] >> (8 * ((i - 1) % 4))) & 0xff;
    };
    uint32_t t[8];
    for (int i = 0; i < 8; i++) t[i] = (r128 && i >= 4) ? 0 : w_.s[(srsrc + i) & 0x7f];
    const uint32_t* smp = &w_.s[ssamp & 0x7f];
    const uint64_t exec = w_.Exec();
    static const int kCoords[8] = {1, 2, 3, 3, 2, 3, 3, 4};
    static const int kDerivs[8] = {1, 2, 3, 2, 1, 2, 2, 2};
    const int ncoord = kCoords[dim];

    // Writes the dmask-selected components of out[4] to vdata (packed halves with d16).
    auto put = [&](int l, const uint32_t out[4], bool is_float) {
        int k = 0;
        uint32_t packed[4] = {0, 0, 0, 0};
        for (int c = 0; c < 4; c++) {
            if (!((dmask >> c) & 1)) continue;
            if (d16) {
                const uint32_t h = is_float ? FloatToHalf(F(out[c])) : (out[c] & 0xffff);
                packed[k / 2] |= h << (16 * (k & 1));
            } else {
                packed[k] = out[c];
            }
            k++;
        }
        const int regs = d16 ? (k + 1) / 2 : k;
        for (int i = 0; i < regs; i++) w_.v[(vdata + i) & 0xff][l] = packed[i];
    };

    const bool sample = (op >= 0x20 && op < 0x61) || (op >= 0x68 && op < 0x70);
    if (sample) {
        bool cl = false, deriv = false, lod = false, bias = false, lz = false, cmp = false, offs = false, gather = false;
        bool get_lod = op == Op::IMAGE_GET_LOD;
        if (op < 0x60) {
            const unsigned low = op & 0xf;
            offs = (op & 0x10) != 0;
            gather = op >= 0x40;
            cmp = low >= 8;
            cl = low == 1 || low == 3 || low == 6 || low == 9 || low == 0xb || low == 0xe;
            deriv = low == 2 || low == 3 || low == 0xa || low == 0xb;
            lod = low == 4 || low == 0xc;
            bias = low == 5 || low == 6 || low == 0xd || low == 0xe;
            lz = low == 7 || low == 0xf;
        } else if (op >= 0x68) {
            const unsigned low = op - 0x68;
            deriv = true;
            cl = (low & 1) != 0;
            cmp = (low & 2) != 0;
            offs = (low & 4) != 0;
        }
        // Address order: offset, bias, compare, derivatives, coordinates, lod/clamp.
        const int nderiv = deriv ? 2 * kDerivs[dim] : 0;
        const int a_off = 0, a_bias = a_off + (offs ? 1 : 0), a_cmp = a_bias + (bias ? 1 : 0);
        const int a_deriv = a_cmp + (cmp ? 1 : 0), a_coord = a_deriv + nderiv, a_lod = a_coord + ncoord;
        float coord[kLanes][4];
        for (int l = 0; l < kLanes; l++)
            for (int c = 0; c < ncoord; c++) coord[l][c] = F(w_.v[areg(a_coord + c)][l]);
        for (int l = 0; l < kLanes; l++) {
            if (!((exec >> l) & 1)) continue;
            SampleRequest req;
            req.dim = (int)dim;
            for (int c = 0; c < ncoord; c++) req.coord[c] = coord[l][c];
            if (offs) {
                const uint32_t o = w_.v[areg(a_off)][l];
                for (int c = 0; c < 3; c++) req.offset[c] = Sext((o >> (8 * c)) & 0x3f, 6);
            }
            if (bias) { req.has_bias = true; req.lod = F(w_.v[areg(a_bias)][l]); }
            if (cmp) { req.has_compare = true; req.compare = F(w_.v[areg(a_cmp)][l]); }
            if (lod) { req.has_lod = true; req.lod = F(w_.v[areg(a_lod)][l]); }
            if (cl) { req.has_clamp = true; req.clamp = F(w_.v[areg(a_lod + (lod ? 1 : 0))][l]); }
            req.level_zero = lz;
            if (deriv) {
                const int nd = kDerivs[dim];
                for (int c = 0; c < nd; c++) {
                    req.dx[c] = F(w_.v[areg(a_deriv + c)][l]);
                    req.dy[c] = F(w_.v[areg(a_deriv + nd + c)][l]);
                }
            } else if (!lod && !lz) {
                // Implicit derivatives from the pixel quad (lanes 0 1 / 2 3).
                const int q = l & ~3;
                for (int c = 0; c < std::min(ncoord, 3); c++) {
                    req.dx[c] = coord[q + 1][c] - coord[q][c];
                    req.dy[c] = coord[q + 2][c] - coord[q][c];
                }
            }
            req.gather = gather;
            if (gather) req.gather_comp = dmask ? __builtin_ctz(dmask) : 0;
            float out[4] = {0, 0, 0, 0};
            if (get_lod) {
                req.gather = false;
                env_.Sample(t, smp, req, out);  // the Env reports the LOD in out[0], out[1] for get_lod via has_lod=false
            } else {
                env_.Sample(t, smp, req, out);
            }
            uint32_t bits[4];
            for (int c = 0; c < 4; c++) bits[c] = U(out[c]);
            if (gather) {
                for (int c = 0; c < 4; c++) w_.v[(vdata + c) & 0xff][l] = bits[c];
            } else {
                put(l, bits, true);
            }
        }
        return Status::kNext;
    }

    switch (op) {
        case Op::IMAGE_LOAD: case Op::IMAGE_LOAD_MIP: case Op::IMAGE_LOAD_PCK: case Op::IMAGE_LOAD_PCK_SGN:
        case Op::IMAGE_LOAD_MIP_PCK: case Op::IMAGE_LOAD_MIP_PCK_SGN:
        case Op::IMAGE_STORE: case Op::IMAGE_STORE_MIP: case Op::IMAGE_STORE_PCK: case Op::IMAGE_STORE_MIP_PCK: {
            const bool mip = op == Op::IMAGE_LOAD_MIP || op == Op::IMAGE_LOAD_MIP_PCK || op == Op::IMAGE_LOAD_MIP_PCK_SGN ||
                             op == Op::IMAGE_STORE_MIP || op == Op::IMAGE_STORE_MIP_PCK;
            const bool store = op >= Op::IMAGE_STORE && op <= Op::IMAGE_STORE_MIP_PCK;
            for (int l = 0; l < kLanes; l++) {
                if (!((exec >> l) & 1)) continue;
                int c4[4] = {0, 0, 0, 0};
                for (int c = 0; c < ncoord; c++) c4[c] = (int)w_.v[areg(c)][l];
                const int level = mip ? (int)w_.v[areg(ncoord)][l] : 0;
                if (store) {
                    uint32_t in[4] = {0, 0, 0, 0};
                    int k = 0;
                    for (int c = 0; c < 4; c++) {
                        if (!((dmask >> c) & 1)) continue;
                        if (d16) {
                            const uint16_t h = (uint16_t)(w_.v[(vdata + k / 2) & 0xff][l] >> (16 * (k & 1)));
                            in[c] = U(HalfToFloat(h));
                        } else {
                            in[c] = w_.v[(vdata + k) & 0xff][l];
                        }
                        k++;
                    }
                    env_.ImageStore(t, (int)dim, c4, level, in, dmask);
                } else {
                    uint32_t out[4] = {0, 0, 0, 0};
                    env_.ImageLoad(t, (int)dim, c4, level, out);
                    put(l, out, true);
                }
            }
            (void)glc;
            return Status::kNext;
        }
        case Op::IMAGE_GET_RESINFO:
            for (int l = 0; l < kLanes; l++) {
                if (!((exec >> l) & 1)) continue;
                uint32_t out[4];
                env_.ResInfo(t, (int)w_.v[areg(0)][l], out);
                put(l, out, false);
            }
            return Status::kNext;
        default: return Unsupported("MIMG", op);
    }
}

// --- exports ---

Status Interp::Exp(uint32_t w0, uint32_t w1) {
    base_ = 2;
    const unsigned en = w0 & 0xf, target = (w0 >> 4) & 0x3f;
    const bool compr = (w0 >> 10) & 1, done = (w0 >> 11) & 1;
    const uint32_t* data[4] = {nullptr, nullptr, nullptr, nullptr};
    if (compr) {
        if (en & 3) data[0] = w_.v[w1 & 0xff];
        if (en & 0xc) data[1] = w_.v[(w1 >> 8) & 0xff];
    } else {
        for (int c = 0; c < 4; c++)
            if ((en >> c) & 1) data[c] = w_.v[(w1 >> (8 * c)) & 0xff];
    }
    uint64_t lanes = w_.Exec();
    if (target <= 8) lanes &= w_.live;
    env_.Export(w_, (int)target, en, compr, done, lanes, data);
    return Status::kNext;
}

}  // namespace

int FormatBytes(uint32_t fmt) {
    FormatInfo f;
    return LookupFormat(fmt, &f) ? FormatBytes(f) : 0;
}

bool DecodeFormat(uint32_t fmt, const uint8_t* p, uint32_t out[4], bool* integer) {
    FormatInfo f;
    if (!LookupFormat(fmt, &f)) return false;
    DecodeFormatted(p, f, out);
    if (integer) *integer = f.type == 4 || f.type == 5;
    return true;
}

bool EncodeFormat(uint32_t fmt, const uint32_t in[4], uint8_t* p) {
    FormatInfo f;
    if (!LookupFormat(fmt, &f)) return false;
    EncodeFormatted(p, f, in);
    return true;
}

Result Run(Wave& w, Env& env, uint64_t max_instructions) {
    Interp interp(w, env);
    return interp.Run(max_instructions);
}

std::string Describe(Env& env, uint64_t addr) {
    const uint8_t* p = env.Read(addr, 12);
    if (!p) return "(unmapped)";
    uint32_t words[3];
    memcpy(words, p, sizeof(words));
    char buf[64];
    snprintf(buf, sizeof(buf), "%#llx: %08x %08x %08x", (unsigned long long)addr, words[0], words[1], words[2]);
    return buf;
}

} // namespace KytyWeb::Rdna
