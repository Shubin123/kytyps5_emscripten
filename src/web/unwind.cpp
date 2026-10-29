// C++ exceptions for guest code: the libc++abi entry points games import (__cxa_throw, __cxa_begin_catch,
// _Unwind_Resume, ...) backed by a host-side unwinder. Guest frames are unwound with each module's .eh_frame
// (found through PT_GNU_EH_FRAME), and their LSDAs are read the way __gxx_personality_v0 reads them. A throw
// runs in two phases like the Itanium ABI: a search for the handler's frame, then the landing pads on the way
// there (cleanups, then the handler) are entered one at a time with that frame's registers restored; cleanups
// end in _Unwind_Resume, which continues from the frame that called it.

#include "hle.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace KytyWeb {
namespace {

HleLayout L;
std::function<uint64_t(const std::string&)> g_stub_of;

uint8_t* Ptr(uint64_t a, uint64_t n) {
    if (a < L.base || n > L.size || a - L.base > L.size - n) return nullptr;
    return L.mem + (a - L.base);
}
template <class T> T Rd(uint64_t a) {
    T v{};
    if (auto* p = Ptr(a, sizeof(T))) memcpy(&v, p, sizeof(T));
    return v;
}
template <class T> void Wr(uint64_t a, T v) {
    if (auto* p = Ptr(a, sizeof(T))) memcpy(p, &v, sizeof(T));
}
std::string Str(uint64_t a) {
    std::string s;
    for (char c; s.size() < 256 && (c = Rd<char>(a + s.size())) != 0;) s += c;
    return s;
}

// IL2CPP throws managed exceptions wrapped in an Il2CppExceptionWrapper whose first field is the Il2CppException;
// its class (name at +0x10, namespace at +0x18) and message (System.String: length at +0x10, UTF-16 at +0x14).
std::string ManagedString(uint64_t s) {
    std::string out;
    const int32_t n = s ? Rd<int32_t>(s + 0x10) : 0;
    for (int32_t i = 0; i < n && i < 400; i++) {
        const uint16_t c = Rd<uint16_t>(s + 0x14 + 2ull * i);
        out += c < 0x80 ? (char)c : '?';
    }
    return out;
}
std::string ManagedException(uint64_t wrapper) {
    const uint64_t ex = Rd<uint64_t>(wrapper), klass = ex ? Rd<uint64_t>(ex) : 0;
    if (!klass) return "(null)";
    return Str(Rd<uint64_t>(klass + 0x18)) + "." + Str(Rd<uint64_t>(klass + 0x10)) + ": " +
           ManagedString(Rd<uint64_t>(ex + 0x18));
}

int g_logs = 0;
#define EH_LOG(...)                        \
    do {                                   \
        if (g_logs < 60) {                 \
            g_logs++;                      \
            printf("[eh] " __VA_ARGS__);   \
        }                                  \
    } while (0)

// --- DWARF readers ---

uint64_t ReadUleb(uint64_t& p) {
    uint64_t v = 0;
    for (int shift = 0; shift < 64; shift += 7) {
        const uint8_t b = Rd<uint8_t>(p++);
        v |= (uint64_t)(b & 0x7f) << shift;
        if (!(b & 0x80)) break;
    }
    return v;
}
int64_t ReadSleb(uint64_t& p) {
    int64_t v = 0;
    int shift = 0;
    uint8_t b;
    do {
        b = Rd<uint8_t>(p++);
        v |= (int64_t)(b & 0x7f) << shift;
        shift += 7;
    } while ((b & 0x80) && shift < 64);
    if (shift < 64 && (b & 0x40)) v |= -((int64_t)1 << shift);
    return v;
}
// DW_EH_PE pointer encodings: low nibble is the format, 0x10 pc-relative, 0x30 relative to data_base,
// 0x80 indirect. 0xff means omitted.
bool ReadEncoded(uint64_t& p, uint8_t enc, uint64_t data_base, uint64_t* out) {
    if (enc == 0xff) { *out = 0; return true; }
    const uint64_t at = p;
    uint64_t v;
    switch (enc & 0x0f) {
        case 0x00: case 0x04: case 0x0c: v = Rd<uint64_t>(p); p += 8; break;
        case 0x01: v = ReadUleb(p); break;
        case 0x02: v = Rd<uint16_t>(p); p += 2; break;
        case 0x03: v = Rd<uint32_t>(p); p += 4; break;
        case 0x09: v = (uint64_t)ReadSleb(p); break;
        case 0x0a: v = (uint64_t)(int64_t)Rd<int16_t>(p); p += 2; break;
        case 0x0b: v = (uint64_t)(int64_t)Rd<int32_t>(p); p += 4; break;
        default: return false;
    }
    if (v) {
        switch (enc & 0x70) {
            case 0x00: break;
            case 0x10: v += at; break;
            case 0x30: v += data_base; break;
            default: return false;
        }
        if (enc & 0x80) v = Rd<uint64_t>(v);
    }
    *out = v;
    return true;
}
int EncodedSize(uint8_t enc) {
    switch (enc & 0x07) {
        case 2: return 2;
        case 3: return 4;
        default: return 8;
    }
}

// --- call frame information ---

struct EhModule { uint64_t base, end, hdr; };
std::vector<EhModule> g_modules;

struct Fde {
    uint64_t pc_begin = 0, pc_end = 0, lsda = 0, personality = 0;
    uint64_t cie_ops = 0, cie_ops_end = 0, ops = 0, ops_end = 0;
    uint64_t code_align = 1;
    int64_t data_align = -8;
    int ra_reg = 16;
    uint8_t fde_enc = 0;
};

bool ParseFde(uint64_t fde, Fde* out) {
    uint64_t p = fde;
    uint64_t len = Rd<uint32_t>(p);
    p += 4;
    if (len == 0xffffffff) { len = Rd<uint64_t>(p); p += 8; }
    const uint64_t fde_end = p + len;
    const uint64_t cie_ptr_at = p;
    const uint32_t cie_off = Rd<uint32_t>(p);
    p += 4;
    if (cie_off == 0) return false;  // a CIE, not an FDE

    uint64_t q = cie_ptr_at - cie_off;
    uint64_t cie_len = Rd<uint32_t>(q);
    q += 4;
    if (cie_len == 0xffffffff) { cie_len = Rd<uint64_t>(q); q += 8; }
    const uint64_t cie_end = q + cie_len;
    q += 4;  // CIE id
    const uint8_t version = Rd<uint8_t>(q++);
    std::string aug;
    for (char c; (c = Rd<char>(q++)) != 0 && aug.size() < 16;) aug += c;
    out->code_align = ReadUleb(q);
    out->data_align = ReadSleb(q);
    out->ra_reg = version == 1 ? Rd<uint8_t>(q++) : (int)ReadUleb(q);
    uint8_t lsda_enc = 0xff;
    const bool z = !aug.empty() && aug[0] == 'z';
    if (z) {
        const uint64_t aug_len = ReadUleb(q), aug_end = q + aug_len;
        for (size_t i = 1; i < aug.size(); i++) {
            if (aug[i] == 'P') {
                const uint8_t enc = Rd<uint8_t>(q++);
                if (!ReadEncoded(q, enc, 0, &out->personality)) return false;
            } else if (aug[i] == 'L') {
                lsda_enc = Rd<uint8_t>(q++);
            } else if (aug[i] == 'R') {
                out->fde_enc = Rd<uint8_t>(q++);
            }
        }
        q = aug_end;
    }
    out->cie_ops = q;
    out->cie_ops_end = cie_end;

    uint64_t range;
    if (!ReadEncoded(p, out->fde_enc, 0, &out->pc_begin) || !ReadEncoded(p, out->fde_enc & 0x0f, 0, &range))
        return false;
    out->pc_end = out->pc_begin + range;
    if (z) {
        const uint64_t aug_len = ReadUleb(p), aug_end = p + aug_len;
        uint64_t lp = p;
        if (lsda_enc != 0xff && !ReadEncoded(lp, lsda_enc, 0, &out->lsda)) return false;
        p = aug_end;
    }
    out->ops = p;
    out->ops_end = fde_end;
    return true;
}

// The eh_frame_hdr search table: sorted (initial location, FDE) pairs, datarel sdata4 as linkers emit them.
bool FindFde(uint64_t pc, Fde* out) {
    for (const EhModule& m : g_modules) {
        if (pc < m.base || pc >= m.end) continue;
        uint64_t p = m.hdr + 4;
        uint64_t eh_frame, count;
        if (Rd<uint8_t>(m.hdr) != 1 || Rd<uint8_t>(m.hdr + 3) != 0x3b ||
            !ReadEncoded(p, Rd<uint8_t>(m.hdr + 1), m.hdr, &eh_frame) ||
            !ReadEncoded(p, Rd<uint8_t>(m.hdr + 2), m.hdr, &count))
            return false;
        uint64_t lo = 0, hi = count;
        while (lo < hi) {
            const uint64_t mid = (lo + hi) / 2;
            if (m.hdr + (int64_t)Rd<int32_t>(p + 8 * mid) <= pc) lo = mid + 1;
            else hi = mid;
        }
        if (!lo) return false;
        const uint64_t fde = m.hdr + (int64_t)Rd<int32_t>(p + 8 * (lo - 1) + 4);
        return ParseFde(fde, out) && pc >= out->pc_begin && pc < out->pc_end;
    }
    return false;
}

// DWARF register numbers: 0 rax, 1 rdx, 2 rcx, 3 rbx, 4 rsi, 5 rdi, 6 rbp, 7 rsp, 8-15 r8-r15, 16 return address.
constexpr int kRegs = 17, kRsp = 7, kRa = 16;
struct Frame { uint64_t r[kRegs]; };

enum class Rule : uint8_t { Unchanged, Undefined, Offset, ValOffset, Register };
struct Row {
    Rule rule[kRegs] = {};
    int64_t arg[kRegs] = {};
    int cfa_reg = kRsp;
    int64_t cfa_off = 8;
};

// Runs CFA instructions until the row covering `pc` is built (pc = UINT64_MAX runs them all).
bool RunCfa(uint64_t p, uint64_t end, const Fde& f, uint64_t pc, Row& row, const Row& initial) {
    uint64_t loc = f.pc_begin;
    std::vector<Row> saved;
    auto set = [&](uint64_t reg, Rule rule, int64_t arg) {
        if (reg < kRegs) { row.rule[reg] = rule; row.arg[reg] = arg; }
    };
    auto advance = [&](uint64_t delta) {
        loc += delta * f.code_align;
        return loc <= pc;
    };
    while (p < end) {
        const uint8_t op = Rd<uint8_t>(p++);
        switch (op >> 6) {
            case 1: if (!advance(op & 0x3f)) return true; continue;
            case 2: set(op & 0x3f, Rule::Offset, (int64_t)ReadUleb(p) * f.data_align); continue;
            case 3: if ((op & 0x3f) < kRegs) { row.rule[op & 0x3f] = initial.rule[op & 0x3f]; row.arg[op & 0x3f] = initial.arg[op & 0x3f]; } continue;
            default: break;
        }
        switch (op) {
            case 0x00: break;
            case 0x01: {
                uint64_t to;
                if (!ReadEncoded(p, f.fde_enc, 0, &to)) return false;
                if (to > pc) return true;
                loc = to;
                break;
            }
            case 0x02: { const uint8_t d = Rd<uint8_t>(p); p += 1; if (!advance(d)) return true; break; }
            case 0x03: { const uint16_t d = Rd<uint16_t>(p); p += 2; if (!advance(d)) return true; break; }
            case 0x04: { const uint32_t d = Rd<uint32_t>(p); p += 4; if (!advance(d)) return true; break; }
            case 0x05: { const uint64_t r = ReadUleb(p); set(r, Rule::Offset, (int64_t)ReadUleb(p) * f.data_align); break; }
            case 0x06: { const uint64_t r = ReadUleb(p); if (r < kRegs) { row.rule[r] = initial.rule[r]; row.arg[r] = initial.arg[r]; } break; }
            case 0x07: set(ReadUleb(p), Rule::Undefined, 0); break;
            case 0x08: set(ReadUleb(p), Rule::Unchanged, 0); break;
            case 0x09: { const uint64_t r = ReadUleb(p); set(r, Rule::Register, (int64_t)ReadUleb(p)); break; }
            case 0x0a: saved.push_back(row); break;
            case 0x0b: if (!saved.empty()) { row = saved.back(); saved.pop_back(); } break;
            case 0x0c: row.cfa_reg = (int)ReadUleb(p); row.cfa_off = (int64_t)ReadUleb(p); break;
            case 0x0d: row.cfa_reg = (int)ReadUleb(p); break;
            case 0x0e: row.cfa_off = (int64_t)ReadUleb(p); break;
            case 0x11: { const uint64_t r = ReadUleb(p); set(r, Rule::Offset, ReadSleb(p) * f.data_align); break; }
            case 0x12: row.cfa_reg = (int)ReadUleb(p); row.cfa_off = ReadSleb(p) * f.data_align; break;
            case 0x13: row.cfa_off = ReadSleb(p) * f.data_align; break;
            case 0x14: { const uint64_t r = ReadUleb(p); set(r, Rule::ValOffset, (int64_t)ReadUleb(p) * f.data_align); break; }
            case 0x15: { const uint64_t r = ReadUleb(p); set(r, Rule::ValOffset, ReadSleb(p) * f.data_align); break; }
            case 0x2e: ReadUleb(p); break;  // GNU_args_size
            case 0x2f: { const uint64_t r = ReadUleb(p); set(r, Rule::Offset, -(int64_t)ReadUleb(p) * f.data_align); break; }
            default: return false;  // DWARF expressions aren't supported
        }
    }
    return true;
}

// Replaces `f` (registers at a call site in some function) with its caller's registers at its call site.
bool UnwindFrame(Frame& f, const Fde& fde) {
    Row initial;
    if (!RunCfa(fde.cie_ops, fde.cie_ops_end, fde, UINT64_MAX, initial, initial)) {
        EH_LOG("  CIE instructions failed for 0x%llx\n", (unsigned long long)fde.pc_begin);
        return false;
    }
    Row row = initial;
    if (!RunCfa(fde.ops, fde.ops_end, fde, f.r[kRa] - 1, row, initial)) {
        EH_LOG("  FDE instructions failed for 0x%llx\n", (unsigned long long)fde.pc_begin);
        return false;
    }
    if (row.cfa_reg < 0 || row.cfa_reg >= kRegs) return false;
    const uint64_t cfa = f.r[row.cfa_reg] + row.cfa_off;
    Frame caller = f;
    for (int i = 0; i < kRegs; i++) {
        switch (row.rule[i]) {
            case Rule::Offset: caller.r[i] = Rd<uint64_t>(cfa + row.arg[i]); break;
            case Rule::ValOffset: caller.r[i] = cfa + row.arg[i]; break;
            case Rule::Register: if (row.arg[i] < kRegs) caller.r[i] = f.r[row.arg[i]]; break;
            default: break;
        }
    }
    caller.r[kRsp] = cfa;
    if (fde.ra_reg != kRa && fde.ra_reg < kRegs) caller.r[kRa] = caller.r[fde.ra_reg];
    f = caller;
    if (!f.r[kRa]) EH_LOG("  return address is 0 (cfa 0x%llx, cfa reg %d + %lld)\n", (unsigned long long)cfa,
                          row.cfa_reg, (long long)row.cfa_off);
    return f.r[kRa] != 0;
}

// The frame that called an import stub, from the registers at the stub ([rsp] is the return address).
Frame CallerOf(const X86Regs& r) {
    Frame f{};
    const uint64_t gp[16] = {r.rax, r.rdx, r.rcx, r.rbx, r.rsi, r.rdi, r.rbp, r.rsp + 8,
                             r.r8,  r.r9,  r.r10, r.r11, r.r12, r.r13, r.r14, r.r15};
    memcpy(f.r, gp, sizeof(gp));
    f.r[kRa] = Rd<uint64_t>(r.rsp);
    return f;
}

// --- C++ type matching ---

enum class Kind { Other, Class, SiClass, VmiClass, Pointer };
Kind KindOf(uint64_t type_info) {
    // type_info vtables are data imports, bound as the import's stub plus the vtable's address point (16)
    static const uint64_t cls = g_stub_of("byV+FWlAnB4"), si = g_stub_of("pZ9WXcClPO8"),
                          vmi = g_stub_of("9ByRMdo7ywg"), ptr = g_stub_of("aeHxLWwq0gQ");
    const uint64_t vtable = Rd<uint64_t>(type_info) - 16;
    if (cls && vtable == cls) return Kind::Class;
    if (si && vtable == si) return Kind::SiClass;
    if (vmi && vtable == vmi) return Kind::VmiClass;
    if (ptr && vtable == ptr) return Kind::Pointer;
    return Kind::Other;
}
std::string TypeName(uint64_t type_info) {
    std::string n = Str(Rd<uint64_t>(type_info + 8));
    return !n.empty() && n[0] == '*' ? n.substr(1) : n;
}
bool SameType(uint64_t a, uint64_t b) { return a == b || (a && b && TypeName(a) == TypeName(b)); }

// Whether an object of type `derived` at `obj` has `base` as a public base; *adjusted gets the base's address.
bool FindBase(uint64_t derived, uint64_t base, uint64_t obj, uint64_t* adjusted, int depth = 0) {
    if (depth > 32) return false;
    if (SameType(derived, base)) { *adjusted = obj; return true; }
    switch (KindOf(derived)) {
        case Kind::SiClass: return FindBase(Rd<uint64_t>(derived + 16), base, obj, adjusted, depth + 1);
        case Kind::VmiClass: {
            const uint32_t count = Rd<uint32_t>(derived + 20);
            for (uint32_t i = 0; i < count && i < 64; i++) {
                const uint64_t base_type = Rd<uint64_t>(derived + 24 + 16ull * i);
                const int64_t flags = Rd<int64_t>(derived + 32 + 16ull * i);
                if (!(flags & 2)) continue;  // only public bases can catch
                int64_t offset = flags >> 8;
                if (flags & 1) offset = Rd<int64_t>(Rd<uint64_t>(obj) + offset);  // virtual base
                if (FindBase(base_type, base, obj + offset, adjusted, depth + 1)) return true;
            }
            return false;
        }
        default: return false;
    }
}
// catch_type 0 is catch (...). For pointers, `obj` holds the thrown pointer and *adjusted is the pointer value.
bool CanCatch(uint64_t catch_type, uint64_t thrown, uint64_t obj, uint64_t* adjusted) {
    if (!catch_type) { *adjusted = obj; return true; }
    if (KindOf(catch_type) == Kind::Pointer && KindOf(thrown) == Kind::Pointer) {
        const uint64_t value = Rd<uint64_t>(obj);
        if (!value) { *adjusted = 0; return true; }
        return FindBase(Rd<uint64_t>(thrown + 24), Rd<uint64_t>(catch_type + 24), value, adjusted);
    }
    return FindBase(thrown, catch_type, obj, adjusted);
}

// --- LSDA (the C++ personality's tables) ---

struct Lsda {
    uint64_t lp_start = 0, ttype_base = 0, call_sites = 0, call_sites_end = 0, actions = 0;
    uint8_t ttype_enc = 0xff, cs_enc = 0x01;
};
bool ParseLsda(uint64_t p, uint64_t func_start, Lsda* l) {
    const uint8_t lp_enc = Rd<uint8_t>(p++);
    l->lp_start = func_start;
    if (lp_enc != 0xff && !ReadEncoded(p, lp_enc, 0, &l->lp_start)) return false;
    l->ttype_enc = Rd<uint8_t>(p++);
    if (l->ttype_enc != 0xff) {
        const uint64_t off = ReadUleb(p);
        l->ttype_base = p + off;
    }
    l->cs_enc = Rd<uint8_t>(p++);
    const uint64_t len = ReadUleb(p);
    l->call_sites = p;
    l->call_sites_end = l->actions = p + len;
    return true;
}
// The call site containing ip: its landing pad (0: none) and first action record (0: cleanup only).
bool FindCallSite(const Lsda& l, uint64_t func_start, uint64_t ip, uint64_t* lp, uint64_t* action) {
    for (uint64_t p = l.call_sites; p < l.call_sites_end;) {
        uint64_t start, len, pad;
        if (!ReadEncoded(p, l.cs_enc, 0, &start) || !ReadEncoded(p, l.cs_enc, 0, &len) ||
            !ReadEncoded(p, l.cs_enc, 0, &pad))
            return false;
        const uint64_t act = ReadUleb(p);
        if (ip < func_start + start) return false;
        if (ip < func_start + start + len) {
            *lp = pad ? l.lp_start + pad : 0;
            *action = act ? l.actions + act - 1 : 0;
            return true;
        }
    }
    return false;
}
enum class Match { None, Cleanup, Handler };
Match EvalActions(const Lsda& l, uint64_t action, uint64_t thrown, uint64_t obj, int64_t* selector,
                  uint64_t* adjusted) {
    if (!action) return Match::Cleanup;
    bool cleanup = false;
    for (uint64_t p = action, guard = 0; guard < 256; guard++) {
        const int64_t filter = ReadSleb(p);
        const uint64_t disp_at = p;
        const int64_t disp = ReadSleb(p);
        if (filter > 0) {
            uint64_t q = l.ttype_base - (uint64_t)filter * EncodedSize(l.ttype_enc), catch_type = 0;
            ReadEncoded(q, l.ttype_enc, 0, &catch_type);
            if (CanCatch(catch_type, thrown, obj, adjusted)) { *selector = filter; return Match::Handler; }
        } else if (filter == 0) {
            cleanup = true;
        }  // negative: exception specifications, which never match here
        if (!disp) break;
        p = disp_at + disp;
    }
    return cleanup ? Match::Cleanup : Match::None;
}

// --- exceptions ---

// libc++abi's x86-64 __cxa_exception, 0x80 bytes in front of the thrown object.
constexpr uint64_t kHeaderSize = 0x80;
constexpr uint64_t kType = 0x08, kDestructor = 0x10, kNext = 0x28, kHandlerCount = 0x30, kSwitchValue = 0x34,
                   kAdjusted = 0x50, kUnwindHeader = 0x60, kHandlerFrame = 0x70;
constexpr uint64_t kGuestClass = 0x434c4e47432b2b00ull;  // "CLNGC++\0", libc++abi's exception class

struct ThreadState { uint64_t caught = 0; int uncaught = 0; };
std::unordered_map<uint64_t, ThreadState> g_threads;
ThreadState& Globals() { return g_threads[HleCurrentThread()]; }

void Terminate(const char* why, uint64_t header) {
    char buf[320];
    snprintf(buf, sizeof(buf), "C++ exception %s (type %s); std::terminate", why,
             header ? TypeName(Rd<uint64_t>(header + kType)).c_str() : "?");
    HleTerminate(buf);
}

// Resumes guest execution at a landing pad in frame f.
void Enter(X86Regs& r, const Frame& f, uint64_t lp, uint64_t header, int64_t selector) {
    r.rax = header + kUnwindHeader;
    r.rdx = (uint64_t)selector;
    r.rip = lp;
    r.rsp = f.r[kRsp];
    r.rbp = f.r[6];
    r.rbx = f.r[3];
    r.r12 = f.r[12];
    r.r13 = f.r[13];
    r.r14 = f.r[14];
    r.r15 = f.r[15];
    L.cpu->RequestYield(true);  // don't execute the stub's RET; the thread continues at the landing pad
}

// Phase 2 from frame f: enters the next landing pad on the way to the handler's frame.
void ContinueUnwind(X86Regs& r, uint64_t header, Frame f) {
    const uint64_t handler_frame = Rd<uint64_t>(header + kHandlerFrame);
    const uint64_t thrown = Rd<uint64_t>(header + kType), obj = header + kHeaderSize;
    for (int depth = 0; depth < 512; depth++) {
        Fde fde;
        const uint64_t ip = f.r[kRa] - 1;
        if (!FindFde(ip, &fde)) return Terminate("unwound into code without unwind information", header);
        const bool is_handler = f.r[kRsp] == handler_frame;
        Lsda l;
        uint64_t lp = 0, action = 0;
        if (fde.lsda && fde.personality && ParseLsda(fde.lsda, fde.pc_begin, &l) &&
            FindCallSite(l, fde.pc_begin, ip, &lp, &action) && lp) {
            if (is_handler) {
                EH_LOG("  handler landing pad 0x%llx (selector %d)\n", (unsigned long long)lp,
                       Rd<int32_t>(header + kSwitchValue));
                return Enter(r, f, lp, header, Rd<int32_t>(header + kSwitchValue));
            }
            int64_t selector = 0;
            uint64_t adjusted = 0;
            if (EvalActions(l, action, thrown, obj, &selector, &adjusted) == Match::Cleanup)
                return Enter(r, f, lp, header, 0);
        }
        if (is_handler) return Terminate("handler frame has no landing pad", header);
        if (!UnwindFrame(f, fde)) return Terminate("could not unwind a frame", header);
    }
    Terminate("unwinding did not reach the handler", header);
}

// Phase 1 from frame f: finds the handler's frame, then starts phase 2.
void RaiseException(X86Regs& r, uint64_t header, const Frame& start) {
    const uint64_t thrown = Rd<uint64_t>(header + kType), obj = header + kHeaderSize;
    Frame f = start;
    for (int depth = 0; depth < 512; depth++) {
        Fde fde;
        const uint64_t ip = f.r[kRa] - 1;
        if (!FindFde(ip, &fde)) {
            EH_LOG("  search: no FDE for 0x%llx\n", (unsigned long long)ip);
            break;
        }
        EH_LOG("  search: frame %d ip 0x%llx in 0x%llx-0x%llx lsda 0x%llx personality 0x%llx rsp 0x%llx\n", depth,
               (unsigned long long)ip, (unsigned long long)fde.pc_begin, (unsigned long long)fde.pc_end,
               (unsigned long long)fde.lsda, (unsigned long long)fde.personality, (unsigned long long)f.r[kRsp]);
        Lsda l;
        uint64_t lp = 0, action = 0;
        if (fde.lsda && fde.personality && ParseLsda(fde.lsda, fde.pc_begin, &l)) {
            if (!FindCallSite(l, fde.pc_begin, ip, &lp, &action))
                return Terminate("thrown through a call site missing from its function's table", header);
            int64_t selector = 0;
            uint64_t adjusted = obj;
            if (lp && EvalActions(l, action, thrown, obj, &selector, &adjusted) == Match::Handler) {
                Wr<uint64_t>(header + kHandlerFrame, f.r[kRsp]);
                Wr<int32_t>(header + kSwitchValue, (int32_t)selector);
                Wr<uint64_t>(header + kAdjusted, adjusted);
                EH_LOG("%s: handler in the function at 0x%llx, %d frames up\n", TypeName(thrown).c_str(),
                       (unsigned long long)fde.pc_begin, depth);
                static int managed_logs = 0;
                if (managed_logs < 200 && TypeName(thrown).find("Il2CppExceptionWrapper") != std::string::npos) {
                    managed_logs++;
                    printf("[eh] managed exception %s (thrown from 0x%llx)\n", ManagedException(obj).c_str(),
                           (unsigned long long)start.r[kRa]);
                }
                return ContinueUnwind(r, header, start);
            }
        }
        if (!UnwindFrame(f, fde)) break;
    }
    Terminate("not caught", header);
}

void AllocateException(X86Regs& r) {  // (thrown size)
    const uint64_t header = HleMalloc(r.rdi + kHeaderSize, 16);
    if (!header) return HleTerminate("out of memory allocating a C++ exception");
    if (auto* p = Ptr(header, kHeaderSize)) memset(p, 0, kHeaderSize);
    r.rax = header + kHeaderSize;
}
void FreeException(X86Regs& r) { HleFree(r.rdi - kHeaderSize); }
void Throw(X86Regs& r) {  // (object, type_info, destructor)
    const uint64_t header = r.rdi - kHeaderSize;
    Wr<uint64_t>(header, 1);  // reference count
    Wr<uint64_t>(header + kType, r.rsi);
    Wr<uint64_t>(header + kDestructor, r.rdx);
    Wr<uint64_t>(header + kUnwindHeader, kGuestClass);
    Globals().uncaught++;
    RaiseException(r, header, CallerOf(r));
}
void Rethrow(X86Regs& r) {
    ThreadState& g = Globals();
    const uint64_t header = g.caught;
    if (!header) return HleTerminate("C++ rethrow with no exception being handled; std::terminate");
    // A negative handler count marks it rethrown, so leaving the catch block doesn't destroy it.
    Wr<int32_t>(header + kHandlerCount, -Rd<int32_t>(header + kHandlerCount));
    g.uncaught++;
    RaiseException(r, header, CallerOf(r));
}
void UnwindResume(X86Regs& r) { ContinueUnwind(r, r.rdi - kUnwindHeader, CallerOf(r)); }
void BeginCatch(X86Regs& r) {  // (_Unwind_Exception*) -> the caught object (or pointer value)
    const uint64_t header = r.rdi - kUnwindHeader;
    ThreadState& g = Globals();
    int32_t count = Rd<int32_t>(header + kHandlerCount);
    count = count < 0 ? -count + 1 : count + 1;
    Wr<int32_t>(header + kHandlerCount, count);
    if (g.caught != header) {
        Wr<uint64_t>(header + kNext, g.caught);
        g.caught = header;
    }
    if (g.uncaught > 0) g.uncaught--;
    r.rax = Rd<uint64_t>(header + kAdjusted);
}
void EndCatch(X86Regs& r) {
    ThreadState& g = Globals();
    const uint64_t header = g.caught;
    if (!header) { r.rax = 0; return; }
    int32_t count = Rd<int32_t>(header + kHandlerCount);
    if (count < 0) {  // rethrown: it lives on, only leave this catch block
        if (++count == 0) g.caught = Rd<uint64_t>(header + kNext);
        Wr<int32_t>(header + kHandlerCount, count);
    } else if (--count == 0) {
        g.caught = Rd<uint64_t>(header + kNext);
        Wr<int32_t>(header + kHandlerCount, 0);
        if (const uint64_t dtor = Rd<uint64_t>(header + kDestructor)) L.call_guest(dtor, header + kHeaderSize, 0, 0);
        HleFree(header);
    } else {
        Wr<int32_t>(header + kHandlerCount, count);
    }
    r.rax = 0;
}
void GetExceptionPtr(X86Regs& r) { r.rax = Rd<uint64_t>(r.rdi - kUnwindHeader + kAdjusted); }
void CurrentExceptionType(X86Regs& r) {
    const uint64_t header = Globals().caught;
    r.rax = header ? Rd<uint64_t>(header + kType) : 0;
}
void UncaughtException(X86Regs& r) { r.rax = Globals().uncaught > 0; }
void UncaughtExceptions(X86Regs& r) { r.rax = (uint64_t)Globals().uncaught; }

struct Entry {
    const char* nid;
    const char* name;
    HleFn fn;
};
const Entry kEntries[] = {
    {"cfAXurvfl5o", "__cxa_allocate_exception", AllocateException},
    {"nOIEswYD4Ig", "__cxa_free_exception", FreeException},
    {"vkuuLfhnSZI", "__cxa_throw", Throw},
    {"ZL9FV4mJXxo", "__cxa_rethrow", Rethrow},
    {"3cUUypQzMiI", "__cxa_begin_catch", BeginCatch},
    {"lX+4FNUklF0", "__cxa_end_catch", EndCatch},
    {"Y6Sl4Xw7gfA", "__cxa_get_exception_ptr", GetExceptionPtr},
    {"BxPeH9TTcs4", "__cxa_current_exception_type", CurrentExceptionType},
    {"vyQXAltCcuI", "__cxa_uncaught_exceptions", UncaughtExceptions},
    {"Q1BL70XVV0o", "_ZSt18uncaught_exceptionv", UncaughtException},
    {"g8p5Q9swEHg", "_ZSt19uncaught_exceptionsv", UncaughtExceptions},
    {"f1zwJ3jAI2k", "_Unwind_Resume", UnwindResume},
};

const Entry* Lookup(const std::string& nid) {
    for (const auto& e : kEntries)
        if (nid == e.nid) return &e;
    return nullptr;
}

} // namespace

void EhSetup(const HleLayout& layout, std::function<uint64_t(const std::string&)> stub_of) {
    L = layout;
    g_stub_of = std::move(stub_of);
}

void EhRegisterModule(uint64_t base, uint64_t size, uint64_t eh_frame_hdr) {
    g_modules.push_back({base, base + size, eh_frame_hdr});
}

HleFn EhFind(const std::string& nid) {
    const Entry* e = Lookup(nid);
    return e ? e->fn : nullptr;
}

const char* EhName(const std::string& nid) {
    const Entry* e = Lookup(nid);
    return e ? e->name : nullptr;
}

} // namespace KytyWeb
