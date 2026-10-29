// PS5 SDK functions implemented against the emulated address space.
// Single-threaded: thread creation is recorded but threads never run, and blocking waits return at once.

#include "hle.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <unordered_map>
#include <vector>

namespace KytyWeb {
namespace {

struct NidName { const char* nid; const char* name; };
const NidName kNames[] = {
#include "hle_names.inc"
};

HleLayout L;
bool g_halt = false;

// --- guest memory access ---

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
size_t StrLen(uint64_t a, size_t max = SIZE_MAX) {
    if (a < L.base || a - L.base >= L.size) return 0;
    const uint8_t* p = L.mem + (a - L.base);
    size_t avail = std::min<uint64_t>(L.size - (a - L.base), max);
    const void* z = memchr(p, 0, avail);
    return z ? (size_t)((const uint8_t*)z - p) : avail;
}
std::string Str(uint64_t a, size_t max = 4096) {
    size_t n = StrLen(a, max);
    return n ? std::string(reinterpret_cast<const char*>(Ptr(a, n)), n) : std::string();
}
void PutStr(uint64_t dst, uint64_t cap, const std::string& s) {
    if (!cap) return;
    size_t n = std::min<size_t>(s.size(), cap - 1);
    if (auto* p = Ptr(dst, n + 1)) { memcpy(p, s.data(), n); p[n] = 0; }
}
uint64_t StackArg(const X86Regs& r, int i) { return Rd<uint64_t>(r.rsp + 8 + 8 * i); }

float F(const X86Regs& r, int i) { float f; memcpy(&f, r.xmm[i], 4); return f; }
double D(const X86Regs& r, int i) { double d; memcpy(&d, r.xmm[i], 8); return d; }
void RetF(X86Regs& r, float f) { memset(r.xmm[0], 0, 16); memcpy(r.xmm[0], &f, 4); }
void RetD(X86Regs& r, double d) { memset(r.xmm[0], 0, 16); memcpy(r.xmm[0], &d, 8); }
void Ret(X86Regs& r, int64_t v) { r.rax = (uint64_t)v; }

// --- error codes ---

constexpr int64_t kSceErrEnoent = (int32_t)0x80020002;
constexpr int64_t kSceErrEnomem = (int32_t)0x8002000C;
constexpr int64_t kSceErrEacces = (int32_t)0x8002000D;
constexpr int64_t kSceErrEinval = (int32_t)0x80020016;
constexpr int64_t kSceErrEagain = (int32_t)0x80020023;
constexpr int64_t kSceErrEtimedout = (int32_t)0x8002003C;

int BsdErrno(int e) {
    switch (e) {
        case ENOENT: return 2;
        case EBADF: return 9;
        case ENOMEM: return 12;
        case EACCES: return 13;
        case EEXIST: return 17;
        case ENOTDIR: return 20;
        case EISDIR: return 21;
        case EINVAL: return 22;
        case ENOSPC: return 28;
        default: return 5;
    }
}
int64_t SceErr(int host_errno) { return (int32_t)(0x80020000u | (uint32_t)BsdErrno(host_errno)); }

uint64_t g_errno_addr = 0;
void SetErrno(int host_errno) { Wr<int32_t>(g_errno_addr, BsdErrno(host_errno)); }

// --- logging ---

int g_game_lines = 0;
void GameLog(const std::string& s) {
    if (s.empty() || g_game_lines >= 3000) return;
    size_t start = 0;
    while (start < s.size() && g_game_lines < 3000) {
        size_t nl = s.find('\n', start);
        std::string line = s.substr(start, nl == std::string::npos ? std::string::npos : nl - start);
        if (!line.empty()) { printf("[game] %s\n", line.c_str()); g_game_lines++; }
        if (nl == std::string::npos) break;
        start = nl + 1;
    }
}

void Halt(const char* why) {
    printf("[hle] %s\n", why);
    g_halt = true;
    L.cpu->Stop();
}

// --- malloc arena ---

struct Heap {
    uint64_t top = 0, end = 0, in_use = 0, peak = 0;
    std::unordered_map<uint64_t, uint64_t> live;
    std::multimap<uint64_t, uint64_t> free_blocks;  // size -> addr

    uint64_t Alloc(uint64_t size, uint64_t align) {
        size = (std::max<uint64_t>(size, 1) + 15) & ~15ULL;
        if (align < 16) align = 16;
        uint64_t a = 0, s = 0;
        auto it = free_blocks.lower_bound(size);
        for (int i = 0; i < 16 && it != free_blocks.end() && it->first <= size * 2; ++i, ++it) {
            if (it->second % align == 0) { a = it->second; s = it->first; free_blocks.erase(it); break; }
        }
        if (!a) {
            a = (top + align - 1) / align * align;
            if (a + size > end || a + size < a) {
                printf("[hle] malloc arena exhausted (request 0x%llx)\n", (unsigned long long)size);
                return 0;
            }
            top = a + size;
            s = size;
        }
        live[a] = s;
        in_use += s;
        peak = std::max(peak, in_use);
        return a;
    }
    void Free(uint64_t a) {
        auto it = live.find(a);
        if (it == live.end()) return;
        in_use -= it->second;
        free_blocks.emplace(it->second, a);
        live.erase(it);
    }
    uint64_t SizeOf(uint64_t a) const {
        auto it = live.find(a);
        return it == live.end() ? 0 : it->second;
    }
};
Heap g_heap;

// --- virtual memory map ---

struct Region {
    uint64_t start, end, phys;
    int prot;
    HleRegion kind;
    int mtype;
    char name[32];
};
std::map<uint64_t, Region> g_regions;
uint64_t g_vm_top = 0;

void Carve(uint64_t start, uint64_t end) {
    auto it = g_regions.lower_bound(start);
    if (it != g_regions.begin()) --it;
    while (it != g_regions.end() && it->second.start < end) {
        Region g = it->second;
        if (g.end <= start) { ++it; continue; }
        it = g_regions.erase(it);
        if (g.start < start) { Region left = g; left.end = start; g_regions[left.start] = left; }
        if (g.end > end) {
            Region right = g;
            right.phys += end - g.start;
            right.start = end;
            g_regions[right.start] = right;
        }
    }
}

void AddRegion(uint64_t start, uint64_t size, int prot, HleRegion kind, uint64_t phys, int mtype, const char* name) {
    if (!size) return;
    Carve(start, start + size);
    Region g{start, start + size, phys, prot, kind, mtype, {}};
    if (name) strncpy(g.name, name, sizeof(g.name) - 1);
    g_regions[start] = g;
    if (start >= L.vm_start && start + size <= L.vm_end) g_vm_top = std::max(g_vm_top, start + size);
}

uint64_t VmAlloc(uint64_t size, uint64_t align, int prot, HleRegion kind, uint64_t phys, int mtype, const char* name) {
    size = (size + 0x3FFF) & ~0x3FFFULL;
    if (align < 0x4000) align = 0x4000;
    uint64_t a = (g_vm_top + align - 1) / align * align;
    if (!size || a + size > L.vm_end) {
        printf("[hle] address space exhausted (request 0x%llx, %s)\n", (unsigned long long)size, name ? name : "");
        return 0;
    }
    AddRegion(a, size, prot, kind, phys, mtype, name);
    return a;
}

int64_t MapCommon(uint64_t addr_ptr, uint64_t len, int prot, int flags, uint64_t align, HleRegion kind,
                  uint64_t phys, int mtype, const char* name) {
    constexpr int kMapFixed = 0x10;
    uint64_t want = Rd<uint64_t>(addr_ptr);
    uint64_t va;
    if ((flags & kMapFixed) && want) {
        if (!Ptr(want, len)) return kSceErrEinval;
        va = want;
        AddRegion(va, len, prot, kind, phys, mtype, name);
    } else {
        va = VmAlloc(len, align, prot, kind, phys, mtype, name);
        if (!va) return kSceErrEnomem;
    }
    Wr<uint64_t>(addr_ptr, va);
    return 0;
}

// Direct memory is a "physical" space mapped into the vm range; report what can actually be mapped.
struct DmemBlock { uint64_t len; int mtype; };
std::map<uint64_t, DmemBlock> g_dmem;
uint64_t g_dmem_top = 0;

int64_t AllocDirect(uint64_t search_start, uint64_t search_end, uint64_t len, uint64_t align, int mtype, uint64_t out) {
    if (align < 0x10000) align = 0x10000;
    uint64_t limit = std::min<uint64_t>(search_end, L.vm_end - L.vm_start);
    uint64_t phys = (std::max(g_dmem_top, search_start) + align - 1) / align * align;
    if (!len || phys + len > limit) {
        printf("[hle] direct memory exhausted (request 0x%llx)\n", (unsigned long long)len);
        return kSceErrEagain;
    }
    g_dmem_top = phys + len;
    g_dmem[phys] = {len, mtype};
    Wr<uint64_t>(out, phys);
    return 0;
}

// --- misc state ---

uint64_t g_thread_handle = 0, g_ctype = 0, g_tolower = 0, g_toupper = 0, g_tm = 0;
std::unordered_map<uint32_t, uint64_t> g_tls_values;
uint32_t g_next_key = 1;
uint64_t g_strtok_next = 0;
const auto g_start = std::chrono::steady_clock::now();

uint64_t NewHandle() {
    uint64_t h = g_heap.Alloc(64, 16);
    if (auto* p = Ptr(h, 64)) memset(p, 0, 64);
    return h;
}
int64_t UsecSinceStart() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - g_start).count();
}
int64_t UnixUsec() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// --- printf ---

struct ArgSrc {
    virtual ~ArgSrc() = default;
    virtual uint64_t Int() = 0;
    virtual double Dbl() = 0;
};

uint64_t GpReg(const X86Regs& r, int i) {
    switch (i) {
        case 0: return r.rdi;
        case 1: return r.rsi;
        case 2: return r.rdx;
        case 3: return r.rcx;
        case 4: return r.r8;
        default: return r.r9;
    }
}

struct RegArgs : ArgSrc {
    const X86Regs& r;
    int gp, fp = 0;
    uint64_t stack;
    RegArgs(const X86Regs& regs, int first_gp) : r(regs), gp(first_gp), stack(regs.rsp + 8) {}
    uint64_t Int() override {
        if (gp < 6) return GpReg(r, gp++);
        uint64_t v = Rd<uint64_t>(stack);
        stack += 8;
        return v;
    }
    double Dbl() override {
        if (fp < 8) return D(r, fp++);
        uint64_t bits = Rd<uint64_t>(stack);
        stack += 8;
        double d;
        memcpy(&d, &bits, 8);
        return d;
    }
};

// SysV va_list: {u32 gp_offset, u32 fp_offset, void* overflow_arg_area, void* reg_save_area}
struct VaArgs : ArgSrc {
    uint64_t va;
    explicit VaArgs(uint64_t v) : va(v) {}
    uint64_t Next8(uint32_t off_field, uint32_t limit, uint32_t step) {
        uint32_t off = Rd<uint32_t>(va + off_field);
        if (off < limit) {
            Wr<uint32_t>(va + off_field, off + step);
            return Rd<uint64_t>(Rd<uint64_t>(va + 16) + off);
        }
        uint64_t ov = Rd<uint64_t>(va + 8);
        Wr<uint64_t>(va + 8, ov + 8);
        return Rd<uint64_t>(ov);
    }
    uint64_t Int() override { return Next8(0, 48, 8); }
    double Dbl() override {
        uint64_t bits = Next8(4, 176, 16);
        double d;
        memcpy(&d, &bits, 8);
        return d;
    }
};

template <class T> void AppendF(std::string& out, const std::string& spec, T v) {
    int n = snprintf(nullptr, 0, spec.c_str(), v);
    if (n <= 0) return;
    std::vector<char> buf(n + 1);
    snprintf(buf.data(), buf.size(), spec.c_str(), v);
    out.append(buf.data(), n);
}

std::string Format(uint64_t fmt_addr, ArgSrc& a) {
    const std::string fmt = Str(fmt_addr, 16384);
    std::string out;
    for (size_t i = 0; i < fmt.size(); i++) {
        if (fmt[i] != '%') { out += fmt[i]; continue; }
        size_t begin = i++;
        std::string spec = "%";
        while (i < fmt.size() && strchr("-+ #0", fmt[i])) spec += fmt[i++];
        if (i < fmt.size() && fmt[i] == '*') { spec += std::to_string((int)a.Int()); i++; }
        while (i < fmt.size() && isdigit((unsigned char)fmt[i])) spec += fmt[i++];
        if (i < fmt.size() && fmt[i] == '.') {
            spec += fmt[i++];
            if (i < fmt.size() && fmt[i] == '*') { spec += std::to_string((int)a.Int()); i++; }
            while (i < fmt.size() && isdigit((unsigned char)fmt[i])) spec += fmt[i++];
        }
        int width_bits = 32;
        while (i < fmt.size() && strchr("hlLqjzt", fmt[i])) {
            if (fmt[i] == 'h') width_bits = width_bits == 16 ? 8 : 16;
            else width_bits = 64;
            i++;
        }
        if (i >= fmt.size()) break;
        const char c = fmt[i];
        switch (c) {
            case 'd': case 'i': {
                int64_t v = (int64_t)a.Int();
                if (width_bits == 32) v = (int32_t)v;
                else if (width_bits == 16) v = (int16_t)v;
                else if (width_bits == 8) v = (int8_t)v;
                AppendF(out, spec + "lld", (long long)v);
                break;
            }
            case 'u': case 'x': case 'X': case 'o': {
                uint64_t v = a.Int();
                if (width_bits == 32) v = (uint32_t)v;
                else if (width_bits == 16) v = (uint16_t)v;
                else if (width_bits == 8) v = (uint8_t)v;
                AppendF(out, spec + "ll" + c, (unsigned long long)v);
                break;
            }
            case 'c': AppendF(out, spec + "c", (int)(uint8_t)a.Int()); break;
            case 's': {
                uint64_t p = a.Int();
                std::string s = p ? Str(p, 65536) : "(null)";
                AppendF(out, spec + "s", s.c_str());
                break;
            }
            case 'p': AppendF(out, std::string("0x%llx"), (unsigned long long)a.Int()); break;
            case 'f': case 'F': case 'e': case 'E': case 'g': case 'G': case 'a': case 'A':
                AppendF(out, spec + c, a.Dbl());
                break;
            case 'n': a.Int(); break;
            case '%': out += '%'; break;
            default: out += fmt.substr(begin, i - begin + 1); break;
        }
    }
    return out;
}

// --- files ---

std::string HostPath(const std::string& guest) {
    static const char* kScratchMounts[] = {"/temp0", "/temp", "/download0", "/savedata0", "/data"};
    if (guest.rfind("/app0", 0) == 0) return "/game" + guest.substr(5);
    for (const char* m : kScratchMounts) {
        size_t n = strlen(m);
        if (guest.compare(0, n, m) == 0 && (guest.size() == n || guest[n] == '/')) {
            std::string dir = std::string("/tmp") + m;
            mkdir("/tmp", 0777);
            mkdir(dir.c_str(), 0777);
            return dir + guest.substr(n);
        }
    }
    return guest;
}

int HostOpenFlags(uint32_t f) {
    int h = (f & 3) == 1 ? O_WRONLY : (f & 3) == 2 ? O_RDWR : O_RDONLY;
    if (f & 0x8) h |= O_APPEND;
    if (f & 0x200) h |= O_CREAT;
    if (f & 0x400) h |= O_TRUNC;
    if (f & 0x800) h |= O_EXCL;
    if (f & 0x20000) h |= O_DIRECTORY;
    return h;
}

int g_open_logs = 0;
void LogOpen(const std::string& path, int result) {
    if (g_open_logs++ < 300)
        printf("[hle] open %s -> %s\n", path.c_str(), result >= 0 ? "ok" : strerror(errno));
}

// FreeBSD struct stat, as used by the PS5 libkernel.
void WriteStat(uint64_t dst, const struct stat& st) {
    uint8_t b[0x78] = {};
    uint32_t mode = (S_ISDIR(st.st_mode) ? 0040000 : 0100000) | (st.st_mode & 0777);
    int64_t size = st.st_size, blocks = (st.st_size + 511) / 512, sec = st.st_mtime;
    uint32_t blksize = 0x10000;
    uint32_t ino = (uint32_t)st.st_ino;
    memcpy(b + 0x04, &ino, 4);
    memcpy(b + 0x08, &mode, 2);
    for (int off : {0x18, 0x28, 0x38, 0x68}) memcpy(b + off, &sec, 8);
    memcpy(b + 0x48, &size, 8);
    memcpy(b + 0x50, &blocks, 8);
    memcpy(b + 0x58, &blksize, 4);
    if (auto* p = Ptr(dst, sizeof(b))) memcpy(p, b, sizeof(b));
}

std::unordered_map<uint64_t, FILE*> g_files;

// ======================= implementations =======================

// libc: memory and strings
void Memcpy(X86Regs& r) {
    auto* d = Ptr(r.rdi, r.rdx);
    auto* s = Ptr(r.rsi, r.rdx);
    if (d && s && r.rdx) memmove(d, s, r.rdx);
    r.rax = r.rdi;
}
void Memset(X86Regs& r) {
    if (auto* d = Ptr(r.rdi, r.rdx)) memset(d, (uint8_t)r.rsi, r.rdx);
    r.rax = r.rdi;
}
void Memcmp(X86Regs& r) {
    auto* a = Ptr(r.rdi, r.rdx);
    auto* b = Ptr(r.rsi, r.rdx);
    Ret(r, a && b ? memcmp(a, b, r.rdx) : 0);
}
void Memchr(X86Regs& r) {
    auto* p = Ptr(r.rdi, r.rdx);
    const void* hit = p ? memchr(p, (uint8_t)r.rsi, r.rdx) : nullptr;
    r.rax = hit ? r.rdi + ((const uint8_t*)hit - p) : 0;
}
void Strlen(X86Regs& r) { r.rax = StrLen(r.rdi); }
void Strnlen(X86Regs& r) { r.rax = StrLen(r.rdi, r.rsi); }
int CmpStr(uint64_t a, uint64_t b, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) {
        uint8_t ca = Rd<uint8_t>(a + i), cb = Rd<uint8_t>(b + i);
        if (ca != cb) return ca - cb;
        if (!ca) return 0;
    }
    return 0;
}
void Strcmp(X86Regs& r) { Ret(r, CmpStr(r.rdi, r.rsi, UINT64_MAX)); }
void Strncmp(X86Regs& r) { Ret(r, CmpStr(r.rdi, r.rsi, r.rdx)); }
void Strcpy(X86Regs& r) {
    size_t n = StrLen(r.rsi) + 1;
    auto* d = Ptr(r.rdi, n);
    auto* s = Ptr(r.rsi, n);
    if (d && s) memmove(d, s, n);
    r.rax = r.rdi;
}
void Strncpy(X86Regs& r) {
    size_t n = StrLen(r.rsi, r.rdx);
    auto* d = Ptr(r.rdi, r.rdx);
    auto* s = Ptr(r.rsi, n);
    if (d && (s || !n)) { if (n) memmove(d, s, n); memset(d + n, 0, r.rdx - n); }
    r.rax = r.rdi;
}
void Strcat(X86Regs& r) {
    size_t dl = StrLen(r.rdi), sl = StrLen(r.rsi) + 1;
    auto* d = Ptr(r.rdi + dl, sl);
    auto* s = Ptr(r.rsi, sl);
    if (d && s) memmove(d, s, sl);
    r.rax = r.rdi;
}
void Strncat(X86Regs& r) {
    size_t dl = StrLen(r.rdi), sl = StrLen(r.rsi, r.rdx);
    auto* d = Ptr(r.rdi + dl, sl + 1);
    auto* s = Ptr(r.rsi, sl);
    if (d && (s || !sl)) { if (sl) memmove(d, s, sl); d[sl] = 0; }
    r.rax = r.rdi;
}
void Strchr(X86Regs& r) {
    size_t n = StrLen(r.rdi) + 1;
    auto* p = Ptr(r.rdi, n);
    const void* hit = p ? memchr(p, (uint8_t)r.rsi, n) : nullptr;
    r.rax = hit ? r.rdi + ((const uint8_t*)hit - p) : 0;
}
void Strrchr(X86Regs& r) {
    size_t n = StrLen(r.rdi) + 1;
    auto* p = Ptr(r.rdi, n);
    r.rax = 0;
    if (!p) return;
    for (size_t i = n; i-- > 0;)
        if (p[i] == (uint8_t)r.rsi) { r.rax = r.rdi + i; return; }
}
void Strstr(X86Regs& r) {
    std::string h = Str(r.rdi, 1 << 20), n = Str(r.rsi, 4096);
    size_t pos = h.find(n);
    r.rax = pos == std::string::npos ? 0 : r.rdi + pos;
}
void Strdup(X86Regs& r) {
    size_t n = StrLen(r.rdi) + 1;
    uint64_t a = g_heap.Alloc(n, 16);
    auto* d = Ptr(a, n);
    auto* s = Ptr(r.rdi, n);
    if (d && s) memcpy(d, s, n);
    r.rax = a;
}
void Strtok(X86Regs& r) {
    uint64_t s = r.rdi ? r.rdi : g_strtok_next;
    std::string delim = Str(r.rsi, 256);
    auto is_delim = [&](uint8_t c) { return c && delim.find((char)c) != std::string::npos; };
    while (s && is_delim(Rd<uint8_t>(s))) s++;
    if (!s || !Rd<uint8_t>(s)) { g_strtok_next = 0; r.rax = 0; return; }
    uint64_t e = s;
    while (Rd<uint8_t>(e) && !is_delim(Rd<uint8_t>(e))) e++;
    if (Rd<uint8_t>(e)) { Wr<uint8_t>(e, 0); g_strtok_next = e + 1; } else { g_strtok_next = 0; }
    r.rax = s;
}
template <class Fn> void StrtoInt(X86Regs& r, Fn fn) {
    std::string s = Str(r.rdi, 256);
    char* end = nullptr;
    errno = 0;
    auto v = fn(s.c_str(), &end, (int)r.rdx);
    if (r.rsi) Wr<uint64_t>(r.rsi, r.rdi + (end - s.c_str()));
    if (errno) SetErrno(errno);
    r.rax = (uint64_t)v;
}
void Strtol(X86Regs& r) { StrtoInt(r, [](const char* s, char** e, int b) { return (int64_t)strtoll(s, e, b); }); }
void Strtoull(X86Regs& r) { StrtoInt(r, [](const char* s, char** e, int b) { return (uint64_t)strtoull(s, e, b); }); }
void Strtod(X86Regs& r) {
    std::string s = Str(r.rdi, 256);
    char* end = nullptr;
    double v = strtod(s.c_str(), &end);
    if (r.rsi) Wr<uint64_t>(r.rsi, r.rdi + (end - s.c_str()));
    RetD(r, v);
}

// libc: allocation
void Malloc(X86Regs& r) { r.rax = g_heap.Alloc(r.rdi, 16); }
void Free(X86Regs& r) { g_heap.Free(r.rdi); }
void Calloc(X86Regs& r) {
    if (r.rsi && r.rdi > UINT64_MAX / r.rsi) { r.rax = 0; return; }
    uint64_t n = r.rdi * r.rsi;
    uint64_t a = g_heap.Alloc(n, 16);
    if (auto* p = Ptr(a, n)) memset(p, 0, n);
    r.rax = a;
}
uint64_t Realloc(uint64_t p, uint64_t n, uint64_t align) {
    if (!p) return g_heap.Alloc(n, align);
    uint64_t old = g_heap.SizeOf(p);
    if (n <= old && p % align == 0) return p;
    uint64_t a = g_heap.Alloc(n, align);
    if (!a) return 0;
    auto* d = Ptr(a, std::min(old, n));
    auto* s = Ptr(p, std::min(old, n));
    if (d && s) memcpy(d, s, std::min(old, n));
    g_heap.Free(p);
    return a;
}
void ReallocFn(X86Regs& r) { r.rax = Realloc(r.rdi, r.rsi, 16); }
void Reallocalign(X86Regs& r) { r.rax = Realloc(r.rdi, r.rsi, r.rdx); }
void Memalign(X86Regs& r) { r.rax = g_heap.Alloc(r.rsi, r.rdi); }
void MallocStatsFast(X86Regs& r) {
    Wr<uint64_t>(r.rdi + 0, g_heap.end - L.heap_start);
    Wr<uint64_t>(r.rdi + 8, g_heap.top - L.heap_start);
    Wr<uint64_t>(r.rdi + 16, g_heap.peak);
    Wr<uint64_t>(r.rdi + 24, g_heap.in_use);
    r.rax = 0;
}
void MspaceUsableSize(X86Regs& r) { r.rax = g_heap.SizeOf(r.rdi); }

// libc: process and runtime support
void Exit(X86Regs& r) {
    char buf[64];
    snprintf(buf, sizeof(buf), "game exited (status %d)", (int)r.rdi);
    Halt(buf);
}
void Abort(X86Regs&) { Halt("game called abort()"); }
void StackChkFail(X86Regs&) { Halt("game failed a stack-protector check"); }
void PureVirtual(X86Regs&) { Halt("game called a pure virtual function"); }
void GuardAcquire(X86Regs& r) { r.rax = Rd<uint8_t>(r.rdi) == 0 ? 1 : 0; }
void GuardRelease(X86Regs& r) { Wr<uint8_t>(r.rdi, 1); }
void ErrorFn(X86Regs& r) { r.rax = g_errno_addr; }
void Getpctype(X86Regs& r) { r.rax = g_ctype + 2; }
void Getptolower(X86Regs& r) { r.rax = g_tolower + 2; }
void Getptoupper(X86Regs& r) { r.rax = g_toupper + 2; }

// libc: math
#define MATH_F1(name, fn) void name(X86Regs& r) { RetF(r, fn(F(r, 0))); }
#define MATH_F2(name, fn) void name(X86Regs& r) { RetF(r, fn(F(r, 0), F(r, 1))); }
#define MATH_D1(name, fn) void name(X86Regs& r) { RetD(r, fn(D(r, 0))); }
#define MATH_D2(name, fn) void name(X86Regs& r) { RetD(r, fn(D(r, 0), D(r, 1))); }
MATH_F1(Sinf, sinf) MATH_F1(Cosf, cosf) MATH_F1(Tanf, tanf) MATH_F1(Asinf, asinf) MATH_F1(Acosf, acosf)
MATH_F1(Atanf, atanf) MATH_F1(Expf, expf) MATH_F1(Exp2f, exp2f) MATH_F1(Logf, logf) MATH_F1(Log2f, log2f)
MATH_F1(Log10f, log10f) MATH_F2(Atan2f, atan2f) MATH_F2(Powf, powf) MATH_F2(Fmodf, fmodf)
MATH_D1(Sin, sin) MATH_D1(Cos, cos) MATH_D1(Tan, tan) MATH_D1(Acos, acos) MATH_D1(Exp, exp) MATH_D1(Log, log)
MATH_D2(Atan2, atan2) MATH_D2(Pow, pow) MATH_D2(Fmod, fmod)
void Modff(X86Regs& r) { float ip; RetF(r, modff(F(r, 0), &ip)); Wr<float>(r.rdi, ip); }
void Modf(X86Regs& r) { double ip; RetD(r, modf(D(r, 0), &ip)); Wr<double>(r.rdi, ip); }
void Ldexpf(X86Regs& r) { RetF(r, ldexpf(F(r, 0), (int)r.rdi)); }
void Ldexp(X86Regs& r) { RetD(r, ldexp(D(r, 0), (int)r.rdi)); }

// libc: time
void Time(X86Regs& r) {
    int64_t t = UnixUsec() / 1000000;
    if (r.rdi) Wr<int64_t>(r.rdi, t);
    Ret(r, t);
}
void Clock(X86Regs& r) { Ret(r, UsecSinceStart()); }
void Gettimeofday(X86Regs& r) {
    int64_t us = UnixUsec();
    Wr<int64_t>(r.rdi, us / 1000000);
    Wr<int64_t>(r.rdi + 8, us % 1000000);
    r.rax = 0;
}
void ClockGettime(X86Regs& r) {
    constexpr int kClockRealtime = 0;
    int64_t us = (int)r.rdi == kClockRealtime ? UnixUsec() : UsecSinceStart();
    Wr<int64_t>(r.rsi, us / 1000000);
    Wr<int64_t>(r.rsi + 8, (us % 1000000) * 1000);
    r.rax = 0;
}
void TmFrom(X86Regs& r, bool local) {
    time_t t = (time_t)Rd<int64_t>(r.rdi);
    struct tm tm{};
    if (local) localtime_r(&t, &tm); else gmtime_r(&t, &tm);
    int32_t f[9] = {tm.tm_sec, tm.tm_min, tm.tm_hour, tm.tm_mday, tm.tm_mon, tm.tm_year, tm.tm_wday, tm.tm_yday, tm.tm_isdst};
    if (auto* p = Ptr(g_tm, 56)) { memset(p, 0, 56); memcpy(p, f, sizeof(f)); }
    r.rax = g_tm;
}
void Localtime(X86Regs& r) { TmFrom(r, true); }
void Gmtime(X86Regs& r) { TmFrom(r, false); }
void KernelGetProcessTime(X86Regs& r) { Ret(r, UsecSinceStart()); }
void KernelGetProcessTimeCounter(X86Regs& r) { Ret(r, UsecSinceStart() * 1000); }
void KernelGetProcessTimeCounterFrequency(X86Regs& r) { r.rax = 1000000000; }

// libc: stdio
void Snprintf(X86Regs& r) {
    RegArgs a(r, 3);
    std::string s = Format(r.rdx, a);
    PutStr(r.rdi, r.rsi, s);
    Ret(r, (int64_t)s.size());
}
void Vsnprintf(X86Regs& r) {
    VaArgs a(r.rcx);
    std::string s = Format(r.rdx, a);
    PutStr(r.rdi, r.rsi, s);
    Ret(r, (int64_t)s.size());
}
void Sprintf(X86Regs& r) {
    RegArgs a(r, 2);
    std::string s = Format(r.rsi, a);
    PutStr(r.rdi, s.size() + 1, s);
    Ret(r, (int64_t)s.size());
}
void EmitTo(uint64_t file, const std::string& s) {
    auto it = g_files.find(file);
    if (it != g_files.end()) fwrite(s.data(), 1, s.size(), it->second);
    else GameLog(s);
}
void Printf(X86Regs& r) { RegArgs a(r, 1); std::string s = Format(r.rdi, a); GameLog(s); Ret(r, (int64_t)s.size()); }
void Vprintf(X86Regs& r) { VaArgs a(r.rsi); std::string s = Format(r.rdi, a); GameLog(s); Ret(r, (int64_t)s.size()); }
void Fprintf(X86Regs& r) { RegArgs a(r, 2); std::string s = Format(r.rsi, a); EmitTo(r.rdi, s); Ret(r, (int64_t)s.size()); }
void Vfprintf(X86Regs& r) { VaArgs a(r.rdx); std::string s = Format(r.rsi, a); EmitTo(r.rdi, s); Ret(r, (int64_t)s.size()); }
void Puts(X86Regs& r) { GameLog(Str(r.rdi, 65536)); r.rax = 1; }
void Fopen(X86Regs& r) {
    std::string path = HostPath(Str(r.rdi)), mode = Str(r.rsi, 8);
    FILE* f = fopen(path.c_str(), mode.c_str());
    LogOpen(path, f ? 0 : -1);
    if (!f) { SetErrno(errno); r.rax = 0; return; }
    uint64_t h = NewHandle();
    g_files[h] = f;
    r.rax = h;
}
void Fclose(X86Regs& r) {
    auto it = g_files.find(r.rdi);
    if (it != g_files.end()) { fclose(it->second); g_files.erase(it); }
    r.rax = 0;
}
void Fread(X86Regs& r) {
    auto it = g_files.find(r.rcx);
    uint64_t total = r.rsi * r.rdx;
    auto* p = Ptr(r.rdi, total);
    r.rax = (it != g_files.end() && p && r.rsi) ? fread(p, r.rsi, r.rdx, it->second) : 0;
}
void Fwrite(X86Regs& r) {
    uint64_t total = r.rsi * r.rdx;
    auto* p = Ptr(r.rdi, total);
    if (!p) { r.rax = 0; return; }
    EmitTo(r.rcx, std::string(reinterpret_cast<const char*>(p), total));
    r.rax = r.rdx;
}
void Fseek(X86Regs& r) {
    auto it = g_files.find(r.rdi);
    Ret(r, it != g_files.end() ? fseek(it->second, (long)r.rsi, (int)r.rdx) : -1);
}
void Ftell(X86Regs& r) {
    auto it = g_files.find(r.rdi);
    Ret(r, it != g_files.end() ? ftell(it->second) : -1);
}

// files (POSIX and sceKernel flavors)
int64_t OpenImpl(X86Regs& r) {
    std::string path = HostPath(Str(r.rdi));
    int fd = open(path.c_str(), HostOpenFlags((uint32_t)r.rsi), 0666);
    LogOpen(path, fd);
    return fd;
}
void Open(X86Regs& r) { int64_t fd = OpenImpl(r); if (fd < 0) SetErrno(errno); Ret(r, fd); }
void KernelOpen(X86Regs& r) { int64_t fd = OpenImpl(r); Ret(r, fd < 0 ? SceErr(errno) : fd); }
void Close(X86Regs& r) { Ret(r, close((int)r.rdi)); }
void KernelClose(X86Regs& r) { Ret(r, close((int)r.rdi) < 0 ? SceErr(errno) : 0); }
int64_t ReadImpl(X86Regs& r, bool positional) {
    auto* p = Ptr(r.rsi, r.rdx);
    if (!p) { errno = EFAULT; return -1; }
    return positional ? pread((int)r.rdi, p, r.rdx, (off_t)r.rcx) : read((int)r.rdi, p, r.rdx);
}
void Read(X86Regs& r) { int64_t n = ReadImpl(r, false); if (n < 0) SetErrno(errno); Ret(r, n); }
void KernelRead(X86Regs& r) { int64_t n = ReadImpl(r, false); Ret(r, n < 0 ? SceErr(errno) : n); }
void KernelPread(X86Regs& r) { int64_t n = ReadImpl(r, true); Ret(r, n < 0 ? SceErr(errno) : n); }
int64_t WriteImpl(X86Regs& r, bool positional) {
    auto* p = Ptr(r.rsi, r.rdx);
    if (!p) { errno = EFAULT; return -1; }
    if (r.rdi == 1 || r.rdi == 2) { GameLog(std::string(reinterpret_cast<const char*>(p), r.rdx)); return (int64_t)r.rdx; }
    return positional ? pwrite((int)r.rdi, p, r.rdx, (off_t)r.rcx) : write((int)r.rdi, p, r.rdx);
}
void KernelWrite(X86Regs& r) { int64_t n = WriteImpl(r, false); Ret(r, n < 0 ? SceErr(errno) : n); }
void KernelPwrite(X86Regs& r) { int64_t n = WriteImpl(r, true); Ret(r, n < 0 ? SceErr(errno) : n); }
void KernelLseek(X86Regs& r) {
    off_t o = lseek((int)r.rdi, (off_t)r.rsi, (int)r.rdx);
    Ret(r, o < 0 ? SceErr(errno) : (int64_t)o);
}
void KernelStat(X86Regs& r) {
    std::string path = HostPath(Str(r.rdi));
    struct stat st{};
    if (stat(path.c_str(), &st) < 0) { LogOpen(path, -1); Ret(r, SceErr(errno)); return; }
    WriteStat(r.rsi, st);
    r.rax = 0;
}
void KernelFstat(X86Regs& r) {
    struct stat st{};
    if (fstat((int)r.rdi, &st) < 0) { Ret(r, SceErr(errno)); return; }
    WriteStat(r.rsi, st);
    r.rax = 0;
}
void KernelMkdir(X86Regs& r) { Ret(r, mkdir(HostPath(Str(r.rdi)).c_str(), 0777) < 0 ? SceErr(errno) : 0); }
void KernelUnlink(X86Regs& r) { Ret(r, unlink(HostPath(Str(r.rdi)).c_str()) < 0 ? SceErr(errno) : 0); }
void KernelRmdir(X86Regs& r) { Ret(r, rmdir(HostPath(Str(r.rdi)).c_str()) < 0 ? SceErr(errno) : 0); }
void KernelFtruncate(X86Regs& r) { Ret(r, ftruncate((int)r.rdi, (off_t)r.rsi) < 0 ? SceErr(errno) : 0); }

// kernel: memory
void KernelGetDirectMemorySize(X86Regs& r) { r.rax = L.vm_end - L.vm_start; }
void KernelAllocateDirectMemory(X86Regs& r) { Ret(r, AllocDirect(r.rdi, r.rsi, r.rdx, r.rcx, (int)r.r8, r.r9)); }
void KernelAllocateMainDirectMemory(X86Regs& r) { Ret(r, AllocDirect(0, UINT64_MAX, r.rdi, r.rsi, (int)r.rdx, r.rcx)); }
void KernelReleaseDirectMemory(X86Regs& r) { g_dmem.erase(r.rdi); r.rax = 0; }
void KernelMapDirectMemory(X86Regs& r) {
    auto it = g_dmem.find(r.r8);
    int mtype = it == g_dmem.end() ? 0 : it->second.mtype;
    Ret(r, MapCommon(r.rdi, r.rsi, (int)r.rdx, (int)r.rcx, r.r9, HleRegion::Direct, r.r8, mtype, nullptr));
}
void KernelMapNamedDirectMemory(X86Regs& r) {
    auto it = g_dmem.find(r.r8);
    int mtype = it == g_dmem.end() ? 0 : it->second.mtype;
    std::string name = Str(StackArg(r, 0), 31);
    Ret(r, MapCommon(r.rdi, r.rsi, (int)r.rdx, (int)r.rcx, r.r9, HleRegion::Direct, r.r8, mtype, name.c_str()));
}
void KernelReserveVirtualRange(X86Regs& r) {
    Ret(r, MapCommon(r.rdi, r.rsi, 0, (int)r.rdx, r.rcx, HleRegion::Reserved, 0, 0, nullptr));
}
void KernelMunmap(X86Regs& r) { Carve(r.rdi, r.rdi + r.rsi); r.rax = 0; }
void KernelMprotect(X86Regs& r) {
    for (auto& [start, g] : g_regions)
        if (g.start < r.rdi + r.rsi && g.end > r.rdi) g.prot = (int)r.rdx;
    r.rax = 0;
}
void KernelSetVirtualRangeName(X86Regs& r) {
    std::string name = Str(r.rdx, 31);
    for (auto& [start, g] : g_regions)
        if (g.start < r.rdi + r.rsi && g.end > r.rdi) { memset(g.name, 0, sizeof(g.name)); strncpy(g.name, name.c_str(), 31); }
    r.rax = 0;
}
void KernelVirtualQuery(X86Regs& r) {
    constexpr uint64_t kFindNext = 1;
    const uint64_t addr = r.rdi;
    const Region* hit = nullptr;
    for (const auto& [start, g] : g_regions) {
        if (g.end <= addr) continue;
        if (g.start <= addr || (r.rsi & kFindNext)) hit = &g;
        break;
    }
    if (!hit) { Ret(r, kSceErrEacces); return; }
    uint8_t b[0x48] = {};
    int32_t prot = hit->prot, mtype = hit->mtype;
    uint32_t bits = (hit->kind == HleRegion::Flexible || hit->kind == HleRegion::Heap ? 1u : 0u) |
                    (hit->kind == HleRegion::Direct ? 2u : 0u) | (hit->kind == HleRegion::Stack ? 4u : 0u) |
                    (hit->kind != HleRegion::Reserved ? 0x10u : 0u);
    memcpy(b + 0x00, &hit->start, 8);
    memcpy(b + 0x08, &hit->end, 8);
    memcpy(b + 0x10, &hit->phys, 8);
    memcpy(b + 0x18, &prot, 4);
    memcpy(b + 0x1c, &mtype, 4);
    memcpy(b + 0x20, &bits, 4);
    memcpy(b + 0x24, hit->name, 32);
    uint64_t n = std::min<uint64_t>(r.rcx, sizeof(b));
    if (auto* p = Ptr(r.rdx, n)) memcpy(p, b, n);
    r.rax = 0;
}
void KernelDirectMemoryQuery(X86Regs& r) {
    constexpr uint64_t kFindNext = 1;
    for (const auto& [phys, blk] : g_dmem) {
        if (phys + blk.len <= r.rdi) continue;
        if (phys > r.rdi && !(r.rsi & kFindNext)) break;
        uint8_t b[24] = {};
        uint64_t end = phys + blk.len;
        memcpy(b, &phys, 8);
        memcpy(b + 8, &end, 8);
        memcpy(b + 16, &blk.mtype, 4);
        uint64_t n = std::min<uint64_t>(r.rcx, sizeof(b));
        if (auto* p = Ptr(r.rdx, n)) memcpy(p, b, n);
        r.rax = 0;
        return;
    }
    Ret(r, kSceErrEacces);
}
void KernelIsStack(X86Regs& r) {
    if (r.rsi) Wr<uint64_t>(r.rsi, L.stack_start);
    if (r.rdx) Wr<uint64_t>(r.rdx, L.stack_end);
    r.rax = 0;
}

// threads and synchronization (single-threaded)
void RetZero(X86Regs& r) { r.rax = 0; }
void ThreadSelf(X86Regs& r) { r.rax = g_thread_handle; }
void ThreadEqual(X86Regs& r) { r.rax = r.rdi == r.rsi; }
void WriteNewHandle(X86Regs& r) { Wr<uint64_t>(r.rdi, NewHandle()); r.rax = 0; }
void KeyCreate(X86Regs& r) { Wr<uint32_t>(r.rdi, g_next_key++); r.rax = 0; }
void GetSpecific(X86Regs& r) { auto it = g_tls_values.find((uint32_t)r.rdi); r.rax = it == g_tls_values.end() ? 0 : it->second; }
void SetSpecific(X86Regs& r) { g_tls_values[(uint32_t)r.rdi] = r.rsi; r.rax = 0; }
void LogThread(uint64_t entry, const std::string& name) {
    printf("[hle] thread %s (entry 0x%llx) not started: threads are not emulated yet\n",
           name.empty() ? "<unnamed>" : name.c_str(), (unsigned long long)entry);
}
void PthreadCreate(X86Regs& r) { LogThread(r.rdx, ""); Wr<uint64_t>(r.rdi, NewHandle()); r.rax = 0; }
void ScePthreadCreate(X86Regs& r) { LogThread(r.rdx, Str(r.r8, 64)); Wr<uint64_t>(r.rdi, NewHandle()); r.rax = 0; }
void ThreadJoin(X86Regs& r) { if (r.rsi) Wr<uint64_t>(r.rsi, 0); r.rax = 0; }
void WaitEventFlag(X86Regs& r) { if (r.rcx) Wr<uint64_t>(r.rcx, r.rsi); r.rax = 0; }
void WaitEqueue(X86Regs& r) { if (r.rcx) Wr<int32_t>(r.rcx, 0); Ret(r, r.r8 ? kSceErrEtimedout : 0); }

// system services
void UserServiceGetInitialUser(X86Regs& r) { Wr<int32_t>(r.rdi, 1); r.rax = 0; }
void UserServiceGetLoginUserIdList(X86Regs& r) {
    int32_t ids[4] = {1, -1, -1, -1};
    if (auto* p = Ptr(r.rdi, sizeof(ids))) memcpy(p, ids, sizeof(ids));
    r.rax = 0;
}
void UserServiceGetUserName(X86Regs& r) { PutStr(r.rsi, r.rdx, "Player"); r.rax = 0; }
void UserServiceGetEvent(X86Regs& r) { Ret(r, (int32_t)0x80960007); }  // no event
void SystemServiceParamGetInt(X86Regs& r) {
    constexpr int kLang = 1, kEnterButtonAssign = 1000;
    int id = (int)r.rdi;
    Wr<int32_t>(r.rsi, id == kLang || id == kEnterButtonAssign ? 1 : 0);
    r.rax = 0;
}
void SystemServiceGetStatus(X86Regs& r) { if (auto* p = Ptr(r.rdi, 136)) memset(p, 0, 136); r.rax = 0; }
void SystemServiceReceiveEvent(X86Regs& r) { Ret(r, (int32_t)0x80A10003); }  // no event
void SystemServiceGetDisplaySafeAreaInfo(X86Regs& r) { Wr<float>(r.rdi, 1.0f); r.rax = 0; }
void AppParamGetInt(X86Regs& r) { Wr<int32_t>(r.rsi, 0); r.rax = 0; }
void NpGetAccountId(X86Regs& r) { Wr<uint64_t>(r.rsi, 1); r.rax = 0; }
void ReturnHandle1(X86Regs& r) { r.rax = 1; }
void LoadStartModule(X86Regs& r) {
    static int next = 0x10;
    printf("[hle] load module %s (not loaded; HLE only)\n", Str(r.rdi, 256).c_str());
    r.rax = next++;
}
void Dlsym(X86Regs& r) {
    printf("[hle] dlsym %s: unresolved\n", Str(r.rsi, 256).c_str());
    Ret(r, kSceErrEnoent);
}
void VideoOutOpen(X86Regs& r) {
    printf("[hle] sceVideoOutOpen(user=%d, bus=%d)\n", (int)r.rdi, (int)r.rsi);
    r.rax = 1;
}

struct Impl { const char* name; HleFn fn; };
const Impl kImpls[] = {
    {"memcpy", Memcpy}, {"memmove", Memcpy}, {"memset", Memset}, {"memcmp", Memcmp}, {"memchr", Memchr},
    {"strlen", Strlen}, {"strnlen", Strnlen}, {"strcmp", Strcmp}, {"strncmp", Strncmp}, {"strcpy", Strcpy},
    {"strncpy", Strncpy}, {"strcat", Strcat}, {"strncat", Strncat}, {"strchr", Strchr}, {"strrchr", Strrchr},
    {"strstr", Strstr}, {"strdup", Strdup}, {"strtok", Strtok}, {"strtol", Strtol}, {"strtoull", Strtoull},
    {"strtod", Strtod},
    {"malloc", Malloc}, {"free", Free}, {"calloc", Calloc}, {"realloc", ReallocFn}, {"reallocalign", Reallocalign},
    {"memalign", Memalign}, {"malloc_stats_fast", MallocStatsFast}, {"sceLibcMspaceMallocUsableSize", MspaceUsableSize},
    {"exit", Exit}, {"catchReturnFromMain", Exit}, {"abort", Abort}, {"__stack_chk_fail", StackChkFail},
    {"__cxa_pure_virtual", PureVirtual}, {"__cxa_guard_acquire", GuardAcquire}, {"__cxa_guard_release", GuardRelease},
    {"atexit", RetZero}, {"__cxa_atexit", RetZero}, {"_init_env", RetZero},
    {"__error", ErrorFn}, {"_Getpctype", Getpctype}, {"_Getptolower", Getptolower}, {"_Getptoupper", Getptoupper},
    {"sinf", Sinf}, {"cosf", Cosf}, {"tanf", Tanf}, {"asinf", Asinf}, {"acosf", Acosf}, {"atanf", Atanf},
    {"expf", Expf}, {"exp2f", Exp2f}, {"logf", Logf}, {"log2f", Log2f}, {"log10f", Log10f}, {"atan2f", Atan2f},
    {"powf", Powf}, {"fmodf", Fmodf}, {"sin", Sin}, {"cos", Cos}, {"tan", Tan}, {"acos", Acos}, {"exp", Exp},
    {"log", Log}, {"atan2", Atan2}, {"pow", Pow}, {"fmod", Fmod}, {"modff", Modff}, {"modf", Modf},
    {"ldexpf", Ldexpf}, {"ldexp", Ldexp},
    {"time", Time}, {"clock", Clock}, {"gettimeofday", Gettimeofday}, {"clock_gettime", ClockGettime},
    {"sceKernelClockGettime", ClockGettime}, {"localtime", Localtime}, {"gmtime", Gmtime},
    {"sceKernelGetProcessTime", KernelGetProcessTime}, {"sceKernelGetProcessTimeCounter", KernelGetProcessTimeCounter},
    {"sceKernelGetProcessTimeCounterFrequency", KernelGetProcessTimeCounterFrequency}, {"sceKernelUsleep", RetZero},
    {"snprintf", Snprintf}, {"vsnprintf", Vsnprintf}, {"sprintf", Sprintf}, {"printf", Printf}, {"vprintf", Vprintf},
    {"fprintf", Fprintf}, {"vfprintf", Vfprintf}, {"puts", Puts}, {"fopen", Fopen}, {"fclose", Fclose},
    {"fread", Fread}, {"fwrite", Fwrite}, {"fseek", Fseek}, {"ftell", Ftell}, {"fflush", RetZero},
    {"open", Open}, {"close", Close}, {"read", Read},
    {"sceKernelOpen", KernelOpen}, {"sceKernelClose", KernelClose}, {"sceKernelRead", KernelRead},
    {"sceKernelPread", KernelPread}, {"sceKernelWrite", KernelWrite}, {"sceKernelPwrite", KernelPwrite},
    {"sceKernelLseek", KernelLseek}, {"sceKernelStat", KernelStat}, {"sceKernelFstat", KernelFstat},
    {"sceKernelMkdir", KernelMkdir}, {"sceKernelUnlink", KernelUnlink}, {"sceKernelRmdir", KernelRmdir},
    {"sceKernelFtruncate", KernelFtruncate},
    {"sceKernelGetDirectMemorySize", KernelGetDirectMemorySize},
    {"sceKernelAllocateDirectMemory", KernelAllocateDirectMemory},
    {"sceKernelAllocateMainDirectMemory", KernelAllocateMainDirectMemory},
    {"sceKernelReleaseDirectMemory", KernelReleaseDirectMemory},
    {"sceKernelCheckedReleaseDirectMemory", KernelReleaseDirectMemory},
    {"sceKernelMapDirectMemory", KernelMapDirectMemory}, {"sceKernelMapNamedDirectMemory", KernelMapNamedDirectMemory},
    {"sceKernelReserveVirtualRange", KernelReserveVirtualRange}, {"sceKernelMunmap", KernelMunmap},
    {"sceKernelMprotect", KernelMprotect}, {"sceKernelSetVirtualRangeName", KernelSetVirtualRangeName},
    {"sceKernelVirtualQuery", KernelVirtualQuery}, {"sceKernelDirectMemoryQuery", KernelDirectMemoryQuery},
    {"sceKernelIsStack", KernelIsStack},
    {"pthread_self", ThreadSelf}, {"scePthreadSelf", ThreadSelf}, {"pthread_equal", ThreadEqual},
    {"pthread_key_create", KeyCreate}, {"pthread_key_delete", RetZero}, {"pthread_getspecific", GetSpecific},
    {"pthread_setspecific", SetSpecific},
    {"pthread_mutex_init", WriteNewHandle}, {"pthread_mutexattr_init", WriteNewHandle},
    {"pthread_cond_init", WriteNewHandle}, {"scePthreadMutexInit", WriteNewHandle},
    {"scePthreadMutexattrInit", WriteNewHandle}, {"scePthreadAttrInit", WriteNewHandle},
    {"sceKernelCreateSema", WriteNewHandle}, {"sceKernelCreateEventFlag", WriteNewHandle},
    {"sceKernelCreateEqueue", WriteNewHandle},
    {"pthread_mutex_lock", RetZero}, {"pthread_mutex_unlock", RetZero}, {"pthread_mutex_destroy", RetZero},
    {"pthread_mutexattr_settype", RetZero}, {"pthread_mutexattr_setprotocol", RetZero},
    {"pthread_mutexattr_destroy", RetZero}, {"pthread_cond_destroy", RetZero}, {"pthread_cond_broadcast", RetZero},
    {"pthread_cond_signal", RetZero}, {"pthread_cond_wait", RetZero}, {"pthread_cond_timedwait", RetZero},
    {"sched_yield", RetZero}, {"scePthreadYield", RetZero}, {"scePthreadMutexLock", RetZero},
    {"scePthreadMutexUnlock", RetZero}, {"scePthreadMutexTrylock", RetZero}, {"scePthreadMutexDestroy", RetZero},
    {"scePthreadMutexattrSettype", RetZero}, {"scePthreadMutexattrSetprotocol", RetZero},
    {"scePthreadMutexattrDestroy", RetZero}, {"scePthreadAttrDestroy", RetZero},
    {"scePthreadAttrSetdetachstate", RetZero}, {"scePthreadAttrSetschedpolicy", RetZero},
    {"scePthreadAttrSetschedparam", RetZero}, {"scePthreadAttrSetinheritsched", RetZero},
    {"scePthreadAttrSetstacksize", RetZero}, {"scePthreadAttrSetaffinity", RetZero},
    {"scePthreadRename", RetZero}, {"scePthreadSetaffinity", RetZero}, {"scePthreadSetprio", RetZero},
    {"pthread_create", PthreadCreate}, {"scePthreadCreate", ScePthreadCreate},
    {"pthread_join", ThreadJoin}, {"scePthreadJoin", ThreadJoin},
    {"sceKernelWaitSema", RetZero}, {"sceKernelSignalSema", RetZero}, {"sceKernelDeleteSema", RetZero},
    {"sceKernelWaitEventFlag", WaitEventFlag}, {"sceKernelSetEventFlag", RetZero},
    {"sceKernelDeleteEventFlag", RetZero}, {"sceKernelWaitEqueue", WaitEqueue},
    {"sceUserServiceInitialize", RetZero}, {"sceUserServiceGetInitialUser", UserServiceGetInitialUser},
    {"sceUserServiceGetLoginUserIdList", UserServiceGetLoginUserIdList},
    {"sceUserServiceGetUserName", UserServiceGetUserName}, {"sceUserServiceGetEvent", UserServiceGetEvent},
    {"sceSystemServiceParamGetInt", SystemServiceParamGetInt}, {"sceSystemServiceGetStatus", SystemServiceGetStatus},
    {"sceSystemServiceReceiveEvent", SystemServiceReceiveEvent},
    {"sceSystemServiceGetDisplaySafeAreaInfo", SystemServiceGetDisplaySafeAreaInfo},
    {"sceSystemServiceHideSplashScreen", RetZero}, {"sceSystemServiceDisableMusicPlayer", RetZero},
    {"sceSysmoduleLoadModule", RetZero}, {"sceSysmoduleIsLoaded", RetZero}, {"sceSysmoduleUnloadModule", RetZero},
    {"sceAppContentInitialize", RetZero}, {"sceAppContentAppParamGetInt", AppParamGetInt},
    {"sceNpGetAccountIdA", NpGetAccountId},
    {"sceKernelLoadStartModule", LoadStartModule}, {"sceKernelDlsym", Dlsym},
    {"sceVideoOutOpen", VideoOutOpen}, {"scePadOpen", ReturnHandle1}, {"sceAudioOutOpen", ReturnHandle1},
};

std::unordered_map<std::string, HleFn> g_by_nid;
std::unordered_map<std::string, const char*> g_names;

} // namespace

void HleInit(const HleLayout& layout) {
    L = layout;
    g_heap.top = layout.heap_start;
    g_heap.end = layout.heap_end;
    g_vm_top = layout.vm_start;
    AddRegion(layout.heap_start, layout.heap_end - layout.heap_start, 3, HleRegion::Heap, 0, 0, "malloc");
    AddRegion(layout.stack_start, layout.stack_end - layout.stack_start, 3, HleRegion::Stack, 0, 0, "main stack");

    g_errno_addr = g_heap.Alloc(16, 16);
    g_thread_handle = NewHandle();
    g_tm = g_heap.Alloc(64, 16);
    g_ctype = g_heap.Alloc(257 * 2, 16);
    g_tolower = g_heap.Alloc(257 * 2, 16);
    g_toupper = g_heap.Alloc(257 * 2, 16);
    // Dinkumware tables are indexed by c in [-1, 255]; the exported pointer is entry 0.
    for (int c = -1; c < 256; c++) {
        int16_t f = 0;
        if (c >= 0 && c < 128) {
            if (isupper(c)) f |= 0x02;
            if (islower(c)) f |= 0x10;
            if (isdigit(c)) f |= 0x20;
            if (isxdigit(c)) f |= 0x01;
            if (ispunct(c)) f |= 0x08;
            if (c == ' ') f |= 0x04;
            if (strchr("\t\n\v\f\r", c)) f |= 0x40;
            else if (iscntrl(c)) f |= 0x80;
        }
        uint64_t off = 2 * (uint64_t)(c + 1);
        Wr<int16_t>(g_ctype + off, f);
        Wr<int16_t>(g_tolower + off, (int16_t)(c >= 0 && c < 128 ? tolower(c) : c));
        Wr<int16_t>(g_toupper + off, (int16_t)(c >= 0 && c < 128 ? toupper(c) : c));
    }

    std::unordered_map<std::string, HleFn> by_name;
    for (const auto& i : kImpls) by_name[i.name] = i.fn;
    for (const auto& n : kNames) {
        g_names[n.nid] = n.name;
        auto it = by_name.find(n.name);
        if (it != by_name.end()) g_by_nid[n.nid] = it->second;
    }
}

void HleMapRegion(uint64_t start, uint64_t size, int prot, HleRegion kind, const char* name) {
    AddRegion(start, size, prot, kind, 0, 0, name);
}

uint64_t HleVmAlloc(uint64_t size, uint64_t align, int prot, HleRegion kind, const char* name) {
    return VmAlloc(size, align, prot, kind, 0, 0, name);
}

uint64_t HleMalloc(uint64_t size, uint64_t align) { return g_heap.Alloc(size, align); }

HleFn HleFind(const std::string& nid) {
    auto it = g_by_nid.find(nid);
    return it == g_by_nid.end() ? nullptr : it->second;
}

const char* HleName(const std::string& nid) {
    auto it = g_names.find(nid);
    return it == g_names.end() ? nullptr : it->second;
}

bool HleHaltRequested() { return g_halt; }

} // namespace KytyWeb
