// PS5 SDK functions implemented against the emulated address space.
// Guest threads run cooperatively on the one interpreter; blocking calls park the calling thread.

#include "hle.h"
#include "web_platform.h"

#include <fcntl.h>
#include <dirent.h>
#include <strings.h>
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
#include <deque>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <tuple>
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

uint64_t VmAlloc(uint64_t size, uint64_t align, int prot, HleRegion kind, uint64_t phys, int mtype, const char* name);
void Carve(uint64_t start, uint64_t end);

// Small blocks come from the malloc arena; large ones are mapped in the vm range, which is bigger and gives the
// space back when they're freed.
struct Heap {
    static constexpr uint64_t kLarge = 4ull << 20;
    uint64_t top = 0, end = 0, in_use = 0, peak = 0;
    std::unordered_map<uint64_t, uint64_t> live, large;
    std::multimap<uint64_t, uint64_t> free_blocks;  // size -> addr

    uint64_t Alloc(uint64_t size, uint64_t align) {
        size = (std::max<uint64_t>(size, 1) + 15) & ~15ULL;
        if (align < 16) align = 16;
        if (size >= kLarge) {
            const uint64_t a = VmAlloc(size, align, 3, HleRegion::Heap, 0, 0, "malloc (large)");
            if (!a) { Exhausted(size); return 0; }
            large[a] = size;
            in_use += size;
            peak = std::max(peak, in_use);
            return a;
        }
        uint64_t a = 0, s = 0;
        auto it = free_blocks.lower_bound(size);
        for (int i = 0; i < 16 && it != free_blocks.end() && it->first <= size * 2; ++i, ++it) {
            if (it->second % align == 0) { a = it->second; s = it->first; free_blocks.erase(it); break; }
        }
        if (!a) {
            a = (top + align - 1) / align * align;
            if (a + size > end || a + size < a) {
                Exhausted(size);
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
        auto big = large.find(a);
        if (big != large.end()) {
            in_use -= big->second;
            Carve(a, a + big->second);
            large.erase(big);
            return;
        }
        auto it = live.find(a);
        if (it == live.end()) return;
        in_use -= it->second;
        free_blocks.emplace(it->second, a);
        live.erase(it);
    }
    uint64_t SizeOf(uint64_t a) const {
        auto it = live.find(a);
        if (it != live.end()) return it->second;
        auto big = large.find(a);
        return big == large.end() ? 0 : big->second;
    }
    // Names the guest code asking (the return address of the SDK call in progress).
    void Exhausted(uint64_t size) const;
};
Heap g_heap;
void Heap::Exhausted(uint64_t size) const {
    const uint64_t rsp = L.cpu->Regs().rsp;
    printf("[hle] out of memory: 0x%llx bytes requested by the code at 0x%llx\n", (unsigned long long)size,
           (unsigned long long)Rd<uint64_t>(rsp));
}

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

// First fit in the vm range, so space that was unmapped gets reused.
uint64_t VmAlloc(uint64_t size, uint64_t align, int prot, HleRegion kind, uint64_t phys, int mtype, const char* name) {
    size = (size + 0x3FFF) & ~0x3FFFULL;
    if (align < 0x4000) align = 0x4000;
    auto fits = [&](uint64_t from, uint64_t to) -> uint64_t {
        const uint64_t a = (from + align - 1) / align * align;
        return a >= from && a + size <= to && a + size > a ? a : 0;
    };
    uint64_t a = 0, cursor = L.vm_start;
    for (auto it = g_regions.lower_bound(L.vm_start); size && !a && it != g_regions.end(); ++it) {
        if (it->second.start >= L.vm_end) break;
        a = fits(cursor, it->second.start);
        cursor = std::max(cursor, it->second.end);
    }
    if (size && !a) a = fits(cursor, L.vm_end);
    if (!a) {
        static bool reported = false;
        printf("[hle] address space exhausted (request 0x%llx, %s)\n", (unsigned long long)size, name ? name : "");
        if (!reported) {
            reported = true;
            std::vector<const Region*> big;
            for (auto& [start, g] : g_regions)
                if (start >= L.vm_start && start < L.vm_end) big.push_back(&g);
            std::sort(big.begin(), big.end(), [](const Region* x, const Region* y) {
                return x->end - x->start > y->end - y->start;
            });
            for (size_t i = 0; i < big.size() && i < 12; i++)
                printf("[hle]   0x%llx-0x%llx %6llu KB %s\n", (unsigned long long)big[i]->start,
                       (unsigned long long)big[i]->end, (unsigned long long)((big[i]->end - big[i]->start) >> 10),
                       big[i]->name);
            std::map<std::string, std::pair<uint64_t, int>> by_name;  // name -> (bytes, regions)
            for (const Region* g : big) {
                auto& e = by_name[g->name[0] ? g->name : "(unnamed)"];
                e.first += g->end - g->start;
                e.second++;
            }
            for (auto& [name, e] : by_name)
                printf("[hle]   total %6llu MB in %d regions: %s\n", (unsigned long long)(e.first >> 20), e.second,
                       name.c_str());
        }
        return 0;
    }
    if (auto* p = Ptr(a, size)) memset(p, 0, size);  // reused space would otherwise keep old contents
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
uint32_t g_next_key = 1;
uint64_t g_strtok_next = 0;
const auto g_start = std::chrono::steady_clock::now();

uint64_t NewHandle() {
    uint64_t h = g_heap.Alloc(64, 16);
    if (auto* p = Ptr(h, 64)) memset(p, 0, 64);
    return h;
}
// Reading the host clock is a call out to JavaScript and games poll it constantly, so guest time queries use a
// reading the scheduler refreshes every time slice, re-read every 64 queries in between.
int64_t g_clock_us = 0;
int g_clock_queries = 0;
void RefreshClock() {
    g_clock_queries = 0;
    g_clock_us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - g_start).count();
}
int64_t UsecSinceStart() {
    if (++g_clock_queries >= 64) RefreshClock();
    return g_clock_us;
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

std::string g_app_root = "/game";  // where /app0 lives on the host: the browser's MEMFS mount or a directory
bool LazyStatPath(const std::string& path, struct stat* st);
std::string LazyMatchName(const std::string& dir, const std::string& name);

// The PS5's file system ignores case and the host's may not: a path that doesn't exist as written is resolved
// one component at a time against the directory listing (and the streamed files).
std::string ResolveCase(const std::string& root, const std::string& rel) {
    static std::unordered_map<std::string, std::string> cache;
    const std::string direct = root + rel;
    struct stat st;
    if (LazyStatPath(direct, &st) || stat(direct.c_str(), &st) == 0) return direct;
    if (auto it = cache.find(direct); it != cache.end()) return it->second;
    std::string cur = root;
    size_t pos = 0;
    while (pos < rel.size()) {
        if (rel[pos] == '/') { pos++; continue; }
        const size_t end = std::min(rel.find('/', pos), rel.size());
        const std::string name = rel.substr(pos, end - pos);
        std::string next = cur + "/" + name;
        if (!LazyStatPath(next, &st) && stat(next.c_str(), &st) != 0) {
            std::string match = LazyMatchName(cur, name);
            if (match.empty()) {
                if (DIR* d = opendir(cur.c_str())) {
                    while (dirent* e = readdir(d))
                        if (!strcasecmp(e->d_name, name.c_str())) { match = e->d_name; break; }
                    closedir(d);
                }
            }
            if (!match.empty()) next = cur + "/" + match;
        }
        cur = next;
        pos = end;
    }
    cache[direct] = cur;
    return cur;
}

std::string HostPath(const std::string& guest) {
    static const char* kScratchMounts[] = {"/temp0", "/temp", "/download0", "/savedata0", "/data"};
    if (guest.rfind("/app0", 0) == 0) return ResolveCase(g_app_root, guest.substr(5));
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

// Streamed files (defined after the thread code): big media read on demand from the frontend.
constexpr int64_t kParked = INT64_MIN;  // the read parked its thread until the data arrives
int LazyOpen(const std::string& path);
bool IsLazyFd(int fd);
int64_t LazyRead(X86Regs& r, int fd, uint64_t buf, uint64_t n, int64_t pos);
int64_t LazySeek(int fd, int64_t off, int whence);
bool LazyStatPath(const std::string& path, struct stat* st);
bool LazyStatFd(int fd, struct stat* st);
void LazyClose(int fd);

// Writes to the game's own log files (Unity's player log) are mirrored into the emulator log.
std::set<int> g_log_fds;
int FileOpen(const std::string& path, int flags) {
    int fd = LazyOpen(path);
    if (fd < 0) fd = open(path.c_str(), flags, 0666);
    else errno = 0;
    LogOpen(path, fd);
    if (fd >= 0 && path.size() > 4 && path.compare(path.size() - 4, 4, ".log") == 0) g_log_fds.insert(fd);
    return fd;
}
int FileClose(int fd) {
    g_log_fds.erase(fd);
    if (IsLazyFd(fd)) { LazyClose(fd); return 0; }
    return close(fd);
}
int64_t FileSeek(int fd, int64_t off, int whence) {
    return IsLazyFd(fd) ? LazySeek(fd, off, whence) : (int64_t)lseek(fd, (off_t)off, whence);
}
int FileStatPath(const std::string& path, struct stat* st) { return LazyStatPath(path, st) ? 0 : stat(path.c_str(), st); }
int FileStatFd(int fd, struct stat* st) {
    if (IsLazyFd(fd)) return LazyStatFd(fd, st) ? 0 : -1;
    return fstat(fd, st);
}

// files (POSIX and sceKernel flavors)
int64_t OpenImpl(X86Regs& r) { return FileOpen(HostPath(Str(r.rdi)), HostOpenFlags((uint32_t)r.rsi)); }
void Open(X86Regs& r) { int64_t fd = OpenImpl(r); if (fd < 0) SetErrno(errno); Ret(r, fd); }
void KernelOpen(X86Regs& r) { int64_t fd = OpenImpl(r); Ret(r, fd < 0 ? SceErr(errno) : fd); }
void Close(X86Regs& r) { Ret(r, FileClose((int)r.rdi)); }
void KernelClose(X86Regs& r) { Ret(r, FileClose((int)r.rdi) < 0 ? SceErr(errno) : 0); }
int64_t ReadImpl(X86Regs& r, bool positional) {
    if (IsLazyFd((int)r.rdi)) return LazyRead(r, (int)r.rdi, r.rsi, r.rdx, positional ? (int64_t)r.rcx : -1);
    auto* p = Ptr(r.rsi, r.rdx);
    if (!p) { errno = EFAULT; return -1; }
    return positional ? pread((int)r.rdi, p, r.rdx, (off_t)r.rcx) : read((int)r.rdi, p, r.rdx);
}
void Read(X86Regs& r) {
    int64_t n = ReadImpl(r, false);
    if (n == kParked) return;
    if (n < 0) SetErrno(errno);
    Ret(r, n);
}
void KernelRead(X86Regs& r) { int64_t n = ReadImpl(r, false); if (n != kParked) Ret(r, n < 0 ? SceErr(errno) : n); }
void KernelPread(X86Regs& r) { int64_t n = ReadImpl(r, true); if (n != kParked) Ret(r, n < 0 ? SceErr(errno) : n); }
int64_t WriteImpl(X86Regs& r, bool positional) {
    auto* p = Ptr(r.rsi, r.rdx);
    if (!p) { errno = EFAULT; return -1; }
    if (r.rdi == 1 || r.rdi == 2) { GameLog(std::string(reinterpret_cast<const char*>(p), r.rdx)); return (int64_t)r.rdx; }
    if (g_log_fds.count((int)r.rdi)) GameLog(std::string(reinterpret_cast<const char*>(p), r.rdx));
    return positional ? pwrite((int)r.rdi, p, r.rdx, (off_t)r.rcx) : write((int)r.rdi, p, r.rdx);
}
void KernelWrite(X86Regs& r) { int64_t n = WriteImpl(r, false); Ret(r, n < 0 ? SceErr(errno) : n); }
void KernelPwrite(X86Regs& r) { int64_t n = WriteImpl(r, true); Ret(r, n < 0 ? SceErr(errno) : n); }
void KernelLseek(X86Regs& r) {
    int64_t o = FileSeek((int)r.rdi, (int64_t)r.rsi, (int)r.rdx);
    Ret(r, o < 0 ? SceErr(errno) : o);
}
void KernelStat(X86Regs& r) {
    std::string path = HostPath(Str(r.rdi));
    struct stat st{};
    if (FileStatPath(path, &st) < 0) { LogOpen(path, -1); Ret(r, SceErr(errno)); return; }
    WriteStat(r.rsi, st);
    r.rax = 0;
}
void KernelFstat(X86Regs& r) {
    struct stat st{};
    if (FileStatFd((int)r.rdi, &st) < 0) { Ret(r, SceErr(errno)); return; }
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

void RetZero(X86Regs& r) { r.rax = 0; }
void RetEinval(X86Regs& r) { Ret(r, kSceErrEinval); }
// Getters that report through their second argument: settings off, an adult signed-out user, default priority.
void OutInt0(X86Regs& r) { Wr<int32_t>(r.rsi, 0); r.rax = 0; }
void UserServiceGetAgeLevel(X86Regs& r) { Wr<int32_t>(r.rsi, 18); r.rax = 0; }
void UserServiceGetGamePresets(X86Regs& r) { if (auto* p = Ptr(r.rsi, 32)) memset(p, 0, 32); r.rax = 0; }
void NpGetState(X86Regs& r) { Wr<int32_t>(r.rsi, 1); r.rax = 0; }  // SCE_NP_STATE_SIGNED_OUT
void NpGetAccountCountry(X86Regs& r) { PutStr(r.rsi, 4, "us"); r.rax = 0; }
void PthreadGetschedparam(X86Regs& r) { Wr<int32_t>(r.rsi, 0); Wr<int32_t>(r.rdx, 700); r.rax = 0; }  // (thread, policy*, param*)
void PthreadAttrGetschedparam(X86Regs& r) { Wr<int32_t>(r.rsi, 700); r.rax = 0; }
void ThreadEqual(X86Regs& r) { r.rax = r.rdi == r.rsi; }
void WriteNewHandle(X86Regs& r) { Wr<uint64_t>(r.rdi, NewHandle()); r.rax = 0; }
void KeyCreate(X86Regs& r) { Wr<uint32_t>(r.rdi, g_next_key++); r.rax = 0; }

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
void Dlsym(X86Regs& r) {
    printf("[hle] dlsym %s: unresolved\n", Str(r.rsi, 256).c_str());
    Ret(r, kSceErrEnoent);
}

// C++ runtime
void PosixMemalign(X86Regs& r) {
    uint64_t a = g_heap.Alloc(r.rdx, r.rsi);
    if (!a) { r.rax = 12; return; }
    Wr<uint64_t>(r.rdi, a);
    r.rax = 0;
}
void AlignedNew(X86Regs& r) { r.rax = g_heap.Alloc(r.rdi, r.rsi); }  // operator new(size, align)
void Terminate(X86Regs&) { Halt("std::terminate called"); }
void LengthError(X86Regs& r) { Halt(("std::length_error: " + Str(r.rdi, 200)).c_str()); }

// libc: bounded and case-insensitive strings, the C11 _s variants, sscanf
int CaseCmp(uint64_t a, uint64_t b, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) {
        const int ca = tolower(Rd<uint8_t>(a + i)), cb = tolower(Rd<uint8_t>(b + i));
        if (ca != cb) return ca - cb;
        if (!ca) return 0;
    }
    return 0;
}
void Strcasecmp(X86Regs& r) { Ret(r, CaseCmp(r.rdi, r.rsi, UINT64_MAX)); }
void Strncasecmp(X86Regs& r) { Ret(r, CaseCmp(r.rdi, r.rsi, r.rdx)); }
void Strlcpy(X86Regs& r) {  // (dst, src, size) -> strlen(src)
    const size_t n = StrLen(r.rsi);
    if (r.rdx) PutStr(r.rdi, r.rdx, Str(r.rsi, n + 1));
    r.rax = n;
}
void Strspn(X86Regs& r) {
    const std::string set = Str(r.rsi);
    uint64_t n = 0;
    for (uint8_t c; (c = Rd<uint8_t>(r.rdi + n)) != 0 && set.find((char)c) != std::string::npos;) n++;
    r.rax = n;
}
uint64_t g_strerror_buf = 0;
void Strerror(X86Regs& r) {
    if (!g_strerror_buf) g_strerror_buf = g_heap.Alloc(128, 16);
    PutStr(g_strerror_buf, 128, "error " + std::to_string((int)r.rdi));
    r.rax = g_strerror_buf;
}
// C11 Annex K: (dst, dst_size, ...) return 0, or an error after clearing the destination.
void MemcpyS(X86Regs& r) {  // (dst, dst_size, src, count)
    if (!r.rdi) { Ret(r, 22); return; }
    if (r.rcx > r.rsi || !r.rdx) { if (auto* d = Ptr(r.rdi, r.rsi)) memset(d, 0, r.rsi); Ret(r, r.rdx ? 34 : 22); return; }
    auto* d = Ptr(r.rdi, r.rcx);
    auto* src = Ptr(r.rdx, r.rcx);
    if (d && src) memmove(d, src, r.rcx);
    r.rax = 0;
}
void StrcpyS(X86Regs& r) {  // (dst, dst_size, src)
    const size_t n = StrLen(r.rdx);
    if (!r.rdi || !r.rsi) { Ret(r, 22); return; }
    if (n >= r.rsi) { Wr<uint8_t>(r.rdi, 0); Ret(r, 34); return; }
    PutStr(r.rdi, r.rsi, Str(r.rdx, n + 1));
    r.rax = 0;
}
void SnprintfS(X86Regs& r) {  // (dst, size, format, ...) like snprintf
    RegArgs a(r, 3);
    std::string out = Format(r.rdx, a);
    PutStr(r.rdi, r.rsi, out);
    Ret(r, (int64_t)out.size());
}
// sscanf with the common conversions: d i u x o c s [ n p f e g a, with hh h l ll z j t L sizes and *.
void Sscanf(X86Regs& r) {
    RegArgs a(r, 2);
    const std::string fmt = Str(r.rsi);
    uint64_t in = r.rdi;
    auto peek = [&]() { return Rd<uint8_t>(in); };
    int assigned = 0;
    bool any_input = false;
    for (size_t i = 0; i < fmt.size(); i++) {
        const char f = fmt[i];
        if (isspace((uint8_t)f)) { while (isspace(peek())) in++; continue; }
        if (f != '%' || (i + 1 < fmt.size() && fmt[i + 1] == '%')) {
            if (f == '%') i++;
            if (peek() != (uint8_t)fmt[i]) break;
            in++;
            continue;
        }
        i++;
        const bool skip = i < fmt.size() && fmt[i] == '*';
        if (skip) i++;
        int width = 0;
        while (i < fmt.size() && isdigit((uint8_t)fmt[i])) width = width * 10 + (fmt[i++] - '0');
        int size = 4;  // bytes stored for integers: hh 1, h 2, (none) 4, l ll z j t 8
        bool long_double = false;
        while (i < fmt.size() && strchr("hlzjtLq", fmt[i])) {
            const char m = fmt[i++];
            if (m == 'h') size = size == 2 ? 1 : 2;
            else if (m == 'L') long_double = true;
            else size = 8;
        }
        if (i >= fmt.size()) break;
        const char conv = fmt[i];
        if (conv == 'n') { if (!skip) Wr<int32_t>(a.Int(), (int32_t)(in - r.rdi)); continue; }
        if (conv != 'c' && conv != '[') while (isspace(peek())) in++;
        if (!peek()) { if (!any_input) assigned = -1; break; }
        const int limit = width ? width : INT32_MAX;
        std::string tok;
        if (conv == 'c') {
            for (int k = 0; k < (width ? width : 1) && peek(); k++) tok += (char)Rd<uint8_t>(in++);
            if (!skip) { const uint64_t dst = a.Int(); if (auto* d = Ptr(dst, tok.size())) memcpy(d, tok.data(), tok.size()); assigned++; }
            any_input = true;
            continue;
        }
        if (conv == 's' || conv == '[') {
            std::string set;
            bool negate = false;
            if (conv == '[') {
                i++;
                if (i < fmt.size() && fmt[i] == '^') { negate = true; i++; }
                if (i < fmt.size() && fmt[i] == ']') set += fmt[i++];
                while (i < fmt.size() && fmt[i] != ']') set += fmt[i++];
            }
            while (peek() && (int)tok.size() < limit) {
                const char c = (char)peek();
                const bool ok = conv == 's' ? !isspace((uint8_t)c) : ((set.find(c) != std::string::npos) != negate);
                if (!ok) break;
                tok += c;
                in++;
            }
            if (tok.empty()) break;
            if (!skip) { PutStr(a.Int(), tok.size() + 1, tok); assigned++; }
            any_input = true;
            continue;
        }
        // numbers: take the longest prefix strtoll/strtod accept (bounded by the width)
        std::string avail;
        for (int k = 0; k < limit && k < 512 && Rd<uint8_t>(in + k); k++) avail += (char)Rd<uint8_t>(in + k);
        char* end = nullptr;
        if (strchr("eEfgGaA", conv)) {
            const double v = strtod(avail.c_str(), &end);
            if (end == avail.c_str()) break;
            if (!skip) {
                const uint64_t dst = a.Int();
                if (size == 8 || long_double) Wr<double>(dst, v); else Wr<float>(dst, (float)v);
                assigned++;
            }
        } else if (strchr("diuxXop", conv)) {
            const int base = conv == 'd' || conv == 'u' ? 10 : conv == 'i' ? 0 : conv == 'o' ? 8 : 16;
            const uint64_t v = conv == 'd' || conv == 'i' ? (uint64_t)strtoll(avail.c_str(), &end, base)
                                                         : strtoull(avail.c_str(), &end, base);
            if (end == avail.c_str()) break;
            if (!skip) {
                const uint64_t dst = a.Int();
                const int n = conv == 'p' ? 8 : size;
                if (n == 1) Wr<uint8_t>(dst, (uint8_t)v);
                else if (n == 2) Wr<uint16_t>(dst, (uint16_t)v);
                else if (n == 4) Wr<uint32_t>(dst, (uint32_t)v);
                else Wr<uint64_t>(dst, v);
                assigned++;
            }
        } else {
            break;
        }
        in += (uint64_t)(end - avail.c_str());
        any_input = true;
    }
    Ret(r, assigned);
}

// libc: math helpers and Dinkumware internals
void Sincosf(X86Regs& r) { const float x = F(r, 0); Wr<float>(r.rdi, sinf(x)); Wr<float>(r.rsi, cosf(x)); }
void Sincos(X86Regs& r) { const double x = D(r, 0); Wr<double>(r.rdi, sin(x)); Wr<double>(r.rsi, cos(x)); }
void IsNan(X86Regs& r) { r.rax = std::isnan(D(r, 0)); }
void IsNanF(X86Regs& r) { r.rax = std::isnan(F(r, 0)); }
void IsInf(X86Regs& r) { r.rax = std::isinf(D(r, 0)); }
void IsFinite(X86Regs& r) { r.rax = std::isfinite(D(r, 0)); }
void IsFiniteF(X86Regs& r) { r.rax = std::isfinite(F(r, 0)); }
void Logb(X86Regs& r) { RetD(r, logb(D(r, 0))); }
void Scalbn(X86Regs& r) { RetD(r, scalbn(D(r, 0), (int)r.rdi)); }
void Nearbyintf(X86Regs& r) { RetF(r, nearbyintf(F(r, 0))); }
void Cosh(X86Regs& r) { RetD(r, D(r, 1) * cosh(D(r, 0))); }  // _Cosh(x, y) = y * cosh(x)
void Sinh(X86Regs& r) { RetD(r, D(r, 1) * sinh(D(r, 0))); }  // _Sinh(x, y) = y * sinh(x)
void Dtest(X86Regs& r) {  // classifies *p: 2 NaN, 1 infinite, -1 finite, -2 denormal, 0 zero
    const double v = Rd<double>(r.rdi);
    Ret(r, std::isnan(v) ? 2 : std::isinf(v) ? 1 : v == 0 ? 0 : std::fpclassify(v) == FP_SUBNORMAL ? -2 : -1);
}
void Powidf2(X86Regs& r) {  // compiler-rt: a ** b for an integer b
    double a = D(r, 0), result = 1;
    int b = (int)r.rdi;
    const bool recip = b < 0;
    for (unsigned e = recip ? -(unsigned)b : (unsigned)b; e; e >>= 1, a *= a)
        if (e & 1) result *= a;
    RetD(r, recip ? 1 / result : result);
}
void Udivti3(X86Regs& r) {  // compiler-rt: unsigned 128-bit a / b, a in rdi:rsi and b in rdx:rcx
    const unsigned __int128 a = ((unsigned __int128)r.rsi << 64) | r.rdi, b = ((unsigned __int128)r.rcx << 64) | r.rdx;
    const unsigned __int128 q = b ? a / b : 0;
    r.rax = (uint64_t)q;
    r.rdx = (uint64_t)(q >> 64);
}
// Dinkumware atomics: (object, value, memory order) -> the previous value. Guest threads don't preempt each other.
void AtomicFetchAdd4(X86Regs& r) { const uint32_t v = Rd<uint32_t>(r.rdi); Wr<uint32_t>(r.rdi, v + (uint32_t)r.rsi); r.rax = v; }
void AtomicFetchSub4(X86Regs& r) { const uint32_t v = Rd<uint32_t>(r.rdi); Wr<uint32_t>(r.rdi, v - (uint32_t)r.rsi); r.rax = v; }
// C11 threads (Dinkumware): _Mtx_init(_Mtx_t*, type) and _Cnd_init(_Cnd_t*) give each object a handle.
void MtxInit(X86Regs& r) { Wr<uint64_t>(r.rdi, g_heap.Alloc(64, 16)); r.rax = 0; }
void CndInit(X86Regs& r) { Wr<uint64_t>(r.rdi, g_heap.Alloc(64, 16)); r.rax = 0; }

// setjmp(buf) saves the callee-saved registers, the stack pointer and the return address; longjmp(buf, v)
// restores them, so setjmp returns again with v (or 1).
void Setjmp(X86Regs& r) {
    const uint64_t regs[8] = {r.rbx, r.rbp, r.r12, r.r13, r.r14, r.r15, r.rsp + 8, Rd<uint64_t>(r.rsp)};
    for (int i = 0; i < 8; i++) Wr<uint64_t>(r.rdi + 8 * i, regs[i]);
    r.rax = 0;
}
void Longjmp(X86Regs& r) {
    const uint64_t buf = r.rdi, value = (uint32_t)r.rsi ? (uint32_t)r.rsi : 1;
    r.rbx = Rd<uint64_t>(buf); r.rbp = Rd<uint64_t>(buf + 8);
    r.r12 = Rd<uint64_t>(buf + 16); r.r13 = Rd<uint64_t>(buf + 24);
    r.r14 = Rd<uint64_t>(buf + 32); r.r15 = Rd<uint64_t>(buf + 40);
    r.rsp = Rd<uint64_t>(buf + 48);
    r.rip = Rd<uint64_t>(buf + 56);
    r.rax = value;
    L.cpu->RequestYield(true);  // continue at the setjmp return address, not at the stub's RET
}

// libc: sorting with guest comparators
void Qsort(X86Regs& r) {
    const uint64_t base = r.rdi, n = r.rsi, size = r.rdx, cmp = r.rcx;
    if (n < 2 || !size || !Ptr(base, n * size)) return;
    std::vector<uint8_t> copy(Ptr(base, n * size), Ptr(base, n * size) + n * size);
    const uint64_t scratch = g_heap.Alloc(2 * size, 16);
    std::vector<uint64_t> idx(n);
    for (uint64_t i = 0; i < n; i++) idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(), [&](uint64_t a, uint64_t b) {
        memcpy(Ptr(scratch, size), copy.data() + a * size, size);
        memcpy(Ptr(scratch + size, size), copy.data() + b * size, size);
        return (int32_t)L.call_guest(cmp, scratch, scratch + size, 0) < 0;
    });
    for (uint64_t i = 0; i < n; i++) memcpy(Ptr(base + i * size, size), copy.data() + idx[i] * size, size);
    g_heap.Free(scratch);
}
void Bsearch(X86Regs& r) {
    const uint64_t key = r.rdi, base = r.rsi, size = r.rcx, cmp = r.r8;
    uint64_t lo = 0, hi = r.rdx;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2;
        int32_t c = (int32_t)L.call_guest(cmp, key, base + mid * size, 0);
        if (c == 0) { r.rax = base + mid * size; return; }
        if (c < 0) hi = mid; else lo = mid + 1;
    }
    r.rax = 0;
}

// libc: time formatting
struct tm GuestTm(uint64_t a) {
    struct tm t{};
    int32_t f[9] = {};
    if (auto* p = Ptr(a, sizeof(f))) memcpy(f, p, sizeof(f));
    t.tm_sec = f[0]; t.tm_min = f[1]; t.tm_hour = f[2]; t.tm_mday = f[3]; t.tm_mon = f[4];
    t.tm_year = f[5]; t.tm_wday = f[6]; t.tm_yday = f[7]; t.tm_isdst = f[8];
    return t;
}
void Strftime(X86Regs& r) {
    struct tm t = GuestTm(r.rcx);
    std::string fmt = Str(r.rdx, 256);
    std::vector<char> out(std::max<uint64_t>(r.rsi, 1));
    size_t n = strftime(out.data(), out.size(), fmt.c_str(), &t);
    if (n && Ptr(r.rdi, n + 1)) memcpy(Ptr(r.rdi, n + 1), out.data(), n + 1);
    r.rax = n;
}
void Mktime(X86Regs& r) { struct tm t = GuestTm(r.rdi); Ret(r, (int64_t)mktime(&t)); }
void Difftime(X86Regs& r) { RetD(r, (double)((int64_t)r.rdi - (int64_t)r.rsi)); }
void RtcGetCurrentTick(X86Regs& r) {
    constexpr uint64_t kUsecFromYear1ToEpoch = 62135596800000000ULL;
    Wr<uint64_t>(r.rdi, (uint64_t)UnixUsec() + kUsecFromYear1ToEpoch);
    r.rax = 0;
}

// libc: locale and stdio odds and ends
uint64_t g_c_locale = 0;
void Setlocale(X86Regs& r) { r.rax = g_c_locale; }
void Fgetc(X86Regs& r) {
    auto it = g_files.find(r.rdi);
    Ret(r, it != g_files.end() ? fgetc(it->second) : -1);
}

// POSIX file calls (errno style)
void PosixStat(X86Regs& r) {
    std::string path = HostPath(Str(r.rdi));
    struct stat st{};
    if (FileStatPath(path, &st) < 0) { LogOpen(path, -1); SetErrno(errno); Ret(r, -1); return; }
    WriteStat(r.rsi, st);
    r.rax = 0;
}
void PosixFstat(X86Regs& r) {
    struct stat st{};
    if (FileStatFd((int)r.rdi, &st) < 0) { SetErrno(errno); Ret(r, -1); return; }
    WriteStat(r.rsi, st);
    r.rax = 0;
}
void PosixLseek(X86Regs& r) {
    int64_t o = FileSeek((int)r.rdi, (int64_t)r.rsi, (int)r.rdx);
    if (o < 0) SetErrno(errno);
    Ret(r, o);
}
void PosixWrite(X86Regs& r) { int64_t n = WriteImpl(r, false); if (n < 0) SetErrno(errno); Ret(r, n); }
void PosixUnlink(X86Regs& r) {
    int rc = unlink(HostPath(Str(r.rdi)).c_str());
    if (rc < 0) SetErrno(errno);
    Ret(r, rc);
}
void PosixRename(X86Regs& r) {
    int rc = rename(HostPath(Str(r.rdi)).c_str(), HostPath(Str(r.rsi)).c_str());
    if (rc < 0) SetErrno(errno);
    Ret(r, rc);
}
void Getpid(X86Regs& r) { r.rax = 1000; }

void KernelGetdents(X86Regs& r) { r.rax = 0; }

MATH_D1(Asin, asin) MATH_D1(Atan, atan) MATH_D1(Tanh, tanh) MATH_D1(Log10, log10)

// --- threads ---
// Guest threads run one at a time in round-robin time slices. A blocking call parks its thread: the
// interpreter leaves RIP on the import stub, so the call runs again once `ready` holds or the timeout passes.

struct GuestThread {
    uint64_t handle = 0;
    std::string name;
    X86Regs regs{};
    uint64_t stack_lo = 0, stack_hi = 0, errno_addr = 0;
    std::unordered_map<uint32_t, uint64_t> specific;
    std::unordered_map<uint64_t, uint64_t> module_tls;
    bool blocked = false, finished = false, timed_out = false;
    std::function<bool()> ready;
    int64_t deadline_us = -1;
    const char* waiting_for = "";
    uint64_t exit_value = 0;
    // A blocking call that spans several retries
    uint64_t call_rip = 0;
    int phase = 0;
    bool signaled = false;
    int saved_lock_count = 0;
    // Signals raised on the thread and, while handlers run, the states they interrupted (innermost last).
    std::vector<int> pending_signals;
    struct Interrupted {
        X86Regs regs;
        bool blocked, timed_out, signaled;
        std::function<bool()> ready;
        int64_t deadline_us;
        const char* waiting_for;
        uint64_t call_rip;
        int phase;
    };
    std::vector<Interrupted> interrupted;
};
std::vector<std::unique_ptr<GuestThread>> g_threads;
size_t g_cur = 0;
GuestThread* Cur() { return g_threads[g_cur].get(); }
GuestThread* ThreadByHandle(uint64_t h) {
    for (auto& t : g_threads) if (t->handle == h) return t.get();
    return nullptr;
}

constexpr int64_t kSceErrEperm = (int32_t)0x80020001, kSceErrEdeadlk = (int32_t)0x8002000B,
                  kSceErrEbusy = (int32_t)0x80020010, kSceErrEsrch = (int32_t)0x80020003;
constexpr int kEperm = 1, kEsrch = 3, kEdeadlk = 11, kEbusy = 16, kEagain = 35, kEtimedout = 60;

void Park(const char* what, std::function<bool()> ready, int64_t timeout_us = -1) {
    GuestThread* t = Cur();
    t->blocked = true;
    t->timed_out = false;
    t->waiting_for = what;
    t->ready = std::move(ready);
    t->deadline_us = timeout_us >= 0 ? UsecSinceStart() + timeout_us : -1;
    L.cpu->RequestYield(true);
}
// True the first time a blocking call executes; false when it is being retried after parking.
bool FirstEntry(const X86Regs& r) {
    GuestThread* t = Cur();
    if (t->call_rip == r.rip) return false;
    t->call_rip = r.rip;
    t->phase = 0;
    t->signaled = false;
    return true;
}
void EndCall() { Cur()->call_rip = 0; Cur()->phase = 0; }
// pthread objects are pointers to an internal object; static initializers leave them NULL.
uint64_t ObjectOf(uint64_t var) {
    uint64_t h = Rd<uint64_t>(var);
    if (!h) { h = NewHandle(); Wr<uint64_t>(var, h); }
    return h;
}

// mutexes (FreeBSD/SCE types: 1 errorcheck, 2 recursive, 3 normal, 4 adaptive)
constexpr int kMutexRecursive = 2;
struct Mutex { GuestThread* owner = nullptr; int count = 0; int type = 1; };
std::unordered_map<uint64_t, Mutex> g_mutexes;
std::unordered_map<uint64_t, int> g_mutex_attr_type;

void MutexInit(X86Regs& r) {
    uint64_t h = NewHandle();
    Wr<uint64_t>(r.rdi, h);
    auto it = g_mutex_attr_type.find(r.rsi ? Rd<uint64_t>(r.rsi) : 0);
    g_mutexes[h] = Mutex{nullptr, 0, it == g_mutex_attr_type.end() ? 1 : it->second};
    r.rax = 0;
}
void MutexattrSettype(X86Regs& r) { g_mutex_attr_type[ObjectOf(r.rdi)] = (int)r.rsi; r.rax = 0; }
void MutexDestroy(X86Regs& r) { g_mutexes.erase(Rd<uint64_t>(r.rdi)); r.rax = 0; }
void MutexLockCommon(X86Regs& r, bool sce, bool try_only) {
    const uint64_t h = ObjectOf(r.rdi);
    Mutex& m = g_mutexes[h];
    GuestThread* t = Cur();
    if (!m.owner) { m.owner = t; m.count = 1; r.rax = 0; return; }
    if (m.owner == t) {
        if (m.type == kMutexRecursive) { m.count++; r.rax = 0; return; }
        Ret(r, sce ? kSceErrEdeadlk : kEdeadlk);
        return;
    }
    if (try_only) { Ret(r, sce ? kSceErrEbusy : kEbusy); return; }
    Park("mutex", [h] { return !g_mutexes[h].owner; });
}
void MutexLock(X86Regs& r) { MutexLockCommon(r, false, false); }
void SceMutexLock(X86Regs& r) { MutexLockCommon(r, true, false); }
void MutexTrylock(X86Regs& r) { MutexLockCommon(r, false, true); }
void SceMutexTrylock(X86Regs& r) { MutexLockCommon(r, true, true); }
void MutexUnlockCommon(X86Regs& r, bool sce) {
    Mutex& m = g_mutexes[ObjectOf(r.rdi)];
    if (m.owner != Cur()) { Ret(r, sce ? kSceErrEperm : kEperm); return; }
    if (--m.count == 0) m.owner = nullptr;
    r.rax = 0;
}
void MutexUnlock(X86Regs& r) { MutexUnlockCommon(r, false); }
void SceMutexUnlock(X86Regs& r) { MutexUnlockCommon(r, true); }

// condition variables
struct Cond { std::deque<GuestThread*> waiters; };
std::unordered_map<uint64_t, Cond> g_conds;

void CondWaitCommon(X86Regs& r, int64_t timeout_us, bool sce) {
    GuestThread* t = Cur();
    const uint64_t c = ObjectOf(r.rdi), mh = ObjectOf(r.rsi);
    if (FirstEntry(r)) {  // release the mutex and wait for a signal
        Mutex& m = g_mutexes[mh];
        if (m.owner != t) { EndCall(); Ret(r, sce ? kSceErrEperm : kEperm); return; }
        t->saved_lock_count = m.count;
        m.owner = nullptr;
        m.count = 0;
        g_conds[c].waiters.push_back(t);
        t->phase = 1;
        Park("condition variable", [t] { return t->signaled; }, timeout_us);
        return;
    }
    if (t->phase == 1) {  // woken by a signal or the timeout
        if (!t->signaled) {
            auto& w = g_conds[c].waiters;
            w.erase(std::remove(w.begin(), w.end(), t), w.end());
        }
        t->phase = t->signaled ? 2 : 3;
    }
    Mutex& m = g_mutexes[mh];  // reacquire before returning
    if (m.owner && m.owner != t) { Park("mutex", [mh] { return !g_mutexes[mh].owner; }); return; }
    m.owner = t;
    m.count = t->saved_lock_count;
    const bool timed_out = t->phase == 3;
    EndCall();
    Ret(r, timed_out ? (sce ? kSceErrEtimedout : kEtimedout) : 0);
}
void CondWait(X86Regs& r) { CondWaitCommon(r, -1, false); }
void SceCondWait(X86Regs& r) { CondWaitCommon(r, -1, true); }
void CondTimedwait(X86Regs& r) {  // absolute CLOCK_REALTIME deadline
    int64_t us = Rd<int64_t>(r.rdx) * 1000000 + Rd<int64_t>(r.rdx + 8) / 1000 - UnixUsec();
    CondWaitCommon(r, std::max<int64_t>(us, 0), false);
}
void SceCondTimedwait(X86Regs& r) { CondWaitCommon(r, (int64_t)(uint32_t)r.rdx, true); }  // relative microseconds
void CondSignal(X86Regs& r) {
    auto& w = g_conds[ObjectOf(r.rdi)].waiters;
    if (!w.empty()) { w.front()->signaled = true; w.pop_front(); }
    r.rax = 0;
}
void CondBroadcast(X86Regs& r) {
    auto& w = g_conds[ObjectOf(r.rdi)].waiters;
    for (GuestThread* t : w) t->signaled = true;
    w.clear();
    r.rax = 0;
}
void CondInit(X86Regs& r) { uint64_t h = NewHandle(); Wr<uint64_t>(r.rdi, h); g_conds[h]; r.rax = 0; }
void CondDestroy(X86Regs& r) { g_conds.erase(Rd<uint64_t>(r.rdi)); r.rax = 0; }

// reader/writer locks
struct RwLock { GuestThread* writer = nullptr; int readers = 0; };
std::unordered_map<uint64_t, RwLock> g_rwlocks;
void RwInit(X86Regs& r) { uint64_t h = NewHandle(); Wr<uint64_t>(r.rdi, h); g_rwlocks[h]; r.rax = 0; }
void RwRdlock(X86Regs& r) {
    const uint64_t h = ObjectOf(r.rdi);
    RwLock& l = g_rwlocks[h];
    if (!l.writer || l.writer == Cur()) { l.readers++; r.rax = 0; return; }
    Park("rwlock", [h] { return !g_rwlocks[h].writer; });
}
void RwWrlock(X86Regs& r) {
    const uint64_t h = ObjectOf(r.rdi);
    RwLock& l = g_rwlocks[h];
    if (!l.writer && !l.readers) { l.writer = Cur(); r.rax = 0; return; }
    Park("rwlock", [h] { return !g_rwlocks[h].writer && !g_rwlocks[h].readers; });
}
void RwUnlock(X86Regs& r) {
    RwLock& l = g_rwlocks[ObjectOf(r.rdi)];
    if (l.writer == Cur()) l.writer = nullptr;
    else if (l.readers) l.readers--;
    r.rax = 0;
}
void RwDestroy(X86Regs& r) { g_rwlocks.erase(Rd<uint64_t>(r.rdi)); r.rax = 0; }

// semaphores: (SceKernelSema* out, name, attr, init, max, opt); wait(sema, need, timeout*)
struct Sema { int64_t count, max; };
std::unordered_map<uint64_t, Sema> g_semas;
void SemaCreate(X86Regs& r) {
    uint64_t h = NewHandle();
    Wr<uint64_t>(r.rdi, h);
    g_semas[h] = {(int32_t)r.rcx, (int32_t)r.r8};
    r.rax = 0;
}
void SemaWait(X86Regs& r) {
    const uint64_t h = r.rdi;
    const int64_t need = (int32_t)r.rsi;
    Sema& s = g_semas[h];
    if (s.count >= need) { s.count -= need; EndCall(); r.rax = 0; return; }
    if (!FirstEntry(r) && Cur()->timed_out) { EndCall(); Ret(r, kSceErrEtimedout); return; }
    Park("semaphore", [h, need] { return g_semas[h].count >= need; }, r.rdx ? (int64_t)Rd<uint32_t>(r.rdx) : -1);
}
void SemaPoll(X86Regs& r) {
    Sema& s = g_semas[r.rdi];
    const int64_t need = (int32_t)r.rsi;
    if (s.count < need) { Ret(r, kSceErrEbusy); return; }
    s.count -= need;
    r.rax = 0;
}
void SemaSignal(X86Regs& r) {
    Sema& s = g_semas[r.rdi];
    s.count = std::min(s.max > 0 ? s.max : INT64_MAX, s.count + (int32_t)r.rsi);
    r.rax = 0;
}
void SemaDelete(X86Regs& r) { g_semas.erase(r.rdi); r.rax = 0; }

// event flags: wait mode 0x01 = all bits, 0x02 = any bit; 0x10 = clear all, 0x20 = clear the pattern
std::unordered_map<uint64_t, uint64_t> g_eventflags;
bool EventFlagMatches(uint64_t h, uint64_t pattern, uint32_t mode) {
    uint64_t bits = g_eventflags[h];
    return (mode & 1) ? (bits & pattern) == pattern : (bits & pattern) != 0;
}
bool EventFlagTake(uint64_t h, uint64_t pattern, uint32_t mode, uint64_t result) {
    if (!EventFlagMatches(h, pattern, mode)) return false;
    uint64_t& bits = g_eventflags[h];
    if (result) Wr<uint64_t>(result, bits);
    if (mode & 0x10) bits = 0;
    else if (mode & 0x20) bits &= ~pattern;
    return true;
}
void EventFlagCreate(X86Regs& r) {
    uint64_t h = NewHandle();
    Wr<uint64_t>(r.rdi, h);
    g_eventflags[h] = r.rcx;
    r.rax = 0;
}
void EventFlagWait(X86Regs& r) {  // (ef, pattern, mode, result*, timeout*)
    const uint64_t h = r.rdi, pattern = r.rsi;
    const uint32_t mode = (uint32_t)r.rdx;
    if (EventFlagTake(h, pattern, mode, r.rcx)) { EndCall(); r.rax = 0; return; }
    if (!FirstEntry(r) && Cur()->timed_out) {
        EndCall();
        if (r.rcx) Wr<uint64_t>(r.rcx, g_eventflags[h]);
        Ret(r, kSceErrEtimedout);
        return;
    }
    Park("event flag", [h, pattern, mode] { return EventFlagMatches(h, pattern, mode); },
         r.r8 ? (int64_t)Rd<uint32_t>(r.r8) : -1);
}
void EventFlagPoll(X86Regs& r) { Ret(r, EventFlagTake(r.rdi, r.rsi, (uint32_t)r.rdx, r.rcx) ? 0 : kSceErrEbusy); }
void EventFlagSet(X86Regs& r) { g_eventflags[r.rdi] |= r.rsi; r.rax = 0; }
void EventFlagClear(X86Regs& r) { g_eventflags[r.rdi] &= r.rsi; r.rax = 0; }
void EventFlagDelete(X86Regs& r) { g_eventflags.erase(r.rdi); r.rax = 0; }

// kernel event queues. A registered event is delivered once per trigger (graphics and video-out
// events behave as EV_CLEAR); triggers that arrive before delivery queue up behind it.
constexpr int64_t kSceErrEbadf = (int32_t)0x80020009, kSceErrEnoentEq = (int32_t)0x80020002;
constexpr int16_t kEvfiltUser = -11;
struct EqRegistration {
    uint64_t ident;
    int16_t filter;
    uint64_t udata;
    bool triggered = false;
    uint32_t fflags = 0;
    int64_t data = 0;
    std::deque<int64_t> pending;
};
std::unordered_map<uint64_t, std::vector<EqRegistration>> g_equeues;

EqRegistration* FindRegistration(uint64_t eq, uint64_t ident, int16_t filter) {
    auto it = g_equeues.find(eq);
    if (it == g_equeues.end()) return nullptr;
    for (auto& reg : it->second)
        if (reg.ident == ident && reg.filter == filter) return &reg;
    return nullptr;
}
int64_t AddRegistration(uint64_t eq, uint64_t ident, int16_t filter, uint64_t udata) {
    auto it = g_equeues.find(eq);
    if (it == g_equeues.end()) return kSceErrEbadf;
    if (EqRegistration* reg = FindRegistration(eq, ident, filter)) reg->udata = udata;
    else it->second.push_back(EqRegistration{ident, filter, udata});
    return 0;
}
void TriggerRegistration(EqRegistration& reg, int64_t data) {
    if (reg.triggered) { reg.pending.push_back(data); return; }
    reg.triggered = true;
    reg.fflags++;
    reg.data = data;
}
bool AnyTriggered(uint64_t eq) {
    auto it = g_equeues.find(eq);
    if (it == g_equeues.end()) return true;  // deleted: let the waiter fail
    for (auto& reg : it->second)
        if (reg.triggered) return true;
    return false;
}

void EqueueCreate(X86Regs& r) {
    uint64_t h = NewHandle();
    Wr<uint64_t>(r.rdi, h);
    g_equeues[h];
    r.rax = 0;
}
void EqueueDelete(X86Regs& r) { Ret(r, g_equeues.erase(r.rdi) ? 0 : kSceErrEbadf); }
void EqueueWait(X86Regs& r) {  // (eq, events*, num, out*, timeout*)
    const uint64_t eq = r.rdi;
    auto it = g_equeues.find(eq);
    if (it == g_equeues.end()) { EndCall(); Ret(r, kSceErrEbadf); return; }
    int delivered = 0;
    for (auto& reg : it->second) {
        if (delivered >= (int)r.rdx) break;
        if (!reg.triggered) continue;
        const uint64_t ev = r.rsi + 32ull * delivered++;  // {ident, filter, flags, fflags, data, udata}
        Wr<uint64_t>(ev, reg.ident);
        Wr<int16_t>(ev + 8, reg.filter);
        Wr<uint16_t>(ev + 10, 0);
        Wr<uint32_t>(ev + 12, reg.fflags);
        Wr<int64_t>(ev + 16, reg.data);
        Wr<uint64_t>(ev + 24, reg.udata);
        if (!reg.pending.empty()) {
            reg.data = reg.pending.front();
            reg.pending.pop_front();
            reg.fflags = 1;
        } else {
            reg.triggered = false;
            reg.fflags = 0;
            reg.data = 0;
        }
    }
    if (delivered) { EndCall(); if (r.rcx) Wr<int32_t>(r.rcx, delivered); r.rax = 0; return; }
    const int64_t timeout = r.r8 ? (int64_t)Rd<uint32_t>(r.r8) : -1;
    if (timeout == 0 || (!FirstEntry(r) && Cur()->timed_out)) {
        EndCall();
        if (r.rcx) Wr<int32_t>(r.rcx, 0);
        Ret(r, kSceErrEtimedout);
        return;
    }
    Park("event queue", [eq] { return AnyTriggered(eq); }, timeout);
}
void AddUserEvent(X86Regs& r) { Ret(r, AddRegistration(r.rdi, (uint32_t)r.rsi, kEvfiltUser, 0)); }
void TriggerUserEvent(X86Regs& r) {  // (eq, id, udata)
    EqRegistration* reg = FindRegistration(r.rdi, (uint32_t)r.rsi, kEvfiltUser);
    if (!reg) { Ret(r, kSceErrEnoentEq); return; }
    reg->udata = r.rdx;
    TriggerRegistration(*reg, 0);
    r.rax = 0;
}
void DeleteUserEvent(X86Regs& r) {
    auto it = g_equeues.find(r.rdi);
    if (it == g_equeues.end()) { Ret(r, kSceErrEbadf); return; }
    auto& v = it->second;
    v.erase(std::remove_if(v.begin(), v.end(), [&](const EqRegistration& reg) {
        return reg.ident == (uint32_t)r.rsi && reg.filter == kEvfiltUser;
    }), v.end());
    r.rax = 0;
}
void GetEventId(X86Regs& r) { r.rax = Rd<uint64_t>(r.rdi); }
void GetEventFilter(X86Regs& r) { Ret(r, Rd<int16_t>(r.rdi + 8)); }
void GetEventFflags(X86Regs& r) { r.rax = Rd<uint32_t>(r.rdi + 12); }
void GetEventData(X86Regs& r) { r.rax = Rd<uint64_t>(r.rdi + 16); }
void GetEventUserData(X86Regs& r) { r.rax = Rd<uint64_t>(r.rdi + 24); }

// sleeping and yielding
void SleepUs(X86Regs& r, int64_t us) {
    if (FirstEntry(r)) { Park("sleep", [] { return false; }, std::max<int64_t>(us, 0)); return; }
    EndCall();
    r.rax = 0;
}
void Usleep(X86Regs& r) { SleepUs(r, (int64_t)(uint32_t)r.rdi); }
void Nanosleep(X86Regs& r) { SleepUs(r, Rd<int64_t>(r.rdi) * 1000000 + Rd<int64_t>(r.rdi + 8) / 1000); }
void SleepSec(X86Regs& r) { SleepUs(r, (int64_t)(uint32_t)r.rdi * 1000000); }
void YieldNow(X86Regs& r) { r.rax = 0; L.cpu->RequestYield(false); }

// signals: sceKernelInstallExceptionHandler(signum, handler(int, Ucontext*)) and sceKernelRaiseException(thread,
// signum), which Unity's garbage collector uses to stop and resume threads. A raised signal runs the handler on
// the target thread the next time it is scheduled, waking it if it is blocked; when the handler returns, the
// interrupted state comes back, including a blocking call that was in progress.
constexpr int kSignals = 128;
uint64_t g_signal_handlers[kSignals] = {};
uint64_t g_sigreturn_trap = 0;
void InstallExceptionHandler(X86Regs& r) {
    const int sig = (int)r.rdi;
    if (sig <= 0 || sig >= kSignals) { Ret(r, kSceErrEinval); return; }
    g_signal_handlers[sig] = r.rsi;
    r.rax = 0;
}
void RemoveExceptionHandler(X86Regs& r) {
    const int sig = (int)r.rdi;
    if (sig <= 0 || sig >= kSignals) { Ret(r, kSceErrEinval); return; }
    g_signal_handlers[sig] = 0;
    r.rax = 0;
}
void RaiseException(X86Regs& r) {
    GuestThread* t = ThreadByHandle(r.rdi);
    const int sig = (int)r.rsi;
    if (!t || t->finished) { Ret(r, kSceErrEsrch); return; }
    if (sig <= 0 || sig >= kSignals) { Ret(r, kSceErrEinval); return; }
    if (g_signal_handlers[sig]) t->pending_signals.push_back(sig);
    r.rax = 0;
    if (t == Cur()) L.cpu->RequestYield(false);  // delivered when the scheduler resumes it
}
// Calls the handler for the current thread's first pending signal. The Ucontext (0x500 bytes, mcontext at
// 0x40 in FreeBSD's layout) goes below the red zone, and the handler returns to the sigreturn trap.
void DeliverSignal(GuestThread& t) {
    const int sig = t.pending_signals.front();
    t.pending_signals.erase(t.pending_signals.begin());
    X86Regs& r = L.cpu->Regs();
    t.interrupted.push_back({r, t.blocked, t.timed_out, t.signaled, t.ready, t.deadline_us, t.waiting_for,
                             t.call_rip, t.phase});
    const uint64_t ctx = (r.rsp - 128 - 0x500) & ~0xfull;
    if (auto* p = Ptr(ctx, 0x500)) memset(p, 0, 0x500);
    const uint64_t mc = ctx + 0x40;
    const uint64_t gprs[] = {r.rdi, r.rsi, r.rdx, r.rcx, r.r8, r.r9, r.rax, r.rbx, r.rbp, r.r10, r.r11, r.r12, r.r13,
                             r.r14, r.r15};
    for (int i = 0; i < 15; i++) Wr<uint64_t>(mc + 8 + 8 * i, gprs[i]);
    Wr<uint64_t>(mc + 0xa0, r.rip);
    Wr<uint64_t>(mc + 0xb0, r.rflags);
    Wr<uint64_t>(mc + 0xb8, r.rsp);
    Wr<uint64_t>(mc + 0x440, r.fs_base);
    r.rsp = ctx - 8;
    Wr<uint64_t>(r.rsp, g_sigreturn_trap);
    r.rip = g_signal_handlers[sig];
    r.rdi = (uint64_t)sig;
    r.rsi = ctx;
    t.blocked = t.timed_out = t.signaled = false;
    t.ready = nullptr;
    t.deadline_us = -1;
    t.waiting_for = "";
    t.call_rip = 0;
    t.phase = 0;
}
void SigReturn(X86Regs& r) {
    GuestThread* t = Cur();
    if (t->interrupted.empty()) { printf("[threads] signal return without a signal on %s\n", t->name.c_str()); return; }
    GuestThread::Interrupted f = std::move(t->interrupted.back());
    t->interrupted.pop_back();
    r = f.regs;
    t->blocked = f.blocked;
    t->timed_out = f.timed_out;
    t->signaled = f.signaled;
    t->ready = std::move(f.ready);
    t->deadline_us = f.deadline_us;
    t->waiting_for = f.waiting_for;
    t->call_rip = f.call_rip;
    t->phase = f.phase;
    // End the slice without executing the trap's RET; a restored blocking call is retried when it can proceed.
    L.cpu->RequestYield(true);
}

// thread lifecycle
uint64_t g_tls_image = 0, g_tls_filesz = 0, g_tls_size = 0, g_tls_align = 16;
struct ModuleTls { uint64_t image, filesz, memsz, align; };
std::unordered_map<uint64_t, ModuleTls> g_module_tls;
std::unordered_map<uint64_t, uint64_t> g_attr_stacksize, g_attr_thread;
uint64_t g_exit_trap = 0;
constexpr uint64_t kStaticTlsModule = 1;  // the eboot

void FinishThread(uint64_t value) {
    GuestThread* t = Cur();
    t->finished = true;
    t->exit_value = value;
    printf("[threads] %s finished\n", t->name.c_str());
    L.cpu->RequestYield(true);
}
void ThreadExit(X86Regs& r) { FinishThread(r.rdi); }

int64_t CreateThread(uint64_t out, uint64_t attr_ptr, uint64_t entry, uint64_t arg, std::string name) {
    auto t = std::make_unique<GuestThread>();
    t->handle = NewHandle();
    t->name = name.empty() ? "thread " + std::to_string(g_threads.size()) : name;
    const uint64_t attr = attr_ptr ? Rd<uint64_t>(attr_ptr) : 0;
    auto ss = g_attr_stacksize.find(attr);
    const uint64_t stack_size = std::clamp<uint64_t>(ss == g_attr_stacksize.end() ? (1 << 20) : ss->second, 64 << 10, 8 << 20);
    const uint64_t stack = VmAlloc(stack_size, 0x4000, 3, HleRegion::Stack, 0, 0, t->name.c_str());
    const uint64_t tls = g_heap.Alloc(g_tls_size + 0x100, g_tls_align);
    if (!stack || !tls) return -1;
    memset(Ptr(tls, g_tls_size + 0x100), 0, g_tls_size + 0x100);
    if (g_tls_filesz) memcpy(Ptr(tls, g_tls_filesz), Ptr(g_tls_image, g_tls_filesz), g_tls_filesz);
    const uint64_t tp = tls + g_tls_size;  // fs:[0] holds the thread pointer itself
    Wr<uint64_t>(tp, tp);
    t->stack_lo = stack;
    t->stack_hi = stack + stack_size;
    t->errno_addr = g_heap.Alloc(16, 16);
    t->regs.rflags = 0x202;
    t->regs.rip = entry;
    t->regs.rdi = arg;
    t->regs.rsp = t->stack_hi - 16 - 8;  // entered as if called; returning lands on the exit trap
    Wr<uint64_t>(t->regs.rsp, g_exit_trap);
    t->regs.fs_base = tp;
    Wr<uint64_t>(out, t->handle);
    printf("[threads] created %s (entry 0x%llx, stack 0x%llx)\n", t->name.c_str(), (unsigned long long)entry,
           (unsigned long long)stack_size);
    g_threads.push_back(std::move(t));
    return 0;
}
void PthreadCreate(X86Regs& r) { Ret(r, CreateThread(r.rdi, r.rsi, r.rdx, r.rcx, "") ? kEagain : 0); }
void ScePthreadCreate(X86Regs& r) { Ret(r, CreateThread(r.rdi, r.rsi, r.rdx, r.rcx, Str(r.r8, 31)) ? kSceErrEagain : 0); }
void ThreadJoinCommon(X86Regs& r, bool sce) {
    GuestThread* t = ThreadByHandle(r.rdi);
    if (!t) { Ret(r, sce ? kSceErrEsrch : kEsrch); return; }
    if (!t->finished) { Park("thread join", [t] { return t->finished; }); return; }
    if (r.rsi) Wr<uint64_t>(r.rsi, t->exit_value);
    r.rax = 0;
}
void ThreadJoin(X86Regs& r) { ThreadJoinCommon(r, false); }
void SceThreadJoin(X86Regs& r) { ThreadJoinCommon(r, true); }
void ThreadSelf(X86Regs& r) { r.rax = Cur()->handle; }
void GetSpecific(X86Regs& r) {
    auto& s = Cur()->specific;
    auto it = s.find((uint32_t)r.rdi);
    r.rax = it == s.end() ? 0 : it->second;
}
void SetSpecific(X86Regs& r) { Cur()->specific[(uint32_t)r.rdi] = r.rsi; r.rax = 0; }
void AttrSetStacksize(X86Regs& r) { g_attr_stacksize[ObjectOf(r.rdi)] = r.rsi; r.rax = 0; }
void AttrGet(X86Regs& r) { g_attr_thread[ObjectOf(r.rsi)] = r.rdi; r.rax = 0; }  // (thread, attr*)
GuestThread* AttrThread(uint64_t attr_ptr) {
    auto it = g_attr_thread.find(ObjectOf(attr_ptr));
    GuestThread* t = it == g_attr_thread.end() ? nullptr : ThreadByHandle(it->second);
    return t ? t : Cur();
}
void AttrGetStackaddr(X86Regs& r) { Wr<uint64_t>(r.rsi, AttrThread(r.rdi)->stack_lo); r.rax = 0; }
void AttrGetStacksize(X86Regs& r) {
    GuestThread* t = AttrThread(r.rdi);
    Wr<uint64_t>(r.rsi, t->stack_hi - t->stack_lo);
    r.rax = 0;
}

// Dynamic TLS: one block per (thread, module), created from the module's TLS image on first use.
void TlsGetAddr(X86Regs& r) {
    const uint64_t mod = Rd<uint64_t>(r.rdi), off = Rd<uint64_t>(r.rdi + 8);
    if (mod == kStaticTlsModule) { r.rax = r.fs_base - g_tls_size + off; return; }
    GuestThread* t = Cur();
    auto it = t->module_tls.find(mod);
    if (it == t->module_tls.end()) {
        auto m = g_module_tls.find(mod);
        if (m == g_module_tls.end()) {
            printf("[hle] __tls_get_addr: module %llu has no TLS\n", (unsigned long long)mod);
            r.rax = 0;
            return;
        }
        const uint64_t block = g_heap.Alloc(m->second.memsz, m->second.align);
        memset(Ptr(block, m->second.memsz), 0, m->second.memsz);
        if (m->second.filesz) memcpy(Ptr(block, m->second.filesz), Ptr(m->second.image, m->second.filesz), m->second.filesz);
        it = t->module_tls.emplace(mod, block).first;
    }
    r.rax = it->second + off;
}

void SwitchTo(size_t i) {
    if (i == g_cur) return;
    g_threads[g_cur]->regs = L.cpu->Regs();
    g_cur = i;
    L.cpu->Regs() = g_threads[i]->regs;
    g_errno_addr = g_threads[i]->errno_addr;
}

// --- streamed files ---
// Big media stays in the frontend. A read requests the 1 MB chunks it lacks and parks its thread
// until they arrive; chunks are evicted least-recently-used beyond a memory budget.
constexpr uint64_t kChunkSize = 1 << 20;  // keep in sync with web/app.js
constexpr uint64_t kChunkBudget = 384ull << 20;
constexpr int kLazyFdBase = 0x40000;
struct LazyFile {
    std::string path;
    uint64_t size;
    std::unordered_map<uint64_t, std::vector<uint8_t>> chunks;
};
std::vector<LazyFile> g_lazy;
std::unordered_map<std::string, int> g_lazy_by_path;
struct LazyFdState { int file; uint64_t offset; };
std::unordered_map<int, LazyFdState> g_lazy_fds;
int g_next_lazy_fd = kLazyFdBase;
std::deque<std::pair<int, uint64_t>> g_lazy_queue;     // chunks to fetch
std::set<std::pair<int, uint64_t>> g_lazy_requested;   // queued or being fetched
std::list<std::pair<int, uint64_t>> g_lazy_lru;        // most recently used first
uint64_t g_lazy_bytes = 0;

// The name of a streamed file or directory inside dir that matches name ignoring case, or "".
std::string LazyMatchName(const std::string& dir, const std::string& name) {
    const std::string prefix = dir + "/";
    for (const auto& [path, file] : g_lazy_by_path) {
        if (path.size() <= prefix.size() || path.compare(0, prefix.size(), prefix) != 0) continue;
        const size_t end = std::min(path.find('/', prefix.size()), path.size());
        const std::string component = path.substr(prefix.size(), end - prefix.size());
        if (!strcasecmp(component.c_str(), name.c_str())) return component;
    }
    return "";
}

int LazyOpen(const std::string& path) {
    auto it = g_lazy_by_path.find(path);
    if (it == g_lazy_by_path.end()) return -1;
    const int fd = g_next_lazy_fd++;
    g_lazy_fds[fd] = {it->second, 0};
    return fd;
}
bool IsLazyFd(int fd) { return fd >= kLazyFdBase && g_lazy_fds.count(fd); }
void LazyClose(int fd) { g_lazy_fds.erase(fd); }
bool ChunksPresent(int file, uint64_t off, uint64_t n, bool request) {
    bool all = true;
    LazyFile& f = g_lazy[file];
    for (uint64_t c = off / kChunkSize; c <= (off + n - 1) / kChunkSize; c++) {
        if (f.chunks.count(c)) continue;
        all = false;
        if (request && g_lazy_requested.insert({file, c}).second) g_lazy_queue.emplace_back(file, c);
    }
    return all;
}
void CopyFromChunks(int file, uint64_t off, uint64_t n, uint8_t* dst) {
    LazyFile& f = g_lazy[file];
    for (uint64_t done = 0; done < n;) {
        const uint64_t c = (off + done) / kChunkSize, in = (off + done) % kChunkSize;
        const auto& chunk = f.chunks[c];
        const uint64_t take = in < chunk.size() ? std::min<uint64_t>(n - done, chunk.size() - in) : 0;
        if (!take) break;
        memcpy(dst + done, chunk.data() + in, take);
        done += take;
        g_lazy_lru.remove({file, c});
        g_lazy_lru.emplace_front(file, c);
    }
}
int64_t LazyRead(X86Regs& r, int fd, uint64_t buf, uint64_t n, int64_t pos) {
    (void)r;
    LazyFdState& s = g_lazy_fds[fd];
    LazyFile& f = g_lazy[s.file];
    const uint64_t off = pos >= 0 ? (uint64_t)pos : s.offset;
    if (off >= f.size || n == 0) return 0;
    n = std::min(n, f.size - off);
    uint8_t* dst = Ptr(buf, n);
    if (!dst) { errno = EFAULT; return -1; }
    const int file = s.file;
    if (!ChunksPresent(file, off, n, true)) {
        Park("streamed file", [file, off, n] { return ChunksPresent(file, off, n, false); });
        return kParked;
    }
    CopyFromChunks(file, off, n, dst);
    if (pos < 0) s.offset = off + n;
    return (int64_t)n;
}
int64_t LazySeek(int fd, int64_t off, int whence) {
    LazyFdState& s = g_lazy_fds[fd];
    const int64_t base = whence == SEEK_CUR ? (int64_t)s.offset : whence == SEEK_END ? (int64_t)g_lazy[s.file].size : 0;
    if (base + off < 0) { errno = EINVAL; return -1; }
    s.offset = (uint64_t)(base + off);
    return (int64_t)s.offset;
}
void LazyStatOf(int file, struct stat* st) {
    *st = {};
    st->st_mode = S_IFREG | 0444;
    st->st_size = (off_t)g_lazy[file].size;
    st->st_ino = (ino_t)(0x10000 + file);
}
bool LazyStatPath(const std::string& path, struct stat* st) {
    auto it = g_lazy_by_path.find(path);
    if (it == g_lazy_by_path.end()) return false;
    LazyStatOf(it->second, st);
    return true;
}
bool LazyStatFd(int fd, struct stat* st) {
    auto it = g_lazy_fds.find(fd);
    if (it == g_lazy_fds.end()) return false;
    LazyStatOf(it->second.file, st);
    return true;
}

// controller (ScePadData is 120 bytes; see KytyPS5 src/libs/padData.h)
struct PadState { uint32_t buttons = 0; uint8_t lx = 128, ly = 128, rx = 128, ry = 128, l2 = 0, r2 = 0; } g_pad;
void FillPadData(uint64_t dst) {
    uint8_t d[120] = {};
    const float orientation_w = 1.0f;
    const uint64_t timestamp = (uint64_t)UsecSinceStart();
    memcpy(d + 0, &g_pad.buttons, 4);
    d[4] = g_pad.lx; d[5] = g_pad.ly; d[6] = g_pad.rx; d[7] = g_pad.ry; d[8] = g_pad.l2; d[9] = g_pad.r2;
    memcpy(d + 24, &orientation_w, 4);
    d[76] = 1;  // connected
    memcpy(d + 80, &timestamp, 8);
    d[104] = 1;  // connected count
    if (auto* p = Ptr(dst, sizeof(d))) memcpy(p, d, sizeof(d));
}
void PadReadState(X86Regs& r) { FillPadData(r.rsi); r.rax = 0; }
void PadRead(X86Regs& r) { FillPadData(r.rsi); r.rax = 1; }  // one sample
void PadGetControllerInformation(X86Regs& r) {
    uint8_t d[28] = {};
    const float pixel_density = 44.86f;
    const uint16_t res_x = 1920, res_y = 943;
    memcpy(d + 0, &pixel_density, 4);
    memcpy(d + 4, &res_x, 2);
    memcpy(d + 6, &res_y, 2);
    d[8] = d[9] = 13;  // stick dead zones
    d[11] = 1;         // connected count
    d[12] = 1;         // connected
    if (auto* p = Ptr(r.rsi, sizeof(d))) memcpy(p, d, sizeof(d));
    r.rax = 0;
}

// audio out: (user, type, index, len, freq, param) opens a port; each output call blocks until the previous
// buffer of `len` samples would have finished playing. Ports of type 0 (main) and 1 (BGM) are mixed to stereo.
struct AudioPort { int type = 0; uint32_t len = 256, freq = 48000, format = 1; int64_t next_us = 0; };
std::map<int, AudioPort> g_audio_ports;
int g_next_audio_port = 1;
std::vector<float> g_audio_mix;
void AudioOutOpen(X86Regs& r) {
    AudioPort p;
    p.type = (int)r.rsi;
    p.len = std::max<uint32_t>((uint32_t)r.rcx, 1);
    p.freq = (uint32_t)r.r8 ? (uint32_t)r.r8 : 48000;
    p.format = (uint32_t)r.r9 & 0xff;
    if (g_audio_ports.empty()) InitAudio(48000, 2);
    const int handle = g_next_audio_port++;
    g_audio_ports[handle] = p;
    printf("[audio] port %d: type %d, %u samples at %u Hz, format %u\n", handle, p.type, p.len, p.freq, p.format);
    r.rax = handle;
}
void AudioOutClose(X86Regs& r) { g_audio_ports.erase((int)r.rdi); r.rax = 0; }
// Converts one buffer to interleaved stereo float (formats: 0-2 s16 mono/stereo/8ch, 3-5 float, 6/7 8ch std).
void MixAudio(const AudioPort& p, uint64_t src) {
    const int fmt = (int)p.format;
    const int channels = (fmt == 0 || fmt == 3) ? 1 : (fmt == 1 || fmt == 4) ? 2 : 8;
    const bool is_float = fmt == 3 || fmt == 4 || fmt == 5 || fmt == 7;
    const uint64_t bytes = (uint64_t)p.len * channels * (is_float ? 4 : 2);
    const uint8_t* s = Ptr(src, bytes);
    if (!s) return;
    g_audio_mix.resize((size_t)p.len * 2);
    auto sample = [&](uint32_t i, int c) -> float {
        const uint64_t at = ((uint64_t)i * channels + c) * (is_float ? 4 : 2);
        if (is_float) { float f; memcpy(&f, s + at, 4); return f; }
        int16_t v; memcpy(&v, s + at, 2); return v / 32768.0f;
    };
    for (uint32_t i = 0; i < p.len; i++) {
        g_audio_mix[2 * i] = sample(i, 0);
        g_audio_mix[2 * i + 1] = sample(i, channels > 1 ? 1 : 0);
    }
    PushAudioSamples(g_audio_mix.data(), (int)p.len);
}
void AudioOutOutput(X86Regs& r) {
    auto it = g_audio_ports.find((int)r.rdi);
    if (it == g_audio_ports.end()) { Ret(r, (int32_t)0x80260004); return; }  // SCE_AUDIO_OUT_ERROR_INVALID_PORT
    AudioPort& p = it->second;
    const int64_t now = UsecSinceStart();
    if (FirstEntry(r) && now < p.next_us) {
        Park("audio output", [] { return false; }, p.next_us - now);
        return;
    }
    EndCall();
    // Fall behind by more than a few buffers (the emulator is slower than real time) and the clock restarts.
    const int64_t period = (int64_t)p.len * 1000000 / p.freq;
    p.next_us = std::max(p.next_us, now - 4 * period) + period;
    if (r.rsi && (p.type == 0 || p.type == 1)) MixAudio(p, r.rsi);
    r.rax = 0;
}
void AudioOutOutputs(X86Regs& r) {  // (SceAudioOutOutputParam {handle, ptr}[num], num)
    if (FirstEntry(r)) {
        int64_t wait = 0;
        const int64_t now = UsecSinceStart();
        for (uint32_t i = 0; i < (uint32_t)r.rsi; i++) {
            auto it = g_audio_ports.find(Rd<int32_t>(r.rdi + 16ull * i));
            if (it != g_audio_ports.end()) wait = std::max(wait, it->second.next_us - now);
        }
        if (wait > 0) { Park("audio output", [] { return false; }, wait); return; }
    }
    EndCall();
    const int64_t now = UsecSinceStart();
    for (uint32_t i = 0; i < (uint32_t)r.rsi; i++) {
        auto it = g_audio_ports.find(Rd<int32_t>(r.rdi + 16ull * i));
        if (it == g_audio_ports.end()) continue;
        AudioPort& p = it->second;
        const int64_t period = (int64_t)p.len * 1000000 / p.freq;
        p.next_us = std::max(p.next_us, now - 4 * period) + period;
        const uint64_t ptr = Rd<uint64_t>(r.rdi + 16ull * i + 8);
        if (ptr && (p.type == 0 || p.type == 1)) MixAudio(p, ptr);
    }
    r.rax = 0;
}
void AudioOutGetPortState(X86Regs& r) {  // SceAudioOutPortState: output, channel, volume, reroute counter, flag
    uint8_t d[32] = {};
    const uint16_t output = 1, volume = 127;  // connected primary
    memcpy(d + 0, &output, 2);
    d[2] = 2;
    memcpy(d + 4, &volume, 2);
    if (auto* p = Ptr(r.rsi, sizeof(d))) memcpy(p, d, sizeof(d));
    r.rax = 0;
}
void AjmInitialize(X86Regs& r) { Wr<uint32_t>(r.rsi, 1); r.rax = 0; }  // (reserved, SceAjmContextId*)

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
    {"sceKernelGetProcessTimeCounterFrequency", KernelGetProcessTimeCounterFrequency},
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
    {"pthread_setspecific", SetSpecific}, {"scePthreadGetspecific", GetSpecific},
    {"scePthreadSetspecific", SetSpecific}, {"scePthreadKeyCreate", KeyCreate},
    {"pthread_create", PthreadCreate}, {"scePthreadCreate", ScePthreadCreate},
    {"pthread_join", ThreadJoin}, {"scePthreadJoin", SceThreadJoin}, {"pthread_detach", RetZero},
    {"scePthreadDetach", RetZero}, {"pthread_exit", ThreadExit}, {"scePthreadExit", ThreadExit},
    {"sched_yield", YieldNow}, {"scePthreadYield", YieldNow}, {"pthread_yield", YieldNow},
    {"pthread_attr_init", WriteNewHandle}, {"pthread_attr_destroy", RetZero},
    {"pthread_attr_setstacksize", AttrSetStacksize}, {"pthread_attr_setdetachstate", RetZero},
    {"scePthreadAttrInit", WriteNewHandle}, {"scePthreadAttrDestroy", RetZero},
    {"scePthreadAttrSetstacksize", AttrSetStacksize}, {"scePthreadAttrSetdetachstate", RetZero},
    {"scePthreadAttrSetschedpolicy", RetZero}, {"scePthreadAttrSetschedparam", RetZero},
    {"scePthreadAttrSetinheritsched", RetZero}, {"scePthreadAttrSetaffinity", RetZero},
    {"scePthreadAttrGet", AttrGet}, {"scePthreadAttrGetstackaddr", AttrGetStackaddr},
    {"scePthreadAttrGetstacksize", AttrGetStacksize},
    {"scePthreadRename", RetZero}, {"scePthreadSetaffinity", RetZero}, {"scePthreadSetprio", RetZero},
    {"pthread_mutex_init", MutexInit}, {"scePthreadMutexInit", MutexInit},
    {"pthread_mutex_lock", MutexLock}, {"pthread_mutex_unlock", MutexUnlock}, {"pthread_mutex_trylock", MutexTrylock},
    {"pthread_mutex_destroy", MutexDestroy}, {"scePthreadMutexLock", SceMutexLock},
    {"scePthreadMutexUnlock", SceMutexUnlock}, {"scePthreadMutexTrylock", SceMutexTrylock},
    {"scePthreadMutexDestroy", MutexDestroy},
    {"pthread_mutexattr_init", WriteNewHandle}, {"pthread_mutexattr_settype", MutexattrSettype},
    {"pthread_mutexattr_setprotocol", RetZero}, {"pthread_mutexattr_destroy", RetZero},
    {"scePthreadMutexattrInit", WriteNewHandle}, {"scePthreadMutexattrSettype", MutexattrSettype},
    {"scePthreadMutexattrSetprotocol", RetZero}, {"scePthreadMutexattrDestroy", RetZero},
    {"pthread_cond_init", CondInit}, {"pthread_cond_destroy", CondDestroy}, {"pthread_cond_wait", CondWait},
    {"pthread_cond_timedwait", CondTimedwait}, {"pthread_cond_signal", CondSignal},
    {"pthread_cond_broadcast", CondBroadcast}, {"pthread_condattr_init", WriteNewHandle},
    {"pthread_condattr_destroy", RetZero},
    {"scePthreadCondInit", CondInit}, {"scePthreadCondDestroy", CondDestroy}, {"scePthreadCondWait", SceCondWait},
    {"scePthreadCondTimedwait", SceCondTimedwait}, {"scePthreadCondSignal", CondSignal},
    {"scePthreadCondBroadcast", CondBroadcast}, {"scePthreadCondattrInit", WriteNewHandle},
    {"scePthreadCondattrDestroy", RetZero},
    {"pthread_rwlock_init", RwInit}, {"pthread_rwlock_destroy", RwDestroy}, {"pthread_rwlock_rdlock", RwRdlock},
    {"pthread_rwlock_wrlock", RwWrlock}, {"pthread_rwlock_unlock", RwUnlock},
    {"scePthreadRwlockInit", RwInit}, {"scePthreadRwlockDestroy", RwDestroy}, {"scePthreadRwlockRdlock", RwRdlock},
    {"scePthreadRwlockWrlock", RwWrlock}, {"scePthreadRwlockUnlock", RwUnlock},
    {"sceKernelCreateSema", SemaCreate}, {"sceKernelWaitSema", SemaWait}, {"sceKernelPollSema", SemaPoll},
    {"sceKernelSignalSema", SemaSignal}, {"sceKernelDeleteSema", SemaDelete},
    {"sceKernelCreateEventFlag", EventFlagCreate}, {"sceKernelWaitEventFlag", EventFlagWait},
    {"sceKernelPollEventFlag", EventFlagPoll}, {"sceKernelSetEventFlag", EventFlagSet},
    {"sceKernelClearEventFlag", EventFlagClear}, {"sceKernelDeleteEventFlag", EventFlagDelete},
    {"sceKernelCancelEventFlag", RetZero},
    {"sceKernelCreateEqueue", EqueueCreate}, {"sceKernelDeleteEqueue", EqueueDelete},
    {"sceKernelWaitEqueue", EqueueWait}, {"sceKernelAddUserEvent", AddUserEvent},
    {"sceKernelAddUserEventEdge", AddUserEvent}, {"sceKernelTriggerUserEvent", TriggerUserEvent},
    {"sceKernelDeleteUserEvent", DeleteUserEvent}, {"sceKernelGetEventId", GetEventId},
    {"sceKernelGetEventFilter", GetEventFilter}, {"sceKernelGetEventFflags", GetEventFflags},
    {"sceKernelGetEventData", GetEventData}, {"sceKernelGetEventUserData", GetEventUserData},
    {"usleep", Usleep}, {"sceKernelUsleep", Usleep}, {"nanosleep", Nanosleep}, {"sceKernelNanosleep", Nanosleep},
    {"sleep", SleepSec}, {"sceKernelSleep", SleepSec}, {"__tls_get_addr", TlsGetAddr},
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
    {"sceKernelDlsym", Dlsym},
    {"scePadOpen", ReturnHandle1}, {"scePadInit", RetZero},
    {"strcasecmp", Strcasecmp}, {"strncasecmp", Strncasecmp}, {"strlcpy", Strlcpy}, {"strspn", Strspn},
    {"strerror", Strerror}, {"memcpy_s", MemcpyS}, {"strcpy_s", StrcpyS}, {"snprintf_s", SnprintfS},
    {"sprintf_s", SnprintfS}, {"sscanf", Sscanf},
    {"sincosf", Sincosf}, {"sincos", Sincos}, {"__isnan", IsNan}, {"__isnanf", IsNanF}, {"__isinf", IsInf},
    {"__isfinite", IsFinite}, {"__isfinitef", IsFiniteF}, {"logb", Logb}, {"scalbn", Scalbn},
    {"nearbyintf", Nearbyintf}, {"_Cosh", Cosh}, {"_Sinh", Sinh}, {"_Dtest", Dtest}, {"__powidf2", Powidf2},
    {"__udivti3", Udivti3}, {"_Atomic_fetch_add_4", AtomicFetchAdd4}, {"_Atomic_fetch_sub_4", AtomicFetchSub4},
    {"_Mtx_init", MtxInit}, {"_Cnd_init", CndInit}, {"_Mtx_destroy", RetZero}, {"_Cnd_destroy", RetZero},
    {"setjmp", Setjmp}, {"longjmp", Longjmp},
    {"sceUserServiceGetAccessibilityTriggerEffect", OutInt0}, {"sceUserServiceGetAccessibilityVibration", OutInt0},
    {"sceUserServiceGetAccessibilityChatTranscription", OutInt0},
    {"sceUserServiceGetAccessibilityPressAndHoldDelay", OutInt0},
    {"sceUserServiceGetAgeLevel", UserServiceGetAgeLevel}, {"sceUserServiceGetGamePresets", UserServiceGetGamePresets},
    {"sceNpGetState", NpGetState}, {"sceNpGetNpReachabilityState", OutInt0},
    {"sceNpGetAccountCountryA", NpGetAccountCountry},
    {"scePthreadGetschedparam", PthreadGetschedparam}, {"scePthreadAttrGetschedparam", PthreadAttrGetschedparam},
    {"scePthreadSetschedparam", RetZero},
    // No native backtraces: callers (Il2Cpp's exception stack traces) see the error and skip them.
    {"sceLibcBacktraceGetBufferSize", RetEinval}, {"sceLibcBacktraceSelf", RetEinval},
    {"sceKernelInstallExceptionHandler", InstallExceptionHandler},
    {"sceKernelRemoveExceptionHandler", RemoveExceptionHandler}, {"sceKernelRaiseException", RaiseException},
    {"sceAudioOutInit", RetZero}, {"sceAudioOutOpen", AudioOutOpen}, {"sceAudioOutClose", AudioOutClose},
    {"sceAudioOutOutput", AudioOutOutput}, {"sceAudioOutOutputs", AudioOutOutputs},
    {"sceAudioOutSetVolume", RetZero}, {"sceAudioOutGetPortState", AudioOutGetPortState},
    {"sceAjmInitialize", AjmInitialize}, {"sceAjmModuleRegister", RetZero},
    {"scePadReadState", PadReadState}, {"scePadRead", PadRead},
    {"scePadGetControllerInformation", PadGetControllerInformation},
    {"_Znwm", Malloc}, {"_Znam", Malloc}, {"_ZnwmRKSt9nothrow_t", Malloc}, {"_ZnamRKSt9nothrow_t", Malloc},
    {"_ZdlPv", Free}, {"_ZdaPv", Free}, {"_ZdlPvm", Free}, {"_ZdaPvm", Free},
    {"_ZnwmSt11align_val_t", AlignedNew}, {"_ZnamSt11align_val_t", AlignedNew},
    {"_ZdlPvSt11align_val_t", Free}, {"_ZdlPvmSt11align_val_t", Free}, {"_ZdaPvSt11align_val_t", Free},
    {"posix_memalign", PosixMemalign}, {"_ZSt9terminatev", Terminate}, {"_ZSt14_Xlength_errorPKc", LengthError},
    {"__cxa_guard_abort", RetZero}, {"__cxa_finalize", RetZero},
    {"qsort", Qsort}, {"bsearch", Bsearch}, {"strftime", Strftime}, {"mktime", Mktime}, {"difftime", Difftime},
    {"sceRtcGetCurrentTick", RtcGetCurrentTick}, {"sceKernelGettimeofday", Gettimeofday},
    {"setlocale", Setlocale}, {"setvbuf", RetZero}, {"fgetc", Fgetc},
    {"stat", PosixStat}, {"fstat", PosixFstat}, {"lseek", PosixLseek}, {"write", PosixWrite},
    {"unlink", PosixUnlink}, {"rename", PosixRename}, {"fcntl", RetZero}, {"getpid", Getpid},
    {"asin", Asin}, {"atan", Atan}, {"tanh", Tanh}, {"log10", Log10},
    {"sceKernelGetdents", KernelGetdents},
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
    g_c_locale = g_heap.Alloc(16, 16);
    PutStr(g_c_locale, 2, "C");
    g_thread_handle = NewHandle();
    g_threads.clear();
    auto main_thread = std::make_unique<GuestThread>();
    main_thread->handle = g_thread_handle;
    main_thread->name = "main";
    main_thread->stack_lo = layout.stack_start;
    main_thread->stack_hi = layout.stack_end;
    main_thread->errno_addr = g_errno_addr;
    g_threads.push_back(std::move(main_thread));
    g_cur = 0;
    g_exit_trap = g_heap.Alloc(16, 16);
    Wr<uint8_t>(g_exit_trap, 0xC3);
    L.cpu->RegisterHLEFunction(g_exit_trap, [](X86Regs& r, uint64_t) { FinishThread(r.rax); });
    g_sigreturn_trap = g_heap.Alloc(16, 16);
    Wr<uint8_t>(g_sigreturn_trap, 0xC3);
    L.cpu->RegisterHLEFunction(g_sigreturn_trap, [](X86Regs& r, uint64_t) { SigReturn(r); });
    AgcSetup(layout);
    GpuSetup(layout);
    AmprSetup(layout);
    AvPlayerSetup(layout);
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
    if (HleFn f = EhFind(nid)) return f;
    if (HleFn f = AgcFind(nid)) return f;
    if (HleFn f = GpuFind(nid)) return f;
    if (HleFn f = AmprFind(nid)) return f;
    if (HleFn f = AvPlayerFind(nid)) return f;
    auto it = g_by_nid.find(nid);
    return it == g_by_nid.end() ? nullptr : it->second;
}

const char* HleName(const std::string& nid) {
    if (const char* n = EhName(nid)) return n;
    if (const char* n = AgcName(nid)) return n;
    if (const char* n = GpuName(nid)) return n;
    if (const char* n = AmprName(nid)) return n;
    if (const char* n = AvPlayerName(nid)) return n;
    auto it = g_names.find(nid);
    return it == g_names.end() ? nullptr : it->second;
}

bool HleHaltRequested() { return g_halt; }
void HleSetAppRoot(const std::string& host_dir) { g_app_root = host_dir; }
std::string HleHostPath(const std::string& guest_path) { return HostPath(guest_path); }
uint64_t HleCurrentThread() { return Cur()->handle; }
void HleFree(uint64_t addr) { g_heap.Free(addr); }
void HleTerminate(const char* why) { Halt(why); }

void HleSetStaticTls(uint64_t image, uint64_t filesz, uint64_t size, uint64_t align) {
    g_tls_image = image;
    g_tls_filesz = filesz;
    g_tls_size = size;
    g_tls_align = align;
}

void HleRegisterModuleTls(uint64_t module_id, uint64_t image, uint64_t filesz, uint64_t memsz, uint64_t align) {
    g_module_tls[module_id] = {image, filesz, memsz, align};
}

void HlePark(const char* what, std::function<bool()> ready) { Park(what, std::move(ready)); }
bool HleFirstEntry(const X86Regs& r) { return FirstEntry(r); }
void HleEndCall() { EndCall(); }

int64_t HleUsecSinceStart() { return UsecSinceStart(); }

bool HleFileSize(const std::string& guest_path, uint64_t* size) {
    struct stat st{};
    if (FileStatPath(HostPath(guest_path), &st) < 0) return false;
    *size = (uint64_t)st.st_size;
    return true;
}

int64_t HleReadFileRange(const std::string& guest_path, uint64_t off, uint64_t n, uint64_t guest_dst, bool* pending) {
    *pending = false;
    const std::string path = HostPath(guest_path);
    auto lazy = g_lazy_by_path.find(path);
    if (lazy != g_lazy_by_path.end()) {
        const int file = lazy->second;
        if (off >= g_lazy[file].size || n == 0) return 0;
        n = std::min(n, g_lazy[file].size - off);
        uint8_t* dst = Ptr(guest_dst, n);
        if (!dst) return -1;
        if (!ChunksPresent(file, off, n, true)) { *pending = true; return 0; }
        CopyFromChunks(file, off, n, dst);
        return (int64_t)n;
    }
    uint8_t* dst = Ptr(guest_dst, n);
    if (!dst) return -1;
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return -1;
    const int64_t got = pread(fd, dst, n, (off_t)off);
    close(fd);
    return got;
}

int HleLazyRegister(const std::string& path, uint64_t size) {
    g_lazy.push_back({path, size, {}});
    g_lazy_by_path[path] = (int)g_lazy.size() - 1;
    return (int)g_lazy.size() - 1;
}

bool HleLazyNextRequest(int* file, uint64_t* chunk) {
    if (g_lazy_queue.empty()) return false;
    std::tie(*file, *chunk) = g_lazy_queue.front();
    g_lazy_queue.pop_front();
    return true;
}

void HleLazyProvide(int file, uint64_t chunk, const uint8_t* data, uint64_t length) {
    g_lazy_requested.erase({file, chunk});
    if (file < 0 || file >= (int)g_lazy.size()) return;
    auto& slot = g_lazy[file].chunks[chunk];
    g_lazy_bytes -= slot.size();
    slot.assign(data, data + length);
    g_lazy_bytes += length;
    g_lazy_lru.remove({file, chunk});
    g_lazy_lru.emplace_front(file, chunk);
    while (g_lazy_bytes > kChunkBudget && g_lazy_lru.size() > 1) {
        auto [vf, vc] = g_lazy_lru.back();
        g_lazy_lru.pop_back();
        auto it = g_lazy[vf].chunks.find(vc);
        if (it == g_lazy[vf].chunks.end()) continue;
        g_lazy_bytes -= it->second.size();
        g_lazy[vf].chunks.erase(it);
    }
}

void HleSetPadState(uint32_t buttons, float lx, float ly, float rx, float ry, float l2, float r2) {
    // frontend bit -> SCE_PAD_BUTTON_* (cross, circle, square, triangle, L1, R1, L2, R2, options,
    // L3, R3, up, down, left, right, touch pad)
    static const uint32_t kMap[][2] = {
        {1u << 0, 0x4000}, {1u << 1, 0x2000}, {1u << 2, 0x8000}, {1u << 3, 0x1000}, {1u << 4, 0x400},
        {1u << 5, 0x800}, {1u << 6, 0x100}, {1u << 7, 0x200}, {1u << 9, 0x8}, {1u << 10, 0x2},
        {1u << 11, 0x4}, {1u << 12, 0x10}, {1u << 13, 0x40}, {1u << 14, 0x80}, {1u << 15, 0x20},
        {1u << 17, 0x100000},
    };
    uint32_t sce = 0;
    for (const auto& m : kMap)
        if (buttons & m[0]) sce |= m[1];
    auto axis = [](float v) { return (uint8_t)std::clamp(128.0f + v * 127.5f, 0.0f, 255.0f); };
    auto trigger = [](float v) { return (uint8_t)std::clamp(v * 255.0f, 0.0f, 255.0f); };
    g_pad.buttons = sce;
    g_pad.lx = axis(lx);
    g_pad.ly = axis(ly);
    g_pad.rx = axis(rx);
    g_pad.ry = axis(ry);
    g_pad.l2 = trigger(l2);
    g_pad.r2 = trigger(r2);
}

void HleEqueueAddEvent(uint64_t eq, uint64_t ident, int16_t filter, uint64_t udata) {
    AddRegistration(eq, ident, filter, udata);
}

void HleEqueueDeleteEvent(uint64_t eq, uint64_t ident, int16_t filter) {
    auto it = g_equeues.find(eq);
    if (it == g_equeues.end()) return;
    auto& v = it->second;
    v.erase(std::remove_if(v.begin(), v.end(), [&](const EqRegistration& reg) {
        return reg.ident == ident && reg.filter == filter;
    }), v.end());
}

void HleEqueueTrigger(uint64_t ident, int16_t filter, uint64_t data) {
    for (auto& [eq, regs] : g_equeues)
        for (auto& reg : regs)
            if (reg.ident == ident && reg.filter == filter) TriggerRegistration(reg, (int64_t)data);
}

void HleDumpThreads(const std::function<std::string(uint64_t)>& name_of) {
    for (size_t i = 0; i < g_threads.size(); i++) {
        GuestThread& t = *g_threads[i];
        // The current thread's registers live in the interpreter until the next switch.
        const X86Regs& r = i == g_cur ? L.cpu->Regs() : t.regs;
        const std::string at = name_of(r.rip);
        std::string frames;
        uint64_t fp = r.rbp;
        for (int k = 0; k < 10 && fp && !(fp & 7) && Ptr(fp, 16); k++, fp = Rd<uint64_t>(fp)) {
            char buf[24];
            snprintf(buf, sizeof(buf), " 0x%llx", (unsigned long long)Rd<uint64_t>(fp + 8));
            frames += buf;
        }
        printf("[threads] %-28s %-30s rip=0x%llx%s%s ret=0x%llx frames:%s\n", t.name.c_str(),
               t.finished ? "finished" : t.blocked ? t.waiting_for : "runnable", (unsigned long long)r.rip,
               at.empty() ? "" : " ", at.c_str(), (unsigned long long)Rd<uint64_t>(r.rsp), frames.c_str());
    }
}

X86Exception HleRunThreads(uint64_t budget, uint64_t* executed) {
    constexpr uint64_t kQuantum = 20000;
    static bool reported_all_blocked = false;
    uint64_t done = 0;
    while (done < budget && !g_halt) {
        GpuPoll();
        RefreshClock();
        const int64_t now = g_clock_us;
        bool sleeping = false;
        for (auto& t : g_threads) {
            if (!t->blocked || t->finished) continue;
            if (t->ready && t->ready()) {
                t->blocked = false;
            } else if (t->deadline_us >= 0) {
                if (now >= t->deadline_us) { t->blocked = false; t->timed_out = true; }
                else sleeping = true;
            }
        }
        const size_t n = g_threads.size();
        size_t pick = SIZE_MAX;
        for (size_t k = 1; k <= n && pick == SIZE_MAX; k++) {
            size_t i = (g_cur + k) % n;
            const GuestThread& t = *g_threads[i];
            if ((!t.blocked || !t.pending_signals.empty()) && !t.finished) pick = i;
        }
        if (pick == SIZE_MAX) {
            if (!sleeping && !reported_all_blocked) {
                std::string who;
                for (auto& t : g_threads)
                    if (!t->finished) who += " " + t->name + " (" + t->waiting_for + ")";
                printf("[threads] every thread is blocked:%s\n", who.c_str());
                reported_all_blocked = true;
            }
            break;
        }
        reported_all_blocked = false;
        SwitchTo(pick);
        if (!Cur()->pending_signals.empty()) DeliverSignal(*Cur());
        const uint64_t before = L.cpu->GetInstructionCount();
        const X86Exception ex = L.cpu->Run(std::min(kQuantum, budget - done));
        done += L.cpu->GetInstructionCount() - before;
        if (ex != X86Exception::None && ex != X86Exception::Yield) {
            *executed = done;
            return ex;
        }
    }
    *executed = done;
    return X86Exception::None;
}

} // namespace KytyWeb
