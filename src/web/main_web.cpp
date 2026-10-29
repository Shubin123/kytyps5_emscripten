// Emscripten main entry point for KytyPS5.
// Initializes the emulator subsystems, loads a PS5 game from the virtual filesystem
// (mounted by the frontend JS), and runs the main emulation loop via
// emscripten_set_main_loop.

#include "hle.h"
#include "web_platform.h"
#include "x86_interpreter.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>

#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/html5.h>
#endif

// ---- ELF structures for PS5 eboot.bin loading ----

namespace {

struct Elf64_Ehdr {
    uint8_t  e_ident[16];
    uint16_t e_type;
    uint16_t e_machine;
    uint32_t e_version;
    uint64_t e_entry;
    uint64_t e_phoff;
    uint64_t e_shoff;
    uint32_t e_flags;
    uint16_t e_ehsize;
    uint16_t e_phentsize;
    uint16_t e_phnum;
    uint16_t e_shentsize;
    uint16_t e_shnum;
    uint16_t e_shstrndx;
};

struct Elf64_Phdr {
    uint32_t p_type;
    uint32_t p_flags;
    uint64_t p_offset;
    uint64_t p_vaddr;
    uint64_t p_paddr;
    uint64_t p_filesz;
    uint64_t p_memsz;
    uint64_t p_align;
};

struct Elf64_Dyn {
    int64_t d_tag;
    union {
        uint64_t d_val;
        uint64_t d_ptr;
    } d_un;
};

struct Elf64_Sym {
    uint32_t st_name;
    uint8_t  st_info;
    uint8_t  st_other;
    uint16_t st_shndx;
    uint64_t st_value;
    uint64_t st_size;
};

struct Elf64_Rela {
    uint64_t r_offset;
    uint64_t r_info;
    int64_t  r_addend;
};

constexpr uint32_t PT_LOAD    = 1;
constexpr uint32_t PT_DYNAMIC = 2;
constexpr uint32_t PT_TLS     = 7;
constexpr uint32_t PT_GNU_EH_FRAME = 0x6474e550;
constexpr uint32_t PT_SCE_DYNLIBDATA = 0x61000000;
constexpr uint32_t PT_SCE_RELRO      = 0x61000010;
constexpr uint32_t PT_SCE_COMMENT    = 0x6FFFFF00;
constexpr uint32_t PT_SCE_PROCPARAM  = 0x61000001;

constexpr int64_t DT_NULL    = 0;
constexpr int64_t DT_NEEDED  = 1;
constexpr int64_t DT_STRTAB  = 5;
constexpr int64_t DT_SYMTAB  = 6;
constexpr int64_t DT_STRSZ   = 10;
constexpr int64_t DT_SYMENT  = 11;
constexpr int64_t DT_RELA    = 7;
constexpr int64_t DT_RELASZ  = 8;
constexpr int64_t DT_JMPREL  = 23;
constexpr int64_t DT_PLTRELSZ = 2;
constexpr int64_t DT_PLTREL  = 20;

constexpr uint32_t R_X86_64_JUMP_SLOT = 7;
constexpr uint32_t R_X86_64_GLOB_DAT  = 6;
constexpr uint32_t R_X86_64_RELATIVE  = 8;

#define ELF64_R_SYM(i)  ((i) >> 32)
#define ELF64_R_TYPE(i)  ((uint32_t)((i) & 0xffffffff))

constexpr uint64_t EMU_BASE_ADDR = 0x100000;
// Emulated address space; Unity's allocators map over a gigabyte for a 2D game. wasm32 tops out at 4 GB.
#ifndef KYTY_EMU_MEM_MB
#define KYTY_EMU_MEM_MB 3072
#endif
constexpr uint64_t EMU_MEM_SIZE  = (uint64_t)KYTY_EMU_MEM_MB * 1024 * 1024;
// Layout: image | malloc heap | kernel mappings | main stack | HLE stubs + process args
constexpr uint64_t HLE_STUB_BASE  = EMU_BASE_ADDR + EMU_MEM_SIZE - 0x100000;
constexpr uint64_t PROC_ARGS_ADDR = HLE_STUB_BASE + 0x80000;
constexpr uint64_t STACK_TOP      = HLE_STUB_BASE - 0x1000;
constexpr uint64_t STACK_SIZE     = 8ULL << 20;
constexpr uint64_t HEAP_END       = EMU_BASE_ADDR + 0xC000000;

struct LoadedModule {
    std::string name;
    uint64_t base_addr;
    uint64_t entry_point;
    uint64_t text_start;
    uint64_t text_size;
    std::vector<std::string> needed_libs;
};

static uint8_t* g_emu_memory = nullptr;
static KytyWeb::X86Interpreter* g_cpu = nullptr;
static LoadedModule g_main_module;
static std::vector<uint8_t> g_framebuffer;
static int g_screen_w = 1920;
static int g_screen_h = 1080;
static bool g_running = false;
static uint64_t g_total_insn = 0;
static int g_hle_call_count = 0;

struct PadInputState {
    uint32_t buttons;
    float lx, ly, rx, ry, l2, r2;
};
static PadInputState g_pad_input = {};

using SteadyClock = std::chrono::steady_clock;
static SteadyClock::time_point g_frame_start;
static double g_fps = 0.0;
static int g_frame_count = 0;
static SteadyClock::time_point g_fps_timer;

// ---- Modules, exports and imports ----
// An import binds to another loaded module's export when one exists; otherwise to a 16-byte stub
// holding RET, whose handler (an SDK implementation, or "return 0") runs when RIP reaches it.

struct ModuleImage {
    std::string name;
    uint64_t base, size, init, start;
};
static std::vector<ModuleImage> g_modules;                 // module handle = index + 1; the eboot is 1
static std::unordered_map<std::string, uint64_t> g_exports;  // NID -> address
static std::unordered_map<uint64_t, std::unordered_map<std::string, uint64_t>> g_module_exports;  // by handle

struct Import {
    std::string nid;
    const char* name;
    KytyWeb::HleFn fn;
    uint64_t calls;
    uint64_t stub;
    std::vector<std::pair<uint64_t, int64_t>> slots;  // (address, addend) currently holding stub + addend
    bool linked;
};
static std::vector<Import> g_imports;
static std::unordered_map<std::string, size_t> g_import_index;

static void HleLoadStartModule(KytyWeb::X86Regs& r);
static void HleDlsym(KytyWeb::X86Regs& r);

static bool InEmu(uint64_t a, uint64_t n) { return a >= EMU_BASE_ADDR && n <= EMU_MEM_SIZE && a - EMU_BASE_ADDR <= EMU_MEM_SIZE - n; }
static void W64(uint64_t a, uint64_t v) { if (InEmu(a, 8)) memcpy(g_emu_memory + (a - EMU_BASE_ADDR), &v, 8); }
static void W32(uint64_t a, uint32_t v) { if (InEmu(a, 4)) memcpy(g_emu_memory + (a - EMU_BASE_ADDR), &v, 4); }
static uint64_t R64(uint64_t a) { uint64_t v = 0; if (InEmu(a, 8)) memcpy(&v, g_emu_memory + (a - EMU_BASE_ADDR), 8); return v; }
static std::string GuestStr(uint64_t a) {
    std::string s;
    while (InEmu(a, 1) && g_emu_memory[a - EMU_BASE_ADDR] && s.size() < 1024) s += (char)g_emu_memory[a++ - EMU_BASE_ADDR];
    return s;
}

static size_t StubImport(const std::string& nid) {
    auto it = g_import_index.find(nid);
    if (it != g_import_index.end()) return it->second;
    size_t idx = g_imports.size();
    uint64_t stub = HLE_STUB_BASE + idx * 16;
    const char* name = KytyWeb::HleName(nid);
    KytyWeb::HleFn fn = KytyWeb::HleFind(nid);
    if (name && !strcmp(name, "sceKernelLoadStartModule")) fn = HleLoadStartModule;
    if (name && !strcmp(name, "sceKernelDlsym")) fn = HleDlsym;
    g_imports.push_back({nid, name, fn, 0, stub, {}, false});
    g_import_index[nid] = idx;
    g_emu_memory[stub - EMU_BASE_ADDR] = 0xC3;
    g_cpu->RegisterHLEFunction(stub, [idx](KytyWeb::X86Regs& r, uint64_t) {
        Import& imp = g_imports[idx];
        g_hle_call_count++;
        if (imp.calls++ == 0)
            printf("[hle] first call: %s%s\n", imp.name ? imp.name : imp.nid.c_str(),
                   imp.fn ? "" : " (not implemented, returns 0)");
        if (imp.fn) imp.fn(r);
        else r.rax = 0;
    });
    return idx;
}

// The stub an NID's import is bound to, or 0 if nothing imports it.
static uint64_t StubOf(const std::string& nid) {
    auto it = g_import_index.find(nid);
    return it == g_import_index.end() ? 0 : g_imports[it->second].stub;
}

// The SDK function an import stub address belongs to, or "" for other addresses.
static std::string StubName(uint64_t a) {
    if (a < HLE_STUB_BASE || (a - HLE_STUB_BASE) / 16 >= g_imports.size()) return {};
    const Import& imp = g_imports[(a - HLE_STUB_BASE) / 16];
    return imp.name ? imp.name : imp.nid;
}

static void BindSymbol(uint64_t slot, const char* symbol, int64_t addend) {
    std::string nid(symbol, strcspn(symbol, "#"));
    auto e = g_exports.find(nid);
    if (e != g_exports.end()) { W64(slot, e->second + addend); return; }
    Import& imp = g_imports[StubImport(nid)];
    imp.slots.emplace_back(slot, addend);
    W64(slot, imp.stub + addend);
}

// Repoints slots bound to stubs at exports from modules loaded since.
static void RelinkImports() {
    for (auto& imp : g_imports) {
        if (imp.linked || imp.slots.empty()) continue;
        auto e = g_exports.find(imp.nid);
        if (e == g_exports.end()) continue;
        for (auto& [slot, addend] : imp.slots) W64(slot, e->second + addend);
        imp.linked = true;
        printf("[loader] bound %s to a module export\n", imp.name ? imp.name : imp.nid.c_str());
    }
}

struct DynInfo {
    uint64_t symtab = 0, strtab = 0, strsz = 0, syment = sizeof(Elf64_Sym), symtab_bytes = 0, hash = 0;
    uint64_t rela = 0, relasz = 0, jmprel = 0, pltrelsz = 0, init = 0;
};
constexpr int64_t DT_HASH_TAG = 4, DT_INIT_TAG = 12, DT_SCE_SYMTABSZ = 0x6100003f;
constexpr uint32_t R_X86_64_64 = 1, R_X86_64_DTPMOD64 = 16, R_X86_64_DTPOFF64 = 17;
constexpr const char* kModuleStartNid = "BaOKcng8g88";  // module_start

static DynInfo ReadDynamic(const uint8_t* file, const Elf64_Phdr* phdrs, int phnum) {
    DynInfo di;
    for (int i = 0; i < phnum; i++) {
        if (phdrs[i].p_type != PT_DYNAMIC) continue;
        for (auto* d = reinterpret_cast<const Elf64_Dyn*>(file + phdrs[i].p_offset); d->d_tag != DT_NULL; d++) {
            switch (d->d_tag) {
                case DT_SYMTAB:       di.symtab = d->d_un.d_ptr; break;
                case DT_STRTAB:       di.strtab = d->d_un.d_ptr; break;
                case DT_STRSZ:        di.strsz = d->d_un.d_val; break;
                case DT_SYMENT:       di.syment = d->d_un.d_val; break;
                case DT_RELA:         di.rela = d->d_un.d_ptr; break;
                case DT_RELASZ:       di.relasz = d->d_un.d_val; break;
                case DT_JMPREL:       di.jmprel = d->d_un.d_ptr; break;
                case DT_PLTRELSZ:     di.pltrelsz = d->d_un.d_val; break;
                case DT_INIT_TAG:     di.init = d->d_un.d_ptr; break;
                case DT_HASH_TAG:     di.hash = d->d_un.d_ptr; break;
                case DT_SCE_SYMTABSZ: di.symtab_bytes = d->d_un.d_val; break;
            }
        }
    }
    return di;
}

// Registers the image's exports and applies its relocations. `bias` maps image vaddrs to emulated addresses.
static size_t LinkImage(const DynInfo& di, uint64_t bias, uint64_t module_id, ModuleImage& m) {
    uint64_t nsyms = di.symtab_bytes ? di.symtab_bytes / di.syment
                   : di.hash ? (R64(bias + di.hash) >> 32) : 0;  // DT_HASH: {nbucket, nchain}
    auto sym_at = [&](uint64_t idx) -> const Elf64_Sym* {
        uint64_t a = bias + di.symtab + idx * di.syment;
        return di.symtab && InEmu(a, sizeof(Elf64_Sym)) ? reinterpret_cast<const Elf64_Sym*>(g_emu_memory + (a - EMU_BASE_ADDR)) : nullptr;
    };
    auto name_of = [&](const Elf64_Sym* s) -> const char* {
        uint64_t a = s ? bias + di.strtab + s->st_name : 0;
        return s && s->st_name < di.strsz && InEmu(a, 1) ? reinterpret_cast<const char*>(g_emu_memory + (a - EMU_BASE_ADDR)) : "";
    };
    size_t exported = 0;
    for (uint64_t k = 0; k < nsyms; k++) {
        const Elf64_Sym* s = sym_at(k);
        const char* name = name_of(s);
        if (!s || !s->st_shndx || !name[0]) continue;
        std::string nid(name, strcspn(name, "#"));
        if (nid == kModuleStartNid) { m.start = bias + s->st_value; continue; }
        g_module_exports[module_id][nid] = bias + s->st_value;
        if (g_exports.emplace(nid, bias + s->st_value).second) exported++;
    }
    size_t bound = 0, unsupported = 0;
    auto apply = [&](uint64_t table, uint64_t size) {
        for (uint64_t off = 0; off + sizeof(Elf64_Rela) <= size; off += sizeof(Elf64_Rela)) {
            uint64_t a = bias + table + off;
            if (!InEmu(a, sizeof(Elf64_Rela))) break;
            Elf64_Rela rel;
            memcpy(&rel, g_emu_memory + (a - EMU_BASE_ADDR), sizeof(rel));
            const uint32_t type = ELF64_R_TYPE(rel.r_info);
            const uint64_t slot = bias + rel.r_offset;
            const Elf64_Sym* s = ELF64_R_SYM(rel.r_info) ? sym_at(ELF64_R_SYM(rel.r_info)) : nullptr;
            switch (type) {
                case R_X86_64_RELATIVE: W64(slot, bias + rel.r_addend); break;
                case R_X86_64_64: case R_X86_64_GLOB_DAT: case R_X86_64_JUMP_SLOT: {
                    int64_t addend = type == R_X86_64_64 ? rel.r_addend : 0;
                    if (!s) W64(slot, addend);
                    else if (s->st_shndx) W64(slot, bias + s->st_value + addend);
                    else { BindSymbol(slot, name_of(s), addend); bound++; }
                    break;
                }
                case R_X86_64_DTPMOD64: W64(slot, module_id); break;
                case R_X86_64_DTPOFF64: W64(slot, (s ? s->st_value : 0) + rel.r_addend); break;
                default: unsupported++; break;
            }
        }
    };
    if (di.rela) apply(di.rela, di.relasz);
    if (di.jmprel) apply(di.jmprel, di.pltrelsz);
    printf("[loader] %s: %zu exports, %zu import slots bound%s\n", m.name.c_str(), exported, bound,
           unsupported ? " (some relocation types unsupported)" : "");
    return bound;
}

// Runs guest code at `fn` to completion from inside an SDK call, then restores the caller's registers.
constexpr uint64_t GUEST_RETURN_TRAP = PROC_ARGS_ADDR - 0x10;
static bool g_guest_returned = false;

static uint64_t CallGuest(uint64_t fn, uint64_t a0, uint64_t a1, uint64_t a2 = 0, uint64_t a3 = 0) {
    static bool trap_ready = false;
    if (!trap_ready) {
        g_emu_memory[GUEST_RETURN_TRAP - EMU_BASE_ADDR] = 0xC3;
        g_cpu->RegisterHLEFunction(GUEST_RETURN_TRAP, [](KytyWeb::X86Regs&, uint64_t) { g_guest_returned = true; });
        trap_ready = true;
    }
    KytyWeb::X86Regs saved = g_cpu->Regs();
    auto& r = g_cpu->Regs();
    r.rdi = a0;
    r.rsi = a1;
    r.rdx = a2;
    r.rcx = a3;
    r.rsp = ((saved.rsp - 512) & ~0xFULL) - 8;  // below the caller's red zone; aligned as after a call
    W64(r.rsp, GUEST_RETURN_TRAP);
    r.rip = fn;
    const bool outer = g_guest_returned;
    g_guest_returned = false;
    // Other threads can't run during a nested call, so a call that keeps parking can never finish.
    uint64_t parked_rip = 0, parked_retries = 0;
    for (uint64_t steps = 0; !g_guest_returned; steps++) {
        const uint64_t rip = r.rip;
        auto ex = g_cpu->Step();
        if (ex == KytyWeb::X86Exception::Yield) {
            if (r.rip == rip && rip == parked_rip && ++parked_retries > 100000) {
                printf("[loader] guest call to 0x%llx is blocked waiting on another thread, which can't run here\n",
                       (unsigned long long)fn);
                break;
            }
            if (r.rip == rip) parked_rip = rip;
            continue;
        }
        parked_retries = 0;
        if ((ex != KytyWeb::X86Exception::None && ex != KytyWeb::X86Exception::HLECall &&
             ex != KytyWeb::X86Exception::Syscall) || KytyWeb::HleHaltRequested()) {
            printf("[loader] guest call to 0x%llx stopped (exception %d at RIP=0x%llx)\n",
                   (unsigned long long)fn, (int)ex, (unsigned long long)r.rip);
            break;
        }
        if (steps > 3000000000ULL) { printf("[loader] guest call to 0x%llx did not return\n", (unsigned long long)fn); break; }
    }
    uint64_t ret = r.rax;
    r = saved;
    g_guest_returned = outer;
    return ret;
}

static int LoadPrx(const std::string& host_path, const std::string& name) {
    FILE* f = fopen(host_path.c_str(), "rb");
    if (!f) return -1;
    std::vector<uint8_t> file;
    fseek(f, 0, SEEK_END);
    file.resize(ftell(f));
    fseek(f, 0, SEEK_SET);
    size_t got = fread(file.data(), 1, file.size(), f);
    fclose(f);
    if (got != file.size() || file.size() < sizeof(Elf64_Ehdr) || memcmp(file.data(), "\x7f" "ELF", 4) != 0) {
        printf("[loader] %s is not a decrypted ELF\n", host_path.c_str());
        return -1;
    }
    auto* eh = reinterpret_cast<const Elf64_Ehdr*>(file.data());
    auto* ph = reinterpret_cast<const Elf64_Phdr*>(file.data() + eh->e_phoff);
    uint64_t lo = UINT64_MAX, hi = 0;
    for (int i = 0; i < eh->e_phnum; i++) {
        if (ph[i].p_type != PT_LOAD) continue;
        lo = std::min(lo, ph[i].p_vaddr);
        hi = std::max(hi, ph[i].p_vaddr + ph[i].p_memsz);
    }
    if (lo == UINT64_MAX) return -1;
    uint64_t base = KytyWeb::HleVmAlloc(hi - lo, 0x10000, 7, KytyWeb::HleRegion::Image, name.c_str());
    if (!base) return -1;
    const uint64_t bias = base - lo;
    memset(g_emu_memory + (base - EMU_BASE_ADDR), 0, hi - lo);
    for (int i = 0; i < eh->e_phnum; i++) {
        if (ph[i].p_type != PT_LOAD || !ph[i].p_filesz) continue;
        uint64_t n = std::min<uint64_t>(ph[i].p_filesz, file.size() - std::min<uint64_t>(ph[i].p_offset, file.size()));
        memcpy(g_emu_memory + (bias + ph[i].p_vaddr - EMU_BASE_ADDR), file.data() + ph[i].p_offset, n);
    }
    ModuleImage m{name, base, hi - lo, 0, 0};
    const uint64_t id = g_modules.size() + 1;
    for (int i = 0; i < eh->e_phnum; i++) {
        if (ph[i].p_type != PT_TLS || !ph[i].p_memsz) continue;
        // Each thread gets its own copy of this image on its first __tls_get_addr for the module.
        uint64_t filesz = ph[i].p_offset + ph[i].p_filesz <= file.size() ? ph[i].p_filesz : 0;
        uint64_t image = KytyWeb::HleMalloc(std::max<uint64_t>(filesz, 1), 16);
        memcpy(g_emu_memory + (image - EMU_BASE_ADDR), file.data() + ph[i].p_offset, filesz);
        KytyWeb::HleRegisterModuleTls(id, image, filesz, ph[i].p_memsz, std::max<uint64_t>(ph[i].p_align, 16));
    }
    for (int i = 0; i < eh->e_phnum; i++)
        if (ph[i].p_type == PT_GNU_EH_FRAME) KytyWeb::EhRegisterModule(base, hi - lo, bias + ph[i].p_vaddr);
    DynInfo di = ReadDynamic(file.data(), ph, eh->e_phnum);
    m.init = di.init ? bias + di.init : 0;
    g_modules.push_back(m);
    LinkImage(di, bias, id, g_modules.back());
    RelinkImports();
    printf("[loader] loaded %s at 0x%llx (0x%llx bytes)\n", name.c_str(), (unsigned long long)base,
           (unsigned long long)(hi - lo));
    return (int)id;
}

static void HleLoadStartModule(KytyWeb::X86Regs& r) {
    const std::string guest = GuestStr(r.rdi);
    const uint64_t args = r.rsi, argp = r.rdx, res = r.r9;
    const std::string host = KytyWeb::HleHostPath(guest);
    const std::string name = guest.substr(guest.find_last_of('/') + 1);
    // Modules are never unloaded (sceKernelStopUnloadModule does nothing), so loading one again returns the
    // handle it already has; Unity loads its native plugins over and over to look up symbols.
    static std::unordered_map<std::string, int> loaded;
    if (auto it = loaded.find(host); it != loaded.end()) {
        if (res) W32(res, 0);
        r.rax = (uint64_t)it->second;
        return;
    }
    int id = -1;
    for (const std::string& p : {host + ".esbak", host}) {
        struct stat st;
        if (stat(p.c_str(), &st) == 0 && (id = LoadPrx(p, name)) > 0) break;
    }
    if (id <= 0) {
        printf("[loader] cannot load module %s\n", guest.c_str());
        r.rax = (uint64_t)(int64_t)(int32_t)0x80020002;
        return;
    }
    const ModuleImage m = g_modules[id - 1];
    // module_start runs the module's own initialization (including DT_INIT), like _start does for the eboot.
    // Without one, DT_INIT is the PRX's start routine: it takes the same (args, argp) and returns the result.
    int64_t start_ret = 0;
    if (m.start) start_ret = (int32_t)CallGuest(m.start, args, argp);
    else if (m.init) start_ret = (int32_t)CallGuest(m.init, args, argp);
    printf("[loader] started %s (%s returned %lld)\n", name.c_str(), m.start ? "module_start" : "DT_INIT",
           (long long)start_ret);
    if (res) W32(res, (uint32_t)start_ret);
    loaded[host] = id;
    r.rax = (uint64_t)id;
}


// Symbol NIDs: the first 8 bytes of SHA-1(name + suffix), byte-reversed, base64 with '-' for '/'.
static std::string SymbolNid(const std::string& name) {
    static const uint8_t kSuffix[16] = {0x51, 0x8D, 0x64, 0xA6, 0x35, 0xDE, 0xD8, 0xC1,
                                        0xE6, 0xB0, 0x39, 0xB1, 0xC3, 0xE5, 0x52, 0x30};
    std::vector<uint8_t> msg(name.begin(), name.end());
    msg.insert(msg.end(), kSuffix, kSuffix + 16);
    const uint64_t bits = (uint64_t)msg.size() * 8;
    msg.push_back(0x80);
    while (msg.size() % 64 != 56) msg.push_back(0);
    for (int i = 7; i >= 0; i--) msg.push_back((uint8_t)(bits >> (8 * i)));
    uint32_t h[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
    auto rol = [](uint32_t v, int n) { return (v << n) | (v >> (32 - n)); };
    for (size_t off = 0; off < msg.size(); off += 64) {
        uint32_t w[80];
        for (int i = 0; i < 16; i++)
            w[i] = (uint32_t)msg[off + 4 * i] << 24 | (uint32_t)msg[off + 4 * i + 1] << 16 |
                   (uint32_t)msg[off + 4 * i + 2] << 8 | msg[off + 4 * i + 3];
        for (int i = 16; i < 80; i++) w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
        for (int i = 0; i < 80; i++) {
            uint32_t f, k;
            if (i < 20) { f = (b & c) | (~b & d); k = 0x5A827999; }
            else if (i < 40) { f = b ^ c ^ d; k = 0x6ED9EBA1; }
            else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
            else { f = b ^ c ^ d; k = 0xCA62C1D6; }
            const uint32_t t = rol(a, 5) + f + e + k + w[i];
            e = d; d = c; c = rol(b, 30); b = a; a = t;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e;
    }
    uint8_t digest[8];
    for (int i = 0; i < 8; i++) digest[i] = (uint8_t)(h[i / 4] >> (24 - 8 * (i % 4)));
    uint8_t raw[8];
    for (int i = 0; i < 8; i++) raw[i] = digest[7 - i];
    static const char* kB64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";
    std::string out;
    uint32_t acc = 0;
    int nbits = 0;
    for (uint8_t byte : raw) {
        acc = (acc << 8) | byte;
        nbits += 8;
        while (nbits >= 6) { out += kB64[(acc >> (nbits - 6)) & 63]; nbits -= 6; }
    }
    if (nbits) out += kB64[(acc << (6 - nbits)) & 63];
    return out;
}

// sceKernelDlsym(handle, name, address*): the module's export, then any module's, then an SDK function.
static void HleDlsym(KytyWeb::X86Regs& r) {
    const std::string name = GuestStr(r.rsi);
    const std::string nid = SymbolNid(name);
    uint64_t addr = 0;
    if (auto m = g_module_exports.find(r.rdi & 0xffffffff); m != g_module_exports.end()) {
        if (auto e = m->second.find(nid); e != m->second.end()) addr = e->second;
    }
    if (!addr) {
        if (auto e = g_exports.find(nid); e != g_exports.end()) addr = e->second;
    }
    if (!addr && KytyWeb::HleFind(nid)) addr = g_imports[StubImport(nid)].stub;
    if (!addr) {
        static int logged = 0;
        if (logged++ < 40) printf("[hle] dlsym %s (%s) in module %llu: not found\n", name.c_str(), nid.c_str(),
                                  (unsigned long long)r.rdi);
        r.rax = (uint64_t)(int64_t)(int32_t)0x80020002;
        return;
    }
    W64(r.rdx, addr);
    r.rax = 0;
}

// Printed once when execution makes no SDK calls for several frames, to diagnose spin loops.
static void DumpStall() {
    const auto& r = g_cpu->Regs();
    const char* names[16] = {"rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
                             "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"};
    const uint64_t vals[16] = {r.rax, r.rcx, r.rdx, r.rbx, r.rsp, r.rbp, r.rsi, r.rdi,
                               r.r8, r.r9, r.r10, r.r11, r.r12, r.r13, r.r14, r.r15};
    printf("[stall] register state at RIP=0x%llx\n", (unsigned long long)r.rip);
    for (int i = 0; i < 16; i++) {
        printf("[stall] %-3s = 0x%llx", names[i], (unsigned long long)vals[i]);
        uint64_t v = vals[i];
        if (InEmu(v, 32)) {
            printf("  ->");
            for (int b = 0; b < 32; b++) printf(" %02x", g_emu_memory[v - EMU_BASE_ADDR + b]);
        }
        printf("\n");
    }
    std::string trace;
    uint64_t fp = r.rbp;
    for (int depth = 0; depth < 16 && InEmu(fp, 16); depth++) {
        uint64_t next, ret;
        memcpy(&next, g_emu_memory + (fp - EMU_BASE_ADDR), 8);
        memcpy(&ret, g_emu_memory + (fp + 8 - EMU_BASE_ADDR), 8);
        char buf[24];
        snprintf(buf, sizeof(buf), " 0x%llx", (unsigned long long)ret);
        trace += buf;
        if (next <= fp) break;
        fp = next;
    }
    printf("[stall] return addresses (frame pointers):%s\n", trace.c_str());
}

static void LogTopImports() {
    std::vector<const Import*> called;
    for (const auto& i : g_imports)
        if (i.calls) called.push_back(&i);
    std::sort(called.begin(), called.end(), [](const Import* a, const Import* b) { return a->calls > b->calls; });
    std::string top;
    for (size_t i = 0; i < called.size() && i < 6; i++) {
        char buf[96];
        snprintf(buf, sizeof(buf), "%s%s x%llu", i ? ", " : "",
                 called[i]->name ? called[i]->name : called[i]->nid.c_str(), (unsigned long long)called[i]->calls);
        top += buf;
    }
    printf("[hle] %zu SDK functions reached; most called: %s\n", called.size(), top.c_str());
}

static void HLE_syscall(KytyWeb::X86Regs& regs, uint64_t) {
    uint64_t syscall_num = regs.rax;
    static int syscall_log_count = 0;
    if (syscall_log_count < 100) {
        printf("[hle] syscall %llu (rdi=0x%llx rsi=0x%llx rdx=0x%llx)\n",
               (unsigned long long)syscall_num,
               (unsigned long long)regs.rdi,
               (unsigned long long)regs.rsi,
               (unsigned long long)regs.rdx);
        syscall_log_count++;
    }

    switch (syscall_num) {
        case 4:   regs.rax = 0; break; // write
        case 5:   regs.rax = 3; break; // open -> return fd 3
        case 6:   regs.rax = 0; break; // close
        case 477: { // mmap
            uint64_t a = KytyWeb::HleVmAlloc(regs.rsi, 0, (int)regs.rdx, KytyWeb::HleRegion::Flexible, "mmap");
            regs.rax = a ? a : (uint64_t)-1;
            break;
        }
        case 12:  regs.rax = 0; break; // chdir
        case 20:  regs.rax = 1000; break; // getpid
        case 54:  regs.rax = 0; break; // ioctl
        case 73:  regs.rax = 0; break; // munmap
        case 74:  regs.rax = 0; break; // mprotect
        case 97:  regs.rax = 0; break; // socket
        case 165: regs.rax = 0; break; // sysarch (used for TLS setup)
        case 324: regs.rax = 0; break; // mlock
        case 557: regs.rax = 0; break; // thr_self
        case 587: { // sceKernelGetProcParam
            regs.rax = 0;
            break;
        }
        default:
            regs.rax = 0;
            break;
    }
}

// ---- ELF loader ----

static bool LoadELF(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[loader] cannot open %s\n", path);
        return false;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    std::vector<uint8_t> elf_data(file_size);
    fread(elf_data.data(), 1, file_size, f);
    fclose(f);

    auto* ehdr = reinterpret_cast<Elf64_Ehdr*>(elf_data.data());

    // Validate ELF magic
    if (ehdr->e_ident[0] != 0x7f || ehdr->e_ident[1] != 'E' ||
        ehdr->e_ident[2] != 'L'  || ehdr->e_ident[3] != 'F') {
        // PS5 ELFs may be in SCE format (signed) - try offset 0x100
        if (file_size > 0x200) {
            ehdr = reinterpret_cast<Elf64_Ehdr*>(elf_data.data() + 0x100);
            if (ehdr->e_ident[0] != 0x7f || ehdr->e_ident[1] != 'E' ||
                ehdr->e_ident[2] != 'L'  || ehdr->e_ident[3] != 'F') {
                fprintf(stderr, "[loader] not a valid ELF: %s\n", path);
                return false;
            }
            printf("[loader] detected SCE-signed ELF (offset 0x100)\n");
        } else {
            fprintf(stderr, "[loader] not a valid ELF: %s\n", path);
            return false;
        }
    }

    printf("[loader] ELF: type=%d machine=%d entry=0x%llx phnum=%d\n",
           ehdr->e_type, ehdr->e_machine,
           (unsigned long long)ehdr->e_entry, ehdr->e_phnum);

    uint8_t* elf_base = reinterpret_cast<uint8_t*>(ehdr);

    // Load PT_LOAD segments into emulated memory
    uint64_t load_base = EMU_BASE_ADDR;
    uint64_t lowest_vaddr = UINT64_MAX;
    uint64_t highest_end = 0;

    auto* phdrs = reinterpret_cast<Elf64_Phdr*>(elf_base + ehdr->e_phoff);
    for (int i = 0; i < ehdr->e_phnum; i++) {
        if (phdrs[i].p_type == PT_LOAD) {
            if (phdrs[i].p_vaddr < lowest_vaddr) lowest_vaddr = phdrs[i].p_vaddr;
            uint64_t end = phdrs[i].p_vaddr + phdrs[i].p_memsz;
            if (end > highest_end) highest_end = end;
        }
    }

    if (lowest_vaddr == UINT64_MAX) {
        fprintf(stderr, "[loader] no PT_LOAD segments found\n");
        return false;
    }

    uint64_t reloc_offset = load_base - lowest_vaddr;
    printf("[loader] relocation offset: 0x%llx (base=0x%llx, lowest_vaddr=0x%llx)\n",
           (unsigned long long)reloc_offset,
           (unsigned long long)load_base,
           (unsigned long long)lowest_vaddr);

    for (int i = 0; i < ehdr->e_phnum; i++) {
        auto& ph = phdrs[i];
        if (ph.p_type == PT_LOAD) {
            uint64_t dest = ph.p_vaddr + reloc_offset - EMU_BASE_ADDR;
            if (dest + ph.p_memsz > EMU_MEM_SIZE) {
                fprintf(stderr, "[loader] segment %d exceeds memory (dest=0x%llx, memsz=0x%llx)\n",
                        i, (unsigned long long)dest, (unsigned long long)ph.p_memsz);
                continue;
            }
            memset(g_emu_memory + dest, 0, ph.p_memsz);
            if (ph.p_filesz > 0) {
                size_t copy_size = ph.p_filesz;
                if (ph.p_offset + copy_size > (uint64_t)file_size) {
                    copy_size = file_size - ph.p_offset;
                }
                memcpy(g_emu_memory + dest, elf_base + ph.p_offset, copy_size);
            }
            printf("[loader] loaded segment %d: vaddr=0x%llx -> 0x%llx, filesz=0x%llx, memsz=0x%llx\n",
                   i, (unsigned long long)ph.p_vaddr,
                   (unsigned long long)(ph.p_vaddr + reloc_offset),
                   (unsigned long long)ph.p_filesz,
                   (unsigned long long)ph.p_memsz);
        }
    }

    g_main_module.name = path;
    g_main_module.base_addr = load_base;
    g_main_module.entry_point = ehdr->e_entry + reloc_offset;
    g_main_module.text_start = load_base;
    g_main_module.text_size = highest_end - lowest_vaddr;

    printf("[loader] entry point: 0x%llx\n", (unsigned long long)g_main_module.entry_point);

    uint64_t image_end = load_base + (highest_end - lowest_vaddr);
    KytyWeb::HleLayout layout{};
    layout.cpu = g_cpu;
    layout.call_guest = [](uint64_t fn, uint64_t a0, uint64_t a1, uint64_t a2) { return CallGuest(fn, a0, a1, a2); };
    layout.call_guest4 = [](uint64_t fn, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3) {
        return CallGuest(fn, a0, a1, a2, a3);
    };
    layout.mem = g_emu_memory;
    layout.base = EMU_BASE_ADDR;
    layout.size = EMU_MEM_SIZE;
    layout.heap_start = (image_end + 0xFFFFF) & ~0xFFFFFULL;
    layout.heap_end = HEAP_END;
    layout.stack_start = STACK_TOP - STACK_SIZE;
    layout.stack_end = STACK_TOP;
    layout.vm_start = HEAP_END;
    layout.vm_end = layout.stack_start;
    if (layout.heap_start >= layout.heap_end) {
        fprintf(stderr, "[loader] image too large for the emulated address space\n");
        return false;
    }
    KytyWeb::HleInit(layout);
    KytyWeb::EhSetup(layout, StubOf);
    for (int i = 0; i < ehdr->e_phnum; i++)
        if (phdrs[i].p_type == PT_GNU_EH_FRAME)
            KytyWeb::EhRegisterModule(load_base, highest_end - lowest_vaddr, reloc_offset + phdrs[i].p_vaddr);

    for (int i = 0; i < ehdr->e_phnum; i++) {
        const auto& ph = phdrs[i];
        if (ph.p_type != PT_LOAD) continue;
        int prot = ((ph.p_flags & 4) ? 1 : 0) | ((ph.p_flags & 2) ? 2 : 0) | ((ph.p_flags & 1) ? 4 : 0);
        KytyWeb::HleMapRegion(ph.p_vaddr + reloc_offset, ph.p_memsz, prot, KytyWeb::HleRegion::Image, "eboot");
    }

    // Static TLS (x86-64 variant II): the block sits just below the thread pointer, and
    // fs:[0] holds the thread pointer itself.
    uint64_t tls_size = 0, tls_align = 16, tls_filesz = 0, tls_offset = 0;
    for (int i = 0; i < ehdr->e_phnum; i++) {
        if (phdrs[i].p_type != PT_TLS) continue;
        tls_align = std::max<uint64_t>(phdrs[i].p_align, 16);
        tls_size = (phdrs[i].p_memsz + tls_align - 1) / tls_align * tls_align;
        tls_filesz = std::min<uint64_t>(phdrs[i].p_filesz, phdrs[i].p_memsz);
        tls_offset = phdrs[i].p_offset;
    }
    uint64_t tls_block = KytyWeb::HleMalloc(tls_size + 0x100, tls_align);
    uint64_t thread_ptr = tls_block + tls_size;
    memset(g_emu_memory + (tls_block - EMU_BASE_ADDR), 0, tls_size + 0x100);
    if (tls_filesz && tls_offset + tls_filesz <= (uint64_t)file_size)
        memcpy(g_emu_memory + (tls_block - EMU_BASE_ADDR), elf_base + tls_offset, tls_filesz);
    memcpy(g_emu_memory + (thread_ptr - EMU_BASE_ADDR), &thread_ptr, 8);
    g_cpu->Regs().fs_base = thread_ptr;
    uint64_t tls_image = KytyWeb::HleMalloc(std::max<uint64_t>(tls_filesz, 1), 16);
    memcpy(g_emu_memory + (tls_image - EMU_BASE_ADDR), g_emu_memory + (tls_block - EMU_BASE_ADDR), tls_filesz);
    KytyWeb::HleSetStaticTls(tls_image, tls_filesz, tls_size, tls_align);
    printf("[loader] TLS: 0x%llx bytes, thread pointer 0x%llx\n",
           (unsigned long long)tls_size, (unsigned long long)thread_ptr);

    g_modules.push_back({"eboot.bin", load_base, highest_end - lowest_vaddr, 0, 0});
    DynInfo di = ReadDynamic(elf_base, phdrs, ehdr->e_phnum);
    size_t bound = LinkImage(di, reloc_offset, g_modules.size(), g_modules.back());
    printf("[loader] patched %zu GOT entries (%zu distinct imports)\n", bound, g_imports.size());
    return true;
}

// ---- Main loop ----

static void MainLoop() {
    auto now = SteadyClock::now();
    auto elapsed = std::chrono::duration<double>(now - g_fps_timer).count();
    if (elapsed >= 1.0) {
        g_fps = g_frame_count / elapsed;
        g_frame_count = 0;
        g_fps_timer = now;

        auto* state = KytyWeb::GetState();
        state->fps = g_fps;
        state->frame_count++;

#ifdef __EMSCRIPTEN__
        EM_ASM({
            if (typeof window !== 'undefined' && window.kytyOnFPS) {
                window.kytyOnFPS($0);
            }
        }, g_fps);
#endif
    }

    // Run interpreter for a time slice
    if (g_cpu && g_running) {
        static int quiet_frames = 0;
        const int calls_before = g_hle_call_count;
        uint64_t executed = 0;
        auto exc = KytyWeb::HleRunThreads(1000000, &executed);
        g_total_insn += executed;

        static int diag_frames = 0;
        if (diag_frames < 20 || diag_frames % 60 == 0) {
            printf("[cpu] frame %d: executed %llu insns, total=%llu, RIP=0x%llx, HLE calls=%d\n",
                   diag_frames, (unsigned long long)executed, (unsigned long long)g_total_insn,
                   (unsigned long long)g_cpu->Regs().rip, g_hle_call_count);
        }
        if (diag_frames == 5 || diag_frames % 120 == 0) LogTopImports();
        if (diag_frames > 0 && diag_frames % 600 == 0) KytyWeb::HleDumpThreads(StubName);
        diag_frames++;
        quiet_frames = g_hle_call_count == calls_before ? quiet_frames + 1 : 0;
        if (quiet_frames == 3) DumpStall();

        if (KytyWeb::HleHaltRequested()) {
            printf("[cpu] halted at RIP=0x%llx after %llu instructions\n",
                   (unsigned long long)g_cpu->Regs().rip, (unsigned long long)g_total_insn);
            LogTopImports();
            g_running = false;
        } else if (exc == KytyWeb::X86Exception::InvalidOpcode) {
            // The decoder has already consumed an unknown length; resuming would execute garbage.
            uint64_t rip = g_cpu->Regs().rip;
            printf("[cpu] unimplemented instruction near RIP=0x%llx (bytes %02x %02x %02x %02x), halting\n",
                   (unsigned long long)rip, g_cpu->ReadMemory8(rip - 2), g_cpu->ReadMemory8(rip - 1),
                   g_cpu->ReadMemory8(rip), g_cpu->ReadMemory8(rip + 1));
            LogTopImports();
            g_running = false;
        } else if (exc == KytyWeb::X86Exception::PageFault) {
            uint64_t rip = g_cpu->FaultRip();
            uint8_t bytes[8] = {};
            if (InEmu(rip, 8)) memcpy(bytes, g_emu_memory + (rip - EMU_BASE_ADDR), 8);
            printf("[cpu] page fault: access to unmapped address 0x%llx by the instruction at RIP=0x%llx "
                   "(%02x %02x %02x %02x %02x %02x), halting\n",
                   (unsigned long long)g_cpu->FaultAddress(), (unsigned long long)rip,
                   bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5]);
            DumpStall();
            LogTopImports();
            g_running = false;
        } else if (exc == KytyWeb::X86Exception::GeneralProtection) {
            printf("[cpu] RIP out of range (0x%llx), halting interpreter\n",
                   (unsigned long long)g_cpu->Regs().rip);
            DumpStall();
            LogTopImports();
            g_running = false;
        } else if (exc == KytyWeb::X86Exception::Syscall) {
            // Syscall handled, continue execution
        } else if (exc != KytyWeb::X86Exception::None &&
                   exc != KytyWeb::X86Exception::HLECall) {
            static int exc_count = 0;
            exc_count++;
            if (exc_count <= 20) {
                printf("[cpu] exception %d at RIP=0x%llx\n",
                       (int)exc, (unsigned long long)g_cpu->Regs().rip);
            }
            if (exc_count > 100) {
                printf("[cpu] too many exceptions, halting\n");
                g_running = false;
            }
        }
    }

    g_frame_count++;  // the canvas is updated on guest flips (gpu.cpp)
}

} // anonymous namespace

// ---- Exported C API ----

extern "C" {

int kyty_init(const char* config_json) {
    printf("[kytyps5] initializing emulator...\n");

    auto* state = KytyWeb::GetState();
    memset(state, 0, sizeof(*state));

    // Allocate emulated memory
    g_emu_memory = static_cast<uint8_t*>(calloc(1, EMU_MEM_SIZE));
    if (!g_emu_memory) {
        fprintf(stderr, "[kytyps5] failed to allocate %llu bytes of emulated memory\n",
                (unsigned long long)EMU_MEM_SIZE);
        return -1;
    }
    printf("[kytyps5] allocated %llu MB emulated memory\n",
           (unsigned long long)(EMU_MEM_SIZE / (1024 * 1024)));

    // Create CPU interpreter
    g_cpu = new KytyWeb::X86Interpreter();
    g_cpu->SetMemory(g_emu_memory, EMU_MEM_SIZE, EMU_BASE_ADDR);
    g_cpu->SetAddressNamer(StubName);

    // Register syscall handler
    g_cpu->RegisterSyscallHandler(HLE_syscall);

    // Initialize graphics
    g_framebuffer.resize(g_screen_w * g_screen_h * 4, 0);
    KytyWeb::InitGraphics(g_screen_w, g_screen_h);

    // Initialize audio
    KytyWeb::InitAudio(48000, 2);

    // Initialize input
    KytyWeb::InitInput();

    state->initialized = true;
    printf("[kytyps5] emulator initialized\n");
    return 0;
}

int kyty_load_game(const char* game_path) {
    printf("[kytyps5] loading game: %s\n", game_path);
    KytyWeb::HleSetAppRoot(game_path);

    auto* state = KytyWeb::GetState();

    // Try to read param.json for game metadata
    std::string param_path = std::string(game_path) + "/sce_sys/param.json";
    FILE* pf = fopen(param_path.c_str(), "rb");
    if (pf) {
        fseek(pf, 0, SEEK_END);
        long pf_size = ftell(pf);
        fseek(pf, 0, SEEK_SET);
        std::vector<char> param_data(pf_size);
        fread(param_data.data(), 1, pf_size, pf);
        fclose(pf);

        KytyWeb::ParseParamJson(param_data.data(), pf_size, &state->game_info);
        printf("[kytyps5] game: %s (%s) v%s\n",
               state->game_info.title_name,
               state->game_info.title_id,
               state->game_info.content_version);
    }

    // Load the main eboot.bin — try several known paths
    std::string base(game_path);
    const char* eboot_candidates[] = {
        "/eboot.bin",
        "/decrypted/eboot.bin.esbak",
        "/decrypted/eboot.bin",
        "/eboot.bin.esbak",
    };
    std::string eboot_path;
    bool loaded = false;
    for (auto* suffix : eboot_candidates) {
        eboot_path = base + suffix;
        printf("[kytyps5] trying %s ...\n", eboot_path.c_str());
        if (LoadELF(eboot_path.c_str())) { loaded = true; break; }
    }
    if (!loaded) {
        fprintf(stderr, "[kytyps5] failed to load any eboot.bin variant\n");
        return -1;
    }

    // Set up CPU state for entry — PS5 CRT _start(argc_ptr, cleanup_fn)
    // RDI points to a SysV-style process param block: [argc, argv[0], NULL, envp..., NULL, auxv...]
    // RSI = cleanup function pointer (0 = none)
    uint64_t setup_base = PROC_ARGS_ADDR;
    uint64_t argv_str   = setup_base + 0x200;

    auto w64 = [&](uint64_t addr, uint64_t val) {
        uint64_t off = addr - EMU_BASE_ADDR;
        if (off + 8 <= EMU_MEM_SIZE)
            memcpy(g_emu_memory + off, &val, 8);
    };
    auto wb = [&](uint64_t addr, uint8_t val) {
        uint64_t off = addr - EMU_BASE_ADDR;
        if (off < EMU_MEM_SIZE) g_emu_memory[off] = val;
    };

    // Write "eboot.bin" string for argv[0]
    const char* arg0 = "eboot.bin";
    for (int j = 0; arg0[j]; j++) wb(argv_str + j, arg0[j]);
    wb(argv_str + strlen(arg0), 0);

    // Build process param block at setup_base:
    // [0]  argc = 1
    // [8]  argv[0] = pointer to "eboot.bin"
    // [16] NULL (argv terminator)
    // [24] NULL (envp terminator)
    // [32] AT_NULL (auxv terminator)
    w64(setup_base + 0,  1);          // argc
    w64(setup_base + 8,  argv_str);   // argv[0]
    w64(setup_base + 16, 0);          // argv NULL
    w64(setup_base + 24, 0);          // envp NULL
    w64(setup_base + 32, 0);          // auxv AT_NULL
    w64(setup_base + 40, 0);

    // Also set up the stack the way _start might expect it:
    // Some PS5 CRT variants read argc from [RSP] (System V ABI style)
    uint64_t stack_top = STACK_TOP;
    // Push auxv terminator, envp terminator, argv terminator, argv[0], argc
    stack_top -= 8; w64(stack_top, 0);          // auxv NULL
    stack_top -= 8; w64(stack_top, 0);          // envp NULL
    stack_top -= 8; w64(stack_top, 0);          // argv terminator
    stack_top -= 8; w64(stack_top, argv_str);   // argv[0]
    stack_top -= 8; w64(stack_top, 1);          // argc

    // _start(args, cleanup) runs the static constructors itself (DT_INIT) before main.
    g_cpu->SetStackPointer(stack_top);
    g_cpu->SetInstructionPointer(g_main_module.entry_point);
    g_cpu->Regs().rdi = setup_base;
    g_cpu->Regs().rsi = 0;
    g_cpu->SetTraceEnabled(false);

    printf("[kytyps5] entry at 0x%llx, stack at 0x%llx, args at 0x%llx\n",
           (unsigned long long)g_main_module.entry_point,
           (unsigned long long)stack_top,
           (unsigned long long)setup_base);

    state->game_loaded = true;
    state->running = true;
    g_running = true;

    // Start the main loop
    g_fps_timer = SteadyClock::now();
    g_frame_count = 0;
#ifdef __EMSCRIPTEN__
    emscripten_set_main_loop(MainLoop, 0, 0);
#endif

    printf("[kytyps5] game loaded and running\n");
    return 0;
}

// Streamed files. Chunks are 1 MB (kChunkSize in hle.cpp); a request is encoded as
// file * 1e7 + chunk index so it fits a JS number, or -1 when nothing is waiting.
int kyty_lazy_register(const char* path, double size) { return KytyWeb::HleLazyRegister(path, (uint64_t)size); }

double kyty_lazy_request() {
    int file;
    uint64_t chunk;
    return KytyWeb::HleLazyNextRequest(&file, &chunk) ? file * 1e7 + (double)chunk : -1.0;
}

void kyty_lazy_provide(int file, double chunk, const uint8_t* data, double length) {
    KytyWeb::HleLazyProvide(file, (uint64_t)chunk, data, (uint64_t)length);
}

// Debug aid: log when execution reaches a runtime address. Call after kyty_init.
void kyty_watch(uint32_t addr) {
    if (g_cpu) g_cpu->AddWatch(addr);
}

void kyty_watch_memory(uint32_t addr) {
    if (g_cpu) g_cpu->AddMemoryWatch(addr);
}

void kyty_frame() {
    MainLoop();
}

void kyty_input_values(int buttons, float lx, float ly, float rx, float ry, float l2, float r2) {
    g_pad_input.buttons = (uint32_t)buttons;
    g_pad_input.lx = lx;
    g_pad_input.ly = ly;
    g_pad_input.rx = rx;
    g_pad_input.ry = ry;
    g_pad_input.l2 = l2;
    g_pad_input.r2 = r2;
    if (g_cpu) KytyWeb::HleSetPadState((uint32_t)buttons, lx, ly, rx, ry, l2, r2);
}

int kyty_get_screen_width() { return g_screen_w; }
int kyty_get_screen_height() { return g_screen_h; }
// Frames the game has presented, for input scripted by frame (the frontend's ?press=).
double kyty_flip_count() { return (double)KytyWeb::GpuFlipCount(); }

#ifdef __EMSCRIPTEN__
int main(int argc, char** argv) {
    printf("[kytyps5] KytyPS5 WebAssembly Emulator\n");
    printf("[kytyps5] waiting for initialization from frontend...\n");

    // Notify the frontend that the WASM module is ready
    EM_ASM({
        if (typeof window !== 'undefined' && window.kytyOnReady) {
            window.kytyOnReady();
        }
    });

    return 0;
}
#else
// Native headless runner (tools/native): kytyps5 <game dir> [frames], one frame = one MainLoop time slice.
// KYTY_PRESS="flip:buttons:flips,..." holds the frontend's button bits (web/input.js) from a presented frame on.
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <game dir with eboot.bin> [frames]\n", argv[0]);
        return 2;
    }
    setvbuf(stdout, nullptr, _IOLBF, 0);
    const long frames = argc > 2 ? atol(argv[2]) : 100000;
    struct Press { uint64_t flip, buttons, flips; };
    std::vector<Press> presses;
    if (const char* s = getenv("KYTY_PRESS")) {
        for (const char* p = s; *p;) {
            char* end = nullptr;
            Press press{};
            press.flip = strtoull(p, &end, 0);
            if (end == p) break;
            if (*end == ':') press.buttons = strtoull(end + 1, &end, 0);
            if (*end == ':') press.flips = strtoull(end + 1, &end, 0);
            presses.push_back(press);
            p = *end == ',' ? end + 1 : end;
        }
    }
    kyty_init("{}");
    if (kyty_load_game(argv[1]) != 0) return 1;
    uint64_t held = 0;
    for (long i = 0; i < frames && g_running; i++) {
        if (!presses.empty()) {
            const uint64_t flip = KytyWeb::GpuFlipCount();
            uint64_t want = 0;
            for (const Press& p : presses)
                if (flip >= p.flip && flip < p.flip + p.flips) want |= p.buttons;
            if (want != held) {
                printf("[input] flip %llu: buttons 0x%llx\n", (unsigned long long)flip, (unsigned long long)want);
                kyty_input_values((int)want, 0, 0, 0, 0, 0, 0);
                held = want;
            }
        }
        const uint64_t before = g_total_insn;
        MainLoop();
        if (g_total_insn == before) usleep(2000);  // every thread is waiting: let wall-clock time pass
    }
    return 0;
}
#endif

} // extern "C"
