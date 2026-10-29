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

#include <emscripten.h>
#include <emscripten/html5.h>

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
constexpr uint64_t EMU_MEM_SIZE  = 512ULL * 1024 * 1024; // 512 MB emulated address space
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

// ---- Imported SDK functions ----
// Each import gets a 16-byte stub holding RET; the interpreter runs the handler when RIP hits the stub.

struct Import {
    std::string nid;
    const char* name;
    KytyWeb::HleFn fn;
    uint64_t calls;
};
static std::vector<Import> g_imports;

static void RegisterImport(const char* symbol, uint64_t stub_addr) {
    std::string nid(symbol, strcspn(symbol, "#"));
    size_t idx = g_imports.size();
    g_imports.push_back({nid, KytyWeb::HleName(nid), KytyWeb::HleFind(nid), 0});
    g_emu_memory[stub_addr - EMU_BASE_ADDR] = 0xC3;
    g_cpu->RegisterHLEFunction(stub_addr, [idx](KytyWeb::X86Regs& r, uint64_t) {
        Import& imp = g_imports[idx];
        g_hle_call_count++;
        if (imp.calls++ == 0)
            printf("[hle] first call: %s%s\n", imp.name ? imp.name : imp.nid.c_str(),
                   imp.fn ? "" : " (not implemented, returns 0)");
        if (imp.fn) imp.fn(r);
        else r.rax = 0;
    });
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
    printf("[loader] TLS: 0x%llx bytes, thread pointer 0x%llx\n",
           (unsigned long long)tls_size, (unsigned long long)thread_ptr);

    // Process PT_DYNAMIC for relocations and PLT/GOT patching
    for (int i = 0; i < ehdr->e_phnum; i++) {
        if (phdrs[i].p_type != PT_DYNAMIC) continue;

        auto* dyn = reinterpret_cast<Elf64_Dyn*>(elf_base + phdrs[i].p_offset);
        uint64_t jmprel_vaddr = 0, pltrelsz = 0;
        uint64_t rela_vaddr = 0, relasz = 0;
        uint64_t symtab_vaddr = 0, strtab_vaddr = 0, strsz = 0;
        uint64_t syment = sizeof(Elf64_Sym);

        for (auto* d = dyn; d->d_tag != DT_NULL; d++) {
            switch (d->d_tag) {
                case DT_JMPREL:   jmprel_vaddr = d->d_un.d_ptr; break;
                case DT_PLTRELSZ: pltrelsz = d->d_un.d_val; break;
                case DT_RELA:     rela_vaddr = d->d_un.d_ptr; break;
                case DT_RELASZ:   relasz = d->d_un.d_val; break;
                case DT_SYMTAB:   symtab_vaddr = d->d_un.d_ptr; break;
                case DT_STRTAB:   strtab_vaddr = d->d_un.d_ptr; break;
                case DT_STRSZ:    strsz = d->d_un.d_val; break;
                case DT_SYMENT:   syment = d->d_un.d_val; break;
            }
        }

        printf("[loader] dynamic: JMPREL=0x%llx (%llu bytes), SYMTAB=0x%llx, STRTAB=0x%llx\n",
               (unsigned long long)jmprel_vaddr, (unsigned long long)pltrelsz,
               (unsigned long long)symtab_vaddr, (unsigned long long)strtab_vaddr);

        // Memory offsets: data at original vaddr V is at g_emu_memory[V + reloc_offset - EMU_BASE_ADDR]
        auto mem_off = [&](uint64_t vaddr) -> uint64_t {
            return vaddr + reloc_offset - EMU_BASE_ADDR;
        };

        auto sym_name = [&](uint64_t r_info) -> const char* {
            if (!symtab_vaddr || !strtab_vaddr) return "";
            uint64_t sym_off = mem_off(symtab_vaddr) + ELF64_R_SYM(r_info) * syment;
            if (sym_off + sizeof(Elf64_Sym) > EMU_MEM_SIZE) return "";
            auto* sym = reinterpret_cast<Elf64_Sym*>(g_emu_memory + sym_off);
            uint64_t strtab_off = mem_off(strtab_vaddr);
            uint64_t name_off = strtab_off + sym->st_name;
            if (name_off >= strtab_off + strsz || name_off >= EMU_MEM_SIZE) return "";
            return reinterpret_cast<const char*>(g_emu_memory + name_off);
        };
        auto patch_import = [&](const Elf64_Rela* rela) {
            uint64_t stub_addr = HLE_STUB_BASE + g_imports.size() * 16;
            uint64_t got_off = rela->r_offset + reloc_offset - EMU_BASE_ADDR;
            if (got_off + 8 <= EMU_MEM_SIZE)
                memcpy(g_emu_memory + got_off, &stub_addr, 8);
            RegisterImport(sym_name(rela->r_info), stub_addr);
        };

        // Process JUMP_SLOT relocations — patch GOT to point to HLE stubs
        if (jmprel_vaddr && pltrelsz) {
            uint64_t jmprel_off = mem_off(jmprel_vaddr);
            int num_jmprel = pltrelsz / sizeof(Elf64_Rela);
            for (int j = 0; j < num_jmprel; j++) {
                if (jmprel_off + (j + 1) * sizeof(Elf64_Rela) > EMU_MEM_SIZE) break;
                auto* rela = reinterpret_cast<Elf64_Rela*>(g_emu_memory + jmprel_off + j * sizeof(Elf64_Rela));
                if (ELF64_R_TYPE(rela->r_info) == R_X86_64_JUMP_SLOT) patch_import(rela);
            }
            printf("[loader] patched %zu GOT entries\n", g_imports.size());
        }

        // Process R_X86_64_RELATIVE relocations
        if (rela_vaddr && relasz) {
            uint64_t rela_off = mem_off(rela_vaddr);
            int num_rela = relasz / sizeof(Elf64_Rela);
            int patched = 0;

            for (int j = 0; j < num_rela; j++) {
                if (rela_off + (j + 1) * sizeof(Elf64_Rela) > EMU_MEM_SIZE) break;
                auto* rela = reinterpret_cast<Elf64_Rela*>(g_emu_memory + rela_off + j * sizeof(Elf64_Rela));
                uint32_t type = ELF64_R_TYPE(rela->r_info);

                if (type == R_X86_64_RELATIVE) {
                    uint64_t addr_off = rela->r_offset + reloc_offset - EMU_BASE_ADDR;
                    if (addr_off + 8 <= EMU_MEM_SIZE) {
                        uint64_t val = (uint64_t)rela->r_addend + reloc_offset;
                        *reinterpret_cast<uint64_t*>(g_emu_memory + addr_off) = val;
                        patched++;
                    }
                } else if (type == R_X86_64_GLOB_DAT) {
                    patch_import(rela);
                    patched++;
                }
            }
            printf("[loader] processed %d RELA relocations (of %d)\n", patched, num_rela);
        }

        break;
    }

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

        EM_ASM({
            if (typeof window !== 'undefined' && window.kytyOnFPS) {
                window.kytyOnFPS($0);
            }
        }, g_fps);
    }

    // Run interpreter for a time slice
    if (g_cpu && g_running) {
        uint64_t before = g_cpu->GetInstructionCount();
        auto exc = g_cpu->Run(1000000);
        uint64_t executed = g_cpu->GetInstructionCount() - before;
        g_total_insn += executed;

        static int diag_frames = 0;
        if (diag_frames < 20 || diag_frames % 60 == 0) {
            printf("[cpu] frame %d: executed %llu insns, total=%llu, RIP=0x%llx, HLE calls=%d\n",
                   diag_frames, (unsigned long long)executed, (unsigned long long)g_total_insn,
                   (unsigned long long)g_cpu->Regs().rip, g_hle_call_count);
        }
        if (diag_frames == 5 || diag_frames % 120 == 0) LogTopImports();
        diag_frames++;

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
        } else if (exc == KytyWeb::X86Exception::GeneralProtection) {
            printf("[cpu] RIP out of range (0x%llx), halting interpreter\n",
                   (unsigned long long)g_cpu->Regs().rip);
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

    // Present the framebuffer
    KytyWeb::PresentFrame(g_framebuffer.data(), g_screen_w, g_screen_h);
    g_frame_count++;
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

    g_cpu->SetStackPointer(stack_top);
    g_cpu->SetInstructionPointer(g_main_module.entry_point);
    // RDI = pointer to argc block (same format as stack), RSI = 0 (no cleanup)
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
    emscripten_set_main_loop(MainLoop, 0, 0);

    printf("[kytyps5] game loaded and running\n");
    return 0;
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
}

int kyty_get_screen_width() { return g_screen_w; }
int kyty_get_screen_height() { return g_screen_h; }

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

} // extern "C"
