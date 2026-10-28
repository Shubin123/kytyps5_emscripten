// Emscripten main entry point for KytyPS5.
// Initializes the emulator subsystems, loads a PS5 game from the virtual filesystem
// (mounted by the frontend JS), and runs the main emulation loop via
// emscripten_set_main_loop.

#include "web_platform.h"
#include "x86_interpreter.h"

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
constexpr uint64_t EMU_STACK_TOP = EMU_BASE_ADDR + EMU_MEM_SIZE - 0x10000;

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

// ---- PS5 HLE (High-Level Emulation) stubs ----
// These intercept PS5 SDK calls and redirect them to browser APIs.

static void HLE_sceVideoOutOpen(KytyWeb::X86Regs& regs, uint64_t) {
    printf("[hle] sceVideoOutOpen(userId=%lld, busType=%lld)\n",
           (long long)regs.rdi, (long long)regs.rsi);
    regs.rax = 1; // return video handle 1
}

static void HLE_sceVideoOutRegisterBuffers(KytyWeb::X86Regs& regs, uint64_t) {
    printf("[hle] sceVideoOutRegisterBuffers(handle=%lld, startIndex=%lld, count=%lld)\n",
           (long long)regs.rdi, (long long)regs.rsi, (long long)regs.rdx);
    regs.rax = 0; // success
}

static void HLE_sceVideoOutSubmitFlip(KytyWeb::X86Regs& regs, uint64_t) {
    // Framebuffer presentation - copy the render target to our display buffer
    uint64_t buffer_addr = regs.rsi;
    if (buffer_addr >= EMU_BASE_ADDR && buffer_addr < EMU_BASE_ADDR + EMU_MEM_SIZE) {
        size_t offset = buffer_addr - EMU_BASE_ADDR;
        size_t fb_size = g_screen_w * g_screen_h * 4;
        if (offset + fb_size <= EMU_MEM_SIZE) {
            memcpy(g_framebuffer.data(), g_emu_memory + offset, fb_size);
        }
    }
    regs.rax = 0;
}

static void HLE_sceKernelUsleep(KytyWeb::X86Regs& regs, uint64_t) {
    // Don't actually sleep in the browser - just yield
    regs.rax = 0;
}

static void HLE_scePadReadState(KytyWeb::X86Regs& regs, uint64_t) {
    uint64_t data_addr = regs.rsi;
    if (data_addr >= EMU_BASE_ADDR && data_addr < EMU_BASE_ADDR + EMU_MEM_SIZE) {
        uint8_t pad[12];
        KytyWeb::PollInput(pad);
        memcpy(g_emu_memory + (data_addr - EMU_BASE_ADDR), pad, sizeof(pad));
    }
    regs.rax = 0;
}

static void HLE_sceGnmSubmitCommandBuffers(KytyWeb::X86Regs& regs, uint64_t) {
    // GPU command buffer submission - for now just acknowledge it
    regs.rax = 0;
}

static void HLE_scePthreadCreate(KytyWeb::X86Regs& regs, uint64_t) {
    printf("[hle] scePthreadCreate - threading not fully supported in WASM interpreter\n");
    regs.rax = 0;
}

static void HLE_sceKernelLoadStartModule(KytyWeb::X86Regs& regs, uint64_t) {
    uint64_t name_addr = regs.rdi;
    if (name_addr >= EMU_BASE_ADDR && name_addr < EMU_BASE_ADDR + EMU_MEM_SIZE) {
        const char* name = reinterpret_cast<const char*>(g_emu_memory + (name_addr - EMU_BASE_ADDR));
        printf("[hle] sceKernelLoadStartModule(%s)\n", name);
    }
    regs.rax = 2; // return module handle
}

static void HLE_default(KytyWeb::X86Regs& regs, uint64_t func_id) {
    printf("[hle] unhandled PS5 function call id=0x%llx\n", (unsigned long long)func_id);
    regs.rax = 0;
}

static void HLE_syscall(KytyWeb::X86Regs& regs, uint64_t) {
    uint64_t syscall_num = regs.rax;
    printf("[hle] syscall %llu\n", (unsigned long long)syscall_num);
    regs.rax = 0;
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

        // Process JUMP_SLOT relocations — patch GOT to point to HLE stubs
        if (jmprel_vaddr && pltrelsz && symtab_vaddr && strtab_vaddr) {
            uint64_t jmprel_off = mem_off(jmprel_vaddr);
            uint64_t symtab_off = mem_off(symtab_vaddr);
            uint64_t strtab_off = mem_off(strtab_vaddr);

            int num_jmprel = pltrelsz / sizeof(Elf64_Rela);
            printf("[loader] processing %d JUMP_SLOT relocations\n", num_jmprel);

            uint64_t hle_base = EMU_BASE_ADDR + EMU_MEM_SIZE - 0x100000;
            int hle_idx = 0;

            for (int j = 0; j < num_jmprel; j++) {
                if (jmprel_off + (j + 1) * sizeof(Elf64_Rela) > EMU_MEM_SIZE) break;
                auto* rela = reinterpret_cast<Elf64_Rela*>(g_emu_memory + jmprel_off + j * sizeof(Elf64_Rela));
                uint32_t type = ELF64_R_TYPE(rela->r_info);
                if (type != R_X86_64_JUMP_SLOT) continue;

                uint32_t sym_idx = ELF64_R_SYM(rela->r_info);
                uint64_t sym_off = symtab_off + sym_idx * syment;
                if (sym_off + sizeof(Elf64_Sym) > EMU_MEM_SIZE) continue;
                auto* sym = reinterpret_cast<Elf64_Sym*>(g_emu_memory + sym_off);

                const char* nid = "";
                uint64_t name_off = strtab_off + sym->st_name;
                if (name_off < strtab_off + strsz && name_off < EMU_MEM_SIZE) {
                    nid = reinterpret_cast<const char*>(g_emu_memory + name_off);
                }

                uint64_t stub_addr = hle_base + hle_idx * 16;
                hle_idx++;

                // Patch GOT entry (virtual address includes reloc_offset)
                uint64_t got_vaddr = rela->r_offset + reloc_offset;
                uint64_t got_off = got_vaddr - EMU_BASE_ADDR;
                if (got_off + 8 <= EMU_MEM_SIZE) {
                    *reinterpret_cast<uint64_t*>(g_emu_memory + got_off) = stub_addr;
                }

                // Write RET at stub
                uint64_t stub_off = stub_addr - EMU_BASE_ADDR;
                if (stub_off < EMU_MEM_SIZE) {
                    g_emu_memory[stub_off] = 0xC3;
                }

                // Register HLE handler
                std::string nid_str(nid);
                g_cpu->RegisterHLEFunction(stub_addr,
                    [nid_str, stub_addr](KytyWeb::X86Regs& r, uint64_t) {
                        g_hle_call_count++;
                        if (g_hle_call_count <= 100) {
                            printf("[hle] PLT call #%d: NID=%s addr=0x%llx\n",
                                   g_hle_call_count, nid_str.c_str(),
                                   (unsigned long long)stub_addr);
                        }
                        r.rax = 0;
                    });

                if (hle_idx <= 10) {
                    printf("[loader] GOT 0x%llx -> stub 0x%llx (NID: %.30s)\n",
                           (unsigned long long)got_vaddr, (unsigned long long)stub_addr, nid);
                }
            }
            printf("[loader] patched %d GOT entries\n", hle_idx);
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
                    // Similar to JUMP_SLOT — patch to an HLE stub
                    uint64_t addr_off = rela->r_offset + reloc_offset - EMU_BASE_ADDR;
                    if (addr_off + 8 <= EMU_MEM_SIZE) {
                        *reinterpret_cast<uint64_t*>(g_emu_memory + addr_off) = 0;
                        patched++;
                    }
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

        // Log first few frames of execution
        static int diag_frames = 0;
        if (diag_frames < 5) {
            printf("[cpu] frame %d: executed %llu insns, total=%llu, RIP=0x%llx, HLE calls=%d\n",
                   diag_frames, (unsigned long long)executed, (unsigned long long)g_total_insn,
                   (unsigned long long)g_cpu->Regs().rip, g_hle_call_count);
            diag_frames++;
        }

        if (exc == KytyWeb::X86Exception::InvalidOpcode) {
            static int invalid_count = 0;
            invalid_count++;
            if (invalid_count <= 10) {
                printf("[cpu] invalid opcode at RIP=0x%llx\n",
                       (unsigned long long)g_cpu->Regs().rip);
            }
            g_cpu->Regs().rip++;
        } else if (exc == KytyWeb::X86Exception::GeneralProtection) {
            printf("[cpu] RIP out of range (0x%llx), halting interpreter\n",
                   (unsigned long long)g_cpu->Regs().rip);
            g_running = false;
        } else if (exc != KytyWeb::X86Exception::None &&
                   exc != KytyWeb::X86Exception::HLECall) {
            static int exc_count = 0;
            exc_count++;
            if (exc_count <= 10) {
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

    // Set up CPU state for entry
    // Write a boot shim that calls a few HLE stubs to prove the system works,
    // then jumps to the real entry point with proper process arguments.
    uint64_t setup_base = EMU_BASE_ADDR + EMU_MEM_SIZE - 0x200000;
    uint64_t shim_addr  = EMU_BASE_ADDR + EMU_MEM_SIZE - 0x180000;
    uint64_t argv_str   = setup_base + 0x100;

    auto w64 = [&](uint64_t addr, uint64_t val) {
        uint64_t off = addr - EMU_BASE_ADDR;
        if (off + 8 <= EMU_MEM_SIZE)
            memcpy(g_emu_memory + off, &val, 8);
    };
    auto wb = [&](uint64_t addr, uint8_t val) {
        uint64_t off = addr - EMU_BASE_ADDR;
        if (off < EMU_MEM_SIZE) g_emu_memory[off] = val;
    };

    // Write "eboot.bin" string
    const char* arg0 = "eboot.bin";
    for (int j = 0; arg0[j]; j++) wb(argv_str + j, arg0[j]);
    wb(argv_str + strlen(arg0), 0);

    // Process info: argc, argv, envp, auxv
    w64(setup_base + 0, 1);
    w64(setup_base + 8, argv_str);
    w64(setup_base + 16, 0);
    w64(setup_base + 24, 0);
    w64(setup_base + 32, 0);
    w64(setup_base + 40, 0);

    // Build boot shim: call first 3 HLE stubs, then set up args and jump to entry
    // Machine code written at shim_addr
    uint64_t hle_base = EMU_BASE_ADDR + EMU_MEM_SIZE - 0x100000;
    uint64_t p = shim_addr;

    // CALL hle_stub_0  (E8 rel32)
    wb(p, 0xE8);
    int32_t rel0 = (int32_t)((int64_t)(hle_base + 0) - (int64_t)(p + 5));
    memcpy(g_emu_memory + (p + 1 - EMU_BASE_ADDR), &rel0, 4);
    p += 5;

    // CALL hle_stub_1  (E8 rel32)
    wb(p, 0xE8);
    int32_t rel1 = (int32_t)((int64_t)(hle_base + 16) - (int64_t)(p + 5));
    memcpy(g_emu_memory + (p + 1 - EMU_BASE_ADDR), &rel1, 4);
    p += 5;

    // CALL hle_stub_2  (E8 rel32)
    wb(p, 0xE8);
    int32_t rel2 = (int32_t)((int64_t)(hle_base + 32) - (int64_t)(p + 5));
    memcpy(g_emu_memory + (p + 1 - EMU_BASE_ADDR), &rel2, 4);
    p += 5;

    // MOV RDI, imm64 (48 BF imm64) — set up process info pointer
    wb(p, 0x48); wb(p+1, 0xBF);
    memcpy(g_emu_memory + (p + 2 - EMU_BASE_ADDR), &setup_base, 8);
    p += 10;

    // XOR ESI, ESI (31 F6) — RSI = 0
    wb(p, 0x31); wb(p+1, 0xF6);
    p += 2;

    // JMP entry_point (E9 rel32)
    wb(p, 0xE9);
    int32_t rel_entry = (int32_t)((int64_t)g_main_module.entry_point - (int64_t)(p + 5));
    memcpy(g_emu_memory + (p + 1 - EMU_BASE_ADDR), &rel_entry, 4);
    p += 5;

    g_cpu->SetStackPointer(EMU_STACK_TOP);
    g_cpu->SetInstructionPointer(shim_addr);

    printf("[kytyps5] boot shim at 0x%llx, entry at 0x%llx, hle_base at 0x%llx\n",
           (unsigned long long)shim_addr,
           (unsigned long long)g_main_module.entry_point,
           (unsigned long long)hle_base);

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
