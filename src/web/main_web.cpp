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

    // Run interpreter for a time slice (roughly one frame's worth of instructions)
    if (g_cpu && g_running) {
        auto exc = g_cpu->Run(1000000); // ~1M instructions per frame
        if (exc == KytyWeb::X86Exception::InvalidOpcode) {
            printf("[cpu] invalid opcode at RIP=0x%llx\n",
                   (unsigned long long)g_cpu->Regs().rip);
            // Skip the instruction
            g_cpu->Regs().rip++;
        } else if (exc != KytyWeb::X86Exception::None &&
                   exc != KytyWeb::X86Exception::HLECall) {
            printf("[cpu] exception %d at RIP=0x%llx\n",
                   (int)exc, (unsigned long long)g_cpu->Regs().rip);
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
    g_cpu->SetMemory(g_emu_memory, EMU_MEM_SIZE);

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
    g_cpu->SetStackPointer(EMU_STACK_TOP);
    g_cpu->SetInstructionPointer(g_main_module.entry_point);

    // Register HLE stubs for known PS5 SDK functions
    // These addresses will be patched when we process the PLT/GOT
    // For now, register them at sentinel addresses used by our linker stubs
    uint64_t hle_base = EMU_BASE_ADDR + EMU_MEM_SIZE - 0x100000;
    struct HLEEntry {
        const char* name;
        void (*handler)(KytyWeb::X86Regs&, uint64_t);
    };
    HLEEntry hle_table[] = {
        {"sceVideoOutOpen",             HLE_sceVideoOutOpen},
        {"sceVideoOutRegisterBuffers",  HLE_sceVideoOutRegisterBuffers},
        {"sceVideoOutSubmitFlip",       HLE_sceVideoOutSubmitFlip},
        {"sceKernelUsleep",             HLE_sceKernelUsleep},
        {"scePadReadState",             HLE_scePadReadState},
        {"sceGnmSubmitCommandBuffers",  HLE_sceGnmSubmitCommandBuffers},
        {"scePthreadCreate",            HLE_scePthreadCreate},
        {"sceKernelLoadStartModule",    HLE_sceKernelLoadStartModule},
    };

    for (size_t i = 0; i < sizeof(hle_table) / sizeof(hle_table[0]); i++) {
        uint64_t addr = hle_base + i * 16;
        g_cpu->RegisterHLEFunction(addr,
            [handler = hle_table[i].handler](KytyWeb::X86Regs& r, uint64_t id) {
                handler(r, id);
            });
        // Write a RET instruction at the HLE stub address so the interpreter
        // returns to the caller after the HLE handler runs
        g_emu_memory[addr - EMU_BASE_ADDR] = 0xC3; // RET
    }

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
    // Called externally if the frontend wants to drive the loop manually
    MainLoop();
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
