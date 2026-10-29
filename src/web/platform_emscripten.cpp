// Emscripten platform layer — replaces SysLinuxVirtual.cpp / SysWindowsVirtual.cpp
// Provides virtual memory, threading, file I/O, and system info stubs for the WASM target.

#include "web_platform.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/threading.h>
#endif

// Kyty platform headers — we provide the implementations that upstream expects.
// The KYTY_PLATFORM_EMSCRIPTEN guard steers upstream headers to our stubs.

namespace KytyWeb {

static EmuState g_state = {};

EmuState* GetState() { return &g_state; }

bool ParseParamJson(const char* json_data, size_t len, GameInfo* info) {
    if (!json_data || !info) return false;
    memset(info, 0, sizeof(*info));

    std::string json(json_data, len);

    auto extract = [&](const char* key, char* out, size_t out_size) {
        std::string needle = std::string("\"") + key + "\"";
        auto pos = json.find(needle);
        if (pos == std::string::npos) return;
        pos = json.find(':', pos);
        if (pos == std::string::npos) return;
        auto q1 = json.find('"', pos + 1);
        if (q1 == std::string::npos) return;
        auto q2 = json.find('"', q1 + 1);
        if (q2 == std::string::npos) return;
        size_t copy_len = q2 - q1 - 1;
        if (copy_len >= out_size) copy_len = out_size - 1;
        memcpy(out, json.c_str() + q1 + 1, copy_len);
        out[copy_len] = '\0';
    };

    extract("titleId", info->title_id, sizeof(info->title_id));
    extract("contentId", info->content_id, sizeof(info->content_id));
    extract("contentVersion", info->content_version, sizeof(info->content_version));

    // Extract titleName from localizedParameters.en-US.titleName
    auto en_pos = json.find("\"en-US\"");
    if (en_pos != std::string::npos) {
        auto tn_pos = json.find("\"titleName\"", en_pos);
        if (tn_pos != std::string::npos) {
            auto colon = json.find(':', tn_pos);
            if (colon != std::string::npos) {
                auto q1 = json.find('"', colon + 1);
                if (q1 != std::string::npos) {
                    auto q2 = json.find('"', q1 + 1);
                    if (q2 != std::string::npos) {
                        size_t copy_len = q2 - q1 - 1;
                        if (copy_len >= sizeof(info->title_name)) copy_len = sizeof(info->title_name) - 1;
                        memcpy(info->title_name, json.c_str() + q1 + 1, copy_len);
                        info->title_name[copy_len] = '\0';
                    }
                }
            }
        }
    }

    info->screen_width = 1920;
    info->screen_height = 1080;
    return true;
}

} // namespace KytyWeb

// --- Emscripten virtual memory stubs ---
// Kyty's VirtualMemory module expects mmap-like allocation. Under Emscripten we use
// malloc-backed regions since WASM linear memory doesn't support mmap semantics.

namespace Kyty::Core {

struct SystemInfo {
    std::string ProcessorName;
};

static std::mutex g_vm_mutex;
static std::map<uintptr_t, size_t> g_allocs;

void sys_get_system_info(SystemInfo* info) {
    if (info) info->ProcessorName = "WebAssembly (Emscripten)";
}

void sys_virtual_init() {
    // No-op under Emscripten
}

void* sys_virtual_alloc(uint64_t addr, uint64_t size, int mode) {
    std::lock_guard<std::mutex> lock(g_vm_mutex);
    void* ptr = aligned_alloc(4096, (size + 4095) & ~4095ULL);
    if (ptr) {
        memset(ptr, 0, size);
        g_allocs[reinterpret_cast<uintptr_t>(ptr)] = size;
    }
    return ptr;
}

bool sys_virtual_alloc_fixed(uint64_t addr, uint64_t size, int mode) {
    // Fixed-address allocation is not meaningful in WASM.
    // Return true and hope the caller also stored the pointer from a previous alloc.
    return true;
}

bool sys_virtual_free(uint64_t addr) {
    std::lock_guard<std::mutex> lock(g_vm_mutex);
    auto it = g_allocs.find(static_cast<uintptr_t>(addr));
    if (it != g_allocs.end()) {
        free(reinterpret_cast<void*>(it->first));
        g_allocs.erase(it);
        return true;
    }
    return false;
}

bool sys_virtual_protect(uint64_t addr, uint64_t size, int mode, int* old_mode) {
    // No hardware page protection in WASM
    if (old_mode) *old_mode = 7; // RWX
    return true;
}

bool sys_virtual_flush_instruction_cache(uint64_t addr, uint64_t size) {
    return true; // No instruction cache in interpreted mode
}

} // namespace Kyty::Core

// --- Debug stubs ---

namespace Kyty::Core {

void sys_dbg_stack_info(char* buf, int buf_size) {
    if (buf && buf_size > 0) {
        snprintf(buf, buf_size, "[WASM stack]");
    }
}

bool sys_is_debugger_present() {
    return false;
}

void sys_debug_break() {
#ifdef __EMSCRIPTEN__
    emscripten_debugger();
#endif
}

} // namespace Kyty::Core

// --- File I/O stubs (Emscripten MEMFS-backed) ---

namespace Kyty::Core {

void sys_file_init() {
    // Emscripten MEMFS is always available
}

} // namespace Kyty::Core
