#ifndef KYTY_HLE_H
#define KYTY_HLE_H

#include "x86_interpreter.h"

#include <cstdint>
#include <functional>
#include <string>

namespace KytyWeb {

// Address ranges inside the emulated window that SDK implementations allocate from.
struct HleLayout {
    X86Interpreter* cpu;
    // Runs guest code fn(a0, a1, a2) to completion (e.g. qsort comparators) and returns rax.
    uint64_t (*call_guest)(uint64_t fn, uint64_t a0, uint64_t a1, uint64_t a2);
    // The same with a fourth argument (rcx).
    uint64_t (*call_guest4)(uint64_t fn, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3);
    uint8_t* mem;
    uint64_t base, size;
    uint64_t heap_start, heap_end;
    uint64_t vm_start, vm_end;
    uint64_t stack_start, stack_end;
};

enum class HleRegion { Image, Heap, Stack, Flexible, Direct, Reserved };

using HleFn = void (*)(X86Regs&);

void HleInit(const HleLayout& layout);
// Records a mapped range so sceKernelVirtualQuery reports it.
void HleMapRegion(uint64_t start, uint64_t size, int prot, HleRegion kind, const char* name);
uint64_t HleVmAlloc(uint64_t size, uint64_t align, int prot, HleRegion kind, const char* name);
uint64_t HleMalloc(uint64_t size, uint64_t align);
// nid is the 11-character symbol hash without the "#lib#module" suffix.
HleFn HleFind(const std::string& nid);
const char* HleName(const std::string& nid);
// Set when the game calls exit/abort or fails a stack check.
bool HleHaltRequested();
// The host directory holding the game (/app0), and the host path of a guest path.
void HleSetAppRoot(const std::string& host_dir);
std::string HleHostPath(const std::string& guest_path);

// libSceAgc command-buffer builders and shader/state helpers (agc.cpp, ported from KytyPS5).
void AgcSetup(const HleLayout& layout);
HleFn AgcFind(const std::string& nid);
const char* AgcName(const std::string& nid);

// sceAvPlayer without decoders: movies play for their length as black frames (avplayer.cpp).
void AvPlayerSetup(const HleLayout& layout);
HleFn AvPlayerFind(const std::string& nid);
const char* AvPlayerName(const std::string& nid);

// GPU stand-in: AGC driver submission, video out (gpu.cpp).
void GpuSetup(const HleLayout& layout);
HleFn GpuFind(const std::string& nid);
// Resumes command buffers stopped at a wait whose condition now holds.
void GpuPoll();
const char* GpuName(const std::string& nid);
// Frames presented so far.
uint64_t GpuFlipCount();

// libSceAmpr / APR asynchronous file I/O (ampr.cpp, ported from KytyPS5).
void AmprSetup(const HleLayout& layout);
HleFn AmprFind(const std::string& nid);
const char* AmprName(const std::string& nid);

// C++ exceptions (unwind.cpp): __cxa_* and _Unwind_Resume over the guest's .eh_frame unwind tables.
// stub_of gives the import stub an NID is bound to (0 if none); type_info vtables are identified that way.
void EhSetup(const HleLayout& layout, std::function<uint64_t(const std::string&)> stub_of);
void EhRegisterModule(uint64_t base, uint64_t size, uint64_t eh_frame_hdr);
HleFn EhFind(const std::string& nid);
const char* EhName(const std::string& nid);

// The current guest thread's handle; HleFree releases HleMalloc memory; HleTerminate stops the game.
uint64_t HleCurrentThread();
void HleFree(uint64_t addr);
void HleTerminate(const char* why);

// For SDK functions outside hle.cpp. Park: the calling thread blocks and the same call re-executes
// once `ready` holds (call from an HleFn, without modifying registers first). FirstEntry is false
// while a parked call is being retried; EndCall marks it finished.
void HlePark(const char* what, std::function<bool()> ready);
bool HleFirstEntry(const X86Regs& r);
void HleEndCall();
// Game files by guest path ("/app0/..."): size, and reading a byte range into guest memory. A read of a
// streamed file whose data isn't loaded yet returns 0 with *pending set (the chunks are requested).
bool HleFileSize(const std::string& guest_path, uint64_t* size);
int64_t HleReadFileRange(const std::string& guest_path, uint64_t offset, uint64_t size, uint64_t guest_dst,
                         bool* pending);
// The guest's process clock (sceKernelGetProcessTime).
int64_t HleUsecSinceStart();

// Kernel event queues, for producers outside hle.cpp. filter: e.g. -13 graphics, -12 video out.
void HleEqueueAddEvent(uint64_t eq, uint64_t ident, int16_t filter, uint64_t udata);
void HleEqueueDeleteEvent(uint64_t eq, uint64_t ident, int16_t filter);
void HleEqueueTrigger(uint64_t ident, int16_t filter, uint64_t data);

// Streamed files: the frontend registers big files, fetches the 1 MB chunks reads ask for, and provides them.
int HleLazyRegister(const std::string& path, uint64_t size);
bool HleLazyNextRequest(int* file, uint64_t* chunk);
void HleLazyProvide(int file, uint64_t chunk, const uint8_t* data, uint64_t length);

// Controller state from the browser: frontend button bits (web/input.js), sticks in [-1, 1], triggers in [0, 1].
void HleSetPadState(uint32_t buttons, float lx, float ly, float rx, float ry, float l2, float r2);

// Guest threads run cooperatively in time slices. Returns the first fatal exception, or None.
X86Exception HleRunThreads(uint64_t budget, uint64_t* executed);
// Logs every guest thread's state, RIP (named by name_of when it is an import stub) and [rsp].
void HleDumpThreads(const std::function<std::string(uint64_t)>& name_of);
// TLS images new threads are given: the eboot's static block (below fs:0) and each module's dynamic block.
void HleSetStaticTls(uint64_t image, uint64_t filesz, uint64_t size, uint64_t align);
void HleRegisterModuleTls(uint64_t module_id, uint64_t image, uint64_t filesz, uint64_t memsz, uint64_t align);

} // namespace KytyWeb

#endif
