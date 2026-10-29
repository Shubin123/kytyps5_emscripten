#ifndef KYTY_HLE_H
#define KYTY_HLE_H

#include "x86_interpreter.h"

#include <cstdint>
#include <string>

namespace KytyWeb {

// Address ranges inside the emulated window that SDK implementations allocate from.
struct HleLayout {
    X86Interpreter* cpu;
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

} // namespace KytyWeb

#endif
