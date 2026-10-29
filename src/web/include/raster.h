#ifndef KYTY_RASTER_H
#define KYTY_RASTER_H

// Software rendering for the GPU stand-in: draws and dispatches run the game's RDNA shaders in the interpreter
// (rdna.h), rasterize triangles and rectangles, sample textures and write render targets in guest memory.

#include <cstdint>

namespace KytyWeb {

struct HleLayout;

// The GPU register state draws read: context (0xA000+), SH (0x2C00+) and user-config (0xC000+) registers.
struct GpuRegs {
    const uint32_t* cx;
    const uint32_t* sh;
    const uint32_t* uc;
};

struct DrawCall {
    uint32_t count = 0;        // vertices or indices
    uint64_t index_addr = 0;   // 0 for auto-indexed draws
    uint32_t index_type = 0;   // 0 16-bit, 1 32-bit, 2 8-bit
    uint32_t instances = 1;
};

void RasterSetup(const HleLayout& layout);
void RasterDraw(const GpuRegs& regs, const DrawCall& draw);
void RasterDispatch(const GpuRegs& regs, uint32_t x, uint32_t y, uint32_t z, uint32_t initiator);

// True during the frame KYTY_RASTER_FRAME selects for detailed logs.
bool RasterDebugFrame();

// Logs what was rendered since the previous flip.
void RasterFrameDone(uint64_t flip);

// Copies a display buffer into rgba (width * height * 4 bytes, linear RGBA8), undoing the render-target
// tiling it was drawn with. Returns false if the buffer isn't mapped.
bool RasterReadDisplay(uint64_t addr, uint32_t width, uint32_t height, uint8_t* rgba);

} // namespace KytyWeb

#endif
