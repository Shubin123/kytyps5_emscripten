// Software rendering: see raster.h. Surface tiling follows KytyPS5's tile.cpp
// (https://github.com/KytyPS5/KytyPS5, GPL-2.0); block-compressed textures are decoded with bcdec
// (third_party/bcdec.h, MIT).

#include "raster.h"

#include "hle.h"
#include "rdna.h"

#include <pthread.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef __EMSCRIPTEN__
#include <emscripten/threading.h>
#endif

#define BCDEC_STATIC
#define BCDEC_BC4BC5_PRECISE
#define BCDEC_IMPLEMENTATION
#include "third_party/bcdec.h"

namespace KytyWeb {
namespace {

using Rdna::kLanes;

HleLayout L;

uint8_t* Ptr(uint64_t a, uint64_t n) {
    if (a < L.base || n > L.size || a - L.base > L.size - n) return nullptr;
    return L.mem + (a - L.base);
}

inline float F(uint32_t b) { float f; memcpy(&f, &b, 4); return f; }
inline uint32_t U(float f) { uint32_t b; memcpy(&b, &f, 4); return b; }

int g_log_budget = 60;
#define RASTER_LOG(...)                       \
    do {                                      \
        if (g_log_budget > 0) {               \
            g_log_budget--;                   \
            printf("[raster] " __VA_ARGS__);  \
        }                                     \
    } while (0)

// ---------------------------------------------------------------------------------------------------------------
// Surface tiling. Elements are texels, or 4x4 blocks for block-compressed formats.

enum class Family { kLinear, k256B, k4KB, k64KB, kPrt64KB, kRt64KB, kDepth64KB };

// Swizzle modes as T# and CB/DB registers encode them; the PS5 uses linear, 256B_S, 4KB_S, 64KB_S, 64KB_S_T
// (PRT), 64KB_Z_X (depth) and 64KB_R_X (render targets). Other values fall back to the closest of those.
Family FamilyOf(uint32_t sw) {
    switch (sw) {
        case 0: return Family::kLinear;
        case 1: case 2: case 3: return Family::k256B;
        case 4: case 5: case 6: case 7: case 20: case 21: case 22: case 23: return Family::k4KB;
        case 16: case 17: case 18: case 19: return Family::kPrt64KB;
        case 24: return Family::kDepth64KB;
        case 27: return Family::kRt64KB;
        default: return Family::k64KB;
    }
}

int Log2(uint32_t v) { return 31 - __builtin_clz(v); }

uint32_t Std4KB(uint32_t x, uint32_t y, uint32_t bpe) {
    uint32_t o = 0;
    switch (bpe) {
        case 1:
            o ^= (y << 4) & 0x1f0; o ^= (y << 5) & 0x400;
            o ^= x & 0x00f; o ^= (x << 5) & 0x200; o ^= (x << 6) & 0x800;
            return o;
        case 2:
            o ^= (y << 4) & 0x070; o ^= (y << 5) & 0x100; o ^= (y << 6) & 0x400;
            o ^= (x << 1) & 0x00e; o ^= (x << 4) & 0x080; o ^= (x << 5) & 0x200; o ^= (x << 6) & 0x800;
            return o;
        case 4:
            o ^= (y << 4) & 0x070; o ^= (y << 5) & 0x100; o ^= (y << 6) & 0x400;
            o ^= (x << 2) & 0x00c; o ^= (x << 5) & 0x080; o ^= (x << 6) & 0x200; o ^= (x << 7) & 0x800;
            return o;
        case 8:
            o ^= (y << 4) & 0x030; o ^= (y << 6) & 0x100; o ^= (y << 7) & 0x400;
            o ^= (x << 3) & 0x008; o ^= (x << 5) & 0x0c0; o ^= (x << 6) & 0x200; o ^= (x << 7) & 0x800;
            return o;
        default:
            o ^= (y << 4) & 0x030; o ^= (y << 6) & 0x100; o ^= (y << 7) & 0x400;
            o ^= (x << 6) & 0x0c0; o ^= (x << 7) & 0x200; o ^= (x << 8) & 0x800;
            return o;
    }
}

uint32_t Std64KB(uint32_t x, uint32_t y, uint32_t bpe) {
    uint32_t o = 0;
    switch (bpe) {
        case 1:
            o ^= x & 0x000f; o ^= (x << 5) & 0x0200; o ^= (x << 6) & 0x0800; o ^= (x << 7) & 0x2000;
            o ^= (x << 8) & 0x8000;
            o ^= (y << 4) & 0x01f0; o ^= (y << 5) & 0x0400; o ^= (y << 6) & 0x1000; o ^= (y << 7) & 0x4000;
            return o;
        case 2:
            o ^= (x << 1) & 0x000e; o ^= (x << 4) & 0x0080; o ^= (x << 5) & 0x0200; o ^= (x << 6) & 0x0800;
            o ^= (x << 7) & 0x2000; o ^= (x << 8) & 0x8000;
            o ^= (y << 4) & 0x0070; o ^= (y << 5) & 0x0100; o ^= (y << 6) & 0x0400; o ^= (y << 7) & 0x1000;
            o ^= (y << 8) & 0x4000;
            return o;
        case 4:
            o ^= (x << 2) & 0x000c; o ^= (x << 5) & 0x0080; o ^= (x << 6) & 0x0200; o ^= (x << 7) & 0x0800;
            o ^= (x << 8) & 0x2000; o ^= (x << 9) & 0x8000;
            o ^= (y << 4) & 0x0070; o ^= (y << 5) & 0x0100; o ^= (y << 6) & 0x0400; o ^= (y << 7) & 0x1000;
            o ^= (y << 8) & 0x4000;
            return o;
        case 8:
            o ^= (x << 3) & 0x0008; o ^= (x << 5) & 0x00c0; o ^= (x << 6) & 0x0200; o ^= (x << 7) & 0x0800;
            o ^= (x << 8) & 0x2000; o ^= (x << 9) & 0x8000;
            o ^= (y << 4) & 0x0030; o ^= (y << 6) & 0x0100; o ^= (y << 7) & 0x0400; o ^= (y << 8) & 0x1000;
            o ^= (y << 9) & 0x4000;
            return o;
        default:
            o ^= (x << 6) & 0x00c0; o ^= (x << 7) & 0x0200; o ^= (x << 8) & 0x0800; o ^= (x << 9) & 0x2000;
            o ^= (x << 10) & 0x8000;
            o ^= (y << 4) & 0x0030; o ^= (y << 6) & 0x0100; o ^= (y << 7) & 0x0400; o ^= (y << 8) & 0x1000;
            o ^= (y << 9) & 0x4000;
            return o;
    }
}

uint32_t RenderTarget64KB(uint32_t x, uint32_t y, uint32_t bpe) {
    uint32_t o = 0;
    switch (bpe) {
        case 1:
            o ^= (y << 2) & 0x0008; o ^= (y << 4) & 0x0010; o ^= (y << 3) & 0x00a0; o ^= (y << 5) & 0x0f00;
            o ^= (y << 6) & 0x1000; o ^= (y << 7) & 0x4000;
            o ^= x & 0x0007; o ^= (x << 3) & 0x0040; o ^= (x << 5) & 0x0300; o ^= (x << 4) & 0x0400;
            o ^= (x << 6) & 0x0800; o ^= (x << 7) & 0x2000; o ^= (x << 8) & 0x8000;
            return o;
        case 2:
            o ^= (y << 4) & 0x0070; o ^= (y << 5) & 0x0f00; o ^= (y << 8) & 0x5000;
            o ^= (x << 1) & 0x000e; o ^= (x << 4) & 0x0480; o ^= (x << 5) & 0x0300; o ^= (x << 6) & 0x0800;
            o ^= (x << 7) & 0x2000; o ^= (x << 8) & 0x8000;
            return o;
        case 4:
            o ^= (y << 4) & 0x0070; o ^= (y << 5) & 0x0f00; o ^= (y << 9) & 0x1000; o ^= (y << 8) & 0x4000;
            o ^= (x << 2) & 0x000c; o ^= (x << 5) & 0x0380; o ^= (x << 4) & 0x0400; o ^= (x << 6) & 0x0800;
            o ^= (x << 9) & 0xa000;
            return o;
        case 8:
            o ^= (y << 4) & 0x0010; o ^= (y << 6) & 0x0080; o ^= (y << 5) & 0x0f00; o ^= (y << 10) & 0x5000;
            o ^= (x << 3) & 0x0008; o ^= (x << 4) & 0x0460; o ^= (x << 5) & 0x0300; o ^= (x << 6) & 0x0800;
            o ^= (x << 10) & 0x2000; o ^= (x << 9) & 0x8000;
            return o;
        default:
            o ^= (x << 4) & 0x0410; o ^= (x << 5) & 0x0340; o ^= (x << 6) & 0x0800; o ^= (x << 11) & 0xa000;
            o ^= (y << 5) & 0x0f20; o ^= (y << 6) & 0x0080; o ^= (y << 10) & 0x1000; o ^= (y << 11) & 0x4000;
            return o;
    }
}

uint32_t Depth64KB(uint32_t x, uint32_t y, uint32_t bpe) {
    uint32_t o = 0;
    switch (bpe) {
        case 1:
            o ^= x & 0x0001; o ^= (x << 1) & 0x0004; o ^= (x << 2) & 0x0010; o ^= (x << 3) & 0x0040;
            o ^= (x << 5) & 0x0300; o ^= (x << 4) & 0x0400; o ^= (x << 6) & 0x0800; o ^= (x << 7) & 0x2000;
            o ^= (x << 8) & 0x8000;
            o ^= (y << 1) & 0x0002; o ^= (y << 2) & 0x0008; o ^= (y << 3) & 0x00a0; o ^= (y << 5) & 0x0f00;
            o ^= (y << 6) & 0x1000; o ^= (y << 7) & 0x4000;
            return o;
        case 2:
            o ^= (x << 1) & 0x0002; o ^= (x << 2) & 0x0008; o ^= (x << 3) & 0x0020; o ^= (x << 4) & 0x0480;
            o ^= (x << 5) & 0x0300; o ^= (x << 6) & 0x0800; o ^= (x << 7) & 0x2000; o ^= (x << 8) & 0x8000;
            o ^= (y << 2) & 0x0004; o ^= (y << 3) & 0x0010; o ^= (y << 4) & 0x0040; o ^= (y << 5) & 0x0f00;
            o ^= (y << 8) & 0x5000;
            return o;
        case 4:
            o ^= (x << 2) & 0x0004; o ^= (x << 3) & 0x0010; o ^= (x << 4) & 0x0440; o ^= (x << 5) & 0x0300;
            o ^= (x << 6) & 0x0800; o ^= (x << 9) & 0xa000;
            o ^= (y << 3) & 0x0008; o ^= (y << 4) & 0x0020; o ^= (y << 5) & 0x0f80; o ^= (y << 9) & 0x1000;
            o ^= (y << 8) & 0x4000;
            return o;
        default:
            o ^= (x << 3) & 0x0008; o ^= (x << 4) & 0x0420; o ^= (x << 5) & 0x0380; o ^= (x << 6) & 0x0800;
            o ^= (x << 10) & 0x2000; o ^= (x << 9) & 0x8000;
            o ^= (y << 4) & 0x0010; o ^= (y << 5) & 0x0f40; o ^= (y << 10) & 0x5000;
            return o;
    }
}

uint32_t Bit(uint32_t v, int from, int to) { return ((v >> from) & 1) << to; }

uint32_t InBlock(Family f, uint32_t bpe, uint32_t x, uint32_t y) {
    switch (f) {
        case Family::k256B: return Std4KB(x, y, bpe) & 0xff;
        case Family::k4KB: return Std4KB(x, y, bpe);
        case Family::k64KB: return Std64KB(x, y, bpe);
        case Family::kPrt64KB: {
            static const uint8_t kBits[5][4] = {{7, 7, 6, 6}, {7, 6, 6, 5}, {6, 6, 5, 5}, {6, 5, 5, 4}, {5, 5, 4, 4}};
            const uint8_t* b = kBits[Log2(bpe)];
            return Std64KB(x, y, bpe) ^ Bit(x, b[0], 8) ^ Bit(y, b[1], 9) ^ Bit(x, b[2], 10) ^ Bit(y, b[3], 11);
        }
        case Family::kRt64KB: return RenderTarget64KB(x, y, bpe);
        case Family::kDepth64KB: return Depth64KB(x, y, bpe);
        default: return 0;
    }
}

// Each in-block offset bit is an XOR of x and y bits, so the offset is the XOR of a part from x and a part from y,
// tabulated per family and element size.
struct Swizzle { uint32_t x[256], y[256]; };
const Swizzle* SwizzleFor(Family f, uint32_t bpe) {
    using Tables = Swizzle[7][5];
    static const Tables* tables = [] {
        auto* t = new Tables[1];
        for (int fi = 0; fi < 7; fi++)
            for (int lb = 0; lb < 5; lb++)
                for (uint32_t i = 0; i < 256; i++) {
                    (*t)[fi][lb].x[i] = InBlock((Family)fi, 1u << lb, i, 0);
                    (*t)[fi][lb].y[i] = InBlock((Family)fi, 1u << lb, 0, i);
                }
        return t;
    }();
    return &(*tables)[(int)f][Log2(bpe)];
}

// Block dimensions (log2 width, height) by log2 bytes per element.
const uint8_t kBlock256B[5][2] = {{4, 4}, {4, 3}, {3, 3}, {3, 2}, {2, 2}};
const uint8_t kBlock4KB[5][2] = {{6, 6}, {6, 5}, {5, 5}, {5, 4}, {4, 4}};
const uint8_t kBlock64KB[5][2] = {{8, 8}, {8, 7}, {7, 7}, {7, 6}, {6, 6}};

// Where each mip of the tail sits inside the tail block (the tail holds the smallest levels).
const uint8_t kTail4KB[5][8][2] = {
    {{32, 0}, {16, 32}, {0, 48}, {0, 32}, {16, 16}, {16, 0}, {0, 16}, {0, 0}},
    {{32, 0}, {16, 16}, {0, 24}, {0, 16}, {16, 8}, {16, 0}, {0, 8}, {0, 0}},
    {{16, 0}, {8, 16}, {0, 24}, {0, 16}, {8, 8}, {8, 0}, {0, 8}, {0, 0}},
    {{16, 0}, {8, 8}, {0, 12}, {0, 8}, {8, 4}, {8, 0}, {0, 4}, {0, 0}},
    {{8, 0}, {4, 8}, {0, 12}, {0, 8}, {4, 4}, {4, 0}, {0, 4}, {0, 0}},
};
const uint8_t kTail64KB[5][12][2] = {
    {{128, 0}, {0, 128}, {64, 0}, {0, 64}, {32, 0}, {16, 32}, {0, 48}, {0, 32}, {16, 16}, {16, 0}, {0, 16}, {0, 0}},
    {{128, 0}, {0, 64}, {64, 0}, {0, 32}, {32, 0}, {16, 16}, {0, 24}, {0, 16}, {16, 8}, {16, 0}, {0, 8}, {0, 0}},
    {{64, 0}, {0, 64}, {32, 0}, {0, 32}, {16, 0}, {8, 16}, {0, 24}, {0, 16}, {8, 8}, {8, 0}, {0, 8}, {0, 0}},
    {{64, 0}, {0, 32}, {32, 0}, {0, 16}, {16, 0}, {8, 8}, {0, 12}, {0, 8}, {8, 4}, {8, 0}, {0, 4}, {0, 0}},
    {{32, 0}, {0, 32}, {16, 0}, {0, 16}, {8, 0}, {4, 8}, {0, 12}, {0, 8}, {4, 4}, {4, 0}, {0, 4}, {0, 0}},
};

uint32_t ShiftCeil(uint32_t v, uint32_t s) { return (uint32_t)(((uint64_t)v + (1ull << s) - 1) >> s); }
uint32_t AlignUp(uint32_t v, uint32_t a) { return (v + a - 1) / a * a; }

// One mip level of one slice of a surface.
struct Layout {
    uint64_t base = 0;  // address of the level
    Family family = Family::kLinear;
    uint32_t bpe = 4;
    int lbw = 0, lbh = 0;  // log2 block width/height (elements)
    uint32_t block_size = 0;
    uint32_t pitch = 0;  // padded width, elements
    uint32_t tail_x = 0, tail_y = 0;
    const Swizzle* swizzle = nullptr;

    uint64_t Offset(uint32_t x, uint32_t y) const {
        if (family == Family::kLinear) return ((uint64_t)y * pitch + x) * bpe;
        x += tail_x;
        y += tail_y;
        const uint32_t bx = x >> lbw, by = y >> lbh;
        const uint64_t block = (uint64_t)by * (pitch >> lbw) + bx;
        return block * block_size + (swizzle->x[x & ((1u << lbw) - 1)] ^ swizzle->y[y & ((1u << lbh) - 1)]);
    }
};

// Places level `level` of slice `slice` of a width0 x height0 (elements) surface with `levels` mips.
bool MakeLayout(uint64_t base, Family family, uint32_t bpe, uint32_t width0, uint32_t height0, uint32_t levels,
                uint32_t level, uint32_t slice, Layout* out) {
    if (bpe == 0 || (bpe & (bpe - 1)) || bpe > 16 || width0 == 0 || height0 == 0) return false;
    levels = std::clamp(levels, 1u, 16u);
    level = std::min(level, levels - 1);
    Layout r;
    r.family = family;
    r.bpe = bpe;
    const int lb = Log2(bpe);
    if (family == Family::kLinear) {
        // Smaller levels come first; level 0 is last. Rows are padded to 256 bytes.
        uint64_t offset = 0, level_offset = 0;
        for (int l = (int)levels - 1; l >= 0; l--) {
            const uint32_t w = std::max(ShiftCeil(width0, (uint32_t)l), 1u), h = std::max(ShiftCeil(height0, (uint32_t)l), 1u);
            const uint32_t pitch = AlignUp(w, 256 / bpe);
            if ((uint32_t)l == level) {
                level_offset = offset;
                r.pitch = pitch;
            }
            offset += (uint64_t)pitch * h * bpe;
        }
        const uint64_t slice_size = (offset + 255) & ~255ull;
        r.base = base + slice_size * slice + level_offset;
        *out = r;
        return true;
    }
    const uint8_t* dims;
    switch (family) {
        case Family::k256B: dims = kBlock256B[lb]; r.block_size = 256; break;
        case Family::k4KB: dims = kBlock4KB[lb]; r.block_size = 4096; break;
        default: dims = kBlock64KB[lb]; r.block_size = 65536; break;
    }
    if (family == Family::kDepth64KB && bpe > 8) return false;
    r.swizzle = SwizzleFor(family, bpe);
    r.lbw = dims[0];
    r.lbh = dims[1];
    const uint32_t bw = 1u << r.lbw, bh = 1u << r.lbh;
    // The mip tail: the first level small enough (and with few enough levels after it) shares one block.
    uint32_t first_tail = levels;
    const bool has_tail = family != Family::k256B;
    uint32_t tail_w = bw / 2, tail_h = bh, tail_max = family == Family::k4KB ? 8 : 12;
    if (family == Family::kDepth64KB && bpe < 4) {
        tail_w = 64;
        tail_h = 128;
    }
    if (has_tail && levels > 1) {
        for (uint32_t l = 0; l < levels; l++) {
            if (ShiftCeil(width0, l) <= tail_w && ShiftCeil(height0, l) <= tail_h && levels - l <= tail_max) {
                first_tail = l;
                break;
            }
        }
    }
    uint64_t sizes[16] = {};
    uint64_t slice_size = first_tail < levels ? r.block_size : 0;
    for (uint32_t l = 0; l < first_tail; l++) {
        const uint64_t pw = AlignUp(std::max(ShiftCeil(width0, l), 1u), bw);
        const uint64_t ph = AlignUp(std::max(ShiftCeil(height0, l), 1u), bh);
        sizes[l] = pw * ph * bpe;
        slice_size += sizes[l];
    }
    uint64_t level_offset = 0;
    if (level >= first_tail) {
        const uint32_t i = level - first_tail;
        const uint8_t* t = family == Family::k4KB ? kTail4KB[lb][std::min(i, 7u)] : kTail64KB[lb][std::min(i, 11u)];
        r.tail_x = t[0];
        r.tail_y = t[1];
        r.pitch = bw;
    } else {
        level_offset = first_tail < levels ? r.block_size : 0;
        for (uint32_t l = first_tail; l-- > level + 1;) level_offset += sizes[l];
        r.pitch = AlignUp(std::max(ShiftCeil(width0, level), 1u), bw);
    }
    r.base = base + slice_size * slice + level_offset;
    *out = r;
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// Texel formats (gfx10 unified image format numbers).

bool IsBc(uint32_t fmt) { return fmt >= 169 && fmt <= 182; }
bool IsSrgb(uint32_t fmt) {
    return fmt == 128 || fmt == 129 || fmt == 130 || fmt == 170 || fmt == 172 || fmt == 174 || fmt == 182;
}
uint32_t TexelBytes(uint32_t fmt) {
    if (fmt >= 1 && fmt <= 77) return (uint32_t)Rdna::FormatBytes(fmt);
    switch (fmt) {
        case 128: return 1;
        case 129: case 133: case 134: case 135: case 136: return 2;
        case 130: case 131: case 132: return 4;
        case 169: case 170: case 175: case 176: return 8;
        case 171: case 172: case 173: case 174: case 177: case 178: case 179: case 180: case 181: case 182: return 16;
        default: return 0;
    }
}

float SrgbToLinear(float c) {
    return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}
float LinearToSrgb(float c) {
    c = std::clamp(c, 0.0f, 1.0f);
    return c <= 0.0031308f ? c * 12.92f : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
}
const float* SrgbTable() {
    static const float* table = [] {
        static float t[256];
        for (int i = 0; i < 256; i++) t[i] = SrgbToLinear((float)i / 255.0f);
        return t;
    }();
    return table;
}

// Unpacks small packed unorm formats: component bit widths from the low bits.
void UnpackUnorm(uint32_t v, const int* bits, int n, uint32_t out[4]) {
    int shift = 0;
    for (int c = 0; c < 4; c++) {
        if (c >= n) { out[c] = c == 3 ? U(1.0f) : 0; continue; }
        const uint32_t m = (1u << bits[c]) - 1;
        out[c] = U((float)((v >> shift) & m) / (float)m);
        shift += bits[c];
    }
}

// Decoded block-compressed blocks, by address: textures are sampled texel by texel. Each shading thread has its
// own; a new draw or dispatch (which may sample memory written since) starts a new generation.
struct BcCacheEntry { uint64_t addr = ~0ull; uint32_t fmt = 0, generation = 0; float texel[16][4]; };
thread_local BcCacheEntry g_bc_cache[256];
uint32_t g_bc_generation = 1;

const BcCacheEntry& DecodeBcBlock(uint32_t fmt, uint64_t addr, const uint8_t* p) {
    BcCacheEntry& e = g_bc_cache[(addr >> 4) & 255];
    if (e.addr == addr && e.fmt == fmt && e.generation == g_bc_generation) return e;
    e.addr = addr;
    e.fmt = fmt;
    e.generation = g_bc_generation;
    uint8_t rgba[16 * 4];
    float f[16 * 4];
    switch (fmt) {
        case 169: case 170: bcdec_bc1(p, rgba, 16); break;
        case 171: case 172: bcdec_bc2(p, rgba, 16); break;
        case 173: case 174: bcdec_bc3(p, rgba, 16); break;
        case 181: case 182: bcdec_bc7(p, rgba, 16); break;
        case 175: case 176:
            bcdec_bc4_float(p, f, 4, fmt == 176);
            for (int i = 0; i < 16; i++) { e.texel[i][0] = f[i]; e.texel[i][1] = 0; e.texel[i][2] = 0; e.texel[i][3] = 1; }
            return e;
        case 177: case 178:
            bcdec_bc5_float(p, f, 8, fmt == 178);
            for (int i = 0; i < 16; i++) { e.texel[i][0] = f[2 * i]; e.texel[i][1] = f[2 * i + 1]; e.texel[i][2] = 0; e.texel[i][3] = 1; }
            return e;
        case 179: case 180:
            bcdec_bc6h_float(p, f, 12, fmt == 180);
            for (int i = 0; i < 16; i++) {
                e.texel[i][0] = f[3 * i]; e.texel[i][1] = f[3 * i + 1]; e.texel[i][2] = f[3 * i + 2]; e.texel[i][3] = 1;
            }
            return e;
        default: memset(rgba, 0, sizeof(rgba)); break;
    }
    const bool srgb = IsSrgb(fmt);
    const float* table = SrgbTable();
    for (int i = 0; i < 16; i++) {
        for (int c = 0; c < 4; c++) {
            const uint8_t v = rgba[4 * i + c];
            e.texel[i][c] = srgb && c < 3 ? table[v] : (float)v / 255.0f;
        }
    }
    return e;
}

// Decodes one texel (out: float bits, or integer bits for integer formats).
void DecodeTexel(uint32_t fmt, const uint8_t* p, uint32_t out[4]) {
    if (fmt >= 1 && fmt <= 77) {
        bool integer;
        if (!Rdna::DecodeFormat(fmt, p, out, &integer)) out[0] = out[1] = out[2] = out[3] = 0;
        return;
    }
    const float* table = SrgbTable();
    switch (fmt) {
        case 128: out[0] = U(table[p[0]]); out[1] = out[2] = 0; out[3] = U(1.0f); return;
        case 129: out[0] = U(table[p[0]]); out[1] = U(table[p[1]]); out[2] = 0; out[3] = U(1.0f); return;
        case 130:
            for (int c = 0; c < 3; c++) out[c] = U(table[p[c]]);
            out[3] = U((float)p[3] / 255.0f);
            return;
        case 133: { static const int b[] = {5, 6, 5}; UnpackUnorm(p[0] | p[1] << 8, b, 3, out); return; }
        case 134: { static const int b[] = {1, 5, 5, 5}; UnpackUnorm(p[0] | p[1] << 8, b, 4, out); return; }
        case 135: { static const int b[] = {5, 5, 5, 1}; UnpackUnorm(p[0] | p[1] << 8, b, 4, out); return; }
        case 136: { static const int b[] = {4, 4, 4, 4}; UnpackUnorm(p[0] | p[1] << 8, b, 4, out); return; }
        case 132: {  // 9_9_9_5 shared exponent
            uint32_t v;
            memcpy(&v, p, 4);
            const float scale = std::ldexp(1.0f, (int)(v >> 27) - 15 - 9);
            for (int c = 0; c < 3; c++) out[c] = U((float)((v >> (9 * c)) & 0x1ff) * scale);
            out[3] = U(1.0f);
            return;
        }
        default: out[0] = out[1] = out[2] = 0; out[3] = U(1.0f); return;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Image and sampler descriptors.

struct Texture {
    bool valid = false;
    uint64_t base = 0;
    uint32_t fmt = 0;
    uint32_t width = 1, height = 1, depth = 1;  // texels, level 0
    uint32_t base_level = 0, last_level = 0, levels = 1;
    uint32_t sw = 0, type = 9, dst_sel = 0xfac, base_array = 0, last_array = 0;
    uint32_t bpe = 4;
    bool bc = false;
    Family family = Family::kLinear;
};

Texture DecodeT(const uint32_t* t) {
    Texture x;
    x.base = ((uint64_t)t[0] << 8) | ((uint64_t)(t[1] & 0xff) << 40);
    x.fmt = (t[1] >> 20) & 0x1ff;
    x.width = (((t[1] >> 30) & 3) | ((t[2] & 0xfff) << 2)) + 1;
    x.height = ((t[2] >> 14) & 0x3fff) + 1;
    x.dst_sel = t[3] & 0xfff;
    x.base_level = (t[3] >> 12) & 0xf;
    x.last_level = (t[3] >> 16) & 0xf;
    x.sw = (t[3] >> 20) & 0x1f;
    x.type = (t[3] >> 28) & 0xf;
    x.depth = (t[4] & 0x1fff) + 1;
    x.base_array = (t[4] >> 16) & 0x1fff;
    x.last_array = x.depth - 1;
    const uint32_t max_mip = (t[5] >> 4) & 0xf;
    x.levels = std::max(max_mip, x.last_level) + 1;
    x.bpe = TexelBytes(x.fmt);
    x.bc = IsBc(x.fmt);
    x.family = FamilyOf(x.sw);
    x.valid = x.base != 0 && x.bpe != 0 && x.type >= 8;
    return x;
}

struct Sampler {
    uint32_t clamp[3] = {0, 0, 0};
    bool unnormalized = false;
    float min_lod = 0, max_lod = 15, lod_bias = 0;
    uint32_t mag = 0, min = 0, mip = 0;
    uint32_t border = 0;
};

Sampler DecodeS(const uint32_t* s) {
    Sampler x;
    x.clamp[0] = s[0] & 7;
    x.clamp[1] = (s[0] >> 3) & 7;
    x.clamp[2] = (s[0] >> 6) & 7;
    x.unnormalized = (s[0] >> 15) & 1;
    x.min_lod = (float)(s[1] & 0xfff) / 256.0f;
    x.max_lod = (float)((s[1] >> 12) & 0xfff) / 256.0f;
    x.lod_bias = (float)((int32_t)(s[2] << 18) >> 18) / 256.0f;
    x.mag = (s[2] >> 20) & 3;
    x.min = (s[2] >> 22) & 3;
    x.mip = (s[2] >> 26) & 3;
    x.border = (s[3] >> 30) & 3;
    return x;
}

// Texture coordinate wrapping; returns -1 for the border.
int WrapCoord(int i, int size, uint32_t mode) {
    switch (mode) {
        case 0: { int r = i % size; return r < 0 ? r + size : r; }                      // wrap
        case 1: {                                                                      // mirror
            const int period = 2 * size;
            int r = i % period;
            if (r < 0) r += period;
            return r < size ? r : period - 1 - r;
        }
        case 2: return std::clamp(i, 0, size - 1);                                     // clamp to edge
        case 3: { const int m = i < 0 ? -i - 1 : i; return std::min(m, size - 1); }    // mirror once
        case 4: case 6: return i < 0 || i >= size ? -1 : i;                             // border
        case 5: case 7: { const int m = i < 0 ? -i - 1 : i; return m >= size ? -1 : m; }
        default: return std::clamp(i, 0, size - 1);
    }
}

// Reads texel (x, y) of a level; out is float bits (integer bits for integer formats).
void ReadTexel(const Texture& tex, const Layout& lay, uint32_t x, uint32_t y, uint32_t out[4]) {
    if (tex.bc) {
        const uint64_t addr = lay.base + lay.Offset(x >> 2, y >> 2);
        const uint8_t* p = Ptr(addr, tex.bpe);
        if (!p) { out[0] = out[1] = out[2] = out[3] = 0; return; }
        const BcCacheEntry& e = DecodeBcBlock(tex.fmt, addr, p);
        const float* t = e.texel[(y & 3) * 4 + (x & 3)];
        for (int c = 0; c < 4; c++) out[c] = U(t[c]);
        return;
    }
    const uint8_t* p = Ptr(lay.base + lay.Offset(x, y), tex.bpe);
    if (!p) { out[0] = out[1] = out[2] = out[3] = 0; return; }
    static const float* unorm8 = [] {
        static float t[256];
        for (int i = 0; i < 256; i++) t[i] = (float)((double)i / 255.0);
        return t;
    }();
    switch (tex.fmt) {  // the common 8_8_8_8 UNORM and SRGB, without the general decoder
        case 56:
            for (int c = 0; c < 4; c++) out[c] = U(unorm8[p[c]]);
            return;
        case 130: {
            const float* srgb = SrgbTable();
            for (int c = 0; c < 3; c++) out[c] = U(srgb[p[c]]);
            out[3] = U((float)p[3] / 255.0f);
            return;
        }
        default: DecodeTexel(tex.fmt, p, out); return;
    }
}

bool LevelLayout(const Texture& tex, uint32_t level, uint32_t slice, Layout* lay, uint32_t* w, uint32_t* h) {
    const uint32_t ew = tex.bc ? (tex.width + 3) / 4 : tex.width, eh = tex.bc ? (tex.height + 3) / 4 : tex.height;
    *w = std::max(tex.width >> level, 1u);
    *h = std::max(tex.height >> level, 1u);
    return MakeLayout(tex.base, tex.family, tex.bpe, ew, eh, tex.levels, level, slice, lay);
}

bool IsIntegerFormat(uint32_t fmt);

// Decoded image (and sampler) descriptors with the layouts of their levels, by descriptor contents: every lane of
// a wave, and every wave of a draw, usually uses the same few.
struct ImageSetup {
    bool used = false;
    uint32_t key[12] = {};
    Texture tex;
    Sampler smp;
    bool integer = false;
    uint32_t slice = 0, made = 0, ok = 0;  // the slice the layouts are for; bits by level
    Layout lay[16];
    uint32_t w[16] = {}, h[16] = {};
};
thread_local ImageSetup g_images[32];

ImageSetup& Image(const uint32_t* t, const uint32_t* s) {
    uint32_t key[12] = {};
    memcpy(key, t, 32);
    if (s) memcpy(key + 8, s, 16);
    uint32_t hash = 2166136261u;
    for (uint32_t k : key) hash = (hash ^ k) * 16777619u;
    ImageSetup& e = g_images[(hash ^ (hash >> 15)) & 31];
    if (!e.used || memcmp(e.key, key, sizeof(key)) != 0) {
        e.used = true;
        memcpy(e.key, key, sizeof(key));
        e.tex = DecodeT(t);
        e.smp = s ? DecodeS(s) : Sampler();
        e.integer = IsIntegerFormat(e.tex.fmt);
        e.slice = 0;
        e.made = e.ok = 0;
    }
    return e;
}

const Layout* ImageLevel(ImageSetup& e, uint32_t level, uint32_t slice, uint32_t* w, uint32_t* h) {
    if (level >= 16) return nullptr;
    if (slice != e.slice) {
        e.slice = slice;
        e.made = 0;
    }
    const uint32_t bit = 1u << level;
    if (!(e.made & bit)) {
        e.made |= bit;
        if (LevelLayout(e.tex, level, slice, &e.lay[level], &e.w[level], &e.h[level])) e.ok |= bit;
        else e.ok &= ~bit;
    }
    if (!(e.ok & bit)) return nullptr;
    *w = e.w[level];
    *h = e.h[level];
    return &e.lay[level];
}

void ApplyTexSwizzle(uint32_t dst_sel, uint32_t v[4], bool integer) {
    uint32_t in[4];
    memcpy(in, v, sizeof(in));
    for (int c = 0; c < 4; c++) {
        const uint32_t sel = (dst_sel >> (3 * c)) & 7;
        v[c] = sel >= 4 ? in[sel - 4] : sel == 1 ? (integer ? 1u : U(1.0f)) : 0u;
    }
}

bool IsIntegerFormat(uint32_t fmt) {
    if (fmt == 0 || fmt > 77) return false;
    static const uint8_t kInts[] = {5, 6, 11, 12, 18, 19, 20, 21, 27, 28, 34, 35, 41, 42, 48, 49, 54, 55, 60, 61, 62, 63,
                                    69, 70, 72, 73, 75, 76};
    for (uint8_t f : kInts)
        if (f == fmt) return true;
    return false;
}

void SampleTexture(const uint32_t* t, const uint32_t* s, const Rdna::SampleRequest& req, float out[4]) {
    ImageSetup& img = Image(t, s);
    const Texture& tex = img.tex;
    out[0] = out[1] = out[2] = out[3] = 0;
    if (!tex.valid) return;
    const Sampler& smp = img.smp;
    const bool array = req.dim == 4 || req.dim == 5 || req.dim == 7;
    uint32_t slice = 0;
    if (array) {
        const int layer_coord = req.dim == 4 ? 1 : 2;
        slice = (uint32_t)std::clamp((int)std::lround(req.coord[layer_coord]), 0, (int)tex.last_array) + tex.base_array;
    }
    // Level of detail.
    float lod;
    const float w0 = (float)tex.width, h0 = (float)tex.height;
    if (req.has_lod) {
        lod = req.lod;
    } else if (req.level_zero) {
        lod = 0;
    } else {
        const float sx = smp.unnormalized ? 1.0f : w0, sy = smp.unnormalized ? 1.0f : h0;
        const float dx0 = req.dx[0] * sx, dx1 = req.dx[1] * sy, dy0 = req.dy[0] * sx, dy1 = req.dy[1] * sy;
        const float rho2 = std::max(dx0 * dx0 + dx1 * dx1, dy0 * dy0 + dy1 * dy1);
        lod = rho2 > 0 ? 0.5f * std::log2(rho2) : -16.0f;
        if (req.has_bias) lod += req.lod;
    }
    lod += smp.lod_bias;
    lod = std::clamp(lod, smp.min_lod, smp.max_lod);
    if (req.has_clamp) lod = std::min(lod, req.clamp);
    const bool magnify = lod <= 0;
    const uint32_t filter = magnify ? smp.mag : smp.min;
    uint32_t level = tex.base_level;
    if (smp.mip != 0 && !magnify) level = tex.base_level + (uint32_t)std::max(0, (int)std::lround(lod));
    level = std::min(level, std::max(tex.last_level, tex.base_level));

    uint32_t w, h;
    const Layout* level_layout = ImageLevel(img, level, slice, &w, &h);
    if (!level_layout) return;
    const Layout& lay = *level_layout;
    const bool integer = img.integer;
    float u = req.coord[0], v = req.dim == 0 || req.dim == 4 ? 0.0f : req.coord[1];
    if (!smp.unnormalized) {
        u *= (float)w;
        v *= (float)h;
    }
    u += (float)req.offset[0];
    v += (float)req.offset[1];
    auto fetch = [&](int ix, int iy, uint32_t texel[4]) {
        // Every wrap mode leaves coordinates inside the image as they are.
        const int x = (uint32_t)ix < w ? ix : WrapCoord(ix, (int)w, smp.clamp[0]);
        const int y = (uint32_t)iy < h ? iy : WrapCoord(iy, (int)h, smp.clamp[1]);
        if (x < 0 || y < 0) {
            const float b = smp.border == 2 ? 1.0f : 0.0f, a = smp.border == 0 ? 0.0f : 1.0f;
            texel[0] = texel[1] = texel[2] = U(b);
            texel[3] = U(a);
            return;
        }
        ReadTexel(tex, lay, (uint32_t)x, (uint32_t)y, texel);
    };
    if (req.gather) {
        const int x0 = (int)std::floor(u - 0.5f), y0 = (int)std::floor(v - 0.5f);
        static const int kOrder[4][2] = {{0, 1}, {1, 1}, {1, 0}, {0, 0}};
        for (int i = 0; i < 4; i++) {
            uint32_t texel[4];
            fetch(x0 + kOrder[i][0], y0 + kOrder[i][1], texel);
            ApplyTexSwizzle(tex.dst_sel, texel, integer);
            out[i] = F(texel[req.gather_comp & 3]);
        }
        return;
    }
    uint32_t r[4];
    if (filter == 0 || integer) {  // point
        fetch((int)std::floor(u), (int)std::floor(v), r);
        ApplyTexSwizzle(tex.dst_sel, r, integer);
        for (int c = 0; c < 4; c++) out[c] = F(r[c]);
        return;
    }
    const float fu = u - 0.5f, fv = v - 0.5f;
    const int x0 = (int)std::floor(fu), y0 = (int)std::floor(fv);
    const float ax = fu - (float)x0, ay = fv - (float)y0;
    uint32_t t00[4], t10[4], t01[4], t11[4];
    fetch(x0, y0, t00);
    fetch(x0 + 1, y0, t10);
    fetch(x0, y0 + 1, t01);
    fetch(x0 + 1, y0 + 1, t11);
    for (int c = 0; c < 4; c++) {
        const float top = F(t00[c]) + (F(t10[c]) - F(t00[c])) * ax;
        const float bottom = F(t01[c]) + (F(t11[c]) - F(t01[c])) * ax;
        r[c] = U(top + (bottom - top) * ay);
    }
    ApplyTexSwizzle(tex.dst_sel, r, false);
    for (int c = 0; c < 4; c++) out[c] = F(r[c]);
}

// ---------------------------------------------------------------------------------------------------------------
// Color buffers.

struct ColorFormat {
    int comps = 0;
    int bits[4] = {0, 0, 0, 0};
    uint32_t bpe = 0;
};

bool LookupColorFormat(uint32_t format, ColorFormat* f) {
    auto set = [&](int n, int b0, int b1, int b2, int b3, uint32_t bpe) {
        f->comps = n; f->bits[0] = b0; f->bits[1] = b1; f->bits[2] = b2; f->bits[3] = b3; f->bpe = bpe;
        return true;
    };
    switch (format) {
        case 1: return set(1, 8, 0, 0, 0, 1);
        case 2: return set(1, 16, 0, 0, 0, 2);
        case 3: return set(2, 8, 8, 0, 0, 2);
        case 4: return set(1, 32, 0, 0, 0, 4);
        case 5: return set(2, 16, 16, 0, 0, 4);
        case 6: return set(3, 11, 11, 10, 0, 4);
        case 7: return set(3, 10, 11, 11, 0, 4);
        case 8: return set(4, 2, 10, 10, 10, 4);
        case 9: return set(4, 10, 10, 10, 2, 4);
        case 10: return set(4, 8, 8, 8, 8, 4);
        case 11: return set(2, 32, 32, 0, 0, 8);
        case 12: return set(4, 16, 16, 16, 16, 8);
        case 14: return set(4, 32, 32, 32, 32, 16);
        case 16: return set(3, 5, 6, 5, 0, 2);
        case 17: return set(4, 5, 5, 5, 1, 2);
        case 18: return set(4, 1, 5, 5, 5, 2);
        case 19: return set(4, 4, 4, 4, 4, 2);
        default: return false;
    }
}

// Which RGBA channel each memory component holds, by component count and COMP_SWAP.
int SwapChannel(int comps, uint32_t swap, int component) {
    static const int k4[4][4] = {{0, 1, 2, 3}, {2, 1, 0, 3}, {3, 2, 1, 0}, {3, 0, 1, 2}};
    static const int k3[4][3] = {{0, 1, 2}, {0, 1, 3}, {2, 1, 0}, {3, 1, 0}};
    static const int k2[4][2] = {{0, 1}, {0, 3}, {1, 0}, {3, 0}};
    static const int k1[4] = {0, 1, 2, 3};
    switch (comps) {
        case 4: return k4[swap & 3][component];
        case 3: return k3[swap & 3][component];
        case 2: return k2[swap & 3][component];
        default: return k1[swap & 3];
    }
}

float SmallFloatToFloat(uint32_t v, int bits) {
    const int mbits = bits == 11 ? 6 : 5;
    const uint32_t e = v >> mbits, m = v & ((1u << mbits) - 1);
    if (e == 0) return std::ldexp((float)m, -14 - mbits);
    if (e == 31) return m ? NAN : INFINITY;
    return std::ldexp(1.0f + (float)m / (float)(1 << mbits), (int)e - 15);
}
uint32_t FloatToSmallFloat(float f, int bits) {
    const int mbits = bits == 11 ? 6 : 5;
    if (!(f > 0)) return 0;
    if (std::isinf(f)) return 31u << mbits;
    int e;
    const float m = std::frexp(f, &e);  // f = m * 2^e, m in [0.5, 1)
    int be = e - 1 + 15;
    if (be >= 31) return (30u << mbits) | ((1u << mbits) - 1);
    if (be <= 0) {
        const uint32_t mant = (uint32_t)std::lround(std::ldexp(f, 14 + mbits));
        return std::min(mant, (1u << mbits) - 1);
    }
    uint32_t mant = (uint32_t)std::lround((m * 2.0f - 1.0f) * (float)(1 << mbits));
    if (mant >> mbits) { mant = 0; be++; }
    return ((uint32_t)be << mbits) | mant;
}

struct Target {
    bool enabled = false;
    uint64_t base = 0;
    Layout layout;
    ColorFormat cf;
    uint32_t number = 0, swap = 0;
    uint32_t width = 0, height = 0;
    uint32_t mask = 0xf;   // RGBA write mask
    uint32_t blend = 0;    // CB_BLEND_CONTROL
    uint32_t export_format = 0;
    bool fast_rgba8 = false;  // 8_8_8_8 UNORM with every channel written: packed directly
    int swap_of[4] = {0, 1, 2, 3};  // RGBA channel held by each memory component
};

void ReadColor(const Target& t, uint32_t x, uint32_t y, float rgba[4]) {
    rgba[0] = rgba[1] = rgba[2] = 0;
    rgba[3] = 1;
    const uint8_t* p = Ptr(t.layout.base + t.layout.Offset(x, y), t.cf.bpe);
    if (!p) return;
    uint64_t raw[2] = {0, 0};
    memcpy(raw, p, t.cf.bpe);
    int shift = 0;
    for (int c = 0; c < t.cf.comps; c++) {
        const int bits = t.cf.bits[c];
        const uint64_t word = raw[shift / 64] >> (shift % 64);
        const uint32_t v = bits == 32 ? (uint32_t)word : (uint32_t)(word & ((1ull << bits) - 1));
        shift += bits;
        float f;
        switch (t.number) {
            case 1: f = std::max(-1.0f, (float)((int32_t)(v << (32 - bits)) >> (32 - bits)) / (float)((1u << (bits - 1)) - 1)); break;
            case 4: f = F(v); break;
            case 5: f = F(v); break;
            case 6: f = c < 3 || t.cf.comps < 4 ? SrgbToLinear((float)v / (float)((1ull << bits) - 1)) : (float)v / (float)((1ull << bits) - 1); break;
            case 7: f = bits == 32 ? F(v) : bits == 16 ? Rdna::HalfToFloat((uint16_t)v) : SmallFloatToFloat(v, bits); break;
            default: f = bits == 32 ? (float)v / 4294967295.0f : (float)v / (float)((1ull << bits) - 1); break;
        }
        rgba[SwapChannel(t.cf.comps, t.swap, c)] = f;
    }
}

void WriteColor(const Target& t, uint32_t x, uint32_t y, const float rgba[4], const uint32_t* ints) {
    uint8_t* p = Ptr(t.layout.base + t.layout.Offset(x, y), t.cf.bpe);
    if (!p) return;
    uint64_t raw[2] = {0, 0};
    uint64_t keep[2] = {0, 0};  // bits of masked-off channels
    memcpy(raw, p, t.cf.bpe);
    int shift = 0;
    uint64_t out[2] = {0, 0};
    for (int c = 0; c < t.cf.comps; c++) {
        const int bits = t.cf.bits[c];
        const int ch = SwapChannel(t.cf.comps, t.swap, c);
        const uint64_t m = bits == 32 ? 0xffffffffull : (1ull << bits) - 1;
        uint64_t v;
        const float f = rgba[ch];
        switch (t.number) {
            case 1: v = (uint64_t)(int64_t)std::lround(std::clamp(f, -1.0f, 1.0f) * (float)((1u << (bits - 1)) - 1)); break;
            case 4: case 5: v = ints ? ints[ch] : (uint64_t)(int64_t)f; break;
            case 6:
                v = (uint64_t)std::lround((c < 3 || t.cf.comps < 4 ? LinearToSrgb(f) : std::clamp(f, 0.0f, 1.0f)) * (float)m);
                break;
            case 7: v = bits == 32 ? U(f) : bits == 16 ? Rdna::FloatToHalf(f) : FloatToSmallFloat(f, bits); break;
            default: v = (uint64_t)std::lround(std::clamp(std::isnan(f) ? 0.0f : f, 0.0f, 1.0f) * (float)m); break;
        }
        v &= m;
        if (((t.mask >> ch) & 1) == 0) keep[shift / 64] |= m << (shift % 64);
        out[shift / 64] |= v << (shift % 64);
        shift += bits;
    }
    for (int i = 0; i < 2; i++) raw[i] = (raw[i] & keep[i]) | (out[i] & ~keep[i]);
    memcpy(p, raw, t.cf.bpe);
}

float BlendFactor(uint32_t f, int c, const float src[4], const float dst[4], const float k[4]) {
    switch (f) {
        case 0: return 0;
        case 1: return 1;
        case 2: return src[c];
        case 3: return 1 - src[c];
        case 4: return src[3];
        case 5: return 1 - src[3];
        case 6: return dst[3];
        case 7: return 1 - dst[3];
        case 8: return dst[c];
        case 9: return 1 - dst[c];
        case 10: return c == 3 ? 1.0f : std::min(src[3], 1 - dst[3]);
        case 13: return k[c];
        case 14: return 1 - k[c];
        case 19: return k[3];
        case 20: return 1 - k[3];
        default: return 1;
    }
}
float BlendCombine(uint32_t fn, float s, float d) {
    switch (fn) {
        case 1: return s - d;
        case 2: return std::min(s, d);
        case 3: return std::max(s, d);
        case 4: return d - s;
        default: return s + d;
    }
}

// Per-draw statistics for KYTY_RASTER_DEBUG.
struct DrawStats {
    uint64_t pixels = 0, waves = 0, triangles = 0;
    double sum[4] = {0, 0, 0, 0};
} g_stats;

// Totals since the last flip, for the per-frame log line.
struct FrameStats {
    uint64_t draws = 0, empty_draws = 0, dispatches = 0, pixels = 0;
} g_frame;
uint64_t g_flips = 0;  // frames presented so far
// KYTY_RASTER_FRAME=n: every draw of the frame after flip n is summarized.
bool DebugFrame() {
    static const uint64_t frame = getenv("KYTY_RASTER_FRAME") ? strtoull(getenv("KYTY_RASTER_FRAME"), nullptr, 10) : ~0ull;
    return g_flips == frame;
}

// Display buffers remember how they were last rendered, for presentation.
struct DisplayInfo { Layout layout; ColorFormat cf; uint32_t number = 0, swap = 0; };
std::unordered_map<uint64_t, DisplayInfo> g_targets_by_base;

// ---------------------------------------------------------------------------------------------------------------
// Shader environments.

uint64_t ShaderAddr(const uint32_t* sh, uint32_t lo) {
    return ((uint64_t)sh[lo] << 8) | ((uint64_t)(sh[lo + 1] & 0xff) << 40);
}

class BaseEnv : public Rdna::Env {
public:
    const uint8_t* Read(uint64_t addr, uint64_t size) override { return Ptr(addr, size); }
    uint8_t* Write(uint64_t addr, uint64_t size) override { return Ptr(addr, size); }
    void Sample(const uint32_t* t, const uint32_t* s, const Rdna::SampleRequest& req, float out[4]) override {
        SampleTexture(t, s, req, out);
    }
    void ImageLoad(const uint32_t* t, int dim, const int coord[4], int mip, uint32_t out[4]) override {
        ImageSetup& img = Image(t, nullptr);
        const Texture& tex = img.tex;
        out[0] = out[1] = out[2] = out[3] = 0;
        if (!tex.valid) return;
        const uint32_t level = std::min((uint32_t)std::max(mip, 0) + tex.base_level, tex.levels - 1);
        const uint32_t slice = (dim == 5 || dim == 7) ? (uint32_t)std::max(coord[2], 0) + tex.base_array
                             : dim == 4 ? (uint32_t)std::max(coord[1], 0) + tex.base_array : 0;
        uint32_t w, h;
        const Layout* lay = ImageLevel(img, level, slice, &w, &h);
        if (!lay) return;
        const int y = dim == 0 || dim == 4 ? 0 : coord[1];
        if (coord[0] < 0 || y < 0 || (uint32_t)coord[0] >= w || (uint32_t)y >= h) return;
        ReadTexel(tex, *lay, (uint32_t)coord[0], (uint32_t)y, out);
        ApplyTexSwizzle(tex.dst_sel, out, img.integer);
    }
    void ImageStore(const uint32_t* t, int dim, const int coord[4], int mip, const uint32_t in[4], uint32_t dmask) override {
        (void)dmask;
        const Texture tex = DecodeT(t);
        if (!tex.valid || tex.bc || tex.fmt > 77) return;
        const uint32_t level = std::min((uint32_t)std::max(mip, 0) + tex.base_level, tex.levels - 1);
        const uint32_t slice = (dim == 5 || dim == 7) ? (uint32_t)std::max(coord[2], 0) + tex.base_array : 0;
        Layout lay;
        uint32_t w, h;
        if (!LevelLayout(tex, level, slice, &lay, &w, &h)) return;
        const int y = dim == 0 || dim == 4 ? 0 : coord[1];
        if (coord[0] < 0 || y < 0 || (uint32_t)coord[0] >= w || (uint32_t)y >= h) return;
        if (uint8_t* p = Ptr(lay.base + lay.Offset((uint32_t)coord[0], (uint32_t)y), tex.bpe))
            Rdna::EncodeFormat(tex.fmt, in, p);
    }
    void ResInfo(const uint32_t* t, int mip, uint32_t out[4]) override {
        const Texture tex = DecodeT(t);
        const uint32_t level = (uint32_t)std::max(mip, 0) + tex.base_level;
        out[0] = std::max(tex.width >> level, 1u);
        out[1] = std::max(tex.height >> level, 1u);
        out[2] = tex.depth;
        out[3] = tex.last_level - tex.base_level + 1;
    }
    void Export(const Rdna::Wave&, int, uint32_t, bool, bool, uint64_t, const uint32_t* const[4]) override {}
    void Attribute(int, int, float out[3]) override { out[0] = out[1] = out[2] = 0; }
};

// Vertex shader outputs of one batch.
struct VertexOut {
    float pos[4];
    float param[32][4];
};

class VsEnv : public BaseEnv {
public:
    VertexOut* out = nullptr;
    uint32_t prim[kLanes] = {};
    uint64_t prim_lanes = 0;
    uint32_t params_written = 0;
    void Export(const Rdna::Wave&, int target, uint32_t en, bool compr, bool, uint64_t lanes,
                const uint32_t* const data[4]) override {
        if (target == 20) {  // primitive connectivity
            for (int l = 0; l < kLanes; l++)
                if (((lanes >> l) & 1) && data[0]) prim[l] = data[0][l];
            prim_lanes |= lanes;
            return;
        }
        float* dst_base;
        if (target == 12) dst_base = nullptr;  // pos0
        else if (target >= 32 && target < 64) params_written |= 1u << (target - 32);
        else return;
        for (int l = 0; l < kLanes; l++) {
            if (!((lanes >> l) & 1)) continue;
            dst_base = target == 12 ? out[l].pos : out[l].param[target - 32];
            for (int c = 0; c < 4; c++) {
                if (compr) {
                    const uint32_t* row = data[c / 2];
                    dst_base[c] = row ? Rdna::HalfToFloat((uint16_t)(row[l] >> (16 * (c & 1)))) : 0.0f;
                } else if ((en >> c) & 1) {
                    dst_base[c] = data[c] ? F(data[c][l]) : 0.0f;
                }
            }
        }
    }
};

// A triangle ready for pixel shading.
struct ScreenTri {
    float x[3], y[3], z[3], invw[3];
    const VertexOut* v[3];
    bool front = true;
};

class PsEnv : public BaseEnv {
public:
    const uint32_t* cx = nullptr;
    const ScreenTri* tri = nullptr;
    const Target* targets = nullptr;
    DrawStats* stats = nullptr;
    int tile_x = 0, tile_y = 0;
    float blend_const[4] = {0, 0, 0, 0};
    bool rop_copy = true;

    void Attribute(int attr, int chan, float out[3]) override {
        const uint32_t cntl = cx[0x191 + (attr & 31)];
        const uint32_t offset = cntl & 0x3f;
        if (offset & 0x20) {
            const uint32_t def = (cntl >> 8) & 3;
            const float v = chan < 3 ? (def >= 2 ? 1.0f : 0.0f) : ((def & 1) ? 1.0f : 0.0f);
            out[0] = out[1] = out[2] = v;
            return;
        }
        const bool flat = (cntl >> 10) & 1;
        for (int i = 0; i < 3; i++) out[i] = tri->v[flat ? 0 : i]->param[offset & 31][chan & 3];
    }

    void Export(const Rdna::Wave&, int target, uint32_t en, bool compr, bool, uint64_t lanes,
                const uint32_t* const data[4]) override {
        if (target > 7) return;
        const Target& t = targets[target];
        if (!t.enabled || !lanes) return;
        const uint32_t fmt = t.export_format;
        const bool blend = (t.blend >> 30) & 1;
        const bool integer = t.number == 4 || t.number == 5;
        for (int l = 0; l < kLanes; l++) {
            if (!((lanes >> l) & 1)) continue;
            const int q = l >> 2, i = l & 3;
            const int px = tile_x + (q & 3) * 2 + (i & 1), py = tile_y + (q >> 2) * 2 + (i >> 1);
            if ((uint32_t)px >= t.width || (uint32_t)py >= t.height) continue;
            float src[4] = {0, 0, 0, 1};
            uint32_t ints[4] = {0, 0, 0, 1};
            auto half = [&](int c) {
                const uint32_t* row = data[c / 2];
                return row ? (uint16_t)(row[l] >> (16 * (c & 1))) : (uint16_t)0;
            };
            switch (fmt) {
                case 1: src[0] = data[0] ? F(data[0][l]) : 0; ints[0] = data[0] ? data[0][l] : 0; break;
                case 2:
                    for (int c = 0; c < 2; c++) { src[c] = data[c] ? F(data[c][l]) : 0; ints[c] = data[c] ? data[c][l] : 0; }
                    break;
                case 3:
                    src[0] = data[0] ? F(data[0][l]) : 0;
                    src[3] = data[1] ? F(data[1][l]) : (data[3] ? F(data[3][l]) : 1.0f);
                    break;
                case 4:
                    for (int c = 0; c < 4; c++) src[c] = Rdna::HalfToFloat(half(c));
                    break;
                case 5:
                    for (int c = 0; c < 4; c++) src[c] = (float)half(c) / 65535.0f;
                    break;
                case 6:
                    for (int c = 0; c < 4; c++) src[c] = std::max(-1.0f, (float)(int16_t)half(c) / 32767.0f);
                    break;
                case 7: case 8:
                    for (int c = 0; c < 4; c++) {
                        ints[c] = fmt == 7 ? half(c) : (uint32_t)(int32_t)(int16_t)half(c);
                        src[c] = (float)ints[c];
                    }
                    break;
                case 9:
                    for (int c = 0; c < 4; c++) {
                        if (compr) { src[c] = Rdna::HalfToFloat(half(c)); continue; }
                        if ((en >> c) & 1 && data[c]) { src[c] = F(data[c][l]); ints[c] = data[c][l]; }
                    }
                    break;
                default: continue;  // SPI_SHADER_ZERO: nothing written
            }
            float color[4];
            uint8_t* fast_ptr = nullptr;
            uint32_t fast_dst = 0;
            if (t.fast_rgba8) {
                fast_ptr = Ptr(t.layout.base + t.layout.Offset((uint32_t)px, (uint32_t)py), 4);
                if (!fast_ptr) continue;
                memcpy(&fast_dst, fast_ptr, 4);
            }
            if (blend && !integer) {
                float dst[4];
                if (fast_ptr) {
                    for (int c = 0; c < 4; c++) dst[t.swap_of[c]] = (float)((fast_dst >> (8 * c)) & 0xff) * (1.0f / 255.0f);
                } else {
                    ReadColor(t, (uint32_t)px, (uint32_t)py, dst);
                }
                const uint32_t b = t.blend;
                const uint32_t csrc = b & 0x1f, cfn = (b >> 5) & 7, cdst = (b >> 8) & 0x1f;
                const bool separate = (b >> 29) & 1;
                const uint32_t asrc = separate ? (b >> 16) & 0x1f : csrc, afn = separate ? (b >> 21) & 7 : cfn;
                const uint32_t adst = separate ? (b >> 24) & 0x1f : cdst;
                for (int c = 0; c < 4; c++) {
                    const uint32_t fs = c == 3 ? asrc : csrc, fd = c == 3 ? adst : cdst, fn = c == 3 ? afn : cfn;
                    float s = src[c], d = dst[c];
                    if (t.number == 0 || t.number == 6) {
                        s = std::clamp(s, 0.0f, 1.0f);
                    }
                    if (fn == 2 || fn == 3) color[c] = BlendCombine(fn, s, d);
                    else color[c] = BlendCombine(fn, s * BlendFactor(fs, c, src, dst, blend_const),
                                                 d * BlendFactor(fd, c, src, dst, blend_const));
                }
            } else {
                memcpy(color, src, sizeof(color));
            }
            if (fast_ptr) {
                uint32_t v = 0;
                for (int c = 0; c < 4; c++) {
                    const float f = color[t.swap_of[c]];
                    const float cl = f > 0.0f ? (f < 1.0f ? f : 1.0f) : 0.0f;  // NaN -> 0
                    v |= (uint32_t)(cl * 255.0f + 0.5f) << (8 * c);
                }
                memcpy(fast_ptr, &v, 4);
            } else {
                WriteColor(t, (uint32_t)px, (uint32_t)py, color, integer ? ints : nullptr);
            }
            stats->pixels++;
            for (int c = 0; c < 4; c++) stats->sum[c] += std::isfinite(color[c]) ? color[c] : 0;
        }
    }
};

// ---------------------------------------------------------------------------------------------------------------
// Draws.

Rdna::Wave* g_wave = nullptr;
Rdna::Wave& Wave() {
    if (!g_wave) g_wave = new Rdna::Wave();
    return *g_wave;
}

std::mutex g_failed_mutex;
std::unordered_map<uint64_t, bool> g_failed_shaders;

bool RunWave(Rdna::Wave& w, Rdna::Env& env, uint64_t shader, const char* what) {
    const Rdna::Result r = Rdna::Run(w, env);
    if (!r.ok) {
        std::lock_guard<std::mutex> lock(g_failed_mutex);
        if (!g_failed_shaders[shader]) {
            g_failed_shaders[shader] = true;
            printf("[raster] %s shader 0x%llx failed: %s\n", what, (unsigned long long)shader, r.error.c_str());
        }
        return false;
    }
    return true;
}

uint32_t UserSgprCount(uint32_t rsrc2) { return ((rsrc2 >> 1) & 0x1f) | (((rsrc2 >> 27) & 1) << 5); }

// Threads that pixel-shade together; KYTY_RASTER_THREADS sets how many (the caller included). Jobs are claimed
// from a counter, so a call finishes even if a worker has not started yet.
class Workers {
public:
    static Workers& Get() {
        static Workers* w = new Workers();
        return *w;
    }
    int Threads() const { return threads_; }
    // Runs job(i) for every i < jobs, partly on the calling thread, and returns once all have finished.
    void Run(int jobs, const std::function<void(int)>& job) {
        std::unique_lock<std::mutex> lock(m_);
        job_ = &job;
        jobs_ = jobs;
        next_ = finished_ = 0;
        wake_.notify_all();
        Drain(lock);
        done_.wait(lock, [&] { return finished_ == jobs_; });
        job_ = nullptr;
        jobs_ = next_ = finished_ = 0;
    }

private:
    Workers() {
        int want = 0;
        if (const char* s = getenv("KYTY_RASTER_THREADS")) {
            want = atoi(s);
        } else {
#ifdef __EMSCRIPTEN__
            want = emscripten_num_logical_cores();
#else
            want = (int)std::thread::hardware_concurrency();
#endif
        }
        want = std::clamp(want, 1, kMaxThreads);
        for (int i = 1; i < want; i++) {
            pthread_attr_t attr;
            pthread_attr_init(&attr);
            pthread_attr_setstacksize(&attr, 1 << 20);
            pthread_t t;
            if (pthread_create(&t, &attr, Main, this) == 0) {
                pthread_detach(t);
                threads_++;
            }
            pthread_attr_destroy(&attr);
        }
        printf("[raster] pixel shading on %d threads\n", threads_);
    }
    static void* Main(void* self) {
        auto* w = static_cast<Workers*>(self);
        std::unique_lock<std::mutex> lock(w->m_);
        for (;;) {
            w->wake_.wait(lock, [&] { return w->next_ < w->jobs_; });
            w->Drain(lock);
        }
        return nullptr;
    }
    // Claims and runs jobs until none are left unclaimed.
    void Drain(std::unique_lock<std::mutex>& lock) {
        while (next_ < jobs_) {
            const int i = next_++;
            const std::function<void(int)>* job = job_;
            lock.unlock();
            (*job)(i);
            lock.lock();
            if (++finished_ == jobs_) done_.notify_all();
        }
    }

    static constexpr int kMaxThreads = 8;  // the WASM build's pthread pool holds the workers
    int threads_ = 1;
    std::mutex m_;
    std::condition_variable wake_, done_;
    const std::function<void(int)>* job_ = nullptr;
    int jobs_ = 0, next_ = 0, finished_ = 0;
};

// One shading job's wave and statistics.
struct ShadeContext {
    Rdna::Wave wave;
    DrawStats stats;
};

struct DrawState {
    const uint32_t* cx;
    const uint32_t* sh;
    Target targets[8];
    int sc_x0, sc_y0, sc_x1, sc_y1;  // scissor, inclusive-exclusive
    float vp_xscale, vp_xoffset, vp_yscale, vp_yoffset, vp_zscale, vp_zoffset;
    uint64_t ps_addr;
    uint32_t ps_user_count;
    uint32_t ps_input_addr;
    uint32_t cull;  // PA_SU_SC_MODE_CNTL
    float blend_const[4];
    bool any_target;
    int debug_tris;  // triangles left to log (KYTY_RASTER_DEBUG)
};

void SetupTargets(DrawState& ds) {
    const uint32_t* cx = ds.cx;
    const uint32_t target_mask = cx[0x8E];
    const uint32_t col_format = cx[0x1C5];
    const uint32_t mode = (cx[0x202] >> 4) & 7;
    ds.any_target = false;
    for (int i = 0; i < 8; i++) {
        Target& t = ds.targets[i];
        t = Target();
        const uint32_t r = 0x318 + 15 * i;
        const uint32_t info = cx[r + 4];
        const uint32_t format = (info >> 2) & 0x1f;
        t.mask = (target_mask >> (4 * i)) & 0xf;
        t.export_format = (col_format >> (4 * i)) & 0xf;
        if (format == 0 || t.mask == 0 || mode == 0 || !LookupColorFormat(format, &t.cf)) continue;
        t.base = ((uint64_t)cx[r] << 8) | ((uint64_t)(cx[0x390 + i] & 0xff) << 40);
        t.number = (info >> 8) & 7;
        t.swap = (info >> 11) & 3;
        const uint32_t attrib2 = cx[0x3B0 + i], attrib3 = cx[0x3B8 + i];
        t.width = ((attrib2 >> 14) & 0x3fff) + 1;
        t.height = (attrib2 & 0x3fff) + 1;
        const uint32_t sw = (attrib3 >> 14) & 0x1f;
        const uint32_t view = cx[r + 3];
        const uint32_t slice = view & 0x1fff, level = (view >> 26) & 0xf;
        const uint32_t max_mip = (attrib2 >> 28) & 0xf;
        if (!t.base || !MakeLayout(t.base, FamilyOf(sw), t.cf.bpe, t.width, t.height, max_mip + 1, level, slice, &t.layout))
            continue;
        t.width = std::max(t.width >> level, 1u);
        t.height = std::max(t.height >> level, 1u);
        t.blend = cx[0x1E0 + i];
        for (int c = 0; c < t.cf.comps; c++) t.swap_of[c] = SwapChannel(t.cf.comps, t.swap, c);
        t.fast_rgba8 = t.cf.comps == 4 && t.cf.bpe == 4 && t.cf.bits[0] == 8 && t.number == 0 && t.mask == 0xf;
        t.enabled = true;
        ds.any_target = true;
        g_targets_by_base[t.base] = DisplayInfo{t.layout, t.cf, t.number, t.swap};
    }
}

void SetupScissor(DrawState& ds) {
    const uint32_t* cx = ds.cx;
    int x0 = 0, y0 = 0, x1 = 16384, y1 = 16384;
    auto apply = [&](uint32_t tl, uint32_t br) {
        const int ax = tl & 0x7fff, ay = (tl >> 16) & 0x7fff, bx = br & 0x7fff, by = (br >> 16) & 0x7fff;
        if (bx == 0 && by == 0) return;
        x0 = std::max(x0, ax); y0 = std::max(y0, ay); x1 = std::min(x1, bx); y1 = std::min(y1, by);
    };
    apply(cx[0x0C], cx[0x0D]);  // screen scissor
    apply(cx[0x81], cx[0x82]);  // window scissor
    apply(cx[0x90], cx[0x91]);  // generic scissor
    apply(cx[0x94], cx[0x95]);  // viewport 0 scissor
    for (const Target& t : ds.targets) {
        if (!t.enabled) continue;
        x1 = std::min(x1, (int)t.width);
        y1 = std::min(y1, (int)t.height);
    }
    ds.sc_x0 = x0; ds.sc_y0 = y0; ds.sc_x1 = x1; ds.sc_y1 = y1;
}

// Pixel-shades the part of one triangle in band `band` of `bands` (8-pixel tile rows, interleaved): 8x8 tiles, one
// wave each, lanes in 2x2 quads.
void ShadeTriangle(const DrawState& ds, const ScreenTri& tri, ShadeContext& ctx, int band, int bands) {
    const double x0 = tri.x[0], y0 = tri.y[0], x1 = tri.x[1], y1 = tri.y[1], x2 = tri.x[2], y2 = tri.y[2];
    double area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
    if (area == 0 || std::isnan(area)) return;
    const bool ccw = area > 0;
    const uint32_t cull = ds.cull;
    // FACE: 0 = counter-clockwise front. Screen y points down, so a positive area is clockwise on screen.
    const bool front = ((cull >> 2) & 1) ? ccw : !ccw;
    if ((front && (cull & 1)) || (!front && (cull & 2))) return;
    int bx0 = (int)std::floor(std::min({x0, x1, x2})), by0 = (int)std::floor(std::min({y0, y1, y2}));
    int bx1 = (int)std::ceil(std::max({x0, x1, x2})), by1 = (int)std::ceil(std::max({y0, y1, y2}));
    bx0 = std::max(bx0, ds.sc_x0); by0 = std::max(by0, ds.sc_y0);
    bx1 = std::min(bx1, ds.sc_x1); by1 = std::min(by1, ds.sc_y1);
    if (bx0 >= bx1 || by0 >= by1) return;

    // Edge functions E_i(p) for the edge opposite vertex i, oriented so the inside is positive.
    const double sgn = ccw ? 1.0 : -1.0;
    struct Edge { double a, b, c; bool tl; } e[3];
    const double vx[3] = {x0, x1, x2}, vy[3] = {y0, y1, y2};
    for (int i = 0; i < 3; i++) {
        const int j = (i + 1) % 3, k = (i + 2) % 3;
        // E(p) = (xk - xj) * (py - yj) - (yk - yj) * (px - xj)
        const double dx = vx[k] - vx[j], dy = vy[k] - vy[j];
        e[i].a = -dy * sgn;
        e[i].b = dx * sgn;
        e[i].c = (dy * vx[j] - dx * vy[j]) * sgn;
        // Shared edges belong to exactly one of the two triangles (they traverse it in opposite directions).
        const double odx = dx * sgn, ody = dy * sgn;
        e[i].tl = ody > 0 || (ody == 0 && odx < 0);
    }
    const double inv_area = 1.0 / std::fabs(area);

    PsEnv env;
    env.cx = ds.cx;
    env.tri = &tri;
    env.targets = ds.targets;
    env.stats = &ctx.stats;
    memcpy(env.blend_const, ds.blend_const, sizeof(env.blend_const));
    Rdna::Wave& w = ctx.wave;
    const uint32_t input = ds.ps_input_addr;
    static const int kInputSize[16] = {2, 2, 2, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    int slot[16];
    int nv = 0;
    for (int b = 0; b < 16; b++) {
        slot[b] = -1;
        if ((input >> b) & 1) { slot[b] = nv; nv += kInputSize[b]; }
    }
    const int tx0 = bx0 & ~7, ty0 = by0 & ~7;
    for (int ty = ty0; ty < by1; ty += 8) {
        if ((ty >> 3) % bands != band) continue;
        for (int tx = tx0; tx < bx1; tx += 8) {
            uint64_t cover = 0;
            float bi[kLanes], bj[kLanes], li[kLanes], lj[kLanes], pz[kLanes], pw[kLanes];
            for (int l = 0; l < kLanes; l++) {
                const int q = l >> 2, i = l & 3;
                const int px = tx + (q & 3) * 2 + (i & 1), py = ty + (q >> 2) * 2 + (i >> 1);
                const double cxp = px + 0.5, cyp = py + 0.5;
                const double e0 = e[0].a * cxp + e[0].b * cyp + e[0].c;
                const double e1 = e[1].a * cxp + e[1].b * cyp + e[1].c;
                const double e2 = e[2].a * cxp + e[2].b * cyp + e[2].c;
                const bool in = px >= bx0 && px < bx1 && py >= by0 && py < by1 &&
                                (e0 > 0 || (e0 == 0 && e[0].tl)) && (e1 > 0 || (e1 == 0 && e[1].tl)) &&
                                (e2 > 0 || (e2 == 0 && e[2].tl));
                if (in) cover |= 1ull << l;
                const double l0 = e0 * inv_area, l1 = e1 * inv_area, l2 = e2 * inv_area;
                li[l] = (float)l1;
                lj[l] = (float)l2;
                const double q0 = l0 * tri.invw[0], q1 = l1 * tri.invw[1], q2 = l2 * tri.invw[2];
                const double qs = q0 + q1 + q2;
                bi[l] = qs != 0 ? (float)(q1 / qs) : (float)l1;
                bj[l] = qs != 0 ? (float)(q2 / qs) : (float)l2;
                pz[l] = (float)(l0 * tri.z[0] + l1 * tri.z[1] + l2 * tri.z[2]);
                pw[l] = (float)qs;
            }
            if (!cover) continue;
            memset(w.s, 0, sizeof(w.s));
            for (uint32_t i = 0; i < ds.ps_user_count && i < 32; i++) w.s[i] = ds.sh[0x0C + i];
            w.scc = false;
            w.pc = ds.ps_addr;
            w.live = cover;
            w.SetExec(cover);
            for (int l = 0; l < kLanes; l++) {
                const int q = l >> 2, i = l & 3;
                const int px = tx + (q & 3) * 2 + (i & 1), py = ty + (q >> 2) * 2 + (i >> 1);
                for (int b = 0; b < 3; b++)
                    if (slot[b] >= 0) { w.v[slot[b]][l] = U(bi[l]); w.v[slot[b] + 1][l] = U(bj[l]); }
                if (slot[3] >= 0) {
                    w.v[slot[3]][l] = U(bi[l] * pw[l]); w.v[slot[3] + 1][l] = U(bj[l] * pw[l]); w.v[slot[3] + 2][l] = U(pw[l]);
                }
                for (int b = 4; b < 7; b++)
                    if (slot[b] >= 0) { w.v[slot[b]][l] = U(li[l]); w.v[slot[b] + 1][l] = U(lj[l]); }
                if (slot[7] >= 0) w.v[slot[7]][l] = 0;
                if (slot[8] >= 0) w.v[slot[8]][l] = U((float)px + 0.5f);
                if (slot[9] >= 0) w.v[slot[9]][l] = U((float)py + 0.5f);
                if (slot[10] >= 0) w.v[slot[10]][l] = U(pz[l]);
                if (slot[11] >= 0) w.v[slot[11]][l] = U(pw[l]);
                if (slot[12] >= 0) w.v[slot[12]][l] = U(tri.front ? 1.0f : -1.0f);
                if (slot[13] >= 0) w.v[slot[13]][l] = 0;
                if (slot[14] >= 0) w.v[slot[14]][l] = 1;
                if (slot[15] >= 0) w.v[slot[15]][l] = (uint32_t)px | ((uint32_t)py << 16);
            }
            env.tile_x = tx;
            env.tile_y = ty;
            ctx.stats.waves++;
            if (!RunWave(w, env, ds.ps_addr, "pixel")) return;
        }
    }
}

// Pixels a triangle may cover, inside the scissor.
int64_t BoundsArea(const DrawState& ds, const ScreenTri& t) {
    const float x0 = std::max(std::min({t.x[0], t.x[1], t.x[2]}), (float)ds.sc_x0);
    const float x1 = std::min(std::max({t.x[0], t.x[1], t.x[2]}), (float)ds.sc_x1);
    const float y0 = std::max(std::min({t.y[0], t.y[1], t.y[2]}), (float)ds.sc_y0);
    const float y1 = std::min(std::max({t.y[0], t.y[1], t.y[2]}), (float)ds.sc_y1);
    return x1 > x0 && y1 > y0 ? (int64_t)(x1 - x0 + 1) * (int64_t)(y1 - y0 + 1) : 0;
}

std::vector<std::unique_ptr<ShadeContext>> g_contexts;

// Shades triangles in order. Big batches are split into bands of tile rows shaded in parallel: each pixel is in
// one band, so it still sees the triangles (and blends) in order.
void ShadeBatch(const DrawState& ds, const std::vector<ScreenTri>& tris) {
    if (tris.empty()) return;
    int64_t area = 0;
    for (const ScreenTri& t : tris) area += BoundsArea(ds, t);
    Workers& workers = Workers::Get();
    const int bands = area >= 16384 && workers.Threads() > 1 ? 2 * workers.Threads() : 1;
    while ((int)g_contexts.size() < bands) g_contexts.push_back(std::make_unique<ShadeContext>());
    auto job = [&](int band) {
        for (const ScreenTri& t : tris) ShadeTriangle(ds, t, *g_contexts[band], band, bands);
    };
    if (bands == 1) job(0);
    else workers.Run(bands, job);
    for (int b = 0; b < bands; b++) {
        DrawStats& s = g_contexts[b]->stats;
        g_stats.pixels += s.pixels;
        g_stats.waves += s.waves;
        for (int c = 0; c < 4; c++) g_stats.sum[c] += s.sum[c];
        g_frame.pixels += s.pixels;
        s = DrawStats();
    }
}

bool ToScreen(const DrawState& ds, const VertexOut& v, float* sx, float* sy, float* sz, float* invw) {
    const float wc = v.pos[3];
    if (!(wc > 0)) return false;
    const float iw = 1.0f / wc;
    // Snap to the 1/256-pixel grid the hardware rasterizes on.
    *sx = std::round((v.pos[0] * iw * ds.vp_xscale + ds.vp_xoffset) * 256.0f) / 256.0f;
    *sy = std::round((v.pos[1] * iw * ds.vp_yscale + ds.vp_yoffset) * 256.0f) / 256.0f;
    *sz = v.pos[2] * iw * ds.vp_zscale + ds.vp_zoffset;
    *invw = iw;
    return std::isfinite(*sx) && std::isfinite(*sy);
}

// The draw's first triangle, for KYTY_RASTER_EMPTY.
ScreenTri g_first_tri;
VertexOut g_first_vertex[3];
std::array<float, 2> g_first_corners[16];  // the first vertex of the first 16 triangles, on screen

// Projects a triangle to the screen and queues it for shading.
void EmitTriangle(DrawState& ds, std::vector<ScreenTri>& out, const VertexOut* a, const VertexOut* b, const VertexOut* c) {
    ScreenTri t;
    const VertexOut* v[3] = {a, b, c};
    if (ds.debug_tris > 0) {
        ds.debug_tris--;
        for (int i = 0; i < 3; i++)
            printf("[raster]   v%d pos (%g %g %g %g) param0 (%g %g %g %g) param1 (%g %g %g %g)\n", i, v[i]->pos[0], v[i]->pos[1],
                   v[i]->pos[2], v[i]->pos[3], v[i]->param[0][0], v[i]->param[0][1], v[i]->param[0][2], v[i]->param[0][3],
                   v[i]->param[1][0], v[i]->param[1][1], v[i]->param[1][2], v[i]->param[1][3]);
    }
    for (int i = 0; i < 3; i++) {
        if (!ToScreen(ds, *v[i], &t.x[i], &t.y[i], &t.z[i], &t.invw[i])) return;
        t.v[i] = v[i];
    }
    const double area = ((double)t.x[1] - t.x[0]) * ((double)t.y[2] - t.y[0]) - ((double)t.y[1] - t.y[0]) * ((double)t.x[2] - t.x[0]);
    t.front = ((ds.cull >> 2) & 1) ? area > 0 : area < 0;
    if (g_stats.triangles == 0) {
        g_first_tri = t;
        for (int i = 0; i < 3; i++) g_first_vertex[i] = *v[i];
    }
    if (g_stats.triangles < 16) g_first_corners[g_stats.triangles] = {t.x[0], t.y[0]};
    g_stats.triangles++;
    out.push_back(t);
}

uint32_t g_draw_count = 0;

} // namespace

void RasterSetup(const HleLayout& layout) { L = layout; }

void RasterDraw(const GpuRegs& regs, const DrawCall& draw) {
    const uint32_t* cx = regs.cx;
    const uint32_t* sh = regs.sh;
    g_draw_count++;
    g_frame.draws++;
    g_bc_generation++;
    DrawState ds{};
    ds.cx = cx;
    ds.sh = sh;
    uint64_t vs = ShaderAddr(sh, 0xC8);
    if (!vs) vs = ShaderAddr(sh, 0x88);
    ds.ps_addr = ShaderAddr(sh, 0x08);
    if (!vs || !ds.ps_addr || draw.count == 0) return;
    SetupTargets(ds);
    if (!ds.any_target) {
        RASTER_LOG("draw without render targets: CB0 info 0x%x base 0x%x attrib2 0x%x attrib3 0x%x target mask 0x%x "
                   "color control 0x%x\n", cx[0x31C], cx[0x318], cx[0x3B0], cx[0x3B8], cx[0x8E], cx[0x202]);
        return;
    }
    SetupScissor(ds);
    ds.vp_xscale = F(cx[0x10F]); ds.vp_xoffset = F(cx[0x110]);
    ds.vp_yscale = F(cx[0x111]); ds.vp_yoffset = F(cx[0x112]);
    ds.vp_zscale = F(cx[0x113]); ds.vp_zoffset = F(cx[0x114]);
    ds.ps_user_count = UserSgprCount(sh[0x0B]);
    ds.ps_input_addr = cx[0x1B4] ? cx[0x1B4] : cx[0x1B3];
    ds.cull = cx[0x205];
    for (int i = 0; i < 4; i++) ds.blend_const[i] = F(cx[0x105 + i]);
    const uint32_t prim_type = regs.uc[0x242] & 0x3f;
    if (g_draw_count <= 3) {
        RASTER_LOG("draw %u: prim %u count %u vs 0x%llx ps 0x%llx rt0 base 0x%llx %ux%u fmt %u/%u swap %u sw %u blend 0x%x "
                   "scissor (%d,%d)-(%d,%d) viewport scale (%g,%g) offset (%g,%g) ps input 0x%x user %u\n",
                   g_draw_count, prim_type, draw.count, (unsigned long long)vs, (unsigned long long)ds.ps_addr,
                   (unsigned long long)ds.targets[0].base, ds.targets[0].width, ds.targets[0].height,
                   (cx[0x31C] >> 2) & 0x1f, ds.targets[0].number, ds.targets[0].swap, (cx[0x3B8] >> 14) & 0x1f,
                   ds.targets[0].blend, ds.sc_x0, ds.sc_y0, ds.sc_x1, ds.sc_y1, ds.vp_xscale, ds.vp_yscale,
                   ds.vp_xoffset, ds.vp_yoffset, ds.ps_input_addr, ds.ps_user_count);
    }

    // Primitive vertex lists (draw-relative vertex numbers).
    auto index_of = [&](uint32_t i) -> uint32_t {
        if (!draw.index_addr) return i;
        switch (draw.index_type) {
            case 1: { uint32_t v = 0; if (auto* p = Ptr(draw.index_addr + 4ull * i, 4)) memcpy(&v, p, 4); return v; }
            case 2: { auto* p = Ptr(draw.index_addr + i, 1); return p ? *p : 0; }
            default: { uint16_t v = 0; if (auto* p = Ptr(draw.index_addr + 2ull * i, 2)) memcpy(&v, p, 2); return v; }
        }
    };
    const bool rects = prim_type == 7 || prim_type == 17;
    std::vector<uint32_t> prims;  // 3 indices per primitive
    switch (prim_type) {
        case 4: case 7: case 17:
            for (uint32_t i = 0; i + 2 < draw.count; i += 3)
                for (int k = 0; k < 3; k++) prims.push_back(index_of(i + k));
            break;
        case 6:  // strip
            for (uint32_t i = 0; i + 2 < draw.count; i++) {
                const uint32_t a = index_of(i), b = index_of(i + 1), c = index_of(i + 2);
                if (i & 1) { prims.push_back(b); prims.push_back(a); prims.push_back(c); }
                else { prims.push_back(a); prims.push_back(b); prims.push_back(c); }
            }
            break;
        case 5:  // fan
            for (uint32_t i = 1; i + 1 < draw.count; i++) {
                prims.push_back(index_of(0)); prims.push_back(index_of(i)); prims.push_back(index_of(i + 1));
            }
            break;
        default:
            RASTER_LOG("primitive type %u not drawn\n", prim_type);
            return;
    }

    static int debug_draws = getenv("KYTY_RASTER_DEBUG") ? atoi(getenv("KYTY_RASTER_DEBUG")) : 0;
    g_stats = DrawStats();
    struct Report {
        bool on;
        uint32_t n, prims;
        uint64_t vs, ps;
        const DrawState* ds;
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        ~Report() {
            // KYTY_RASTER_EMPTY=n: the first triangle of the first n draws (from draw KYTY_RASTER_EMPTY_FROM on) whose
            // triangles shaded nothing.
            static int empty_left = getenv("KYTY_RASTER_EMPTY") ? atoi(getenv("KYTY_RASTER_EMPTY")) : 0;
            static uint32_t empty_from = getenv("KYTY_RASTER_EMPTY_FROM") ? atoi(getenv("KYTY_RASTER_EMPTY_FROM")) : 0;
            if (empty_left > 0 && n >= empty_from && g_stats.triangles && !g_stats.pixels) {
                empty_left--;
                printf("[raster] draw %u (vs 0x%llx ps 0x%llx, %llu triangles) shaded nothing; scissor (%d %d)-(%d %d) "
                       "cull 0x%x\n", n, (unsigned long long)vs, (unsigned long long)ps,
                       (unsigned long long)g_stats.triangles, ds->sc_x0, ds->sc_y0, ds->sc_x1, ds->sc_y1, ds->cull);
                for (int i = 0; i < 3; i++) {
                    const VertexOut& v = g_first_vertex[i];
                    printf("[raster]   v%d clip (%g %g %g %g) screen (%g %g) param0 (%g %g %g %g) param1 (%g %g %g %g)\n", i,
                           v.pos[0], v.pos[1], v.pos[2], v.pos[3], g_first_tri.x[i], g_first_tri.y[i], v.param[0][0],
                           v.param[0][1], v.param[0][2], v.param[0][3], v.param[1][0], v.param[1][1], v.param[1][2],
                           v.param[1][3]);
                }
                std::string corners;
                char buf[40];
                for (uint64_t i = 0; i < std::min<uint64_t>(g_stats.triangles, 16); i++) {
                    snprintf(buf, sizeof(buf), " (%.0f %.0f)", g_first_corners[i][0], g_first_corners[i][1]);
                    corners += buf;
                }
                printf("[raster]   triangles start at%s\n", corners.c_str());
            }
            if (!on) return;
            const uint32_t* c = ds->cx;
            if (DebugFrame()) {
                // Context registers that changed since the previous logged draw.
                static uint32_t prev[0x400];
                std::string diff;
                char buf[40];
                for (int i = 0; i < 0x400; i++)
                    if (c[i] != prev[i]) { snprintf(buf, sizeof(buf), " %03x=%08x", i, c[i]); diff += buf; }
                memcpy(prev, c, sizeof(prev));
                printf("[raster]   cx changes:%s\n", diff.c_str());
                std::string shd;
                for (int i = 0; i < 0x20; i++) { snprintf(buf, sizeof(buf), " %08x", ds->sh[0x0C + i]); shd += buf; }
                printf("[raster]   ps user:%s\n", shd.c_str());
            }
            printf("[raster]   db: render 0x%x depth_control 0x%x z_info 0x%x stencil_info 0x%x z_read 0x%x stencil_control 0x%x "
                   "stencil refmask 0x%x/0x%x shader_control 0x%x depth_clear 0x%x zscale %g zoffset %g\n", c[0x000], c[0x200],
                   c[0x010], c[0x011], c[0x012], c[0x10B], c[0x10C], c[0x10D], c[0x203], c[0x00B], ds->vp_zscale, ds->vp_zoffset);
            const double k = g_stats.pixels ? 1.0 / (double)g_stats.pixels : 0;
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            printf("[raster] draw %u: %u prims, vs 0x%llx ps 0x%llx rt 0x%llx %ux%u mask 0x%x blend 0x%x exp %u cb_shader_mask 0x%x "
                   "vp (%g %g %g %g): %llu triangles, %llu waves, %llu pixels, mean (%.3f %.3f %.3f %.3f), %.1f ms\n",
                   n, prims, (unsigned long long)vs, (unsigned long long)ps, (unsigned long long)ds->targets[0].base,
                   ds->targets[0].width, ds->targets[0].height, ds->targets[0].mask, ds->targets[0].blend,
                   ds->targets[0].export_format, ds->cx[0x8F], ds->vp_xscale, ds->vp_yscale, ds->vp_xoffset, ds->vp_yoffset,
                   (unsigned long long)g_stats.triangles, (unsigned long long)g_stats.waves,
                   (unsigned long long)g_stats.pixels, g_stats.sum[0] * k, g_stats.sum[1] * k, g_stats.sum[2] * k,
                   g_stats.sum[3] * k, ms);
        }
    };
    // The first draw of each shader pair is always summarized; KYTY_RASTER_DEBUG=n summarizes the first n draws.
    static std::unordered_map<uint64_t, bool> seen_pairs;
    const uint64_t pair_key = vs * 1000003u ^ ds.ps_addr;
    const bool first_of_pair = !seen_pairs[pair_key];
    seen_pairs[pair_key] = true;
    Report report{(int)g_draw_count <= debug_draws || first_of_pair || DebugFrame(), g_draw_count, (uint32_t)(prims.size() / 3), vs,
                  ds.ps_addr, &ds};
    ds.debug_tris = (int)g_draw_count <= debug_draws && g_draw_count <= 24 ? 2 : 0;
    const uint32_t gs_user = UserSgprCount(sh[0x8B]) ? UserSgprCount(sh[0x8B]) : 32;
    static VertexOut outs[kLanes], corners[kLanes];
    static std::vector<ScreenTri> tris;
    Rdna::Wave& w = Wave();
    const size_t nprims = prims.size() / 3;
    for (uint32_t inst = 0; inst < std::max(draw.instances, 1u); inst++) {
        size_t p = 0;
        while (p < nprims) {
            // Batch: up to 64 primitives using up to 64 distinct vertices.
            uint32_t lane_vertex[kLanes];
            uint32_t prim_lanes[kLanes][3];
            int nverts = 0, nb = 0;
            while (p < nprims && nb < kLanes) {
                int lanes[3];
                int added = 0;
                for (int k = 0; k < 3; k++) {
                    const uint32_t vi = prims[3 * p + k];
                    int found = -1;
                    for (int j = 0; j < nverts; j++)
                        if (lane_vertex[j] == vi) { found = j; break; }
                    if (found < 0) {
                        for (int j = 0; j < k; j++)
                            if (prims[3 * p + j] == vi) found = lanes[j];
                    }
                    if (found < 0) { found = nverts + added; added++; }
                    lanes[k] = found;
                }
                if (nverts + added > kLanes) break;
                for (int k = 0; k < 3; k++) {
                    if (lanes[k] >= nverts) lane_vertex[lanes[k]] = prims[3 * p + k];
                    prim_lanes[nb][k] = (uint32_t)lanes[k];
                }
                nverts += added;
                nb++;
                p++;
            }
            if (nb == 0) break;
            // Vertex (NGG primitive) shader wave: one lane per vertex, one per primitive for its export.
            memset(w.s, 0, sizeof(w.s));
            w.s[3] = (uint32_t)nverts | ((uint32_t)nb << 8);
            for (uint32_t i = 0; i < gs_user && i < 32; i++) w.s[8 + i] = sh[0x8C + i];
            w.scc = false;
            w.pc = vs;
            w.live = ~0ull;
            w.SetExec(~0ull);
            for (int l = 0; l < kLanes; l++) {
                w.v[0][l] = l < nb ? prim_lanes[l][0] | (prim_lanes[l][1] << 10) | (prim_lanes[l][2] << 20) : 0;
                w.v[1][l] = 0;
                w.v[2][l] = (uint32_t)l;
                w.v[3][l] = 0;
                w.v[4][l] = 0;
                w.v[5][l] = l < nverts ? lane_vertex[l] : 0;
                w.v[6][l] = 0;
                w.v[7][l] = 0;
                w.v[8][l] = inst;
            }
            VsEnv venv;
            venv.out = outs;
            static const char* trace_vs = getenv("KYTY_TRACE_VS");
            static int traced = 0;
            if (trace_vs && strtoull(trace_vs, nullptr, 16) == vs && traced < 1) {
                traced++;
                venv.trace = true;
                printf("[raster] tracing vs 0x%llx: s3=%08x user:", (unsigned long long)vs, w.s[3]);
                for (uint32_t i = 0; i < gs_user && i < 32; i++) printf(" %08x", w.s[8 + i]);
                printf(" lane0 vertex %u\n", lane_vertex[0]);
            }
            for (int l = 0; l < kLanes; l++) outs[l].pos[0] = outs[l].pos[1] = outs[l].pos[2] = 0, outs[l].pos[3] = 1;
            if (!RunWave(w, venv, vs, "vertex")) return;
            tris.clear();
            for (int k = 0; k < nb; k++) {
                if (((venv.prim_lanes >> k) & 1) && (venv.prim[k] >> 31)) continue;  // culled by the shader
                const VertexOut* a = &outs[prim_lanes[k][0]];
                const VertexOut* b = &outs[prim_lanes[k][1]];
                const VertexOut* c = &outs[prim_lanes[k][2]];
                if (rects) {
                    // The fourth corner completes the parallelogram opposite the first vertex.
                    VertexOut& d = corners[k];
                    for (int i = 0; i < 4; i++) d.pos[i] = b->pos[i] + c->pos[i] - a->pos[i];
                    for (int j = 0; j < 32; j++)
                        for (int i = 0; i < 4; i++) d.param[j][i] = b->param[j][i] + c->param[j][i] - a->param[j][i];
                    EmitTriangle(ds, tris, a, b, c);
                    EmitTriangle(ds, tris, b, &d, c);
                } else {
                    EmitTriangle(ds, tris, a, b, c);
                }
            }
            ShadeBatch(ds, tris);
        }
    }
}

void RasterDispatch(const GpuRegs& regs, uint32_t gx, uint32_t gy, uint32_t gz, uint32_t initiator) {
    const uint32_t* sh = regs.sh;
    const uint64_t cs = ShaderAddr(sh, 0x20C);
    if (!cs || !gx || !gy || !gz) return;
    g_frame.dispatches++;
    g_bc_generation++;
    const uint32_t nx = std::max(sh[0x207] & 0x3ff, 1u), ny = std::max(sh[0x208] & 0x3ff, 1u), nz = std::max(sh[0x209] & 0x3ff, 1u);
    const uint32_t rsrc2 = sh[0x213];
    const uint32_t user = UserSgprCount(rsrc2);
    const bool tgid[3] = {((rsrc2 >> 7) & 1) != 0, ((rsrc2 >> 8) & 1) != 0, ((rsrc2 >> 9) & 1) != 0};
    const bool tg_size = (rsrc2 >> 10) & 1;
    const uint32_t tid_comps = (rsrc2 >> 11) & 3;
    const uint32_t lds_bytes = std::min(((rsrc2 >> 15) & 0x1ff) * 512u, 65536u);
    const bool wave32 = (initiator >> 15) & 1;
    static std::unordered_map<uint64_t, bool> seen;
    if (!seen[cs] || DebugFrame()) {
        seen[cs] = true;
        printf("[raster] dispatch %ux%ux%u groups of %ux%ux%u, cs 0x%llx user %u lds %u%s\n", gx, gy, gz, nx, ny, nz,
                   (unsigned long long)cs, user, lds_bytes, wave32 ? " wave32" : "");
    }
    if (wave32) {
        RASTER_LOG("wave32 dispatches are not supported\n");
        return;
    }
    const uint32_t threads = nx * ny * nz, waves = (threads + kLanes - 1) / kLanes;
    std::vector<uint8_t> lds(std::max(lds_bytes, 4u));
    std::vector<Rdna::Wave> wv(waves);
    BaseEnv env;
    for (uint32_t z = 0; z < gz; z++) {
        for (uint32_t y = 0; y < gy; y++) {
            for (uint32_t x = 0; x < gx; x++) {
                std::fill(lds.begin(), lds.end(), 0);
                std::vector<bool> done(waves, false);
                for (uint32_t i = 0; i < waves; i++) {
                    Rdna::Wave& w = wv[i];
                    memset(w.s, 0, sizeof(w.s));
                    uint32_t s = 0;
                    for (; s < user && s < 16; s++) w.s[s] = sh[0x240 + s];
                    s = user;
                    const uint32_t ids[3] = {x + sh[0x204], y + sh[0x205], z + sh[0x206]};
                    for (int d = 0; d < 3; d++)
                        if (tgid[d]) w.s[s++] = ids[d];
                    if (tg_size) w.s[s++] = (waves << 6) | i;
                    uint64_t exec = 0;
                    for (int l = 0; l < kLanes; l++) {
                        const uint32_t t = i * kLanes + (uint32_t)l;
                        if (t < threads) exec |= 1ull << l;
                        w.v[0][l] = t % nx;
                        w.v[1][l] = tid_comps >= 1 ? (t / nx) % ny : 0;
                        w.v[2][l] = tid_comps >= 2 ? t / (nx * ny) : 0;
                    }
                    w.SetExec(exec);
                    w.scc = false;
                    w.pc = cs;
                    w.live = exec;
                    w.lds = lds.data();
                    w.lds_bytes = lds_bytes;
                    w.first_thread = i * kLanes;
                }
                // Waves run until they end or reach a barrier, round-robin.
                uint32_t remaining = waves;
                while (remaining) {
                    for (uint32_t i = 0; i < waves; i++) {
                        if (done[i]) continue;
                        const Rdna::Result r = Rdna::Run(wv[i], env);
                        if (!r.ok) {
                            if (!g_failed_shaders[cs]) {
                                g_failed_shaders[cs] = true;
                                printf("[raster] compute shader 0x%llx failed: %s\n", (unsigned long long)cs, r.error.c_str());
                            }
                            return;
                        }
                        if (!r.barrier) { done[i] = true; remaining--; }
                    }
                }
            }
        }
    }
}

bool RasterDebugFrame() { return DebugFrame(); }

void RasterFrameDone(uint64_t flip) {
    g_flips = flip;
    if (flip <= 60 || flip % 60 == 0)
        printf("[raster] frame %llu: %llu draws, %llu dispatches, %llu pixels shaded\n", (unsigned long long)flip,
               (unsigned long long)g_frame.draws, (unsigned long long)g_frame.dispatches,
               (unsigned long long)g_frame.pixels);
    g_frame = FrameStats();
}

bool RasterReadDisplay(uint64_t addr, uint32_t width, uint32_t height, uint8_t* rgba) {
    auto it = g_targets_by_base.find(addr);
    static int debug = getenv("KYTY_RASTER_DEBUG") ? 10 : 0;
    if (debug > 0) {
        debug--;
        printf("[raster] present 0x%llx %ux%u (%s)\n", (unsigned long long)addr, width, height,
               it != g_targets_by_base.end() ? "rendered" : "not rendered");
    }
    DisplayInfo info;
    if (it != g_targets_by_base.end()) {
        info = it->second;
    } else {
        // Not rendered by us: assume the display layout (64KB render-target tiling, BGRA8).
        LookupColorFormat(10, &info.cf);
        info.swap = 1;
        if (!MakeLayout(addr, Family::kRt64KB, 4, width, height, 1, 0, 0, &info.layout)) return false;
    }
    Target t;
    t.layout = info.layout;
    t.cf = info.cf;
    t.number = info.number;
    t.swap = info.swap;
    if (!Ptr(addr, 4)) return false;
    const bool fast = info.cf.comps == 4 && info.cf.bpe == 4 && info.cf.bits[0] == 8 && (info.number == 0 || info.number == 6);
    int swap_of[4];
    for (int c = 0; c < 4; c++) swap_of[c] = SwapChannel(4, info.swap, c);
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            if (fast) {
                // sRGB targets hold encoded values already: their bytes are what the display shows.
                uint8_t* o = rgba + 4ull * (y * width + x);
                const uint8_t* p = Ptr(info.layout.base + info.layout.Offset(x, y), 4);
                if (!p) { o[0] = o[1] = o[2] = 0; o[3] = 255; continue; }
                for (int c = 0; c < 4; c++) if (swap_of[c] < 3) o[swap_of[c]] = p[c];
                o[3] = 255;
                continue;
            }
            float c[4];
            ReadColor(t, x, y, c);
            uint8_t* o = rgba + 4ull * (y * width + x);
            for (int i = 0; i < 3; i++) o[i] = (uint8_t)std::lround(std::clamp(info.number == 6 ? LinearToSrgb(c[i]) : c[i], 0.0f, 1.0f) * 255.0f);
            o[3] = 255;
        }
    }
    return true;
}

} // namespace KytyWeb
