// GPU stand-in for submitted AGC command buffers, plus the AGC driver and sceVideoOut entry points.
// Packets execute in order per queue: register state, memory writes, copies, fills, waits on memory, end-of-pipe
// label writes and interrupts, flips, and draws and dispatches (rendered in software by raster.cpp).
// Packet and struct layouts follow KytyPS5 (https://github.com/KytyPS5/KytyPS5, GPL-2.0).

#include "hle.h"
#include "raster.h"
#include "web_platform.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace KytyWeb {
namespace {

HleLayout L;

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
uint64_t StackArg(const X86Regs& r, int i) { return Rd<uint64_t>(r.rsp + 8 + 8 * i); }
void Ret(X86Regs& r, int64_t v) { r.rax = (uint64_t)v; }

const auto g_start = std::chrono::steady_clock::now();
uint64_t NowUs() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - g_start).count();
}
uint64_t ReferenceClock() { return NowUs() * 100; }  // the GPU reference clock runs at 100 MHz

constexpr int16_t kEvfiltVideoOut = -13, kEvfiltGraphics = -14;

// Command-buffer traffic is logged for the first submissions only.
int g_trace_budget = 40;
#define GPU_TRACE(...)                                   \
    do {                                                 \
        if (g_trace_budget > 0) {                        \
            g_trace_budget--;                            \
            printf("[gpu] " __VA_ARGS__);                \
        }                                                \
    } while (0)
constexpr uint64_t kVideoOutEventFlip = 0;

// --- video out ---

struct VideoOutState {
    bool open = false;
    uint64_t buffers[16] = {};
    uint32_t width = 1920, height = 1080, tiling = 0;
    uint64_t pixel_format = 0;
    uint64_t flip_count = 0, flip_time_us = 0, submit_time_us = 0;
    int64_t flip_arg = 0;
    int32_t current_buffer = -1;
} g_vo;

void Flip(int index, int64_t arg, const char* source) {
    g_vo.flip_count++;
    RasterFrameDone(g_vo.flip_count);
    if (g_vo.flip_count <= 60)
        printf("[gpu] flip %llu from %s: buffer %d (0x%llx)\n", (unsigned long long)g_vo.flip_count, source, index,
               (unsigned long long)(index >= 0 && index < 16 ? g_vo.buffers[index] : 0));
    g_vo.flip_arg = arg;
    g_vo.current_buffer = index;
    g_vo.flip_time_us = NowUs();
    if (g_vo.flip_count == 1 || g_vo.flip_count % 600 == 0)
        printf("[gpu] flip #%llu (buffer %d, %ux%u)\n", (unsigned long long)g_vo.flip_count, index, g_vo.width,
               g_vo.height);
    // The display buffer is detiled and converted to RGBA, then scaled to the canvas texture size.
    constexpr uint32_t kOutW = 1920, kOutH = 1080;
    static std::vector<uint8_t> rgba, scaled(kOutW * kOutH * 4);
    rgba.resize((size_t)g_vo.width * g_vo.height * 4);
    if (index >= 0 && index < 16 && g_vo.buffers[index] &&
        RasterReadDisplay(g_vo.buffers[index], g_vo.width, g_vo.height, rgba.data())) {
        if (g_vo.width == kOutW && g_vo.height == kOutH) {
            PresentFrame(rgba.data(), kOutW, kOutH);
        } else {
            for (uint32_t y = 0; y < kOutH; y++)
                for (uint32_t x = 0; x < kOutW; x++)
                    memcpy(&scaled[(y * kOutW + x) * 4],
                           &rgba[((uint64_t)(y * g_vo.height / kOutH) * g_vo.width + x * g_vo.width / kOutW) * 4], 4);
            PresentFrame(scaled.data(), kOutW, kOutH);
        }
    }
    const uint64_t data = (NowUs() & 0xfff) | (std::min<uint64_t>(g_vo.flip_count, 0xf) << 12) |
                          (((uint64_t)arg & 0xffffffffffffULL) << 16);
    HleEqueueTrigger(kVideoOutEventFlip, kEvfiltVideoOut, data);
}

void VideoOutOpen(X86Regs& r) {
    printf("[gpu] sceVideoOutOpen(user=%d, bus=%d)\n", (int)r.rdi, (int)r.rsi);
    g_vo.open = true;
    r.rax = 1;
}
void VideoOutClose(X86Regs& r) { g_vo.open = false; r.rax = 0; }
void VideoOutRegisterBuffers2(X86Regs& r) {  // (handle, set, start, buffers*, num, attr*, category, option)
    const int start = (int)r.rdx, num = (int)r.r8;
    for (int i = 0; i < num && start + i < 16; i++) g_vo.buffers[start + i] = Rd<uint64_t>(r.rcx + 32ull * i);
    if (r.r9) {  // VideoOutBufferAttribute2
        g_vo.tiling = Rd<uint32_t>(r.r9 + 4);
        g_vo.width = std::max<uint32_t>(Rd<uint32_t>(r.r9 + 0xC), 1);
        g_vo.height = std::max<uint32_t>(Rd<uint32_t>(r.r9 + 0x10), 1);
        g_vo.pixel_format = Rd<uint64_t>(r.r9 + 0x20);
    }
    printf("[gpu] registered %d display buffers (%ux%u, format 0x%llx, tiling %u)\n", num, g_vo.width, g_vo.height,
           (unsigned long long)g_vo.pixel_format, g_vo.tiling);
    r.rax = 0;
}
void VideoOutSubmitFlip(X86Regs& r) {  // (handle, index, mode, arg)
    g_vo.submit_time_us = NowUs();
    Flip((int)r.rsi, (int64_t)r.rcx, "sceVideoOutSubmitFlip");
    r.rax = 0;
}
void VideoOutAddFlipEvent(X86Regs& r) {  // (eq, handle, udata)
    HleEqueueAddEvent(r.rdi, kVideoOutEventFlip, kEvfiltVideoOut, r.rdx);
    r.rax = 0;
}
void VideoOutGetFlipStatus(X86Regs& r) {
    uint8_t s[0x80] = {};
    const uint64_t time = g_vo.flip_time_us, counter = g_vo.flip_time_us * 100, submit = g_vo.submit_time_us * 100;
    const int32_t current = g_vo.current_buffer;
    memcpy(s + 0x00, &g_vo.flip_count, 8);
    memcpy(s + 0x08, &time, 8);
    memcpy(s + 0x18, &g_vo.flip_arg, 8);
    memcpy(s + 0x28, &counter, 8);
    memcpy(s + 0x38, &current, 4);
    memcpy(s + 0x40, &submit, 8);
    if (auto* p = Ptr(r.rsi, sizeof(s))) memcpy(p, s, sizeof(s));
    r.rax = 0;
}
void VideoOutGetVblankStatus(X86Regs& r) {
    uint8_t s[0x28] = {};
    const uint64_t now = NowUs(), count = now / 16667, counter = now * 100;
    memcpy(s + 0x00, &count, 8);
    memcpy(s + 0x08, &now, 8);
    memcpy(s + 0x18, &counter, 8);
    if (auto* p = Ptr(r.rsi, sizeof(s))) memcpy(p, s, sizeof(s));
    r.rax = 0;
}
void VideoOutSetBufferAttribute2(X86Regs& r) {  // (attr*, pixel_format, tiling, width, height, option, dcc_control, dcc_clear)
    uint8_t a[0x50] = {};
    const uint32_t tiling = (uint32_t)r.rdx, width = (uint32_t)r.rcx, height = (uint32_t)r.r8;
    const uint64_t option = r.r9, format = r.rsi, dcc_clear = StackArg(r, 1);
    const uint32_t dcc_control = (uint32_t)StackArg(r, 0);
    memcpy(a + 0x04, &tiling, 4);
    memcpy(a + 0x0C, &width, 4);
    memcpy(a + 0x10, &height, 4);
    memcpy(a + 0x18, &option, 8);
    memcpy(a + 0x20, &format, 8);
    memcpy(a + 0x28, &dcc_clear, 8);
    memcpy(a + 0x30, &dcc_control, 4);
    if (auto* p = Ptr(r.rdi, sizeof(a))) memcpy(p, a, sizeof(a));
}
void VideoOutGetOutputStatus(X86Regs& r) {  // (handle, status*): resolution, dynamic range, refresh rate, flags
    uint8_t s[0x30] = {};
    const uint32_t resolution = 1, dynamic_range = 1;  // the 1080p output, SDR
    const uint64_t refresh_rate = 3;                   // 59.94 Hz
    memcpy(s + 0x00, &resolution, 4);
    memcpy(s + 0x04, &dynamic_range, 4);
    memcpy(s + 0x08, &refresh_rate, 8);
    if (auto* p = Ptr(r.rsi, sizeof(s))) memcpy(p, s, sizeof(s));
    r.rax = 0;
}
void RetZero(X86Regs& r) { r.rax = 0; }

// --- command processor ---

std::unordered_map<uint32_t, bool> g_skipped_ops;
std::string g_trace_ops;  // packet opcodes of a traced submission

void ReleaseMem(const uint32_t* b, int event_id) {
    uint32_t data_sel = (b[1] >> 29) & 7;
    const uint32_t int_sel = (b[1] >> 24) & 7;
    const uint64_t addr = b[2] | ((uint64_t)b[3] << 32);
    uint64_t data = b[4] | ((uint64_t)b[5] << 32);
    // Interrupt selectors: 0 none, 1 interrupt only, 2/3 after the write, 4 send context id only.
    const bool interrupt = int_sel != 0 && int_sel != 3;
    if (!addr || int_sel == 4 || (int_sel == 1 && event_id < 0x20)) data_sel = 0;
    switch (data_sel) {
        case 1: Wr<uint32_t>(addr, (uint32_t)data); break;
        case 3: data = ReferenceClock(); [[fallthrough]];
        case 2: Wr<uint64_t>(addr, data); break;
        default: break;
    }
    GPU_TRACE("release mem: data_sel %u int_sel %u addr 0x%llx data 0x%llx event %d ctx 0x%x\n", data_sel, int_sel,
              (unsigned long long)addr, (unsigned long long)data, event_id, b[6]);
    if (interrupt) HleEqueueTrigger((uint64_t)event_id, kEvfiltGraphics, b[6]);
}

void CopyData(const uint32_t* b) {
    const uint32_t src_sel = b[0] & 0xf, dst_sel = (b[0] >> 8) & 0xf;
    const bool wide = (b[0] >> 16) & 1;
    const uint64_t src = b[1] | ((uint64_t)b[2] << 32), dst = b[3] | ((uint64_t)b[4] << 32);
    uint64_t value;
    switch (src_sel) {
        case 1: case 2: value = wide ? Rd<uint64_t>(src) : Rd<uint32_t>(src); break;  // memory
        case 5: value = src; break;                                               // immediate
        case 9: value = ReferenceClock(); break;                                  // GPU clock
        default: return;                                                          // registers, counters
    }
    if (dst_sel == 0) return;  // register destination
    if (wide) Wr<uint64_t>(dst, value);
    else Wr<uint32_t>(dst, (uint32_t)value);
}

void DmaData(const uint32_t* b) {
    const uint32_t src_sel = (b[0] >> 29) & 3, dst_sel = (b[0] >> 20) & 3;
    const uint64_t src = b[1] | ((uint64_t)b[2] << 32), dst = b[3] | ((uint64_t)b[4] << 32);
    const uint64_t n = b[5] & 0x3ffffff;
    if (dst_sel == 1 || src_sel == 1) return;  // GDS
    uint8_t* d = Ptr(dst, n);
    if (!d) return;
    if (src_sel == 2) {  // fill with the 32-bit value in src
        const uint32_t v = b[1];
        for (uint64_t i = 0; i + 4 <= n; i += 4) memcpy(d + i, &v, 4);
    } else if (const uint8_t* s = Ptr(src, n)) {
        memmove(d, s, n);
    }
}

// --- GPU register state ---
// Context (0xA000+), SH (0x2C00+) and user-config (0xC000+) registers as the command buffers set them,
// directly or through PS5 "indirect" packets that point at (offset, value) pairs in memory.
uint32_t g_cx[0x400], g_sh[0x300], g_uc[0x4000];
// Draw state outside the registers.
uint64_t g_index_base = 0;
uint32_t g_index_type = 0, g_instances = 1;
constexpr uint32_t kRegSelector = 0x70000000u;  // high bits of indirect offsets select the register bank
void SetCx(uint32_t off, uint32_t v) { off &= ~kRegSelector; if (off < 0x400) g_cx[off] = v; }
void SetSh(uint32_t off, uint32_t v) { off &= ~kRegSelector; if (off < 0x300) g_sh[off] = v; }
void SetUc(uint32_t off, uint32_t v) {
    off &= ~kRegSelector;
    if (off < 0x4000) g_uc[off] = v;
    if (off == 0x243) g_index_type = v & 3;  // VGT_INDEX_TYPE
}
void SetIndirect(const uint32_t* b, void (*set)(uint32_t, uint32_t)) {
    const uint64_t pairs = (b[0] & ~3u) | ((uint64_t)b[1] << 32);
    const uint32_t n = b[3] & 0x3fff;
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t off = Rd<uint32_t>(pairs + 8ull * i), v = Rd<uint32_t>(pairs + 8ull * i + 4);
        if (off != 0xffffffffu) set(off, v);
    }
}
uint64_t ShaderAddress(uint32_t lo_reg) { return ((uint64_t)g_sh[lo_reg] << 8) | ((uint64_t)(g_sh[lo_reg + 1] & 0xff) << 40); }

uint32_t IndexType() { return g_index_type; }

// CLEAR_STATE restores the context registers' power-on values; the ones draws depend on aren't zero.
void ResetContext() {
    memset(g_cx, 0, sizeof(g_cx));
    const float one = 1.0f;
    uint32_t one_bits;
    memcpy(&one_bits, &one, 4);
    g_cx[0x202] = 0x00CC0010;  // CB_COLOR_CONTROL: normal mode, ROP3 copy
    for (uint32_t r : {0x10Fu, 0x111u, 0x113u}) g_cx[r] = one_bits;  // viewport scales
    g_cx[0x206] = 0x43F;  // PA_CL_VTE_CNTL: viewport transform enabled
    for (uint32_t r : {0x0Du, 0x82u, 0x91u, 0x95u}) g_cx[r] = 0x40004000;  // scissor bottom-right corners
    for (uint32_t i = 0; i < 8; i++) g_cx[0x1E0 + i] = 0x20010001;  // CB_BLEND_CONTROL: ONE, ONE (disabled)
}
void Draw(uint32_t count, uint64_t index_addr) {
    DrawCall d;
    d.count = count;
    d.index_addr = index_addr;
    d.index_type = IndexType();
    d.instances = g_instances;
    RasterDraw(GpuRegs{g_cx, g_sh, g_uc}, d);
}

// With KYTY_DUMP=<dir> (native runs), the first draws are described in the log and their shaders written to
// <dir>/shader_<address>.bin, for disassembly with llvm-mc -arch=amdgcn -mcpu=gfx1013.
int g_draws_described = 0;
void DumpShader(uint64_t addr) {
    static std::unordered_map<uint64_t, bool> dumped;
    const char* dir = getenv("KYTY_DUMP");
    if (!dir || !addr || dumped[addr]) return;
    dumped[addr] = true;
    uint32_t n = 0;  // up to and including s_endpgm
    while (n < 16384 && Rd<uint32_t>(addr + 4ull * n) != 0xBF810000u) n++;
    char path[512];
    snprintf(path, sizeof(path), "%s/shader_%llx.bin", dir, (unsigned long long)addr);
    if (FILE* f = fopen(path, "wb")) {
        if (const uint8_t* p = Ptr(addr, 4ull * (n + 1))) fwrite(p, 4, n + 1, f);
        fclose(f);
    }
}
void DescribeDraw(const char* what, uint32_t count) {
    if (!getenv("KYTY_DUMP")) return;
    DumpShader(ShaderAddress(0x8));
    DumpShader(ShaderAddress(0xC8));
    DumpShader(ShaderAddress(0x20C));
    if (g_draws_described >= 200) return;
    g_draws_described++;
    const uint64_t gs = ShaderAddress(0x88), ps = ShaderAddress(0x8), es = ShaderAddress(0xC8), hs = ShaderAddress(0x108);
    printf("[gpu] %s count %u prim %u | GS 0x%llx PS 0x%llx ES 0x%llx HS 0x%llx | stages 0x%x | CB0 base 0x%llx view 0x%x info 0x%x attrib 0x%x | target mask 0x%x col fmt 0x%x\n",
           what, count, g_uc[0x242] & 0x3f, (unsigned long long)gs, (unsigned long long)ps, (unsigned long long)es,
           (unsigned long long)hs, g_cx[0x2D5], (unsigned long long)g_cx[0x318] << 8, g_cx[0x31B], g_cx[0x31C],
           g_cx[0x31D], g_cx[0x8E], g_cx[0x1C5]);
    printf("[gpu]   GS user data:");
    for (int i = 0; i < 16; i++) printf(" %08x", g_sh[0x8C + i]);
    printf("\n[gpu]   PS user data:");
    for (int i = 0; i < 16; i++) printf(" %08x", g_sh[0xC + i]);
    printf("\n");
    DumpShader(gs);
    DumpShader(ps);
    DumpShader(es);
    DumpShader(hs);
}

// Each queue runs its submissions in order. A WAIT_REG_MEM on memory whose condition doesn't hold yet stops the
// queue (as the hardware would stall) until GpuPoll() sees the value change, typically a label the CPU writes.
struct Stream { uint64_t addr; uint32_t pos, dw_num; };
struct Submission { std::vector<Stream> stack; };  // innermost indirect buffer last
struct Queue { std::deque<Submission> pending; uint64_t stalled_on = 0; };
std::map<int, Queue> g_queues;  // by interrupt event id: 0 graphics, 0x20.. compute

bool WaitSatisfied(const uint32_t* b, bool wide) {
    const uint32_t control = b[0];
    if (!(control & 0x10)) return true;  // register space isn't modeled
    const uint64_t addr = (b[1] | ((uint64_t)b[2] << 32)) & ~(wide ? 7ull : 3ull);
    const uint64_t value = wide ? Rd<uint64_t>(addr) : Rd<uint32_t>(addr);
    const uint64_t ref = wide ? (b[3] | ((uint64_t)b[4] << 32)) : b[3];
    const uint64_t mask = wide ? (b[5] | ((uint64_t)b[6] << 32)) : b[4];
    const uint64_t v = value & mask;
    switch (control & 7) {
        case 0: return true;
        case 1: return v < ref;
        case 2: return v <= ref;
        case 3: return v == ref;
        case 4: return v != ref;
        case 5: return v >= ref;
        case 6: return v > ref;
        default: return true;
    }
}

// Runs a submission until it ends (true) or stops at an unsatisfied wait (false).
bool Run(Submission& sub, int event_id, Queue& q) {
    while (!sub.stack.empty()) {
        Stream& st = sub.stack.back();
        if (st.pos >= st.dw_num) { sub.stack.pop_back(); continue; }
        const uint64_t at = st.addr + 4ull * st.pos;
        const uint32_t hdr = Rd<uint32_t>(at);
        const uint32_t type = hdr >> 30;
        if (type == 2) { st.pos++; continue; }
        uint32_t len = ((hdr >> 16) & 0x3fff) + 2;
        if (type != 3) { st.pos += len; continue; }
        if ((hdr & 0x3fffff00u) == 0x3fff1000u) len = 1;  // one-dword NOP filler
        const uint32_t* b = reinterpret_cast<const uint32_t*>(Ptr(at + 4, 4ull * (len - 1)));
        if (!b) { sub.stack.pop_back(); continue; }
        const uint32_t op = (hdr >> 8) & 0xff;
        if (g_trace_budget > 0) g_trace_ops += " " + std::to_string(op);
        if (RasterDebugFrame())
            printf("[gpu] queue %d @0x%llx op 0x%02x len %u: %08x %08x %08x %08x\n", event_id, (unsigned long long)at, op, len,
                   len > 1 ? b[0] : 0, len > 2 ? b[1] : 0, len > 3 ? b[2] : 0, len > 4 ? b[3] : 0);
        constexpr uint32_t kFlipMarker = 0x17;
        switch (op) {
            case 0x10:  // NOP; AgcDcbSetFlip marks its flip in the header's register field
                if (((hdr >> 2) & 0x3f) == kFlipMarker && len >= 6) Flip((int32_t)b[1], (int64_t)(b[3] | ((uint64_t)b[4] << 32)), "command buffer");
                break;
            case 0x37: {  // WRITE_DATA
                const uint64_t dst = b[1] | ((uint64_t)b[2] << 32);
                const bool increment = (b[0] >> 16) & 1;
                for (uint32_t i = 3; i < len - 1; i++) Wr<uint32_t>(dst + (increment ? 4ull * (i - 3) : 0), b[i]);
                break;
            }
            case 0x3C: case 0x93:  // WAIT_REG_MEM, WAIT_REG_MEM_64
                if (!WaitSatisfied(b, op == 0x93)) {
                    if (q.stalled_on != at) {
                        q.stalled_on = at;
                        GPU_TRACE("queue %d waits at 0x%llx for [0x%llx] (function %u, reference 0x%x, mask 0x%x)\n",
                                  event_id, (unsigned long long)at, (unsigned long long)(b[1] | ((uint64_t)b[2] << 32)),
                                  b[0] & 7, b[3], op == 0x93 ? b[5] : b[4]);
                    }
                    return false;
                }
                q.stalled_on = 0;
                break;
            case 0x40: CopyData(b); break;
            case 0x50: DmaData(b); break;
            case 0x49: ReleaseMem(b, event_id); break;
            case 0x12: ResetContext(); break;  // CLEAR_STATE
            case 0x69: for (uint32_t i = 1; i + 1 < len; i++) SetCx(b[0] + i - 1, b[i]); break;  // SET_CONTEXT_REG
            case 0x76: for (uint32_t i = 1; i + 1 < len; i++) SetSh(b[0] + i - 1, b[i]); break;  // SET_SH_REG
            case 0x79: case 0x7A: for (uint32_t i = 1; i + 1 < len; i++) SetUc((b[0] & 0xffff) + i - 1, b[i]); break;
            case 0x9F: if (len == 5) SetIndirect(b, SetCx); break;  // SET_CONTEXT_REG_INDIRECT
            case 0x63: if (len == 5) SetIndirect(b, SetSh); break;  // SET_SH_REG_INDIRECT
            case 0x64: if (len == 5) SetIndirect(b, SetUc); break;  // SET_UCONFIG_REG_INDIRECT
            case 0x2D: DescribeDraw("DRAW_INDEX_AUTO", b[0]); Draw(b[0], 0); break;
            case 0x27:  // DRAW_INDEX_2: max size, address, count
                DescribeDraw("DRAW_INDEX_2", b[3]);
                Draw(b[3], b[1] | ((uint64_t)(b[2] & 0xffff) << 32));
                break;
            case 0x3A:  // DRAW_INDEX with instances: count, address, ..., instances
                if (len >= 8) {
                    const uint32_t saved = g_instances;
                    g_instances = b[6];
                    Draw(b[0], b[1] | ((uint64_t)(b[2] & 0xffff) << 32));
                    g_instances = saved;
                }
                break;
            case 0x35:  // DRAW_INDEX_OFFSET_2: max size, offset, count
                Draw(b[2], g_index_base + (uint64_t)b[1] * (IndexType() == 1 ? 4 : IndexType() == 2 ? 1 : 2));
                break;
            case 0x15:  // DISPATCH_DIRECT: x, y, z, initiator
                DescribeDraw("DISPATCH_DIRECT", b[0] * b[1] * b[2]);
                RasterDispatch(GpuRegs{g_cx, g_sh, g_uc}, b[0], b[1], b[2], b[3]);
                break;
            case 0x26: g_index_base = b[0] | ((uint64_t)(b[1] & 0xffff) << 32); break;  // INDEX_BASE
            case 0x2A: g_index_type = b[0] & 3; break;                                  // INDEX_TYPE
            case 0x2F: g_instances = std::max(b[0], 1u); break;                         // NUM_INSTANCES
            case 0x3F: {  // INDIRECT_BUFFER
                const Stream ib{(b[0] & ~3u) | ((uint64_t)(b[1] & 0xffff) << 32), 0, b[2] & 0xfffff};
                st.pos += len;
                if (sub.stack.size() < 8) sub.stack.push_back(ib);
                continue;
            }
            default:
                if (!g_skipped_ops[op]) {
                    g_skipped_ops[op] = true;
                    if (op == 0x2D || op == 0x27 || op == 0x15)
                        printf("[gpu] draw/dispatch packets (first: opcode 0x%02x) are not rendered yet\n", op);
                }
                break;
        }
        st.pos += len;
    }
    return true;
}

void Drain(int event_id, Queue& q) {
    while (!q.pending.empty() && Run(q.pending.front(), event_id, q)) q.pending.pop_front();
}

void Execute(uint64_t addr, uint32_t dw_num, int event_id) {
    Queue& q = g_queues[event_id];
    q.pending.push_back(Submission{{Stream{addr, 0, dw_num}}});
    Drain(event_id, q);
}

// --- AGC driver ---

void SubmitDcb(X86Regs& r) {  // (packet*: {u32* addr; u32 dw_num; u8 flags})
    g_trace_ops.clear();
    Execute(Rd<uint64_t>(r.rdi), Rd<uint32_t>(r.rdi + 8), 0);
    GPU_TRACE("submitted dcb 0x%llx (%u dwords), opcodes:%s\n", (unsigned long long)Rd<uint64_t>(r.rdi),
              Rd<uint32_t>(r.rdi + 8), g_trace_ops.c_str());
    r.rax = 0;
}
void SubmitMultiDcbs(X86Regs& r) {  // (addrs**, sizes*, count)
    for (uint32_t i = 0; i < (uint32_t)r.rdx; i++)
        Execute(Rd<uint64_t>(r.rdi + 8ull * i), Rd<uint32_t>(r.rsi + 4ull * i), 0);
    r.rax = 0;
}
int QueueEventId(uint32_t queue) { return queue >= 0x20 && queue < 0x58 ? (int)queue : 0; }
void SubmitCommandBuffer(X86Regs& r) {  // (queue_context*, packet*)
    const uint32_t queue = Rd<uint32_t>(r.rdi + 4);
    Execute(Rd<uint64_t>(r.rsi), Rd<uint32_t>(r.rsi + 8), QueueEventId(queue));
    r.rax = 0;
}
void SubmitMultiCommandBuffers(X86Regs& r) {  // (queue_context*, buffers**, sizes*, count)
    const int id = QueueEventId(Rd<uint32_t>(r.rdi + 4));
    for (uint32_t i = 0; i < (uint32_t)r.rcx; i++)
        Execute(Rd<uint64_t>(r.rsi + 8ull * i), Rd<uint32_t>(r.rdx + 4ull * i), id);
    r.rax = 0;
}
void SubmitAcb(X86Regs& r) {  // (queue, packet*)
    if (r.rsi) Execute(Rd<uint64_t>(r.rsi), Rd<uint32_t>(r.rsi + 8), QueueEventId((uint32_t)r.rdi));
    r.rax = 0;
}
void SubmitMultiAcbs(X86Regs& r) {  // (queue, acbs**, sizes*, count)
    for (uint32_t i = 0; i < (uint32_t)r.rcx; i++)
        Execute(Rd<uint64_t>(r.rsi + 8ull * i), Rd<uint32_t>(r.rdx + 4ull * i), QueueEventId((uint32_t)r.rdi));
    r.rax = 0;
}
void AddEqEvent(X86Regs& r) {  // (eq, id, udata)
    printf("[gpu] equeue 0x%llx gets graphics event %u (udata 0x%llx)\n", (unsigned long long)r.rdi, (uint32_t)r.rsi,
           (unsigned long long)r.rdx);
    HleEqueueAddEvent(r.rdi, (uint32_t)r.rsi, kEvfiltGraphics, r.rdx);
    r.rax = 0;
}
void DeleteEqEvent(X86Regs& r) { HleEqueueDeleteEvent(r.rdi, (uint32_t)r.rsi, kEvfiltGraphics); r.rax = 0; }
void GetEqEventType(X86Regs& r) {
    Ret(r, Rd<int16_t>(r.rdi + 8) == kEvfiltGraphics ? (int32_t)Rd<uint64_t>(r.rdi) : (int32_t)Rd<int64_t>(r.rdi + 16));
}
void GetEqContextId(X86Regs& r) {
    r.rax = Rd<int16_t>(r.rdi + 8) == kEvfiltGraphics ? (uint32_t)Rd<int64_t>(r.rdi + 16) : (uint32_t)Rd<uint64_t>(r.rdi);
}
void RegisterDefaultOwner(X86Regs& r) { Ret(r, (int32_t)0x8a6c9018u); }

struct Entry { const char* nid; const char* name; HleFn fn; };
const Entry kEntries[] = {
    {"Up36PTk687E", "sceVideoOutOpen", VideoOutOpen},
    {"uquVH4-Du78", "sceVideoOutClose", VideoOutClose},
    {"rKBUtgRrtbk", "sceVideoOutRegisterBuffers2", VideoOutRegisterBuffers2},
    {"N5KDtkIjjJ4", "sceVideoOutUnregisterBuffers", RetZero},
    {"U46NwOiJpys", "sceVideoOutSubmitFlip", VideoOutSubmitFlip},
    {"HXzjK9yI30k", "sceVideoOutAddFlipEvent", VideoOutAddFlipEvent},
    {"SbU3dwp80lQ", "sceVideoOutGetFlipStatus", VideoOutGetFlipStatus},
    {"zgXifHT9ErY", "sceVideoOutIsFlipPending", RetZero},
    {"1FZBKy8HeNU", "sceVideoOutGetVblankStatus", VideoOutGetVblankStatus},
    {"PjS5uASwcV8", "sceVideoOutSetBufferAttribute2", VideoOutSetBufferAttribute2},
    {"CBiu4mCE1DA", "sceVideoOutSetFlipRate", RetZero},
    {"utPrVdxio-8", "sceVideoOutGetOutputStatus", VideoOutGetOutputStatus},
    {"w0hLuNarQxY", "sceVideoOutConfigureOutput", RetZero},
    {"UglJIZjGssM", "sceAgcDriverSubmitDcb", SubmitDcb},
    {"AhGvpITrf4M", "sceAgcDriverSubmitDcb", SubmitDcb},
    {"6UzEidRZwkg", "sceAgcDriverSubmitMultiDcbs", SubmitMultiDcbs},
    {"+T8Xo6LtFJI", "sceAgcDriverSubmitMultiDcbs", SubmitMultiDcbs},
    {"b4fpgH5ZXxQ", "sceAgcDriverSubmitCommandBuffer", SubmitCommandBuffer},
    {"Fj7r9EHzF38", "sceAgcDriverSubmitMultiCommandBuffers", SubmitMultiCommandBuffers},
    {"gSRnr79F8tQ", "sceAgcDriverSubmitAcb", SubmitAcb},
    {"HF3YllT3mXU", "sceAgcDriverSubmitMultiAcbs", SubmitMultiAcbs},
    {"w2rJhmD+dsE", "sceAgcDriverAddEqEvent", AddEqEvent},
    {"DL2RXaXOy88", "sceAgcDriverDeleteEqEvent", DeleteEqEvent},
    {"5CdQTZIQPxM", "sceAgcDriverGetEqEventType", GetEqEventType},
    {"Zw7uUVPulbw", "sceAgcDriverGetEqContextId", GetEqContextId},
    {"XlNp7jzGiPo", "sceAgcDriverSetTFRing", RetZero},
    {"MM4IZSEYytQ", "sceAgcDriverSetHsOffchipParam", RetZero},
    {"Ddwk4gLT5j0", "sceAgcDriverIsCaptureInProgress", RetZero},
    {"U9ueyEhSkF4", "sceAgcDriverRegisterDefaultOwner", RegisterDefaultOwner},
};

const Entry* Lookup(const std::string& nid) {
    for (const auto& e : kEntries)
        if (nid == e.nid) return &e;
    return nullptr;
}

} // namespace

void GpuSetup(const HleLayout& layout) {
    L = layout;
    ResetContext();
    RasterSetup(layout);
}

void GpuPoll() {
    for (auto& [event_id, q] : g_queues)
        if (!q.pending.empty()) Drain(event_id, q);
}

uint64_t GpuFlipCount() { return g_vo.flip_count; }

HleFn GpuFind(const std::string& nid) {
    const Entry* e = Lookup(nid);
    return e ? e->fn : nullptr;
}

const char* GpuName(const std::string& nid) {
    const Entry* e = Lookup(nid);
    return e ? e->name : nullptr;
}

} // namespace KytyWeb
