// GPU stand-in for submitted AGC command buffers, plus the AGC driver and sceVideoOut entry points.
// Packets the game synchronizes on are executed: memory writes, copies, fills, waits on memory, end-of-pipe
// label writes and interrupts, and flips. Draws and dispatches are skipped, so nothing is rendered yet.
// Packet and struct layouts follow KytyPS5 (https://github.com/KytyPS5/KytyPS5, GPL-2.0).

#include "hle.h"
#include "web_platform.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
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

void Flip(int index, int64_t arg) {
    g_vo.flip_count++;
    g_vo.flip_arg = arg;
    g_vo.current_buffer = index;
    g_vo.flip_time_us = NowUs();
    if (g_vo.flip_count == 1 || g_vo.flip_count % 600 == 0)
        printf("[gpu] flip #%llu (buffer %d, %ux%u)\n", (unsigned long long)g_vo.flip_count, index, g_vo.width,
               g_vo.height);
    // Display buffers are shown as-is: nothing has drawn into them yet, and tiling is not undone.
    constexpr uint32_t kOutW = 1920, kOutH = 1080;  // the canvas texture size
    const uint8_t* p = index >= 0 && index < 16 && g_vo.buffers[index]
                           ? Ptr(g_vo.buffers[index], (uint64_t)g_vo.width * g_vo.height * 4) : nullptr;
    if (p && g_vo.width == kOutW && g_vo.height == kOutH) {
        PresentFrame(p, kOutW, kOutH);
    } else if (p) {
        static std::vector<uint8_t> scaled(kOutW * kOutH * 4);
        for (uint32_t y = 0; y < kOutH; y++)
            for (uint32_t x = 0; x < kOutW; x++)
                memcpy(&scaled[(y * kOutW + x) * 4],
                       p + ((uint64_t)(y * g_vo.height / kOutH) * g_vo.width + x * g_vo.width / kOutW) * 4, 4);
        PresentFrame(scaled.data(), kOutW, kOutH);
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
    Flip((int)r.rsi, (int64_t)r.rcx);
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
void VideoOutGetOutputStatus(X86Regs& r) {
    if (auto* p = Ptr(r.rsi, 0x30)) memset(p, 0, 0x30);
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
        constexpr uint32_t kFlipMarker = 0x17;
        switch (op) {
            case 0x10:  // NOP; AgcDcbSetFlip marks its flip in the header's register field
                if (((hdr >> 2) & 0x3f) == kFlipMarker && len >= 6) Flip((int32_t)b[1], (int64_t)(b[3] | ((uint64_t)b[4] << 32)));
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

void GpuSetup(const HleLayout& layout) { L = layout; }

void GpuPoll() {
    for (auto& [event_id, q] : g_queues)
        if (!q.pending.empty()) Drain(event_id, q);
}

HleFn GpuFind(const std::string& nid) {
    const Entry* e = Lookup(nid);
    return e ? e->fn : nullptr;
}

const char* GpuName(const std::string& nid) {
    const Entry* e = Lookup(nid);
    return e ? e->name : nullptr;
}

} // namespace KytyWeb
