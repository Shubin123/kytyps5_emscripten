// Browser input backend — maps Gamepad API / keyboard events to PS5 DualSense pad state.
// The frontend JS polls navigator.getGamepads() and posts the state; this module merges it
// with any emscripten_get_gamepad_status calls from the emulator.

#include "web_platform.h"

#include <cstdio>
#include <cstring>
#include <mutex>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/html5.h>
#endif

namespace KytyWeb {

// PS5 DualSense button bit layout (matches Kyty's scePadReadState output)
enum PadButton : uint32_t {
    PAD_L3        = 0x0002,
    PAD_R3        = 0x0004,
    PAD_OPTIONS   = 0x0008,
    PAD_UP        = 0x0010,
    PAD_RIGHT     = 0x0020,
    PAD_DOWN      = 0x0040,
    PAD_LEFT      = 0x0080,
    PAD_L2        = 0x0100,
    PAD_R2        = 0x0200,
    PAD_L1        = 0x0400,
    PAD_R1        = 0x0800,
    PAD_TRIANGLE  = 0x1000,
    PAD_CIRCLE    = 0x2000,
    PAD_CROSS     = 0x4000,
    PAD_SQUARE    = 0x8000,
    PAD_TOUCHPAD  = 0x100000,
};

#pragma pack(push, 1)
struct PadState {
    uint32_t buttons;
    uint8_t lx, ly;   // left stick 0-255, 128=center
    uint8_t rx, ry;   // right stick
    uint8_t l2, r2;   // trigger analog
    uint8_t _pad[2];
};
#pragma pack(pop)

static PadState g_pad = {};
static std::mutex g_input_mutex;
static bool g_initialized = false;

void InitInput() {
    memset(&g_pad, 0, sizeof(g_pad));
    g_pad.lx = g_pad.ly = g_pad.rx = g_pad.ry = 128;
    g_initialized = true;
    printf("[input] initialized\n");
}

void PollInput(uint8_t* pad_state_out) {
    if (!g_initialized || !pad_state_out) return;
    std::lock_guard<std::mutex> lock(g_input_mutex);
    memcpy(pad_state_out, &g_pad, sizeof(g_pad));
}

} // namespace KytyWeb

extern "C" {

// Called from JS with packed pad state (12 bytes matching PadState)
void kyty_input(const uint8_t* pad_state) {
    if (!pad_state) return;
    std::lock_guard<std::mutex> lock(KytyWeb::g_input_mutex);
    memcpy(&KytyWeb::g_pad, pad_state, sizeof(KytyWeb::PadState));
}

} // extern "C"
