// Web Audio backend — replaces Kyty's native audio output for Emscripten builds.
// Uses Emscripten's SDL2 audio or direct AudioWorklet-based PCM streaming.

#include "web_platform.h"

#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

#include <emscripten.h>

namespace KytyWeb {

static constexpr int RING_SIZE = 8192;
static float g_ring[RING_SIZE * 2]; // stereo interleaved
static int g_ring_write = 0;
static int g_ring_read = 0;
static std::mutex g_audio_mutex;
static bool g_audio_initialized = false;
static int g_sample_rate = 48000;
static int g_channels = 2;

void InitAudio(int sample_rate, int channels) {
    g_sample_rate = sample_rate;
    g_channels = channels;
    g_ring_write = 0;
    g_ring_read = 0;
    memset(g_ring, 0, sizeof(g_ring));
    g_audio_initialized = true;
    printf("[audio] initialized: %d Hz, %d ch\n", sample_rate, channels);
}

void ShutdownAudio() {
    g_audio_initialized = false;
}

void PushAudioSamples(const float* samples, int count) {
    if (!g_audio_initialized || !samples) return;
    std::lock_guard<std::mutex> lock(g_audio_mutex);
    for (int i = 0; i < count * g_channels; i++) {
        g_ring[(g_ring_write * g_channels + i) % (RING_SIZE * 2)] = samples[i];
    }
    g_ring_write = (g_ring_write + count) % RING_SIZE;
}

} // namespace KytyWeb

extern "C" {

void kyty_audio_callback(float* buffer, int frames) {
    if (!KytyWeb::g_audio_initialized || !buffer) {
        if (buffer) memset(buffer, 0, frames * KytyWeb::g_channels * sizeof(float));
        return;
    }

    std::lock_guard<std::mutex> lock(KytyWeb::g_audio_mutex);
    int ch = KytyWeb::g_channels;

    for (int i = 0; i < frames; i++) {
        if (KytyWeb::g_ring_read != KytyWeb::g_ring_write) {
            for (int c = 0; c < ch; c++) {
                buffer[i * ch + c] = KytyWeb::g_ring[(KytyWeb::g_ring_read * ch + c) % (KytyWeb::RING_SIZE * 2)];
            }
            KytyWeb::g_ring_read = (KytyWeb::g_ring_read + 1) % KytyWeb::RING_SIZE;
        } else {
            for (int c = 0; c < ch; c++) {
                buffer[i * ch + c] = 0.0f;
            }
        }
    }
}

} // extern "C"
