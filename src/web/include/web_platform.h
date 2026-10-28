#ifndef KYTY_WEB_PLATFORM_H
#define KYTY_WEB_PLATFORM_H

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

int kyty_init(const char* config_json);
int kyty_load_game(const char* game_path);
void kyty_frame(void);
void kyty_input(const uint8_t* pad_state);
void kyty_audio_callback(float* buffer, int frames);
int kyty_get_screen_width(void);
int kyty_get_screen_height(void);

#ifdef __cplusplus
}
#endif

namespace KytyWeb {

struct GameInfo {
    char title_id[16];
    char title_name[256];
    char content_id[64];
    char content_version[16];
    int screen_width;
    int screen_height;
};

bool ParseParamJson(const char* json_data, size_t len, GameInfo* info);

void InitGraphics(int width, int height);
void ShutdownGraphics();
void PresentFrame(const uint8_t* framebuffer, int width, int height);

void InitAudio(int sample_rate, int channels);
void ShutdownAudio();
void PushAudioSamples(const float* samples, int count);

void InitInput();
void PollInput(uint8_t* pad_state_out);

struct EmuState {
    bool initialized;
    bool game_loaded;
    bool running;
    GameInfo game_info;
    int frame_count;
    double fps;
};

EmuState* GetState();

} // namespace KytyWeb

#endif
