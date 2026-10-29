// Presentation for the native headless runner (tools/native): with KYTY_FRAMES=<dir>, presented frames are
// written there as frame_NNNNN.ppm (every one, or every KYTY_FRAME_EVERY-th).

#include "web_platform.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace KytyWeb {

static int g_presented = 0;

void InitGraphics(int width, int height) { printf("[gfx] native presentation %dx%d\n", width, height); }

void ShutdownGraphics() {}

void PresentFrame(const uint8_t* framebuffer, int width, int height) {
    const char* dir = getenv("KYTY_FRAMES");
    const char* every = getenv("KYTY_FRAME_EVERY");
    const int n = g_presented++;
    if (!dir || !framebuffer || (every && atoi(every) > 1 && n % atoi(every) != 0)) return;
    char path[512];
    snprintf(path, sizeof(path), "%s/frame_%05d.ppm", dir, n);
    FILE* f = fopen(path, "wb");
    if (!f) return;
    fprintf(f, "P6\n%d %d\n255\n", width, height);
    for (int i = 0; i < width * height; i++) fwrite(framebuffer + 4 * i, 1, 3, f);  // RGBA -> RGB
    fclose(f);
}

} // namespace KytyWeb
