# KytyPS5 WebAssembly / Emscripten Port

Backend: https://github.com/Shubin123/kytyps5_emscripten  
Frontend: https://github.com/Shubin123/kytyps5_web  

## Overview

This repository hosts the Emscripten/WebAssembly port of the **KytyPS5** PlayStation 5 emulator. It provides a PS5 runtime environment compiled to WebAssembly with pthread support for execution inside modern web browsers.

Frontend presentation, game file mounting, DualSense input mapping, and browser test harnesses are maintained in the sibling repository [`kytyps5_web`](https://github.com/Shubin123/kytyps5_web).

## Architecture

- **CPU Core**: Software x86-64 interpreter executing PS5 eboot.bin binaries in WASM. The interpreter handles the full x86-64 instruction set needed by IL2CPP/Unity game code, with HLE (High-Level Emulation) interception of PS5 SDK calls.
- **GPU Subsystem**: Software rendering. The AGC command buffers are executed by a command processor (`gpu.cpp`); draws and compute dispatches run their RDNA (gfx10, wave64) shaders in an interpreter (`rdna.cpp`) and are rasterized, textured and blended into the game's tiled render targets (`raster.cpp`), with pixel shading spread over several threads. Flipped display buffers are detiled and shown through a WebGL2 textured quad.
- **Audio Engine**: Web Audio API PCM streaming via AudioWorklet, fed from a ring buffer that receives samples from the emulated audio subsystem.
- **Host Bridge**: Emscripten MEMFS/IDBFS virtual filesystem for game file mounting, save-data persistence, and system module loading.
- **HLE Layer**: PS5 system call interception — sceVideoOut, sceGnmDriver, scePad, sceKernel, sceAudio and other SDK functions are implemented as host-side callbacks invoked by the x86 interpreter.

## Credits

- **Kyty** by InoriRus — the upstream PS5 & PS4 emulator whose HLE architecture and ELF loader this port builds on.
- **WebAssembly port** (x86-64 interpreter, software GPU, browser runtime) by Shubin123.
- **KytyPS5** — the AGC layer and surface tiling are ported from it (GPL-2.0).
- **bcdec** by Sergii Kudlai — block-compressed texture decoding (`src/web/third_party/bcdec.h`, MIT).
- **Emscripten** and its contributors.

## Upstream Provenance

- Upstream Source: https://github.com/InoriRus/Kyty
- Tracking branch: `main`
- The upstream source is cloned into `upstream/` and referenced by the build system.

## Build and Setup

### Prerequisites

- **Emscripten SDK**: >= 6.0.9
- **CMake**: >= 3.25
- **Ninja**: >= 1.10
- **Node.js**: >= 20.x

### Building the WebAssembly Core

```bash
# Clone upstream if not already present:
git clone --depth 1 https://github.com/InoriRus/Kyty.git upstream

# Build:
bash tools/web/build.sh
```

### Syncing to Frontend

```bash
cd ../kytyps5_web
npm run sync
npm start
```

Then visit `http://localhost:8080` to launch the browser runner.

### Native Headless Runner

`bash tools/native/build.sh` builds the same emulator as a host program, which is quicker to test and profile:

```bash
KYTY_FRAMES=/tmp/frames KYTY_FRAME_EVERY=30 KYTY_PRESS="90:0x200:4,140:1:4" \
  build-native/kytyps5 "<game folder>" 200000
```

- `KYTY_FRAMES`, `KYTY_FRAME_EVERY`: write every n-th presented frame as a PPM image.
- `KYTY_PRESS="flip:buttons:flips,..."`: hold buttons (bits as in `kytyps5_web/web/input.js`: 1 cross,
  0x200 options) for some frames starting at a presented frame.
- `KYTY_RASTER_THREADS`: pixel-shading threads (default: one per core, at most 8).
- `KYTY_RASTER_DEBUG=n`, `KYTY_RASTER_FRAME=n`, `KYTY_RASTER_EMPTY=n`, `KYTY_TRACE_VS=<addr>`, `KYTY_DUMP=<dir>`:
  draw summaries, one frame's draws, draws that cover no pixels, a vertex shader trace, shader dumps.

`tools/difftest` checks the x86 interpreter instruction by instruction against a real CPU.

## Game Compatibility

Tested with:
- **River City Girls** (PPSA04075) — Unity/IL2CPP title

The emulator targets decrypted PS5 game dumps containing:
- `eboot.bin` — main executable (ELF)
- `sce_module/` — system modules (.prx)
- `fakelib/` — stub libraries (.sprx)
- `Media/` — game assets
- `sce_sys/param.json` — game metadata
