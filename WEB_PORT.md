# KytyPS5 WebAssembly / Emscripten Port

Backend: https://github.com/Shubin123/kytyps5_emscripten  
Frontend: https://github.com/Shubin123/kytyps5_web  

## Overview

This repository hosts the Emscripten/WebAssembly port of the **KytyPS5** PlayStation 5 emulator. It provides a PS5 runtime environment compiled to WebAssembly with pthread support for execution inside modern web browsers.

Frontend presentation, game file mounting, DualSense input mapping, and browser test harnesses are maintained in the sibling repository [`kytyps5_web`](https://github.com/Shubin123/kytyps5_web).

## Architecture

- **CPU Core**: Software x86-64 interpreter executing PS5 eboot.bin binaries in WASM. The interpreter handles the full x86-64 instruction set needed by IL2CPP/Unity game code, with HLE (High-Level Emulation) interception of PS5 SDK calls.
- **GPU Subsystem**: WebGL2 graphics pipeline. PS5 GNM graphics commands are translated to WebGL2 draw calls with a fullscreen textured-quad presentation path.
- **Audio Engine**: Web Audio API PCM streaming via AudioWorklet, fed from a ring buffer that receives samples from the emulated audio subsystem.
- **Host Bridge**: Emscripten MEMFS/IDBFS virtual filesystem for game file mounting, save-data persistence, and system module loading.
- **HLE Layer**: PS5 system call interception — sceVideoOut, sceGnmDriver, scePad, sceKernel, sceAudio and other SDK functions are implemented as host-side callbacks invoked by the x86 interpreter.

## Credits

- **Kyty** by InoriRus — the upstream PS5 & PS4 emulator whose HLE architecture and ELF loader this port builds on.
- **WebAssembly port** (x86-64 interpreter, WebGL2 renderer, browser runtime) by Shubin123.
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

## Game Compatibility

Tested with:
- **River City Girls** (PPSA04075) — Unity/IL2CPP title

The emulator targets decrypted PS5 game dumps containing:
- `eboot.bin` — main executable (ELF)
- `sce_module/` — system modules (.prx)
- `fakelib/` — stub libraries (.sprx)
- `Media/` — game assets
- `sce_sys/param.json` — game metadata
