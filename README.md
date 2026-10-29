# KytyPS5 Emscripten

Experimental WebAssembly / Emscripten port of a PlayStation 5 emulator: an x86-64 interpreter plus
high-level implementations of the PS5 SDK, running in the browser. The frontend lives in
[kytyps5_web](https://github.com/Shubin123/kytyps5_web).

**Status:** River City Girls (PPSA04075, Unity/IL2CPP) plays from boot into its first stage: intro movies,
title screen, file, mode and character select, the loading screen and the detention-room fight, all driven by
the controller. Frames are drawn by a software GPU: the game's RDNA shaders run in an interpreter and a
multi-threaded rasterizer textures and blends them into its render targets. There is no video decoder, so
movies play for their real length as black frames. It is slow: about a second per frame natively on an Apple
M4 and two in the browser, and about a quarter of an hour natively from boot to the first stage.

See [WEB_PORT.md](WEB_PORT.md) for architecture details and build instructions.

## Quick Start

```bash
git clone --depth 1 https://github.com/InoriRus/Kyty.git upstream
bash tools/web/build.sh
cd ../kytyps5_web && npm run sync && npm start
```

Or run it natively without a browser (`bash tools/native/build.sh`, then `build-native/kytyps5 <game folder>`;
see [WEB_PORT.md](WEB_PORT.md#native-headless-runner)).

## License

This project is licensed under the [GNU General Public License version 2](LICENSE) (`GPL-2.0-only`).

It is based on [Kyty](https://github.com/InoriRus/Kyty), released under the MIT License; Kyty's original
copyright and license notice are preserved in [`LICENSES/Kyty-MIT.txt`](LICENSES/Kyty-MIT.txt). Parts of the
PS5 graphics (AGC) layer are ported from [KytyPS5](https://github.com/KytyPS5/KytyPS5) (GPL-2.0).
