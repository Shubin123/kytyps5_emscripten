# KytyPS5 Emscripten

Experimental WebAssembly / Emscripten port of a PlayStation 5 emulator: an x86-64 interpreter plus
high-level implementations of the PS5 SDK, running in the browser. The frontend lives in
[kytyps5_web](https://github.com/Shubin123/kytyps5_web).

**Status:** River City Girls (PPSA04075, Unity/IL2CPP) boots through Unity's engine and IL2CPP runtime
start-up, runs its job, loading, graphics and FMOD audio threads, and starts loading its first scene. The GPU
is a stand-in that executes the packets games synchronize on (memory writes, waits, end-of-pipe interrupts,
flips) but draws nothing, so the screen stays black. The interpreter runs roughly 7 million guest
instructions per second, far fewer than the game needs to reach its title screen in reasonable time.

See [WEB_PORT.md](WEB_PORT.md) for architecture details and build instructions.

## Quick Start

```bash
git clone --depth 1 https://github.com/InoriRus/Kyty.git upstream
bash tools/web/build.sh
cd ../kytyps5_web && npm run sync && npm start
```

## License

This project is licensed under the [GNU General Public License version 2](LICENSE) (`GPL-2.0-only`).

It is based on [Kyty](https://github.com/InoriRus/Kyty), released under the MIT License; Kyty's original
copyright and license notice are preserved in [`LICENSES/Kyty-MIT.txt`](LICENSES/Kyty-MIT.txt). Parts of the
PS5 graphics (AGC) layer are ported from [KytyPS5](https://github.com/KytyPS5/KytyPS5) (GPL-2.0).
