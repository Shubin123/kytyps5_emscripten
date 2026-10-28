# KytyPS5 Emscripten

PlayStation 5 emulator WebAssembly / Emscripten port — runs PS5 games in the browser.

See [WEB_PORT.md](WEB_PORT.md) for architecture details and build instructions.

## Quick Start

```bash
git clone --depth 1 https://github.com/InoriRus/Kyty.git upstream
bash tools/web/build.sh
cd ../kytyps5_web && npm run sync && npm start
```

## License

Upstream Kyty is MIT-licensed. WebAssembly port additions are MIT-licensed.
