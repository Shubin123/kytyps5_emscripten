#!/usr/bin/env bash
# Builds the emulator as a native headless program (no browser): build-native/kytyps5 <game dir> [frames].
# Frames the game presents are written as PPM files when KYTY_FRAMES=<dir> is set.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
src="$root/src/web"
mkdir -p "$root/build-native"
"${CXX:-c++}" -std=c++17 -O2 -g -I"$src/include" \
  "$src/main_web.cpp" "$src/platform_emscripten.cpp" "$src/audio_web.cpp" "$src/input_web.cpp" \
  "$src/graphics_native.cpp" "$src/x86_interpreter.cpp" "$src/hle.cpp" "$src/agc.cpp" "$src/gpu.cpp" \
  "$src/ampr.cpp" "$src/unwind.cpp" "$src/rdna.cpp" "$src/raster.cpp" "$src/avplayer.cpp" \
  -o "$root/build-native/kytyps5"
echo "built $root/build-native/kytyps5"
