#!/usr/bin/env bash
# KytyPS5 Emscripten / WebAssembly build script
set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
dev="${KYTY_DEV:-0}"
build_dir="$root/build-web"
dev_option=OFF
if [ "$dev" = 1 ]; then
  build_dir="$root/build-web-dev"
  dev_option=ON
fi

# Check upstream source is present
if [ ! -d "$root/upstream/source" ]; then
  echo "Error: upstream Kyty source not found. Run:"
  echo "  git clone --depth 1 https://github.com/InoriRus/Kyty.git upstream"
  exit 1
fi

# Create empty data directory for preloaded files
mkdir -p "$root/data"

# Activate emsdk
if [ -n "${EMSDK:-}" ] && [ -f "$EMSDK/emsdk_env.sh" ]; then
  source "$EMSDK/emsdk_env.sh" >/dev/null 2>&1
elif [ -f "$HOME/emsdk/emsdk_env.sh" ]; then
  source "$HOME/emsdk/emsdk_env.sh" >/dev/null 2>&1
elif ! command -v emcc >/dev/null 2>&1; then
  echo "Error: emcc not found and emsdk_env.sh not located." >&2
  exit 1
fi
echo "Using $(command -v emcc): $(emcc --version | head -1)"

web_flags="-pthread -msimd128 -fwasm-exceptions"

echo "Configuring KytyPS5 WebAssembly build..."
emcmake cmake -S "$root" -B "$build_dir" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  "-DCMAKE_C_FLAGS=$web_flags" \
  "-DCMAKE_CXX_FLAGS=$web_flags" \
  "-DCMAKE_EXE_LINKER_FLAGS=$web_flags" \
  -DKYTY_WEB_DEV=$dev_option \
  "$@"

[ "${KYTY_CONFIGURE_ONLY:-0}" = 1 ] && exit 0

echo "Building KytyPS5 WebAssembly..."
cmake --build "$build_dir" --target kytyps5 --parallel \
  "${KYTY_BUILD_JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

echo "Build complete. Artifacts in $build_dir/bin/; run 'npm run sync' in ../kytyps5_web to install."
