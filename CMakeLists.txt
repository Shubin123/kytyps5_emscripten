cmake_minimum_required(VERSION 3.25)
project(kytyps5_web LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT EMSCRIPTEN)
  message(FATAL_ERROR "This build is Emscripten-only. Use tools/web/build.sh")
endif()

option(KYTY_WEB_DEV "Dev build (faster link, larger wasm)" OFF)

set(KYTY_WEB_SRC "${CMAKE_SOURCE_DIR}/src/web")

include_directories(${KYTY_WEB_SRC}/include)

# --- Web platform sources (self-contained PS5 runtime for WASM) ---

set(WEB_SRC
  ${KYTY_WEB_SRC}/main_web.cpp
  ${KYTY_WEB_SRC}/platform_emscripten.cpp
  ${KYTY_WEB_SRC}/graphics_webgl.cpp
  ${KYTY_WEB_SRC}/audio_web.cpp
  ${KYTY_WEB_SRC}/input_web.cpp
  ${KYTY_WEB_SRC}/x86_interpreter.cpp
  ${KYTY_WEB_SRC}/hle.cpp
)

add_executable(kytyps5 ${WEB_SRC})

# Common compile flags
target_compile_options(kytyps5 PRIVATE
  -pthread
  -msimd128
  -fwasm-exceptions
  -Wall
  -Wno-unused-function
)

# Link flags
set(LINK_FLAGS
  "-sUSE_PTHREADS=1"
  "-sPTHREAD_POOL_SIZE=4"
  "-sALLOW_MEMORY_GROWTH=1"
  "-sINITIAL_MEMORY=268435456"
  "-sMAXIMUM_MEMORY=2147483648"
  "-sSTACK_SIZE=4194304"
  "-sFORCE_FILESYSTEM=1"
  "-sEXPORTED_FUNCTIONS=['_main','_kyty_init','_kyty_load_game','_kyty_frame','_kyty_input','_kyty_input_values','_kyty_audio_callback','_kyty_get_screen_width','_kyty_get_screen_height','_kyty_watch','_malloc','_free']"
  "-sEXPORTED_RUNTIME_METHODS=['ccall','cwrap','FS','MEMFS','setValue','getValue','UTF8ToString','stringToUTF8','addFunction']"
  "-sENVIRONMENT=web,worker"
  "-sMODULARIZE=1"
  "-sEXPORT_NAME=KytyPS5"
  "-sWASM_BIGINT=1"
  "-sUSE_WEBGL2=1"
  "-sFULL_ES3=1"
  "-sMIN_WEBGL_VERSION=2"
  "-sMAX_WEBGL_VERSION=2"
  "-lidbfs.js"
  "-pthread"
)

if(NOT KYTY_WEB_DEV)
  list(APPEND LINK_FLAGS "-O3")
else()
  list(APPEND LINK_FLAGS "-O1" "-gsource-map")
endif()

target_link_options(kytyps5 PRIVATE ${LINK_FLAGS})

set_target_properties(kytyps5 PROPERTIES
  SUFFIX ".js"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
