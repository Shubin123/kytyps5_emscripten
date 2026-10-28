// WebGL2 graphics backend — replaces Kyty's Vulkan renderer for Emscripten builds.
// Provides a minimal GPU pipeline that receives PS5 framebuffer data and presents it
// to the HTML5 canvas via WebGL2 textured quads.

#include "web_platform.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <emscripten.h>
#include <emscripten/html5.h>
#include <GLES3/gl3.h>

namespace KytyWeb {

static GLuint g_program = 0;
static GLuint g_vao = 0;
static GLuint g_vbo = 0;
static GLuint g_texture = 0;
static int g_width = 1920;
static int g_height = 1080;
static bool g_initialized = false;

static const char* VERT_SHADER = R"(#version 300 es
precision highp float;
layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(a_pos, 0.0, 1.0);
    v_uv = a_uv;
}
)";

static const char* FRAG_SHADER = R"(#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_texture;
void main() {
    fragColor = texture(u_texture, v_uv);
}
)";

static GLuint CompileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        fprintf(stderr, "[webgl] shader compile error: %s\n", log);
    }
    return s;
}

void InitGraphics(int width, int height) {
    if (g_initialized) return;
    g_width = width;
    g_height = height;

    EmscriptenWebGLContextAttributes attrs;
    emscripten_webgl_init_context_attributes(&attrs);
    attrs.majorVersion = 2;
    attrs.minorVersion = 0;
    attrs.alpha = false;
    attrs.antialias = false;
    attrs.powerPreference = EM_WEBGL_POWER_PREFERENCE_HIGH_PERFORMANCE;

    EMSCRIPTEN_WEBGL_CONTEXT_HANDLE ctx = emscripten_webgl_create_context("#canvas", &attrs);
    if (ctx <= 0) {
        fprintf(stderr, "[webgl] failed to create WebGL2 context: %d\n", ctx);
        return;
    }
    emscripten_webgl_make_context_current(ctx);

    GLuint vs = CompileShader(GL_VERTEX_SHADER, VERT_SHADER);
    GLuint fs = CompileShader(GL_FRAGMENT_SHADER, FRAG_SHADER);
    g_program = glCreateProgram();
    glAttachShader(g_program, vs);
    glAttachShader(g_program, fs);
    glLinkProgram(g_program);
    glDeleteShader(vs);
    glDeleteShader(fs);

    // Fullscreen quad
    float verts[] = {
        -1, -1, 0, 1,
         1, -1, 1, 1,
        -1,  1, 0, 0,
         1,  1, 1, 0,
    };
    glGenVertexArrays(1, &g_vao);
    glBindVertexArray(g_vao);
    glGenBuffers(1, &g_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, (void*)8);

    glGenTextures(1, &g_texture);
    glBindTexture(GL_TEXTURE_2D, g_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Initial black texture
    std::vector<uint8_t> black(width * height * 4, 0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, black.data());

    glViewport(0, 0, width, height);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    g_initialized = true;
    printf("[webgl] initialized %dx%d\n", width, height);
}

void ShutdownGraphics() {
    if (!g_initialized) return;
    glDeleteTextures(1, &g_texture);
    glDeleteBuffers(1, &g_vbo);
    glDeleteVertexArrays(1, &g_vao);
    glDeleteProgram(g_program);
    g_initialized = false;
}

void PresentFrame(const uint8_t* framebuffer, int width, int height) {
    if (!g_initialized) return;

    if (framebuffer) {
        glBindTexture(GL_TEXTURE_2D, g_texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, framebuffer);
    }

    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(g_program);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, g_texture);
    glUniform1i(glGetUniformLocation(g_program, "u_texture"), 0);
    glBindVertexArray(g_vao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

} // namespace KytyWeb
