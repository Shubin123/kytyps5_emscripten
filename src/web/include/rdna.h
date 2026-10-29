#ifndef KYTY_RDNA_H
#define KYTY_RDNA_H

// An interpreter for AMD RDNA (gfx10, as in the PS5 GPU) shader programs, one wave64 at a time. It executes
// the scalar and vector ALU, scalar and buffer memory, LDS, image, interpolation and export instructions;
// what a shader can't do on its own (reading guest memory, sampling images, receiving exports) goes through Env.

#include <cstdint>
#include <string>

namespace KytyWeb::Rdna {

constexpr int kLanes = 64;

struct Wave {
    // SGPRs as the ISA numbers scalar operands: s0-s105, vcc 106-107, ttmp 108-123, m0 124, exec 126-127.
    uint32_t s[128] = {};
    uint32_t v[256][kLanes];
    bool scc = false;
    uint64_t pc = 0;
    // Pixel shaders: lanes that are real pixels (not helpers); color and depth exports only write these.
    uint64_t live = ~0ull;
    // The workgroup's LDS (shared by its waves); lds_bytes 0 when the shader has none.
    uint8_t* lds = nullptr;
    uint32_t lds_bytes = 0;
    // Lane index within the workgroup of lane 0, for buffer add_tid addressing.
    uint32_t first_thread = 0;

    uint64_t Exec() const { return s[126] | (uint64_t)s[127] << 32; }
    void SetExec(uint64_t e) { s[126] = (uint32_t)e; s[127] = (uint32_t)(e >> 32); }
    uint64_t Vcc() const { return s[106] | (uint64_t)s[107] << 32; }
    void SetVcc(uint64_t m) { s[106] = (uint32_t)m; s[107] = (uint32_t)(m >> 32); }
};

// Per-lane image sampling request; coordinates are in the shader's units (normalized unless unnormalized).
struct SampleRequest {
    float coord[4] = {};   // s, t, r/slice/face, fragment
    float lod = 0;         // explicit LOD (sample_l) or bias (sample_b)
    float clamp = 0;       // LOD clamp (_cl)
    float compare = 0;     // depth reference (_c)
    float dx[3] = {}, dy[3] = {};  // derivatives of the coordinates across the pixel quad, for implicit LOD
    int offset[3] = {};
    int dim = 1;           // MIMG dim: 0 1D, 1 2D, 2 3D, 3 cube, 4 1D array, 5 2D array, 6 MSAA, 7 MSAA array
    bool has_lod = false, has_bias = false, has_clamp = false, has_compare = false, level_zero = false;
    bool gather = false;   // gather4: out receives the 4 texels' values of the one component in gather_comp
    int gather_comp = 0;
};

struct Env {
    virtual ~Env() = default;
    // Guest memory; nullptr when the range isn't mapped.
    virtual const uint8_t* Read(uint64_t addr, uint64_t size) = 0;
    virtual uint8_t* Write(uint64_t addr, uint64_t size) = 0;
    // T# is 8 dwords, S# 4 dwords; out receives RGBA as floats (or raw integer bits for integer formats).
    virtual void Sample(const uint32_t* t, const uint32_t* s, const SampleRequest& req, float out[4]) = 0;
    virtual void ImageLoad(const uint32_t* t, int dim, const int coord[4], int mip, uint32_t out[4]) = 0;
    virtual void ImageStore(const uint32_t* t, int dim, const int coord[4], int mip, const uint32_t in[4],
                            uint32_t dmask) = 0;
    // image_get_resinfo: width, height, depth/layers, mip levels for the given mip.
    virtual void ResInfo(const uint32_t* t, int mip, uint32_t out[4]) = 0;
    // EXP: target as encoded (mrt 0-7, mrtz 8, null 9, pos 12-15, prim 20, param 32-63). data[c] is the VGPR
    // row of each enabled component (en bits), nullptr otherwise; with compr, data[0] and data[1] hold two
    // packed fp16 each. lanes: the lanes being exported.
    virtual void Export(const Wave& w, int target, uint32_t en, bool compr, bool done, uint64_t lanes,
                        const uint32_t* const data[4]) = 0;
    // Attribute interpolation inputs for the wave's primitive: the attribute's value at the three vertices.
    virtual void Attribute(int attr, int chan, float out[3]) = 0;
    // s_sendmsg (e.g. GS_ALLOC_REQ = 9 with counts in m0).
    virtual void SendMsg(uint32_t msg, uint32_t m0) { (void)msg; (void)m0; }
    // Debugging: log each instruction and the registers it changed in lane trace_lane.
    bool trace = false;
    int trace_lane = 0;
};

struct Result {
    bool ok = true;
    bool barrier = false;   // stopped at s_barrier; Run again to continue past it
    std::string error;
    uint64_t instructions = 0;
};

// Runs the wave from w.pc until s_endpgm, s_barrier (Result::barrier, pc after it) or an unsupported
// instruction (Result::error).
Result Run(Wave& w, Env& env, uint64_t max_instructions = 1u << 24);

// The instruction at addr as hex words, for logs.
std::string Describe(Env& env, uint64_t addr);

// Data formats 1-77 of gfx10's unified buffer/image format numbering: bytes per element (0 if unknown),
// decoding to four dwords (floats, or integers for the integer types; missing components 0, alpha 1) and encoding.
int FormatBytes(uint32_t fmt);
bool DecodeFormat(uint32_t fmt, const uint8_t* p, uint32_t out[4], bool* integer);
bool EncodeFormat(uint32_t fmt, const uint32_t in[4], uint8_t* p);

// Converts an IEEE half to float and back (round to nearest even).
float HalfToFloat(uint16_t h);
uint16_t FloatToHalf(float f);

} // namespace KytyWeb::Rdna

#endif
