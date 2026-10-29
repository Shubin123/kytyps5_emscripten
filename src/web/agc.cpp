// libSceAgc (PS5 graphics library): command-buffer builders, shader/state helpers and patch helpers.
// Adapted from KytyPS5 (https://github.com/KytyPS5/KytyPS5, src/libs/agc.cpp), GPL-2.0.
//
// Guest code is interpreted, so guest pointers are not host pointers: every structure the game passes
// lives in guest memory and pointer fields inside it hold guest addresses. Packets are written through a
// host view of guest memory and functions return guest addresses.

#include "hle.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace KytyWeb {
namespace {

HleLayout L;

uint8_t* Host(uint64_t a, uint64_t n) {
    if (a < L.base || n > L.size || a - L.base > L.size - n) return nullptr;
    return L.mem + (a - L.base);
}
template <class T> T Rd(uint64_t a) {
    T v{};
    if (auto* p = Host(a, sizeof(T))) memcpy(&v, p, sizeof(T));
    return v;
}
template <class T> void Wr(uint64_t a, T v) {
    if (auto* p = Host(a, sizeof(T))) memcpy(p, &v, sizeof(T));
}
std::string GuestStr(uint64_t a, size_t max = 4096) {
    std::string s;
    for (auto* p = Host(a, 1); p && *p && s.size() < max; p = Host(++a, 1)) s += static_cast<char>(*p);
    return s;
}
uint64_t Arg(const X86Regs& r, int i) {
    switch (i) {
        case 0: return r.rdi;
        case 1: return r.rsi;
        case 2: return r.rdx;
        case 3: return r.rcx;
        case 4: return r.r8;
        case 5: return r.r9;
        default: return Rd<uint64_t>(r.rsp + 8 + 8 * (i - 6));  // rsp holds the return address
    }
}
void RetInt(X86Regs& r, int v) { r.rax = static_cast<uint64_t>(static_cast<int64_t>(v)); }

#define AGC_ERROR(...)                    \
    do {                                  \
        static bool reported_ = false;    \
        if (!reported_) {                 \
            reported_ = true;             \
            printf("[agc] " __VA_ARGS__); \
        }                                 \
    } while (0)

constexpr int OK = 0;
constexpr int GRAPHICS5_ERROR_INVALID_SHADER_PROGRAM = static_cast<int>(0x8a6c0005u);
constexpr int GRAPHICS5_ERROR_INVALID_SHADER_HALVES = static_cast<int>(0x8a6c0008u);
constexpr int GRAPHICS5_ERROR_INVALID_PACKET = static_cast<int>(0x8a6c000cu);

// --- PM4 ---

namespace Pm4 {
constexpr uint32_t IT_NOP = 0x10;
constexpr uint32_t IT_SET_BASE = 0x11;
constexpr uint32_t IT_CLEAR_STATE = 0x12;
constexpr uint32_t IT_INDEX_BUFFER_SIZE = 0x13;
constexpr uint32_t IT_DISPATCH_DIRECT = 0x15;
constexpr uint32_t IT_DISPATCH_INDIRECT = 0x16;
constexpr uint32_t IT_DRAW_INDIRECT = 0x24;
constexpr uint32_t IT_DRAW_INDEX_INDIRECT = 0x25;
constexpr uint32_t IT_INDEX_BASE = 0x26;
constexpr uint32_t IT_DRAW_INDEX_2 = 0x27;
constexpr uint32_t IT_DRAW_INDEX_AUTO = 0x2D;
constexpr uint32_t IT_NUM_INSTANCES = 0x2F;
constexpr uint32_t IT_WRITE_DATA = 0x37;
constexpr uint32_t IT_WAIT_REG_MEM = 0x3C;
constexpr uint32_t IT_COPY_DATA = 0x40;
constexpr uint32_t IT_PFP_SYNC_ME = 0x42;
constexpr uint32_t IT_EVENT_WRITE = 0x46;
constexpr uint32_t IT_RELEASE_MEM = 0x49;
constexpr uint32_t IT_DMA_DATA = 0x50;
constexpr uint32_t IT_ACQUIRE_MEM = 0x58;
constexpr uint32_t IT_SET_SH_REG_INDIRECT = 0x63;
constexpr uint32_t IT_SET_UCONFIG_REG_INDIRECT = 0x64;
constexpr uint32_t IT_SET_SH_REG = 0x76;
constexpr uint32_t IT_SET_UCONFIG_REG = 0x79;
constexpr uint32_t IT_SET_UCONFIG_REG_INDEX = 0x7A;
constexpr uint32_t IT_WAIT_REG_MEM_64 = 0x93;
constexpr uint32_t IT_SET_CONTEXT_REG_INDIRECT = 0x9F;

constexpr uint32_t R_ZERO = 0x00;
constexpr uint32_t R_WAIT_FLIP_DONE = 0x06;
constexpr uint32_t R_DISPATCH_RESET = 0x09;
constexpr uint32_t R_PUSH_MARKER = 0x0B;
constexpr uint32_t R_POP_MARKER = 0x0C;
constexpr uint32_t R_FLIP = 0x17;
constexpr uint32_t R_NUM = 0x3F + 1;

constexpr uint32_t SPI_PS_INPUT_CNTL_0 = 0x191;
constexpr uint32_t VGT_GS_OUT_PRIM_TYPE = 0x29B;
constexpr uint32_t VGT_SHADER_STAGES_EN = 0x2D5;
constexpr uint32_t SPI_SHADER_PGM_LO_PS = 0x8;
constexpr uint32_t SPI_SHADER_PGM_CHKSUM_GS = 0x80;
constexpr uint32_t SPI_SHADER_PGM_LO_GS = 0x88;
constexpr uint32_t SPI_SHADER_PGM_RSRC1_GS = 0x8A;
constexpr uint32_t SPI_SHADER_PGM_RSRC2_GS = 0x8B;
constexpr uint32_t SPI_SHADER_PGM_LO_ES = 0xC8;
constexpr uint32_t SPI_SHADER_PGM_CHKSUM_HS = 0x100;
constexpr uint32_t SPI_SHADER_PGM_LO_HS = 0x108;
constexpr uint32_t SPI_SHADER_PGM_RSRC1_HS = 0x10A;
constexpr uint32_t SPI_SHADER_PGM_RSRC2_HS = 0x10B;
constexpr uint32_t SPI_SHADER_PGM_LO_LS = 0x148;
constexpr uint32_t COMPUTE_PGM_LO = 0x20C;
constexpr uint32_t VGT_PRIMITIVE_TYPE = 0x242;
constexpr uint32_t VGT_INDEX_TYPE = 0x243;
constexpr uint32_t GE_CNTL = 0x25B;
constexpr uint32_t GE_USER_VGPR_EN = 0x262;
} // namespace Pm4

constexpr uint32_t PM4(uint32_t len, uint32_t op, uint32_t r) {
    return 0xC0000000u | (((static_cast<uint16_t>(len) - 2u) & 0x3fffu) << 16u) | ((op & 0xffu) << 8u) |
           ((r & (Pm4::R_NUM - 1u)) << 2u);
}
constexpr uint32_t PM4_R(uint32_t cmd_id) { return (cmd_id >> 2u) & (Pm4::R_NUM - 1u); }
constexpr uint32_t PM4_LEN(uint32_t cmd_id) { return ((cmd_id >> 16u) & 0x3fffu) + 2u; }

// --- CommandBuffer (guest): bottom @0, top @8, cursor_up @0x10, cursor_down @0x18,
//     callback @0x20, user_data @0x28, reserved_dw (u32) @0x30 ---

constexpr uint64_t kCbCursorUp = 0x10, kCbCursorDown = 0x18, kCbCallback = 0x20, kCbUserData = 0x28,
                   kCbReservedDw = 0x30;

uint64_t AvailableSizeDw(uint64_t buf) {
    const auto distance = static_cast<int64_t>(Rd<uint64_t>(buf + kCbCursorDown) - Rd<uint64_t>(buf + kCbCursorUp));
    const auto available = static_cast<uint64_t>(distance >> 2);
    const uint32_t reserved_dw = Rd<uint32_t>(buf + kCbReservedDw);
    return available <= reserved_dw ? 0 : available - reserved_dw;
}

bool ReserveDw(uint64_t buf, uint32_t num_dw) {
    const uint64_t remaining = AvailableSizeDw(buf);
    if (num_dw <= remaining) return true;
    const uint64_t callback = Rd<uint64_t>(buf + kCbCallback);
    const uint32_t reserved_dw = Rd<uint32_t>(buf + kCbReservedDw);
    if (callback == 0) {
        AGC_ERROR("command buffer exhausted and has no grow callback: requested %u, remaining %llu\n", num_dw,
                  (unsigned long long)remaining);
        return false;
    }
    if ((L.call_guest(callback, buf, num_dw + reserved_dw, Rd<uint64_t>(buf + kCbUserData)) & 0xff) == 0) {
        AGC_ERROR("command buffer grow callback failed: requested %u\n", num_dw);
        return false;
    }
    if (AvailableSizeDw(buf) < num_dw) {
        AGC_ERROR("command buffer grow callback did not provide enough space: requested %u\n", num_dw);
        return false;
    }
    return true;
}

struct Packet {
    uint32_t* p = nullptr;
    uint64_t addr = 0;
    explicit operator bool() const { return p != nullptr; }
    uint32_t& operator[](size_t i) { return p[i]; }
};

Packet AllocateDw(uint64_t buf, uint32_t size_dw) {
    if (size_dw == 0 || !ReserveDw(buf, size_dw)) {
        AGC_ERROR("command buffer AllocateDW failed: size_dw = %u\n", size_dw);
        return {};
    }
    const uint64_t up = Rd<uint64_t>(buf + kCbCursorUp);
    auto* p = reinterpret_cast<uint32_t*>(Host(up, 4ull * size_dw));
    if (!p) {
        AGC_ERROR("command buffer cursor 0x%llx is outside guest memory\n", (unsigned long long)up);
        return {};
    }
    Wr<uint64_t>(buf + kCbCursorUp, up + 4ull * size_dw);
    return {p, up};
}

// Host view of an existing packet in guest memory.
uint32_t* PacketAt(uint64_t cmd, uint32_t size_dw) { return reinterpret_cast<uint32_t*>(Host(cmd, 4ull * size_dw)); }

// --- ShaderRegister arrays and Shader structs in guest memory ---

struct ShaderRegister {
    uint32_t offset;
    uint32_t value;
};

uint32_t RegOffset(uint64_t regs, uint32_t i) { return Rd<uint32_t>(regs + 8ull * i); }
uint32_t RegValue(uint64_t regs, uint32_t i) { return Rd<uint32_t>(regs + 8ull * i + 4); }
void SetReg(uint64_t entry, uint32_t offset, uint32_t value) {
    Wr<uint32_t>(entry, offset);
    Wr<uint32_t>(entry + 4, value);
}
void CopyReg(uint64_t dst, uint64_t src) { Wr<uint64_t>(dst, Rd<uint64_t>(src)); }

// Shader: file_header @0, version @4, user_data @8, code @0x10, cx_registers @0x18, sh_registers @0x20,
// specials @0x28, input_semantics @0x30, output_semantics @0x38, header_size @0x40, shader_size @0x44,
// embedded_constant_buffer_size_dqw @0x48, target @0x4c, num_input_semantics @0x50,
// scratch_size_dw_per_thread @0x54, num_output_semantics @0x56, special_sizes_bytes @0x58, type @0x5a,
// num_cx_registers @0x5b, num_sh_registers @0x5c.
constexpr uint64_t kShFileHeader = 0x00, kShVersion = 0x04, kShUserData = 0x08, kShCode = 0x10, kShCx = 0x18,
                   kShSh = 0x20, kShSpecials = 0x28, kShInputSemantics = 0x30, kShOutputSemantics = 0x38,
                   kShNumInputSemantics = 0x50, kShNumOutputSemantics = 0x56, kShType = 0x5a, kShNumSh = 0x5c,
                   kShaderSize = 0x60;
// ShaderSpecialRegs: ge_cntl @0, vgt_shader_stages_en @8, dispatch_modifier @0x10, user_data_range @0x14,
// draw_modifier @0x18, vgt_gs_out_prim_type @0x20, ge_user_vgpr_en @0x28.
constexpr uint64_t kSpGeCntl = 0x00, kSpVgtShaderStagesEn = 0x08, kSpVgtGsOutPrimType = 0x20, kSpGeUserVgprEn = 0x28;

enum ShaderBinaryType : uint8_t {
    kCs = 0, kPs = 1, kGs = 2, kHs = 3, kGsFront = 4, kHsFront = 5, kGsBack = 6, kHsBack = 7, kFs = 8,
};

bool GetShaderProgramAddressRegister(uint8_t type, uint32_t* lo_offset) {
    switch (type) {
        case kCs: *lo_offset = Pm4::COMPUTE_PGM_LO; return true;
        case kPs: *lo_offset = Pm4::SPI_SHADER_PGM_LO_PS; return true;
        case kGs: *lo_offset = Pm4::SPI_SHADER_PGM_LO_ES; return true;
        case kHs: *lo_offset = Pm4::SPI_SHADER_PGM_LO_LS; return true;
        case kGsBack: *lo_offset = Pm4::SPI_SHADER_PGM_LO_GS; return true;
        case kHsBack: *lo_offset = Pm4::SPI_SHADER_PGM_LO_HS; return true;
        default: return false;
    }
}

int PatchProgramAddressRegister(uint64_t regs, uint32_t num_regs, uint8_t type, uint64_t base) {
    uint32_t lo_offset = 0;
    if (!GetShaderProgramAddressRegister(type, &lo_offset)) return OK;
    if (regs == 0 || num_regs == 0) return GRAPHICS5_ERROR_INVALID_SHADER_PROGRAM;
    for (uint32_t lo = 0; lo < num_regs; lo++) {
        if (RegOffset(regs, lo) != lo_offset) continue;
        const uint32_t hi = lo + 1u;
        if (hi >= num_regs || RegOffset(regs, hi) != lo_offset + 1u) return GRAPHICS5_ERROR_INVALID_SHADER_PROGRAM;
        const uint64_t shader_offset = (static_cast<uint64_t>(RegValue(regs, lo)) << 8u) |
                                       ((static_cast<uint64_t>(RegValue(regs, hi)) & 0xffu) << 40u);
        const uint64_t addr = base + shader_offset;
        Wr<uint32_t>(regs + 8ull * lo + 4, static_cast<uint32_t>((addr >> 8u) & 0xffffffffu));
        Wr<uint32_t>(regs + 8ull * hi + 4,
                     (RegValue(regs, hi) & 0xffffff00u) | static_cast<uint32_t>((addr >> 40u) & 0xffu));
        return OK;
    }
    return GRAPHICS5_ERROR_INVALID_SHADER_PROGRAM;
}

// Returns the guest address of the register entry, or 0.
uint64_t FindShaderRegister(uint64_t regs, uint32_t num_regs, uint32_t offset, uint32_t occurrence = 0) {
    if (regs == 0) return 0;
    for (uint32_t i = 0; i < num_regs; i++) {
        if (RegOffset(regs, i) != offset) continue;
        if (occurrence == 0) return regs + 8ull * i;
        occurrence--;
    }
    return 0;
}

void PatchShaderRegisterAddress(uint64_t regs, uint32_t num_regs, uint32_t lo_offset, uint64_t address) {
    const uint64_t lo = FindShaderRegister(regs, num_regs, lo_offset);
    if (lo == 0) return;
    const uint64_t hi = lo + 8;
    if (!(hi < regs + 8ull * num_regs && Rd<uint32_t>(hi) == lo_offset + 1u)) return;
    Wr<uint32_t>(lo + 4, static_cast<uint32_t>((address >> 8u) & 0xffffffffu));
    Wr<uint32_t>(hi + 4, (Rd<uint32_t>(hi + 4) & 0xffffff00u) | static_cast<uint32_t>((address >> 40u) & 0xffu));
}

// --- register defaults ---

struct CompactRegisterDefaults {
    ShaderRegister* tbl0_regs = nullptr;
    uint32_t tbl0_register_count = 0;
    const uint16_t* tbl0_pointer_offsets = nullptr;
    uint32_t tbl0_pointer_count = 0;
    ShaderRegister* tbl1_regs = nullptr;
    uint32_t tbl1_register_count = 0;
    const uint16_t* tbl1_pointer_offsets = nullptr;
    uint32_t tbl1_pointer_count = 0;
    ShaderRegister* tbl2_regs = nullptr;
    uint32_t tbl2_register_count = 0;
    const uint16_t* tbl2_pointer_offsets = nullptr;
    uint32_t tbl2_pointer_count = 0;
    ShaderRegister* tbl3_regs = nullptr;
    uint32_t tbl3_register_count = 0;
    const uint16_t* tbl3_pointer_offsets = nullptr;
    uint32_t tbl3_pointer_count = 0;
    uint32_t* types = nullptr;
    uint32_t count = 0;
};

#include "agc_register_defaults.inc"

constexpr uint32_t GRAPHICS_REGISTER_DEFAULTS_MAX_VERSION = 13;
constexpr uint32_t GRAPHICS_REGISTER_DEFAULTS_FALLBACK_VERSION = 13;

uint64_t CopyToGuest(const void* src, uint64_t bytes) {
    const uint64_t a = HleMalloc(std::max<uint64_t>(bytes, 8), 16);
    if (auto* p = Host(a, bytes)) memcpy(p, src, bytes);
    return a;
}

// Guest RegisterDefaults: tbl0..tbl3 (ShaderRegister**) @0x00..0x18, tbl0..tbl3 register counts @0x20..0x2c,
// types @0x30, count @0x38.
uint64_t BuildRegisterDefaults(const CompactRegisterDefaults* c) {
    const ShaderRegister* regs[4] = {c->tbl0_regs, c->tbl1_regs, c->tbl2_regs, c->tbl3_regs};
    const uint32_t reg_counts[4] = {c->tbl0_register_count, c->tbl1_register_count, c->tbl2_register_count,
                                    c->tbl3_register_count};
    const uint16_t* offsets[4] = {c->tbl0_pointer_offsets, c->tbl1_pointer_offsets, c->tbl2_pointer_offsets,
                                  c->tbl3_pointer_offsets};
    const uint32_t ptr_counts[4] = {c->tbl0_pointer_count, c->tbl1_pointer_count, c->tbl2_pointer_count,
                                    c->tbl3_pointer_count};
    const uint64_t d = HleMalloc(0x40, 16);
    if (auto* p = Host(d, 0x40)) memset(p, 0, 0x40);
    for (int k = 0; k < 4; k++) {
        Wr<uint32_t>(d + 0x20 + 4 * k, reg_counts[k]);
        if (ptr_counts[k] == 0 || regs[k] == nullptr || offsets[k] == nullptr) continue;
        const uint64_t guest_regs = CopyToGuest(regs[k], 8ull * reg_counts[k]);
        const uint64_t table = HleMalloc(8ull * ptr_counts[k], 16);
        for (uint32_t i = 0; i < ptr_counts[k]; i++) {
            if (offsets[k][i] >= reg_counts[k]) AGC_ERROR("register defaults pointer out of range\n");
            Wr<uint64_t>(table + 8ull * i, guest_regs + 8ull * offsets[k][i]);
        }
        Wr<uint64_t>(d + 8 * k, table);
    }
    // Each type entry is three words.
    if (c->types) Wr<uint64_t>(d + 0x30, CopyToGuest(c->types, 12ull * c->count));
    Wr<uint32_t>(d + 0x38, c->count);
    return d;
}

uint64_t GetRegisterDefaults(bool internal, uint32_t ver) {
    static uint64_t built[2][GRAPHICS_REGISTER_DEFAULTS_MAX_VERSION + 1] = {};
    if (ver > GRAPHICS_REGISTER_DEFAULTS_MAX_VERSION) AGC_ERROR("unsupported register defaults version %u\n", ver);
    const uint32_t index = ver <= GRAPHICS_REGISTER_DEFAULTS_MAX_VERSION ? ver : GRAPHICS_REGISTER_DEFAULTS_FALLBACK_VERSION;
    uint64_t& slot = built[internal ? 1 : 0][index];
    if (!slot)
        slot = BuildRegisterDefaults(internal ? g_agc_internal_reg_defaults_by_version[index]
                                              : g_agc_public_reg_defaults_by_version[index]);
    return slot;
}

// ======================= functions =======================

void AgcInit(X86Regs& r) {
    if ((uint32_t)r.rsi > GRAPHICS_REGISTER_DEFAULTS_MAX_VERSION)
        AGC_ERROR("sceAgcInit: unsupported version %u\n", (uint32_t)r.rsi);
    RetInt(r, OK);
}
void AgcGetRegisterDefaults2(X86Regs& r) { r.rax = GetRegisterDefaults(false, (uint32_t)r.rdi); }
void AgcGetRegisterDefaults2Internal(X86Regs& r) { r.rax = GetRegisterDefaults(true, (uint32_t)r.rdi); }
void AgcSuspendPoint(X86Regs& r) { RetInt(r, OK); }

// Fields in the shader header are offsets relative to the field itself.
void UpdateAddr(uint64_t field) {
    const uint64_t v = Rd<uint64_t>(field);
    if (v != 0) Wr<uint64_t>(field, v + field);
}

void AgcCreateShader(X86Regs& r) {  // (Shader** dst, void* header, const void* code)
    const uint64_t dst = r.rdi, h = r.rsi, code = r.rdx;
    if (!dst || !h || !code) {
        AGC_ERROR("sceAgcCreateShader: null argument\n");
        RetInt(r, GRAPHICS5_ERROR_INVALID_SHADER_PROGRAM);
        return;
    }
    UpdateAddr(h + kShCx);
    UpdateAddr(h + kShSh);
    UpdateAddr(h + kShUserData);
    UpdateAddr(h + kShSpecials);
    UpdateAddr(h + kShInputSemantics);
    UpdateAddr(h + kShOutputSemantics);
    const uint64_t user_data = Rd<uint64_t>(h + kShUserData);
    if (user_data != 0) {
        UpdateAddr(user_data + 0);  // direct_resource_offset
        for (int i = 0; i < 4; i++) UpdateAddr(user_data + 8 + 8 * i);  // sharp_resource_offset[i]
    }
    Wr<uint64_t>(h + kShCode, code);
    if (Rd<uint32_t>(h + kShFileHeader) != 0x34333231 || Rd<uint32_t>(h + kShVersion) != 0x00000018)
        AGC_ERROR("sceAgcCreateShader: unexpected header 0x%08x version 0x%08x\n", Rd<uint32_t>(h + kShFileHeader),
                  Rd<uint32_t>(h + kShVersion));
    if ((code & 0xFFFF0000000000FFull) != 0) AGC_ERROR("sceAgcCreateShader: misaligned code 0x%llx\n", (unsigned long long)code);
    const int patch_result =
        PatchProgramAddressRegister(Rd<uint64_t>(h + kShSh), Rd<uint8_t>(h + kShNumSh), Rd<uint8_t>(h + kShType), code);
    if (patch_result != OK) {
        RetInt(r, patch_result);
        return;
    }
    Wr<uint64_t>(dst, h);
    RetInt(r, OK);
}

void AgcGetFusedShaderSize(X86Regs& r) {  // (SizeAlign* dst, const Shader* front, const Shader* back)
    const uint64_t dst = r.rdi, front = r.rsi, back = r.rdx;
    if (!dst || !front || !back) {
        AGC_ERROR("sceAgcGetFusedShaderSize: null argument\n");
        RetInt(r, GRAPHICS5_ERROR_INVALID_SHADER_HALVES);
        return;
    }
    const uint8_t ft = Rd<uint8_t>(front + kShType), bt = Rd<uint8_t>(back + kShType);
    if (!((ft == kGsFront && bt == kGsBack) || (ft == kHsFront && bt == kHsBack))) {
        RetInt(r, GRAPHICS5_ERROR_INVALID_SHADER_HALVES);
        return;
    }
    Wr<uint64_t>(dst, static_cast<uint64_t>(Rd<uint8_t>(back + kShNumSh)) * sizeof(ShaderRegister));
    Wr<uint64_t>(dst + 8, 4);
    RetInt(r, OK);
}

void MergeShaderRegisterMaxField(uint64_t dst, uint64_t src, uint32_t shift, uint32_t mask) {
    if (!dst || !src) return;
    uint32_t dv = Rd<uint32_t>(dst + 4);
    const uint32_t field = std::max((dv >> shift) & mask, (Rd<uint32_t>(src + 4) >> shift) & mask);
    dv &= ~(mask << shift);
    dv |= field << shift;
    Wr<uint32_t>(dst + 4, dv);
}

int FuseShaderHalves(uint64_t fused, uint64_t front, uint64_t back, uint64_t scratch_mem, bool recompute_shared_vgprs) {
    if (!fused || !front || !back) {
        AGC_ERROR("sceAgcFuseShaderHalves: null argument\n");
        return GRAPHICS5_ERROR_INVALID_SHADER_HALVES;
    }
    const uint8_t front_type = Rd<uint8_t>(front + kShType), back_type = Rd<uint8_t>(back + kShType);
    const bool is_gs = front_type == kGsFront, is_hs = front_type == kHsFront;
    if ((!is_gs && !is_hs) || (is_gs && back_type != kGsBack) || (is_hs && back_type != kHsBack))
        return GRAPHICS5_ERROR_INVALID_SHADER_HALVES;

    auto* back_host = Host(back, kShaderSize);
    auto* fused_host = Host(fused, kShaderSize);
    if (!back_host || !fused_host) return GRAPHICS5_ERROR_INVALID_SHADER_HALVES;
    memmove(fused_host, back_host, kShaderSize);
    Wr<uint8_t>(fused + kShType, is_gs ? kGs : kHs);

    const uint64_t back_specials = Rd<uint64_t>(back + kShSpecials), front_specials = Rd<uint64_t>(front + kShSpecials);
    const uint32_t back_stages = Rd<uint32_t>(back_specials + kSpVgtShaderStagesEn + 4);
    const uint32_t front_stages = Rd<uint32_t>(front_specials + kSpVgtShaderStagesEn + 4);
    const uint32_t mismatch_bit = is_gs ? (1u << 22u) : (1u << 21u);
    if (((front_stages ^ back_stages) & mismatch_bit) != 0) return GRAPHICS5_ERROR_INVALID_SHADER_HALVES;

    if (scratch_mem != 0) {
        const uint64_t bytes = static_cast<uint64_t>(Rd<uint8_t>(back + kShNumSh)) * sizeof(ShaderRegister);
        auto* src = Host(Rd<uint64_t>(back + kShSh), bytes);
        auto* dstp = Host(scratch_mem, bytes);
        if (src && dstp) memmove(dstp, src, bytes);
        Wr<uint64_t>(fused + kShSh, scratch_mem);
    }

    const uint64_t fused_regs = Rd<uint64_t>(fused + kShSh);
    const uint32_t fused_reg_count = Rd<uint8_t>(fused + kShNumSh);
    const uint64_t front_regs = Rd<uint64_t>(front + kShSh);
    const uint32_t front_reg_count = Rd<uint8_t>(front + kShNumSh);
    const uint32_t checksum_offset = is_gs ? Pm4::SPI_SHADER_PGM_CHKSUM_GS : Pm4::SPI_SHADER_PGM_CHKSUM_HS;
    for (uint32_t occurrence = 0; occurrence < 2; occurrence++) {
        const uint64_t src = FindShaderRegister(front_regs, front_reg_count, checksum_offset, occurrence);
        const uint64_t dst = FindShaderRegister(fused_regs, fused_reg_count, checksum_offset, occurrence);
        if (src && dst) Wr<uint32_t>(dst + 4, Rd<uint32_t>(src + 4));
    }

    const uint32_t rsrc1_offset = is_gs ? Pm4::SPI_SHADER_PGM_RSRC1_GS : Pm4::SPI_SHADER_PGM_RSRC1_HS;
    const uint32_t rsrc2_offset = is_gs ? Pm4::SPI_SHADER_PGM_RSRC2_GS : Pm4::SPI_SHADER_PGM_RSRC2_HS;
    const uint64_t front_rsrc1 = FindShaderRegister(front_regs, front_reg_count, rsrc1_offset);
    const uint64_t front_rsrc2 = FindShaderRegister(front_regs, front_reg_count, rsrc2_offset);
    const uint64_t fused_rsrc1 = FindShaderRegister(fused_regs, fused_reg_count, rsrc1_offset);
    const uint64_t fused_rsrc2 = FindShaderRegister(fused_regs, fused_reg_count, rsrc2_offset);
    if (!front_rsrc1 || !front_rsrc2 || !fused_rsrc1 || !fused_rsrc2) {
        AGC_ERROR("sceAgcFuseShaderHalves: missing RSRC registers\n");
        return GRAPHICS5_ERROR_INVALID_SHADER_HALVES;
    }
    auto val = [](uint64_t reg) { return Rd<uint32_t>(reg + 4); };
    auto set = [](uint64_t reg, uint32_t v) { Wr<uint32_t>(reg + 4, v); };

    if (recompute_shared_vgprs) {
        const uint32_t front_vgprs = ((val(front_rsrc1) & 0x3fu) + 1u) * 4u;
        const uint32_t back_vgprs = ((val(fused_rsrc1) & 0x3fu) + 1u) * 4u;
        const uint32_t front_total = front_vgprs + (val(front_rsrc2) >> 28u) * 8u;
        const uint32_t back_total = back_vgprs + (val(fused_rsrc2) >> 28u) * 8u;
        const uint32_t max_total = std::max(front_total, back_total);
        // sceAgcFuseShaderHalves reallocates shared VGPRs; the older export takes the maximum.
        const uint32_t shared = std::max(front_vgprs, back_vgprs) >= max_total
                                    ? 0u
                                    : (max_total - std::min(front_total, back_total) + 7u) / 64u;
        set(fused_rsrc2, (val(fused_rsrc2) & 0x0fffffffu) | ((shared & 0xfu) << 28u));
    } else {
        MergeShaderRegisterMaxField(fused_rsrc2, front_rsrc2, 28, 0x0fu);
    }
    MergeShaderRegisterMaxField(fused_rsrc1, front_rsrc1, 0, 0x3fu);
    if (is_gs) {
        MergeShaderRegisterMaxField(fused_rsrc1, front_rsrc1, 29, 0x03u);
        MergeShaderRegisterMaxField(fused_rsrc2, front_rsrc2, 16, 0x03u);
        set(fused_rsrc2, (val(fused_rsrc2) & 0xfffbffffu) | (val(front_rsrc2) & 0x00040000u));
    } else {
        MergeShaderRegisterMaxField(fused_rsrc1, front_rsrc1, 28, 0x03u);
    }
    set(fused_rsrc2, (val(fused_rsrc2) & 0xf7ffffc1u) | (val(front_rsrc2) & 0x0800003eu));

    PatchShaderRegisterAddress(fused_regs, fused_reg_count, is_gs ? Pm4::SPI_SHADER_PGM_LO_ES : Pm4::SPI_SHADER_PGM_LO_LS,
                               Rd<uint64_t>(front + kShCode));
    Wr<uint64_t>(fused + kShUserData, recompute_shared_vgprs ? 0 : Rd<uint64_t>(front + kShUserData));
    return OK;
}

void AgcFuseShaderHalves_0200(X86Regs& r) { RetInt(r, FuseShaderHalves(r.rdi, r.rsi, r.rdx, r.rcx, true)); }

// ShaderSemantic is a 32-bit bitfield; its packed word equals the raw value in memory.
uint32_t SemanticWord(uint64_t sems, uint32_t i) { return Rd<uint32_t>(sems + 4ull * i); }

uint32_t ApplyInterpolantDefaultValue(uint32_t value, uint32_t ps_word) {
    value &= ~0x00000300u;
    value |= ((ps_word >> 28u) & 0x3u) << 8u;
    return value;
}
uint32_t ApplyInterpolantDefaultValueHi(uint32_t value, uint32_t ps_word) {
    value &= ~0x00600000u;
    value |= ((ps_word >> 30u) & 0x3u) << 21u;
    return value;
}
uint32_t CreateInterpolantMappingValue(uint32_t value, uint32_t ps_word, uint32_t gs_word) {
    const uint32_t flat_shade = ((ps_word & 0x00400000u) != 0 || (ps_word & 0x01000000u) != 0 ? 0x00000400u : 0u);
    value &= ~0x0000001fu;
    value |= (gs_word >> 8u) & 0x1fu;
    value &= ~0x00000400u;
    value |= flat_shade;
    return ApplyInterpolantDefaultValue(value, ps_word);
}
uint32_t CreateInterpolantDefaultValue(uint32_t value, uint32_t ps_word) {
    value &= ~0x0000001fu;
    value &= ~0x00000400u;
    return ApplyInterpolantDefaultValue(value, ps_word);
}
uint32_t CreateInterpolantF16Value(uint32_t ps_word, bool has_gs, uint32_t gs_word) {
    uint32_t value = (ps_word << 4u) & 0x03000000u;
    if (!has_gs) {
        value |= 0x00180020u;
    } else {
        const uint32_t common_word = ps_word & gs_word;
        value &= 0xfff7ffdfu;
        value |= (common_word >> 15u) & 0x20u;
        value ^= 0x00080020u;
        value &= ~0x00100000u;
        value |= (~common_word >> 1u) & 0x00100000u;
    }
    return ApplyInterpolantDefaultValueHi(value, ps_word);
}
uint32_t CreateInterpolantNonF16Value(uint32_t ps_word, bool has_gs) {
    return ((ps_word & 0x01000000u) != 0 || !has_gs) ? 0x20u : 0u;
}
bool FindInterpolantOutputSemantic(uint64_t gs, uint32_t semantic, uint32_t* word) {
    if (gs == 0) return false;
    const uint64_t sems = Rd<uint64_t>(gs + kShOutputSemantics);
    if (sems == 0) return false;
    const uint16_t n = Rd<uint16_t>(gs + kShNumOutputSemantics);
    for (uint16_t i = 0; i < n; i++) {
        const uint32_t w = SemanticWord(sems, i);
        if ((w & 0xffu) == semantic) {
            *word = w;
            return true;
        }
    }
    return false;
}
void SetInterpolantRegister(uint64_t regs, uint32_t index, uint32_t value) {
    SetReg(regs + 8ull * index, Pm4::SPI_PS_INPUT_CNTL_0 + index, value);
}
void SetIdentityInterpolantRegisters(uint64_t regs, uint32_t first_index) {
    for (uint32_t i = first_index; i < 32u; i++) SetInterpolantRegister(regs, i, i);
}

void AgcCreateInterpolantMapping(X86Regs& r) {  // (ShaderRegister* regs, const Shader* gs, const Shader* ps)
    const uint64_t regs = r.rdi, gs = r.rsi, ps = r.rdx;
    if (regs == 0) {
        AGC_ERROR("sceAgcCreateInterpolantMapping: null regs\n");
        RetInt(r, OK);
        return;
    }
    const uint32_t ps_count = ps ? Rd<uint32_t>(ps + kShNumInputSemantics) : 0;
    const uint64_t ps_sems = ps ? Rd<uint64_t>(ps + kShInputSemantics) : 0;
    if (ps_count == 0 || ps_sems == 0) {
        if (ps_count != 0) AGC_ERROR("sceAgcCreateInterpolantMapping: missing input semantics\n");
        SetIdentityInterpolantRegisters(regs, 0);
        RetInt(r, OK);
        return;
    }
    if (gs == 0) AGC_ERROR("sceAgcCreateInterpolantMapping: null gs\n");
    for (uint32_t ps_index = 0; ps_index < ps_count; ps_index++) {
        const uint32_t ps_word = SemanticWord(ps_sems, ps_index);
        uint32_t gs_word = 0;
        const bool has_gs = FindInterpolantOutputSemantic(gs, ps_word & 0xffu, &gs_word);
        uint32_t value = (ps_word & 0x00300000u) != 0 ? CreateInterpolantF16Value(ps_word, has_gs, gs_word)
                                                       : CreateInterpolantNonF16Value(ps_word, has_gs);
        value = has_gs ? CreateInterpolantMappingValue(value, ps_word, gs_word)
                       : CreateInterpolantDefaultValue(value, ps_word);
        SetInterpolantRegister(regs, ps_index, value);
    }
    SetIdentityInterpolantRegisters(regs, ps_count);
    RetInt(r, OK);
}

uint32_t PrimitiveTypeToGsOut(uint32_t prim_type) {
    switch (prim_type) {
        case 1: return 0;                                     // kPointList -> kPoints
        case 2: case 3: case 10: case 11: case 18: return 1;  // line types -> kLines
        case 7: return 3;                                     // kRectList -> k2dRectangle
        case 17: return 4;                                    // kRectListLegacy -> kRectList
        default: return 2;                                    // kTriangles
    }
}

void AgcCreatePrimState(X86Regs& r) {  // (ShaderRegister* cx, ShaderRegister* uc, const Shader* hs, const Shader* gs, u32 prim)
    const uint64_t cx = r.rdi, uc = r.rsi, hs = r.rdx, gs = r.rcx;
    const uint32_t prim_type = static_cast<uint32_t>(r.r8);
    if (cx == 0 && uc == 0) {
        RetInt(r, OK);
        return;
    }
    if (gs == 0) {
        AGC_ERROR("sceAgcCreatePrimState: null gs\n");
        RetInt(r, OK);
        return;
    }
    if (hs != 0 && Rd<uint8_t>(hs + kShType) != kHs) AGC_ERROR("sceAgcCreatePrimState: hs is not a HS shader\n");
    if (Rd<uint8_t>(gs + kShType) != kGs) AGC_ERROR("sceAgcCreatePrimState: gs is not a GS shader\n");
    const uint64_t gs_sp = Rd<uint64_t>(gs + kShSpecials);
    const uint64_t hs_sp = hs ? Rd<uint64_t>(hs + kShSpecials) : 0;

    if (cx != 0) {
        if (Rd<uint32_t>(gs_sp + kSpVgtShaderStagesEn) != Pm4::VGT_SHADER_STAGES_EN ||
            Rd<uint32_t>(gs_sp + kSpVgtGsOutPrimType) != Pm4::VGT_GS_OUT_PRIM_TYPE ||
            (hs && (Rd<uint32_t>(hs_sp + kSpVgtShaderStagesEn) != Pm4::VGT_SHADER_STAGES_EN ||
                    Rd<uint32_t>(hs_sp + kSpVgtGsOutPrimType) != Pm4::VGT_GS_OUT_PRIM_TYPE)))
            AGC_ERROR("sceAgcCreatePrimState: unexpected special register offsets\n");
        CopyReg(cx, gs_sp + kSpVgtShaderStagesEn);
        if ((Rd<uint32_t>(cx + 4) & 0x20u) != 0) CopyReg(cx + 8, gs_sp + kSpVgtGsOutPrimType);
        else SetReg(cx + 8, Pm4::VGT_GS_OUT_PRIM_TYPE, PrimitiveTypeToGsOut(prim_type));
        if (hs != 0) {
            Wr<uint32_t>(cx + 4, Rd<uint32_t>(cx + 4) | Rd<uint32_t>(hs_sp + kSpVgtShaderStagesEn + 4));
            if ((Rd<uint32_t>(cx + 4) & 0x20u) == 0) CopyReg(cx + 8, hs_sp + kSpVgtGsOutPrimType);
        }
    }
    if (uc != 0) {
        if (Rd<uint32_t>(gs_sp + kSpGeCntl) != Pm4::GE_CNTL || Rd<uint32_t>(gs_sp + kSpGeUserVgprEn) != Pm4::GE_USER_VGPR_EN ||
            (hs && Rd<uint32_t>(hs_sp + kSpGeUserVgprEn) != Pm4::GE_USER_VGPR_EN))
            AGC_ERROR("sceAgcCreatePrimState: unexpected uc special register offsets\n");
        CopyReg(uc, gs_sp + kSpGeCntl);
        CopyReg(uc + 8, gs_sp + kSpGeUserVgprEn);
        SetReg(uc + 16, Pm4::VGT_PRIMITIVE_TYPE, prim_type);
        if (hs != 0) CopyReg(uc + 8, hs_sp + kSpGeUserVgprEn);
    }
    RetInt(r, OK);
}

// --- command buffer builders ---

void Out(X86Regs& r, const Packet& p) { r.rax = p.addr; }

void AgcCbDispatch(X86Regs& r) {  // (buf, x, y, z, modifier)
    if (!r.rdi) { r.rax = 0; return; }
    auto cmd = AllocateDw(r.rdi, 5);
    if (!cmd) { r.rax = 0; return; }
    cmd[0] = PM4(5, Pm4::IT_DISPATCH_DIRECT, 0u);
    cmd[1] = (uint32_t)r.rsi;
    cmd[2] = (uint32_t)r.rdx;
    cmd[3] = (uint32_t)r.rcx;
    cmd[4] = ((uint32_t)r.r8 & 0xa038u) | 0x41u;
    Out(r, cmd);
}

void AgcCbReleaseMem(X86Regs& r) {
    const uint64_t buf = Arg(r, 0);
    const uint8_t action = (uint8_t)Arg(r, 1);
    const uint16_t gcr_cntl = (uint16_t)Arg(r, 2);
    const uint8_t dst = (uint8_t)Arg(r, 3), cache_policy = (uint8_t)Arg(r, 4);
    const uint64_t address = Arg(r, 5);
    const uint8_t data_sel = (uint8_t)Arg(r, 6);
    const uint64_t data = Arg(r, 7);
    const uint16_t gds_offset = (uint16_t)Arg(r, 8), gds_size = (uint16_t)Arg(r, 9);
    const uint8_t interrupt = (uint8_t)Arg(r, 10);
    const uint32_t interrupt_ctx_id = (uint32_t)Arg(r, 11);
    if (!buf) { r.rax = 0; return; }
    if (dst > 1 || !(data_sel <= 3 || data_sel == 5) || interrupt > 4)
        AGC_ERROR("sceAgcCbReleaseMem: unsupported dst %u data_sel %u interrupt %u\n", dst, data_sel, interrupt);
    auto cmd = AllocateDw(buf, 8);
    if (!cmd) { r.rax = 0; return; }
    uint32_t packet_gcr_cntl = gcr_cntl;
    if ((packet_gcr_cntl & 0x300u) == 0x100u) packet_gcr_cntl |= 0x200u;
    uint64_t address_value = address, packet_data = data;
    if ((interrupt & 0x7u) == 4u) {
        address_value = 0;
        packet_data = 0;
    } else if ((data_sel & 0x7u) == 5u) {
        packet_data = static_cast<uint64_t>(gds_offset) | (static_cast<uint64_t>(gds_size) << 16u);
    }
    const uint32_t packet_action = static_cast<uint32_t>(action) & 0x3fu;
    const uint32_t packet_event_index = (action >= 0x2fu ? 6u : 5u);
    cmd[0] = PM4(8, Pm4::IT_RELEASE_MEM, 0);
    cmd[1] = packet_action | (packet_event_index << 8u) | ((packet_gcr_cntl & 0xfffu) << 12u) |
             ((static_cast<uint32_t>(cache_policy) & 0x3u) << 25u);
    cmd[2] = ((static_cast<uint32_t>(dst) & 0x3u) << 16u) | ((static_cast<uint32_t>(interrupt) & 0x7u) << 24u) |
             ((static_cast<uint32_t>(data_sel) & 0x7u) << 29u);
    cmd[3] = static_cast<uint32_t>(address_value & 0xfffffffcu);
    cmd[4] = static_cast<uint32_t>((address_value >> 32u) & 0xffffffffu);
    cmd[5] = static_cast<uint32_t>(packet_data & 0xffffffffu);
    cmd[6] = static_cast<uint32_t>((packet_data >> 32u) & 0xffffffffu);
    cmd[7] = interrupt_ctx_id & 0x07ffffffu;
    Out(r, cmd);
}

void AgcCbNop(X86Regs& r) {  // (buf, size_in_dwords)
    const uint32_t size = (uint32_t)r.rsi;
    if (!r.rdi || size == 0) { r.rax = 0; return; }
    auto cmd = AllocateDw(r.rdi, size);
    if (!cmd) { r.rax = 0; return; }
    cmd[0] = PM4(size, Pm4::IT_NOP, Pm4::R_ZERO);
    Out(r, cmd);
}

void AgcCbSetShRegisterRangeDirect(X86Regs& r) {  // (buf, offset, const u32* values, num_values)
    const uint64_t buf = r.rdi, values = r.rdx;
    const uint32_t offset = (uint32_t)r.rsi, num_values = (uint32_t)r.rcx;
    if (!buf) { r.rax = 0; return; }
    auto cmd = AllocateDw(buf, num_values + 2);
    if (!cmd) { r.rax = 0; return; }
    cmd[0] = PM4(num_values + 2, Pm4::IT_SET_SH_REG, 0u);
    cmd[1] = offset & 0xffffu;
    if (values != 0) {
        if (auto* src = Host(values, 4ull * num_values)) memcpy(cmd.p + 2, src, 4ull * num_values);
    }
    Out(r, cmd);
}

void AgcCbSetShRegistersDirect(X86Regs& r) {  // (buf, const ShaderRegister* regs, num_regs)
    const uint64_t buf = r.rdi, regs = r.rsi;
    const uint32_t num_regs = (uint32_t)r.rdx;
    if (num_regs == 0 || !buf || !regs) { r.rax = 0; return; }
    std::vector<ShaderRegister> local(num_regs);
    for (uint32_t i = 0; i < num_regs; i++) local[i] = {RegOffset(regs, i), RegValue(regs, i)};
    uint64_t first_cmd = 0;
    uint32_t run_start_index = 0, run_start_offset = local[0].offset, prev_offset = run_start_offset;
    for (uint32_t i = 1; i <= num_regs; i++) {
        const bool flush = (i == num_regs || local[i].offset != prev_offset + 1u);
        if (flush) {
            const uint32_t run_count = i - run_start_index;
            auto cmd = AllocateDw(buf, run_count + 2u);
            if (!cmd) { r.rax = first_cmd; return; }
            if (first_cmd == 0) first_cmd = cmd.addr;
            cmd[0] = PM4(run_count + 2u, Pm4::IT_SET_SH_REG, 0u);
            cmd[1] = run_start_offset & 0xffffu;
            for (uint32_t j = 0; j < run_count; j++) cmd[j + 2u] = local[run_start_index + j].value;
            if (i < num_regs) {
                run_start_index = i;
                run_start_offset = local[i].offset;
            }
        }
        if (i < num_regs) prev_offset = local[i].offset;
    }
    r.rax = first_cmd;
}

uint64_t DcbAcquireMem(uint64_t buf, uint8_t engine, uint32_t cb_db_op, uint32_t gcr_cntl, uint64_t base,
                       uint64_t size_bytes, uint32_t poll_cycles) {
    if (!buf) return 0;
    auto cmd = AllocateDw(buf, 8);
    if (!cmd) return 0;
    constexpr uint64_t address_mask = 0xffffffffffffull;
    const uint64_t vaddr = base;
    // Cover the byte range in 256-byte units, bounded by the 48-bit address space.
    const uint64_t size =
        std::min((((vaddr & 0xffu) + size_bytes) & address_mask) + 0xffu, (address_mask & ~uint64_t{0xff}) - vaddr) >> 8u;
    cmd[0] = PM4(8, Pm4::IT_ACQUIRE_MEM, 0u);
    cmd[1] = (static_cast<uint32_t>(engine) << 31u) | (cb_db_op & 0x06007fc0u);
    cmd[2] = static_cast<uint32_t>(size);
    cmd[3] = static_cast<uint32_t>((size >> 32u) & 0xffu);
    cmd[4] = static_cast<uint32_t>(vaddr >> 8u);
    cmd[5] = static_cast<uint32_t>(vaddr >> 40u);
    cmd[6] = std::min(poll_cycles >> 4u, 0xffffu);
    cmd[7] = gcr_cntl & 0x7ffffu;
    return cmd.addr;
}
void AgcDcbAcquireMem(X86Regs& r) {  // (buf, engine, cb_db_op, gcr_cntl, base, size_bytes, poll_cycles)
    r.rax = DcbAcquireMem(Arg(r, 0), (uint8_t)Arg(r, 1), (uint32_t)Arg(r, 2), (uint32_t)Arg(r, 3), Arg(r, 4), Arg(r, 5),
                          (uint32_t)Arg(r, 6));
}
void AgcAcbAcquireMem(X86Regs& r) {  // (buf, gcr_cntl, base, size_bytes, poll_cycles)
    r.rax = DcbAcquireMem(r.rdi, 0, 0, (uint32_t)r.rsi, r.rdx, r.rcx, (uint32_t)r.r8);
}

void AgcAcbDispatchIndirect(X86Regs& r) {  // (buf, indirect_args, modifier)
    if (!r.rdi) { r.rax = 0; return; }
    auto cmd = AllocateDw(r.rdi, 4);
    if (!cmd) { r.rax = 0; return; }
    const uint64_t args_addr = r.rsi;
    cmd[0] = PM4(4, Pm4::IT_DISPATCH_INDIRECT, 0u);
    cmd[1] = static_cast<uint32_t>(args_addr & 0xffffffffu);
    cmd[2] = static_cast<uint32_t>((args_addr >> 32u) & 0xffffffffu);
    cmd[3] = ((uint32_t)r.rdx & 0xa038u) | 0x41u;
    Out(r, cmd);
}

void AgcAcbDmaData(X86Regs& r) {  // (buf, dst, dst_cp, dst_addr, src, src_cp, src_addr, num_bytes, wait, write_confirm)
    const uint64_t buf = Arg(r, 0);
    const uint8_t dst = (uint8_t)Arg(r, 1), dst_cache_policy = (uint8_t)Arg(r, 2);
    const uint64_t dst_address_or_offset = Arg(r, 3);
    const uint8_t src = (uint8_t)Arg(r, 4), src_cache_policy = (uint8_t)Arg(r, 5);
    uint64_t src_address = Arg(r, 6);
    const uint32_t num_bytes = (uint32_t)Arg(r, 7);
    const uint8_t wait_for_previous = (uint8_t)Arg(r, 8), write_confirm = (uint8_t)Arg(r, 9);
    if (!buf) { r.rax = 0; return; }
    switch (src) {
        case 0x14: src_address = 0x30174u; break;
        case 0x24: src_address = 0x3017cu; break;
        case 0x25: src_address = 0x30184u; break;
        default: break;
    }
    auto cmd = AllocateDw(buf, 7);
    if (!cmd) { r.rax = 0; return; }
    cmd[0] = PM4(7, Pm4::IT_DMA_DATA, 0u);
    cmd[1] = ((static_cast<uint32_t>(src_cache_policy) & 0x3u) << 13u) | ((static_cast<uint32_t>(dst) & 0x3u) << 20u) |
             ((static_cast<uint32_t>(dst_cache_policy) & 0x3u) << 25u) | ((static_cast<uint32_t>(src) & 0x3u) << 29u);
    cmd[2] = static_cast<uint32_t>(src_address & 0xffffffffu);
    cmd[3] = static_cast<uint32_t>((src_address >> 32u) & 0xffffffffu);
    cmd[4] = static_cast<uint32_t>(dst_address_or_offset & 0xffffffffu);
    cmd[5] = static_cast<uint32_t>((dst_address_or_offset >> 32u) & 0xffffffffu);
    cmd[6] = (num_bytes & 0x03ffffffu) | ((static_cast<uint32_t>(src) & 0x4u) << 24u) |
             ((static_cast<uint32_t>(dst) & 0x4u) << 25u) | ((static_cast<uint32_t>(src) & 0x8u) << 25u) |
             ((static_cast<uint32_t>(dst) & 0x8u) << 26u) | ((static_cast<uint32_t>(wait_for_previous) & 0x1u) << 30u) |
             ((static_cast<uint32_t>(write_confirm) & 0x1u) << 31u);
    Out(r, cmd);
}

void AgcAcbEventWrite(X86Regs& r) {  // (buf, event_type, address)
    const uint8_t event_type = (uint8_t)r.rsi;
    if (!r.rdi) { r.rax = 0; return; }
    auto cmd = AllocateDw(r.rdi, 2);
    if (!cmd) { r.rax = 0; return; }
    cmd[0] = PM4(2, Pm4::IT_EVENT_WRITE, 0u);
    cmd[1] = (static_cast<uint32_t>(event_type) & 0x3fu) | (event_type == 7u ? 0x400u : 0u);
    Out(r, cmd);
}

void AgcAcbWriteData(X86Regs& r) {  // (buf, dst, cache_policy, address, data, num_dwords, increment, write_confirm)
    const uint64_t buf = Arg(r, 0);
    const uint8_t dst = (uint8_t)Arg(r, 1), cache_policy = (uint8_t)Arg(r, 2);
    const uint64_t address_or_offset = Arg(r, 3), data = Arg(r, 4);
    const uint32_t num_dwords = (uint32_t)Arg(r, 5);
    const uint8_t increment = (uint8_t)Arg(r, 6), write_confirm = (uint8_t)Arg(r, 7);
    if (!buf || !data || num_dwords == 0u || num_dwords > 0x3ffdu) { r.rax = 0; return; }
    auto* src = Host(data, 4ull * num_dwords);
    if (!src) { r.rax = 0; return; }
    auto cmd = AllocateDw(buf, 4u + num_dwords);
    if (!cmd) { r.rax = 0; return; }
    cmd[0] = PM4(4u + num_dwords, Pm4::IT_WRITE_DATA, 0u);
    cmd[1] = ((static_cast<uint32_t>(dst) & 0xfu) << 8u) | ((static_cast<uint32_t>(increment) & 0x1u) << 16u) |
             ((dst == 0u ? 0u : static_cast<uint32_t>(write_confirm) & 0x1u) << 20u) |
             ((static_cast<uint32_t>(cache_policy) & 0x3u) << 25u);
    cmd[2] = static_cast<uint32_t>(address_or_offset);
    cmd[3] = static_cast<uint32_t>(address_or_offset >> 32u);
    memcpy(cmd.p + 4, Host(data, 4ull * num_dwords), 4ull * num_dwords);
    Out(r, cmd);
}

// --- WAIT_REG_MEM, wrapped in AGC's internal data markers ---

constexpr uint32_t AGC_INTERNAL_DATA_REGISTER = 0x342u;
constexpr uint32_t AGC_DRAW_INDIRECT_MULTI_BEGIN_TAG = 0xc6000008u;
constexpr uint32_t AGC_DRAW_INDIRECT_MULTI_END_TAG = 0xc6000000u;
constexpr uint32_t AGC_WAIT_USER_DATA_BEGIN_32_TAG = 0xc8010000u;
constexpr uint32_t AGC_WAIT_USER_DATA_BEGIN_64_TAG = 0xc8020000u;
constexpr uint32_t AGC_WAIT_USER_DATA_END_TAG = 0xc8000000u;
constexpr uint32_t AGC_MEM_SEMAPHORE_SIGNAL_BEGIN_TAG = 0xc9010000u;
constexpr uint32_t AGC_MEM_SEMAPHORE_WAIT_BEGIN_TAG = 0xc9020000u;
constexpr uint32_t AGC_MEM_SEMAPHORE_END_TAG = 0xc9000000u;
constexpr uint32_t AGC_WAIT_USER_DATA_BEGIN_DW = 4u;
constexpr uint32_t AGC_INTERNAL_DATA_PACKET_SIZE_DW = 3u;

void WriteAgcInternalDataPacket(uint32_t* cmd, uint32_t tag) {
    cmd[0] = PM4(AGC_INTERNAL_DATA_PACKET_SIZE_DW, Pm4::IT_SET_UCONFIG_REG, 1u);
    cmd[1] = AGC_INTERNAL_DATA_REGISTER;
    cmd[2] = tag;
}

bool AgcIsInternalDataPacket(uint32_t cmd_id, const uint32_t* payload) {
    if (payload == nullptr || ((cmd_id >> 8u) & 0xffu) != Pm4::IT_SET_UCONFIG_REG || PM4_R(cmd_id) != 1u ||
        payload[0] != AGC_INTERNAL_DATA_REGISTER)
        return false;
    const uint32_t size_dw = PM4_LEN(cmd_id);
    if (size_dw == AGC_WAIT_USER_DATA_BEGIN_DW) {
        const uint32_t tag = payload[1] & 0xffff0000u;
        return tag == AGC_WAIT_USER_DATA_BEGIN_32_TAG || tag == AGC_WAIT_USER_DATA_BEGIN_64_TAG ||
               tag == AGC_MEM_SEMAPHORE_SIGNAL_BEGIN_TAG || tag == AGC_MEM_SEMAPHORE_WAIT_BEGIN_TAG;
    }
    return size_dw == AGC_INTERNAL_DATA_PACKET_SIZE_DW &&
           (payload[1] == AGC_WAIT_USER_DATA_END_TAG || payload[1] == AGC_MEM_SEMAPHORE_END_TAG ||
            payload[1] == AGC_DRAW_INDIRECT_MULTI_BEGIN_TAG || payload[1] == AGC_DRAW_INDIRECT_MULTI_END_TAG);
}

uint32_t* GetAgcWaitPacket(uint32_t* cmd) {
    if (cmd == nullptr || PM4_LEN(cmd[0]) != AGC_WAIT_USER_DATA_BEGIN_DW || !AgcIsInternalDataPacket(cmd[0], cmd + 1))
        return nullptr;
    uint32_t* wait = cmd + AGC_WAIT_USER_DATA_BEGIN_DW;
    const uint32_t op = (wait[0] >> 8u) & 0xffu;
    const uint32_t len = PM4_LEN(wait[0]);
    const uint32_t tag = cmd[2] & 0xffff0000u;
    if (PM4_R(wait[0]) != 0u || !((op == Pm4::IT_WAIT_REG_MEM && len == 7u && tag == AGC_WAIT_USER_DATA_BEGIN_32_TAG) ||
                                  (op == Pm4::IT_WAIT_REG_MEM_64 && len == 9u && tag == AGC_WAIT_USER_DATA_BEGIN_64_TAG)))
        return nullptr;
    return wait;
}

uint32_t WaitRegMem32Control(uint8_t compare_function, uint8_t op, uint8_t cache_policy) {
    return 0x10u | (static_cast<uint32_t>(compare_function) & 0x7u) | ((static_cast<uint32_t>(op) & 0x3u) << 8u) |
           ((static_cast<uint32_t>(op) & 0xcu) << 4u) | ((static_cast<uint32_t>(cache_policy) & 0x3u) << 25u);
}
uint32_t WaitRegMem64Control(uint8_t compare_function, uint8_t op, uint8_t cache_policy) {
    return 0x10u | (static_cast<uint32_t>(compare_function) & 0x7u) | ((static_cast<uint32_t>(op) & 0x1u) << 8u) |
           ((static_cast<uint32_t>(op) & 0x6u) << 5u) | ((static_cast<uint32_t>(cache_policy) & 0x3u) << 25u);
}

uint64_t DcbWaitRegMem(uint64_t buf, uint8_t size, uint8_t compare_function, uint8_t op, uint8_t cache_policy,
                       uint64_t address, uint64_t reference, uint64_t mask, uint32_t poll_cycles) {
    if (!buf) return 0;
    if (size != 0 && size != 1) {
        AGC_ERROR("sceAgcDcbWaitRegMem: unsupported size %u\n", size);
        return 0;
    }
    if (cache_policy > 3) AGC_ERROR("sceAgcDcbWaitRegMem: unsupported cache policy %u\n", cache_policy);
    const bool wait32 = (size == 0);
    const uint32_t poll = std::min(poll_cycles >> 4u, 0xffffu);
    const uint32_t wait_dw = wait32 ? 7u : 9u;
    const uint32_t total_dw = AGC_WAIT_USER_DATA_BEGIN_DW + wait_dw + AGC_INTERNAL_DATA_PACKET_SIZE_DW;
    auto cmd = AllocateDw(buf, total_dw);
    if (!cmd) return 0;
    cmd[0] = PM4(AGC_WAIT_USER_DATA_BEGIN_DW, Pm4::IT_SET_UCONFIG_REG, 1u);
    cmd[1] = AGC_INTERNAL_DATA_REGISTER;
    cmd[2] = (wait32 ? AGC_WAIT_USER_DATA_BEGIN_32_TAG : AGC_WAIT_USER_DATA_BEGIN_64_TAG) |
             static_cast<uint32_t>((address >> 32u) & 0xffffu);
    cmd[3] = static_cast<uint32_t>(address);
    uint32_t* wait = cmd.p + AGC_WAIT_USER_DATA_BEGIN_DW;
    WriteAgcInternalDataPacket(wait + wait_dw, AGC_WAIT_USER_DATA_END_TAG);
    if (wait32) {
        wait[0] = PM4(7, Pm4::IT_WAIT_REG_MEM, 0u);
        wait[1] = WaitRegMem32Control(compare_function, op, cache_policy);
        wait[2] = static_cast<uint32_t>(address) & ~0x3u;
        wait[3] = static_cast<uint32_t>(address >> 32u) & 0x3ffffu;
        wait[4] = static_cast<uint32_t>(reference);
        wait[5] = static_cast<uint32_t>(mask);
        wait[6] = poll;
        return cmd.addr;
    }
    wait[0] = PM4(9, Pm4::IT_WAIT_REG_MEM_64, 0u);
    wait[1] = WaitRegMem64Control(compare_function, op, cache_policy);
    wait[2] = static_cast<uint32_t>(address) & ~0x7u;
    wait[3] = static_cast<uint32_t>(address >> 32u) & 0x3ffffu;
    wait[4] = static_cast<uint32_t>(reference);
    wait[5] = static_cast<uint32_t>(reference >> 32u);
    wait[6] = static_cast<uint32_t>(mask);
    wait[7] = static_cast<uint32_t>(mask >> 32u);
    wait[8] = poll;
    return cmd.addr;
}
void AgcDcbWaitRegMem(X86Regs& r) {  // (buf, size, compare, op, cache_policy, address, reference, mask, poll)
    r.rax = DcbWaitRegMem(Arg(r, 0), (uint8_t)Arg(r, 1), (uint8_t)Arg(r, 2), (uint8_t)Arg(r, 3), (uint8_t)Arg(r, 4),
                          Arg(r, 5), Arg(r, 6), Arg(r, 7), (uint32_t)Arg(r, 8));
}
void AgcAcbWaitRegMem(X86Regs& r) {  // (buf, size, compare, cache_policy, address, reference, mask, poll)
    r.rax = DcbWaitRegMem(Arg(r, 0), (uint8_t)Arg(r, 1), (uint8_t)Arg(r, 2), 0, (uint8_t)Arg(r, 3), Arg(r, 4),
                          Arg(r, 5), Arg(r, 6), (uint32_t)Arg(r, 7));
}

void AgcAcbResetQueue(X86Regs& r) {  // (buf, op)
    if (!r.rdi) { r.rax = 0; return; }
    if (((uint32_t)r.rsi & ~0x1c2u) != 0) AGC_ERROR("sceAgcAcbResetQueue: unsupported op 0x%x\n", (uint32_t)r.rsi);
    auto cmd = AllocateDw(r.rdi, 2);
    if (!cmd) { r.rax = 0; return; }
    cmd[0] = PM4(2, Pm4::IT_NOP, Pm4::R_DISPATCH_RESET);
    cmd[1] = 0;
    Out(r, cmd);
}

void AgcDcbCopyData(X86Regs& r) {  // (buf, dst, dst_cp, dst_addr, src, src_cp, src_addr, item_size, write_confirm)
    const uint64_t buf = Arg(r, 0);
    const uint8_t dst = (uint8_t)Arg(r, 1), dst_cache_policy = (uint8_t)Arg(r, 2);
    const uint64_t dst_address = Arg(r, 3);
    const uint8_t src = (uint8_t)Arg(r, 4), src_cache_policy = (uint8_t)Arg(r, 5);
    const uint64_t src_address_or_immediate = Arg(r, 6);
    const uint8_t item_size = (uint8_t)Arg(r, 7), write_confirm = (uint8_t)Arg(r, 8);
    if (!buf) { r.rax = 0; return; }
    auto cmd = AllocateDw(buf, 6);
    if (!cmd) { r.rax = 0; return; }
    cmd[0] = PM4(6, Pm4::IT_COPY_DATA, 0u);
    cmd[1] = ((static_cast<uint32_t>(src) >> 1u) & 0xfu) | (((static_cast<uint32_t>(dst) >> 1u) & 0xfu) << 8u) |
             ((static_cast<uint32_t>(src_cache_policy) & 0x3u) << 13u) |
             ((static_cast<uint32_t>(item_size) & 0x1u) << 16u) | ((static_cast<uint32_t>(write_confirm) & 0x1u) << 20u) |
             ((static_cast<uint32_t>(dst_cache_policy) & 0x3u) << 25u) | ((static_cast<uint32_t>(src) & 0x1u) << 30u);
    cmd[2] = static_cast<uint32_t>(src_address_or_immediate & 0xffffffffu);
    cmd[3] = static_cast<uint32_t>((src_address_or_immediate >> 32u) & 0xffffffffu);
    cmd[4] = static_cast<uint32_t>(dst_address & 0xffffffffu);
    cmd[5] = static_cast<uint32_t>((dst_address >> 32u) & 0xffffffffu);
    Out(r, cmd);
}

void AgcDcbDmaData(X86Regs& r) {
    // (buf, engine, dst, dst_cp, dst_addr, src, src_cp, src_addr, num_bytes, wait, write_confirm, block_engine)
    const uint64_t buf = Arg(r, 0);
    const uint8_t engine = (uint8_t)Arg(r, 1), dst = (uint8_t)Arg(r, 2), dst_cache_policy = (uint8_t)Arg(r, 3);
    const uint64_t dst_address_or_offset = Arg(r, 4);
    const uint8_t src = (uint8_t)Arg(r, 5), src_cache_policy = (uint8_t)Arg(r, 6);
    uint64_t src_address = Arg(r, 7);
    const uint32_t num_bytes = (uint32_t)Arg(r, 8);
    const uint8_t wait_for_previous = (uint8_t)Arg(r, 9), write_confirm = (uint8_t)Arg(r, 10),
                  block_engine = (uint8_t)Arg(r, 11);
    if (!buf) { r.rax = 0; return; }
    switch (src) {
        case 0x14: src_address = (engine == 1u ? 0x30148u : 0x30174u); break;
        case 0x24: src_address = (engine == 1u ? 0x30150u : 0x3017cu); break;
        case 0x64: src_address = (engine == 1u ? 0x30158u : 0x30184u); break;
        default: break;
    }
    auto cmd = AllocateDw(buf, 7);
    if (!cmd) { r.rax = 0; return; }
    cmd[0] = PM4(7, Pm4::IT_DMA_DATA, 0u);
    cmd[1] = (static_cast<uint32_t>(engine) & 0x1u) | ((static_cast<uint32_t>(src_cache_policy) & 0x3u) << 13u) |
             ((static_cast<uint32_t>(dst) & 0x3u) << 20u) | ((static_cast<uint32_t>(dst_cache_policy) & 0x3u) << 25u) |
             ((static_cast<uint32_t>(src) & 0x3u) << 29u) | ((static_cast<uint32_t>(block_engine) & 0x1u) << 31u);
    cmd[2] = static_cast<uint32_t>(src_address & 0xffffffffu);
    cmd[3] = static_cast<uint32_t>((src_address >> 32u) & 0xffffffffu);
    cmd[4] = static_cast<uint32_t>(dst_address_or_offset & 0xffffffffu);
    cmd[5] = static_cast<uint32_t>((dst_address_or_offset >> 32u) & 0xffffffffu);
    cmd[6] = (num_bytes & 0x03ffffffu) | ((static_cast<uint32_t>(src) & 0x4u) << 24u) |
             ((static_cast<uint32_t>(dst) & 0x4u) << 25u) | ((static_cast<uint32_t>(src) & 0x8u) << 25u) |
             ((static_cast<uint32_t>(dst) & 0x8u) << 26u) | ((static_cast<uint32_t>(wait_for_previous) & 0x1u) << 30u) |
             ((static_cast<uint32_t>(write_confirm) & 0x1u) << 31u);
    Out(r, cmd);
}

void AgcDcbWriteData(X86Regs& r) {  // (buf, dst, cache_policy, address, data, num_dwords, increment, write_confirm)
    const uint64_t buf = Arg(r, 0);
    const uint8_t dst = (uint8_t)Arg(r, 1), cache_policy = (uint8_t)Arg(r, 2);
    const uint64_t address_or_offset = Arg(r, 3), data = Arg(r, 4);
    const uint32_t num_dwords = (uint32_t)Arg(r, 5);
    const uint8_t increment = (uint8_t)Arg(r, 6), write_confirm = (uint8_t)Arg(r, 7);
    if (!buf || !data) { r.rax = 0; return; }
    if ((4 + num_dwords - 2u) > 0x3fffu) { r.rax = 0; return; }
    if (!Host(data, 4ull * num_dwords)) { r.rax = 0; return; }
    auto cmd = AllocateDw(buf, 4 + num_dwords);
    if (!cmd) { r.rax = 0; return; }
    const uint32_t dst_value = dst;
    const uint32_t write_confirm_value = (dst == 0u ? 0u : (static_cast<uint32_t>(write_confirm) & 0x1u));
    cmd[0] = PM4(4 + num_dwords, Pm4::IT_WRITE_DATA, 0u);
    cmd[1] = ((dst_value & 0x1u) << 30u) | ((dst_value & 0x1eu) << 7u) | ((static_cast<uint32_t>(increment) & 0x1u) << 16u) |
             (write_confirm_value << 20u) | ((static_cast<uint32_t>(cache_policy) & 0x3u) << 25u);
    cmd[2] = static_cast<uint32_t>(address_or_offset) & ~0x3u;
    cmd[3] = static_cast<uint32_t>(address_or_offset >> 32u);
    memcpy(cmd.p + 4, Host(data, 4ull * num_dwords), 4ull * num_dwords);
    Out(r, cmd);
}

// --- indirect register packets ---

enum class RegIndirectPacket : uint32_t { Cx, Sh, Uc };
constexpr uint32_t RegIndirectPacketSizeDw = 5;

uint32_t RegIndirectNativeOp(RegIndirectPacket type) {
    switch (type) {
        case RegIndirectPacket::Cx: return Pm4::IT_SET_CONTEXT_REG_INDIRECT;
        case RegIndirectPacket::Sh: return Pm4::IT_SET_SH_REG_INDIRECT;
        case RegIndirectPacket::Uc: return Pm4::IT_SET_UCONFIG_REG_INDIRECT;
    }
    return 0;
}
bool IsNativeRegIndirectPacket(const uint32_t* cmd, RegIndirectPacket type) {
    return ((cmd[0] >> 8u) & 0xffu) == RegIndirectNativeOp(type);
}
constexpr bool IsTrinityMode() { return false; }
uint32_t RegIndirectPm4R(RegIndirectPacket type) { return (type == RegIndirectPacket::Uc && IsTrinityMode() ? 1u : 0u); }

void RegIndirectWritePacket(uint32_t* cmd, uint64_t vaddr, uint32_t num_regs, RegIndirectPacket type) {
    cmd[0] = PM4(RegIndirectPacketSizeDw, RegIndirectNativeOp(type), RegIndirectPm4R(type));
    cmd[1] = static_cast<uint32_t>(vaddr) & 0xfffffffcu;
    cmd[2] = static_cast<uint32_t>(vaddr >> 32u);
    cmd[3] = 0x80000000u;
    cmd[4] = num_regs & 0x3fffu;
}

void SetRegistersIndirect(X86Regs& r, RegIndirectPacket type) {  // (buf, regs, num_regs)
    if (!r.rdi) { r.rax = 0; return; }
    auto cmd = AllocateDw(r.rdi, RegIndirectPacketSizeDw);
    if (!cmd) { r.rax = 0; return; }
    RegIndirectWritePacket(cmd.p, r.rsi, (uint32_t)r.rdx, type);
    Out(r, cmd);
}
void AgcDcbSetCxRegistersIndirect(X86Regs& r) { SetRegistersIndirect(r, RegIndirectPacket::Cx); }
void AgcDcbSetShRegistersIndirect(X86Regs& r) { SetRegistersIndirect(r, RegIndirectPacket::Sh); }
void AgcDcbSetUcRegistersIndirect(X86Regs& r) { SetRegistersIndirect(r, RegIndirectPacket::Uc); }

void RegIndirectPatchSetAddress(X86Regs& r, RegIndirectPacket type) {  // (cmd, regs)
    uint32_t* cmd = PacketAt(r.rdi, RegIndirectPacketSizeDw);
    if (!cmd || !r.rsi) {
        AGC_ERROR("RegIndirectPatchSetAddress: null argument\n");
        RetInt(r, GRAPHICS5_ERROR_INVALID_PACKET);
        return;
    }
    if (!IsNativeRegIndirectPacket(cmd, type)) {
        RetInt(r, GRAPHICS5_ERROR_INVALID_PACKET);
        return;
    }
    const uint64_t vaddr = r.rsi;
    cmd[1] = (cmd[1] & 0x3u) | (static_cast<uint32_t>(vaddr) & 0xfffffffcu);
    cmd[2] = static_cast<uint32_t>(vaddr >> 32u);
    RetInt(r, OK);
}
void AgcSetCxRegIndirectPatchSetAddress(X86Regs& r) { RegIndirectPatchSetAddress(r, RegIndirectPacket::Cx); }
void AgcSetShRegIndirectPatchSetAddress(X86Regs& r) { RegIndirectPatchSetAddress(r, RegIndirectPacket::Sh); }
void AgcSetUcRegIndirectPatchSetAddress(X86Regs& r) { RegIndirectPatchSetAddress(r, RegIndirectPacket::Uc); }

void RegIndirectPatchAddRegisters(X86Regs& r, RegIndirectPacket type) {  // (cmd, num_regs)
    uint32_t* cmd = PacketAt(r.rdi, RegIndirectPacketSizeDw);
    if (!cmd) {
        AGC_ERROR("RegIndirectPatchAddRegisters: null cmd\n");
        RetInt(r, GRAPHICS5_ERROR_INVALID_PACKET);
        return;
    }
    if (!IsNativeRegIndirectPacket(cmd, type)) {
        RetInt(r, GRAPHICS5_ERROR_INVALID_PACKET);
        return;
    }
    const uint32_t new_num_regs = ((cmd[4] & 0x3fffu) + (uint32_t)r.rsi) & 0x3fffu;
    cmd[4] = (cmd[4] & ~0x3fffu) | new_num_regs;
    RetInt(r, OK);
}
void AgcSetCxRegIndirectPatchAddRegisters(X86Regs& r) { RegIndirectPatchAddRegisters(r, RegIndirectPacket::Cx); }
void AgcSetShRegIndirectPatchAddRegisters(X86Regs& r) { RegIndirectPatchAddRegisters(r, RegIndirectPacket::Sh); }
void AgcSetUcRegIndirectPatchAddRegisters(X86Regs& r) { RegIndirectPatchAddRegisters(r, RegIndirectPacket::Uc); }

// --- draws and dispatches ---

uint32_t ExtractModifierBits(uint32_t modifier, uint32_t start, uint32_t count) {
    return (modifier >> start) & ((1u << count) - 1u);
}
uint32_t IndirectModifierSgprBase(uint32_t modifier) {
    const uint32_t stage = modifier >> 29u;
    return ((stage == 3u || stage == 5u) ? 0x80u : 0u) + 0x8cu;
}
uint64_t DecodeIndirectModifierPatchOffsets(uint64_t modifier, bool indexed) {
    const auto low = static_cast<uint32_t>(modifier);
    const uint32_t sgpr_base = IndirectModifierSgprBase(low);
    uint64_t base_vtx_loc = 0x280u;
    if ((low & 0x1u) != 0) base_vtx_loc = sgpr_base + ExtractModifierBits(low, 9u, 5u);
    uint64_t start_inst_loc = 0x280u;
    if ((low & 0x4u) != 0) start_inst_loc = sgpr_base + ExtractModifierBits(low, 19u, 5u);
    if (indexed && (low & 0x2u) != 0)
        base_vtx_loc |= static_cast<uint64_t>(sgpr_base + ExtractModifierBits(low, 14u, 5u)) << 16u;
    return base_vtx_loc | (start_inst_loc << 32u);
}
uint32_t DecodeIndirectDrawInitiator(uint64_t modifier) {
    const auto low = static_cast<uint32_t>(modifier);
    uint32_t initiator = 2u;
    if ((modifier & (1ull << 32u)) == 0) initiator = ((low >> 3u) & 0x20u) | 2u;
    return initiator;
}
uint32_t DecodeDrawIndexInitiator(uint64_t modifier) {
    if ((modifier & (1ull << 32u)) != 0) return 0;
    return (static_cast<uint32_t>(modifier) >> 3u) & 0x20u;
}

void AgcDcbDrawIndex(X86Regs& r) {  // (buf, index_count, index_addr, modifier)
    const uint32_t index_count = (uint32_t)r.rsi;
    const uint64_t vaddr = r.rdx, modifier = r.rcx;
    if (!r.rdi) { r.rax = 0; return; }
    if (vaddr == 0 || (vaddr & 1u) != 0) AGC_ERROR("sceAgcDcbDrawIndex: invalid index address 0x%llx\n", (unsigned long long)vaddr);
    auto cmd = AllocateDw(r.rdi, 6);
    if (!cmd) { r.rax = 0; return; }
    cmd[0] = PM4(6, Pm4::IT_DRAW_INDEX_2, 0u);
    cmd[1] = (index_count == 0 ? 1u : index_count);
    cmd[2] = static_cast<uint32_t>(vaddr & 0xffffffffu);
    cmd[3] = static_cast<uint32_t>((vaddr >> 32u) & 0xffffffffu);
    cmd[4] = index_count;
    cmd[5] = DecodeDrawIndexInitiator(modifier);
    Out(r, cmd);
}

void AgcDcbDrawIndexAuto(X86Regs& r) {  // (buf, index_count, modifier)
    if (!r.rdi) { r.rax = 0; return; }
    auto cmd = AllocateDw(r.rdi, 3);
    if (!cmd) { r.rax = 0; return; }
    cmd[0] = PM4(3, Pm4::IT_DRAW_INDEX_AUTO, 0u);
    cmd[1] = (uint32_t)r.rsi;
    cmd[2] = DecodeDrawIndexInitiator(r.rdx) | 0x2u;
    Out(r, cmd);
}

void AgcDcbSetBaseIndirectArgs(X86Regs& r) {  // (buf, shader_type, indirect_base_addr)
    if (!r.rdi) { r.rax = 0; return; }
    auto cmd = AllocateDw(r.rdi, 4);
    if (!cmd) { r.rax = 0; return; }
    const uint64_t addr = r.rdx;
    cmd[0] = PM4(4, Pm4::IT_SET_BASE, 0u) | ((uint32_t)r.rsi << 1u);
    cmd[1] = 1u;
    cmd[2] = static_cast<uint32_t>(addr) & ~0x7u;
    cmd[3] = static_cast<uint32_t>(addr >> 32u);
    Out(r, cmd);
}

void AgcDcbDrawIndexIndirect(X86Regs& r) {  // (buf, data_offset_in_bytes, modifier)
    const uint64_t modifier = r.rdx;
    if (!r.rdi) { r.rax = 0; return; }
    auto cmd = AllocateDw(r.rdi, 5);
    if (!cmd) { r.rax = 0; return; }
    // Single indexed draws store the index-offset enable in word 3, bit 28.
    const uint64_t patch_offsets = DecodeIndirectModifierPatchOffsets(modifier, true) | ((modifier & 0x2ull) << 59u);
    cmd[0] = PM4(5, Pm4::IT_DRAW_INDEX_INDIRECT, 0u);
    cmd[1] = (uint32_t)r.rsi;
    cmd[2] = static_cast<uint32_t>(patch_offsets);
    cmd[3] = static_cast<uint32_t>(patch_offsets >> 32u);
    cmd[4] = DecodeDrawIndexInitiator(modifier);
    Out(r, cmd);
}

void AgcDcbDrawIndirect(X86Regs& r) {  // (buf, data_offset_in_bytes, modifier)
    const uint64_t modifier = r.rdx;
    if (!r.rdi) { r.rax = 0; return; }
    auto cmd = AllocateDw(r.rdi, 5);
    if (!cmd) { r.rax = 0; return; }
    const uint64_t patch_offsets = DecodeIndirectModifierPatchOffsets(modifier, false);
    cmd[0] = PM4(5, Pm4::IT_DRAW_INDIRECT, 0u);
    cmd[1] = (uint32_t)r.rsi;
    cmd[2] = static_cast<uint32_t>(patch_offsets);
    cmd[3] = static_cast<uint32_t>(patch_offsets >> 32u);
    cmd[4] = DecodeIndirectDrawInitiator(modifier);
    Out(r, cmd);
}

void AgcDcbDispatchIndirect(X86Regs& r) {  // (buf, data_offset_in_bytes, flags)
    if (!r.rdi) { r.rax = 0; return; }
    auto cmd = AllocateDw(r.rdi, 3);
    if (!cmd) { r.rax = 0; return; }
    cmd[0] = PM4(3, Pm4::IT_DISPATCH_INDIRECT, 0u);
    cmd[1] = (uint32_t)r.rsi;
    cmd[2] = ((uint32_t)r.rdx & 0xa038u) | 0x41u;
    Out(r, cmd);
}

void AgcDcbEventWrite(X86Regs& r) {  // (buf, event_type, address)
    const uint8_t event_type = (uint8_t)r.rsi;
    if (!r.rdi) { r.rax = 0; return; }
    if (event_type > 0x3fu) AGC_ERROR("sceAgcDcbEventWrite: unsupported event type 0x%x\n", event_type);
    const bool addressed_event = ((event_type & 0xfeu) == 0x38u);
    auto cmd = AllocateDw(r.rdi, addressed_event ? 4u : 2u);
    if (!cmd) { r.rax = 0; return; }
    cmd[0] = PM4(addressed_event ? 4u : 2u, Pm4::IT_EVENT_WRITE, 0u);
    if (event_type == 7u || event_type == 15u || event_type == 16u) {
        cmd[1] = 0x400u | event_type;
    } else if (addressed_event) {
        const uint64_t addr = r.rdx;
        cmd[1] = 0x100u | event_type;
        cmd[2] = static_cast<uint32_t>(addr) & 0xfffffff8u;
        cmd[3] = static_cast<uint32_t>(addr >> 32u);
    } else {
        cmd[1] = event_type & 0x3fu;
    }
    Out(r, cmd);
}

void AgcDcbSetIndexSize(X86Regs& r) {  // (buf, index_size, cache_policy)
    if (!r.rdi) { r.rax = 0; return; }
    auto cmd = AllocateDw(r.rdi, 3);
    if (!cmd) { r.rax = 0; return; }
    cmd[0] = PM4(3, Pm4::IT_SET_UCONFIG_REG_INDEX, 0u);
    cmd[1] = 0x20000000u | Pm4::VGT_INDEX_TYPE;
    cmd[2] = 0x400u | ((uint8_t)r.rsi & 0x3u) | (((uint8_t)r.rdx & 0x3u) << 6u);
    Out(r, cmd);
}

void AgcDcbSetIndexBuffer(X86Regs& r) {  // (buf, index_addr)
    const uint64_t index_addr = r.rsi;
    if (!r.rdi) { r.rax = 0; return; }
    if ((index_addr & 1u) != 0) AGC_ERROR("sceAgcDcbSetIndexBuffer: misaligned index buffer\n");
    auto cmd = AllocateDw(r.rdi, 3);
    if (!cmd) { r.rax = 0; return; }
    cmd[0] = PM4(3, Pm4::IT_INDEX_BASE, 0u);
    cmd[1] = static_cast<uint32_t>(index_addr & 0xffffffffu);
    cmd[2] = static_cast<uint32_t>((index_addr >> 32u) & 0xffffffffu);
    Out(r, cmd);
}

void AgcDcbSetIndexCount(X86Regs& r) {  // (buf, index_count)
    if (!r.rdi) { r.rax = 0; return; }
    auto cmd = AllocateDw(r.rdi, 2);
    if (!cmd) { r.rax = 0; return; }
    cmd[0] = PM4(2, Pm4::IT_INDEX_BUFFER_SIZE, 0u);
    cmd[1] = (uint32_t)r.rsi;
    Out(r, cmd);
}

void AgcDcbSetNumInstances(X86Regs& r) {  // (buf, num_instances)
    if (!r.rdi) { r.rax = 0; return; }
    auto cmd = AllocateDw(r.rdi, 2);
    if (!cmd) { r.rax = 0; return; }
    cmd[0] = PM4(2, Pm4::IT_NUM_INSTANCES, 0u);
    cmd[1] = (uint32_t)r.rsi;
    Out(r, cmd);
}

void AgcDcbStallCommandBufferParser(X86Regs& r) {  // (buf)
    if (!r.rdi) { r.rax = 0; return; }
    auto cmd = AllocateDw(r.rdi, 2);
    if (!cmd) { r.rax = 0; return; }
    cmd[0] = PM4(2, Pm4::IT_PFP_SYNC_ME, 0u);
    cmd[1] = 0;
    Out(r, cmd);
}

void AgcDcbResetQueue(X86Regs& r) {  // (buf, op, state)
    if (!r.rdi) { r.rax = 0; return; }
    if (((uint32_t)r.rsi & ~0xfffu) != 0) AGC_ERROR("sceAgcDcbResetQueue: unsupported op 0x%x\n", (uint32_t)r.rsi);
    auto cmd = AllocateDw(r.rdi, 2);
    if (!cmd) { r.rax = 0; return; }
    cmd[0] = PM4(2, Pm4::IT_CLEAR_STATE, 0u);
    cmd[1] = (uint32_t)r.rdx & 0xfu;
    Out(r, cmd);
}

void AgcDcbWaitUntilSafeForRendering(X86Regs& r) {  // (buf, video_out_handle, display_buffer_index)
    if (!r.rdi) { r.rax = 0; return; }
    auto cmd = AllocateDw(r.rdi, 7);
    if (!cmd) { r.rax = 0; return; }
    cmd[0] = PM4(7, Pm4::IT_NOP, Pm4::R_WAIT_FLIP_DONE);
    cmd[1] = (uint32_t)r.rsi;
    cmd[2] = (uint32_t)r.rdx;
    cmd[3] = cmd[4] = cmd[5] = cmd[6] = 0;
    Out(r, cmd);
}

void AgcDcbPushMarker(X86Regs& r) {  // (buf, const char* str, color)
    if (!r.rdi) { r.rax = 0; return; }
    const std::string str = r.rsi ? GuestStr(r.rsi) : std::string();
    const size_t len = str.size() + 1;
    const auto payload_dwords = static_cast<uint32_t>((len + 3) / 4);
    const uint32_t size = 1u + (payload_dwords == 0 ? 1u : payload_dwords);
    auto cmd = AllocateDw(r.rdi, size);
    if (!cmd) { r.rax = 0; return; }
    cmd[0] = PM4(size, Pm4::IT_NOP, Pm4::R_PUSH_MARKER);
    memset(cmd.p + 1, 0, static_cast<size_t>(size - 1) * sizeof(uint32_t));
    memcpy(cmd.p + 1, str.c_str(), len);
    Out(r, cmd);
}

void AgcDcbPopMarker(X86Regs& r) {  // (buf)
    if (!r.rdi) { r.rax = 0; return; }
    auto cmd = AllocateDw(r.rdi, 2);
    if (!cmd) { r.rax = 0; return; }
    cmd[0] = PM4(2, Pm4::IT_NOP, Pm4::R_POP_MARKER);
    cmd[1] = 0;
    Out(r, cmd);
}

void AgcDcbSetFlip(X86Regs& r) {  // (buf, video_out_handle, display_buffer_index, flip_mode, flip_arg)
    if (!r.rdi) { r.rax = 0; return; }
    auto cmd = AllocateDw(r.rdi, 6);
    if (!cmd) { r.rax = 0; return; }
    const uint64_t flip_arg = r.r8;
    cmd[0] = PM4(6, Pm4::IT_NOP, Pm4::R_FLIP);
    cmd[1] = (uint32_t)r.rsi;
    cmd[2] = (uint32_t)(int32_t)r.rdx;
    cmd[3] = (uint32_t)r.rcx;
    cmd[4] = static_cast<uint32_t>(flip_arg & 0xffffffffu);
    cmd[5] = static_cast<uint32_t>((flip_arg >> 32u) & 0xffffffffu);
    Out(r, cmd);
}

// --- packet patching ---

void AgcGetDataPacketPayloadAddress(X86Regs& r) {  // (uint32_t** addr, uint32_t* cmd, int type)
    const uint64_t addr = r.rdi, cmd = r.rsi;
    const int type = (int)(int32_t)r.rdx;
    if (!addr) {
        AGC_ERROR("sceAgcGetDataPacketPayloadAddress: null addr\n");
        RetInt(r, OK);
        return;
    }
    if (type != 0) {
        Wr<uint64_t>(addr, cmd + 8);
        RetInt(r, OK);
        return;
    }
    const uint32_t cmd_id = Rd<uint32_t>(cmd);
    Wr<uint64_t>(addr, (~cmd_id & 0x3fff0000u) != 0 ? cmd + 4 : 0);
    RetInt(r, OK);
}

void AgcDmaDataPatchSetDstAddressOrOffset(X86Regs& r) {  // (cmd, dst_address_or_offset)
    uint32_t* cmd = PacketAt(r.rdi, 6);
    if (!cmd) {
        AGC_ERROR("sceAgcDmaDataPatchSetDstAddressOrOffset: null cmd\n");
        RetInt(r, GRAPHICS5_ERROR_INVALID_PACKET);
        return;
    }
    if (((cmd[0] >> 8u) & 0xffu) != Pm4::IT_DMA_DATA) {
        RetInt(r, GRAPHICS5_ERROR_INVALID_PACKET);
        return;
    }
    cmd[4] = static_cast<uint32_t>(r.rsi & 0xffffffffu);
    cmd[5] = static_cast<uint32_t>((r.rsi >> 32u) & 0xffffffffu);
    RetInt(r, OK);
}

void AgcWaitRegMemPatchAddress(X86Regs& r) {  // (cmd, address)
    uint32_t* cmd = PacketAt(r.rdi, AGC_WAIT_USER_DATA_BEGIN_DW + 9u);
    uint32_t* wait = GetAgcWaitPacket(cmd);
    if (wait == nullptr) {
        RetInt(r, GRAPHICS5_ERROR_INVALID_PACKET);
        return;
    }
    const uint64_t vaddr = r.rsi;
    const uint32_t op = (wait[0] >> 8u) & 0xffu;
    const uint32_t align_mask = (op == Pm4::IT_WAIT_REG_MEM ? 0x3u : 0x7u);
    cmd[2] = (cmd[2] & 0xffff0000u) | static_cast<uint32_t>((vaddr >> 32u) & 0xffffu);
    cmd[3] = static_cast<uint32_t>(vaddr);
    wait[2] = (wait[2] & align_mask) | (static_cast<uint32_t>(vaddr) & ~align_mask);
    wait[3] = (wait[3] & 0xfffc0000u) | (static_cast<uint32_t>(vaddr >> 32u) & 0x3ffffu);
    RetInt(r, OK);
}

void AgcQueueEndOfPipeActionPatchAddress(X86Regs& r) {  // (cmd, const Label* address)
    uint32_t* cmd = PacketAt(r.rdi, 8);
    if (!cmd) {
        AGC_ERROR("sceAgcQueueEndOfPipeActionPatchAddress: null cmd\n");
        RetInt(r, GRAPHICS5_ERROR_INVALID_PACKET);
        return;
    }
    if (((cmd[0] >> 8u) & 0xffu) != Pm4::IT_RELEASE_MEM) {
        RetInt(r, GRAPHICS5_ERROR_INVALID_PACKET);
        return;
    }
    // A kSendIntCtx release has no address.
    if (((cmd[2] >> 24u) & 0x7u) != 4u) {
        const uint64_t vaddr = r.rsi;
        cmd[3] = (cmd[3] & 0x3u) | (static_cast<uint32_t>(vaddr) & ~0x3u);
        cmd[4] = static_cast<uint32_t>(vaddr >> 32u);
    }
    RetInt(r, OK);
}

struct Export {
    const char* nid;
    const char* name;
    HleFn fn;
};

const Export kExports[] = {
    {"2JtWUUiYBXs", "sceAgcGetRegisterDefaults2", AgcGetRegisterDefaults2},
    {"23LRUSvYu1M", "sceAgcInit", AgcInit},
    {"wRbq6ZjNop4", "sceAgcGetRegisterDefaults2Internal", AgcGetRegisterDefaults2Internal},
    {"h9z6+0hEydk", "sceAgcSuspendPoint", AgcSuspendPoint},
    {"f3dg2CSgRKY", "sceAgcCreateShader", AgcCreateShader},
    {"dolOmWH+huQ", "sceAgcGetFusedShaderSize", AgcGetFusedShaderSize},
    {"fd5Bp5tGTgo", "sceAgcFuseShaderHalves", AgcFuseShaderHalves_0200},
    {"HV4j+E0MBHE", "sceAgcCreateInterpolantMapping", AgcCreateInterpolantMapping},
    {"D9sr1xGUriE", "sceAgcCreatePrimState", AgcCreatePrimState},
    {"k3GhuSNmBLU", "sceAgcCbDispatch", AgcCbDispatch},
    {"wr23dPKyWc0", "sceAgcCbReleaseMem", AgcCbReleaseMem},
    {"LtTouSCZjHM", "sceAgcCbNop", AgcCbNop},
    {"n2fD4A+pb+g", "sceAgcCbSetShRegisterRangeDirect", AgcCbSetShRegisterRangeDirect},
    {"UZbQjYAwwXM", "sceAgcCbSetShRegistersDirect", AgcCbSetShRegistersDirect},
    {"KT-hTp-Ch14", "sceAgcAcbAcquireMem", AgcAcbAcquireMem},
    {"j3EtxFkSIhQ", "sceAgcAcbDispatchIndirect", AgcAcbDispatchIndirect},
    {"-RnpfpxIhec", "sceAgcAcbDmaData", AgcAcbDmaData},
    {"cFazmnXpJOE", "sceAgcAcbEventWrite", AgcAcbEventWrite},
    {"eZ4+17OQz4Q", "sceAgcAcbWriteData", AgcAcbWriteData},
    {"htn36gPnBk4", "sceAgcAcbWaitRegMem", AgcAcbWaitRegMem},
    {"JrtiDtKeS38", "sceAgcAcbResetQueue", AgcAcbResetQueue},
    {"57labkp+rSQ", "sceAgcDcbAcquireMem", AgcDcbAcquireMem},
    {"1rZSWUv1IRc", "sceAgcDcbCopyData", AgcDcbCopyData},
    {"WmAc2MEj6Io", "sceAgcDcbDmaData", AgcDcbDmaData},
    {"i1jyy49AjXU", "sceAgcDcbWriteData", AgcDcbWriteData},
    {"-HOOCn0JY48", "sceAgcDcbSetShRegistersIndirect", AgcDcbSetShRegistersIndirect},
    {"ZvwO9euwYzc", "sceAgcDcbSetCxRegistersIndirect", AgcDcbSetCxRegistersIndirect},
    {"hvUfkUIQcOE", "sceAgcDcbSetUcRegistersIndirect", AgcDcbSetUcRegistersIndirect},
    {"CtB+A9-VxO0", "sceAgcDcbDispatchIndirect", AgcDcbDispatchIndirect},
    {"Yw0jKSqop+E", "sceAgcDcbDrawIndexAuto", AgcDcbDrawIndexAuto},
    {"q88lQ+GP5Yk", "sceAgcDcbDrawIndex", AgcDcbDrawIndex},
    {"1q1titRBL6o", "sceAgcDcbDrawIndirect", AgcDcbDrawIndirect},
    {"t1vNu082-jM", "sceAgcDcbDrawIndexIndirect", AgcDcbDrawIndexIndirect},
    {"aJf+j5yntiU", "sceAgcDcbEventWrite", AgcDcbEventWrite},
    {"RmaJwLtc8rY", "sceAgcDcbSetBaseIndirectArgs", AgcDcbSetBaseIndirectArgs},
    {"GIIW2J37e70", "sceAgcDcbSetIndexSize", AgcDcbSetIndexSize},
    {"l4fM9K-Lyks", "sceAgcDcbSetIndexBuffer", AgcDcbSetIndexBuffer},
    {"8N2tmT3jmC8", "sceAgcDcbSetIndexCount", AgcDcbSetIndexCount},
    {"tSBxhAPyytQ", "sceAgcDcbSetNumInstances", AgcDcbSetNumInstances},
    {"u2T2DiA5hRI", "sceAgcDcbStallCommandBufferParser", AgcDcbStallCommandBufferParser},
    {"VmW0Tdpy420", "sceAgcDcbWaitRegMem", AgcDcbWaitRegMem},
    {"TRO721eVt4g", "sceAgcDcbResetQueue", AgcDcbResetQueue},
    {"MWiElSNE8j8", "sceAgcDcbWaitUntilSafeForRendering", AgcDcbWaitUntilSafeForRendering},
    {"+kSrjIVxKFE", "sceAgcDcbPushMarker", AgcDcbPushMarker},
    {"H7uZqCoNuWk", "sceAgcDcbPopMarker", AgcDcbPopMarker},
    {"YUeqkyT7mEQ", "sceAgcDcbSetFlip", AgcDcbSetFlip},
    {"Qrj4c+61z4A", "sceAgcSetShRegIndirectPatchSetAddress", AgcSetShRegIndirectPatchSetAddress},
    {"z2duB-hHQSM", "sceAgcSetShRegIndirectPatchAddRegisters", AgcSetShRegIndirectPatchAddRegisters},
    {"vcmNN+AAXnY", "sceAgcSetCxRegIndirectPatchSetAddress", AgcSetCxRegIndirectPatchSetAddress},
    {"d-6uF9sZDIU", "sceAgcSetCxRegIndirectPatchAddRegisters", AgcSetCxRegIndirectPatchAddRegisters},
    {"6lNcCp+fxi4", "sceAgcSetUcRegIndirectPatchSetAddress", AgcSetUcRegIndirectPatchSetAddress},
    {"vRoArM9zaIk", "sceAgcSetUcRegIndirectPatchAddRegisters", AgcSetUcRegIndirectPatchAddRegisters},
    {"V++UgBtQhn0", "sceAgcGetDataPacketPayloadAddress", AgcGetDataPacketPayloadAddress},
    {"IxYiarKlXxM", "sceAgcDmaDataPatchSetDstAddressOrOffset", AgcDmaDataPatchSetDstAddressOrOffset},
    {"3KDcnM3lrcU", "sceAgcWaitRegMemPatchAddress", AgcWaitRegMemPatchAddress},
    {"0fWWK5uG9rQ", "sceAgcQueueEndOfPipeActionPatchAddress", AgcQueueEndOfPipeActionPatchAddress},
};

const Export* FindExport(const std::string& nid) {
    for (const auto& e : kExports)
        if (nid == e.nid) return &e;
    return nullptr;
}

} // namespace

void AgcSetup(const HleLayout& layout) { L = layout; }

HleFn AgcFind(const std::string& nid) {
    const Export* e = FindExport(nid);
    return e ? e->fn : nullptr;
}

const char* AgcName(const std::string& nid) {
    const Export* e = FindExport(nid);
    return e ? e->name : nullptr;
}

} // namespace KytyWeb
