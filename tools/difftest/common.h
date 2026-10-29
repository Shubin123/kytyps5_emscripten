// Shared layout for the differential instruction tests (cases.py writes Case records).
#pragma once
#include <stdint.h>

#define CODE_ADDR 0x300000000ull
#define DATA_ADDR 0x300100000ull
#define DATA_SIZE 0x100000ull
#define MAX_CHANGES 64

#pragma pack(push, 1)
typedef struct {
    uint8_t len;
    uint8_t code[15];
    uint64_t gpr[16];  // rax rcx rdx rbx rsp rbp rsi rdi r8..r15
    uint64_t flags;
    uint8_t ymm[16][32];
    uint64_t seed;
} Case;

typedef struct {
    uint32_t status;  // 0 ran, 1 faulted, 2 not runnable here
    uint32_t nchanges;
    uint64_t gpr[16];
    uint64_t flags;
    uint8_t ymm[16][32];
    struct { uint32_t off, val; } changes[MAX_CHANGES];
} Result;
#pragma pack(pop)

static inline void fill_data(uint8_t* p, uint64_t seed) {
    uint64_t x = seed | 1;
    for (uint64_t i = 0; i < DATA_SIZE; i += 8) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        __builtin_memcpy(p + i, &x, 8);
    }
}

static inline void diff_data(const uint8_t* now, uint64_t seed, Result* r) {
    uint64_t x = seed | 1;
    r->nchanges = 0;
    for (uint64_t i = 0; i < DATA_SIZE; i += 8) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        const uint8_t* b = (const uint8_t*)&x;
        for (int k = 0; k < 8; k++)
            if (now[i + k] != b[k] && r->nchanges < MAX_CHANGES) {
                r->changes[r->nchanges].off = (uint32_t)(i + k);
                r->changes[r->nchanges].val = now[i + k];
                r->nchanges++;
            }
    }
}
