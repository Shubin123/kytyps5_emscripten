# Builds differential test cases: one sample of every instruction form found in the given ELFs' code, each run
# from several random states. Memory operands are aimed at a scratch area that both executors map at DATA.
# Usage: cases.py <out.bin> <forms.txt> <elf>...
import random, struct, sys
from collections import OrderedDict
from capstone import Cs, CS_ARCH_X86, CS_MODE_64, CS_GRP_JUMP, CS_GRP_CALL, CS_GRP_RET, CS_GRP_INT, CS_GRP_IRET
from capstone.x86 import X86_OP_REG, X86_OP_MEM, X86_OP_IMM, X86_REG_RIP, X86_REG_INVALID

CODE, DATA, DATA_SIZE = 0x300000000, 0x300100000, 0x100000
DATA_MID = DATA + DATA_SIZE // 2
CASES_PER_FORM = int(__import__("os").environ.get("CASES", "6"))
VARIANTS = int(__import__("os").environ.get("VARIANTS", "1"))
SKIP = {'syscall', 'cpuid', 'rdtsc', 'rdtscp', 'xgetbv', 'ud2', 'hlt', 'int3', 'rdrand', 'rdseed', 'enter',
        'leave', 'xsave', 'xrstor', 'xsaveopt', 'xsavec', 'fxsave', 'fxrstor', 'rdpid', 'rdfsbase', 'rdgsbase',
        'wrfsbase', 'wrgsbase', 'cmpxchg16b', 'cmpxchg8b', 'pause'}
GPR64 = ['rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi', 'rbp', 'rsp', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15']
ORDER = ['rax', 'rcx', 'rdx', 'rbx', 'rsp', 'rbp', 'rsi', 'rdi', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15']

def full(md, reg):
    """The 64-bit GPR containing a register name, or None for non-GPRs."""
    n = md.reg_name(reg)
    alias = {'eax': 'rax', 'ax': 'rax', 'al': 'rax', 'ah': 'rax', 'ebx': 'rbx', 'bx': 'rbx', 'bl': 'rbx', 'bh': 'rbx',
             'ecx': 'rcx', 'cx': 'rcx', 'cl': 'rcx', 'ch': 'rcx', 'edx': 'rdx', 'dx': 'rdx', 'dl': 'rdx', 'dh': 'rdx',
             'esi': 'rsi', 'si': 'rsi', 'sil': 'rsi', 'edi': 'rdi', 'di': 'rdi', 'dil': 'rdi',
             'ebp': 'rbp', 'bp': 'rbp', 'bpl': 'rbp', 'esp': 'rsp', 'sp': 'rsp', 'spl': 'rsp'}
    if n in GPR64: return n
    if n in alias: return alias[n]
    for k in range(8, 16):
        if n in (f'r{k}d', f'r{k}w', f'r{k}b'): return f'r{k}'
    return None

def form_key(md, ins):
    parts = [ins.mnemonic]
    for o in ins.operands:
        if o.type == X86_OP_REG:
            n = md.reg_name(o.reg)
            parts.append(n[:3] if n[:3] in ('xmm', 'ymm') else f'r{o.size * 8}')
        elif o.type == X86_OP_MEM:
            parts.append(f'm{o.size * 8}' + ('rip' if o.mem.base == X86_REG_RIP else ''))
        elif o.type == X86_OP_IMM:
            parts.append(f'i{o.size * 8}')
    return ' '.join(parts)

def code_ranges(path):
    d = open(path, 'rb').read()
    phoff, = struct.unpack_from('<Q', d, 0x20)
    phe, phn = struct.unpack_from('<HH', d, 0x36)
    for i in range(phn):
        t, flags, off, va, _, fs, ms, _ = struct.unpack_from('<IIQQQQQQ', d, phoff + i * phe)
        if t == 1 and flags & 1 and fs:
            yield d[off:off + fs]

def sample_forms(paths):
    md = Cs(CS_ARCH_X86, CS_MODE_64)
    md.detail = True
    md.skipdata = True
    forms = OrderedDict()
    for path in paths:
        for code in code_ranges(path):
            for ins in md.disasm(code, 0):
                if ins.id == 0 or ins.mnemonic in SKIP or ins.mnemonic.startswith('.'): continue
                if any(g in ins.groups for g in (CS_GRP_JUMP, CS_GRP_CALL, CS_GRP_RET, CS_GRP_INT, CS_GRP_IRET)): continue
                raw = bytes(ins.bytes)
                if any(b in (0x64, 0x65) for b in raw[:4]): continue  # fs:/gs: segment overrides
                if 0x67 in raw[:4]: continue                            # 32-bit addressing
                mems = [o for o in ins.operands if o.type == X86_OP_MEM]
                if any(o.mem.base == X86_REG_INVALID and o.mem.index == X86_REG_INVALID for o in mems): continue
                key = form_key(md, ins)
                # Encodings of a form differ in prefixes, REX/VEX register bits, ModRM and SIB: keep up to
                # VARIANTS of them, told apart by the bytes before any displacement or immediate.
                cut = min([o for o, n in ((ins.disp_offset, ins.disp_size), (ins.imm_offset, ins.imm_size)) if n and o]
                          or [len(raw)])
                variants = forms.setdefault(key, OrderedDict())
                if len(variants) < VARIANTS and raw[:cut] not in variants: variants[raw[:cut]] = ins
    return md, forms

def rand_value(rng, kind):
    if kind == 0: return rng.getrandbits(64)
    if kind == 1: return rng.randrange(0, 256)
    if kind == 2: return rng.choice([0, 1, 0xffffffffffffffff, 0x8000000000000000, 0x7fffffffffffffff, 0x80000000,
                                     0xffffffff, 0x7fffffff, 63, 64, 31, 32])
    return rng.getrandbits(32)

def float_lanes(rng, kind):
    out = b''
    for _ in range(8):
        if kind == 0: out += struct.pack('<I', rng.getrandbits(32))
        elif kind == 1: out += struct.pack('<f', rng.choice([0.0, 1.0, -1.0, 0.5, 2.5, -3.75, 100.0, 1e10, -1e-10]))
        else: out += struct.pack('<f', float(rng.randrange(-1000, 1000)) / 8)
    return out

def make_cases(md, ins, rng):
    raw = bytearray(ins.bytes)
    mems = [o for o in ins.operands if o.type == X86_OP_MEM]
    for o in mems:
        if o.mem.base == X86_REG_RIP:
            disp = DATA_MID - (CODE + len(raw))
            raw[ins.disp_offset:ins.disp_offset + 4] = struct.pack('<i', disp)
    cases = []
    for c in range(CASES_PER_FORM):
        kind = c % 4
        regs = {r: rand_value(rng, kind) for r in GPR64}
        regs['rsp'] = DATA_MID - 0x40000 - 8 * rng.randrange(0, 16)
        if ins.mnemonic.startswith(('rep', 'movs', 'stos', 'lods', 'cmps', 'scas')) or 'movs' in ins.mnemonic and ins.mnemonic in ('movsb', 'movsw', 'movsd', 'movsq'):
            regs['rsi'], regs['rdi'], regs['rcx'] = DATA_MID, DATA_MID + 0x4000, rng.randrange(0, 24)
        vector_index = None  # gathers: (register number, index size) of the VSIB index vector
        for o in mems:
            if o.mem.base == X86_REG_RIP: continue
            base = full(md, o.mem.base) if o.mem.base != X86_REG_INVALID else None
            vname = md.reg_name(o.mem.index) if o.mem.index != X86_REG_INVALID else ''
            if vname[:3] in ('xmm', 'ymm'):
                vector_index = (int(vname[3:]), 8 if 'gatherq' in ins.mnemonic else 4)
                regs[base] = (DATA_MID + 16 * rng.randrange(-64, 64) - o.mem.disp) & 0xffffffffffffffff
                continue
            index = full(md, o.mem.index) if o.mem.index != X86_REG_INVALID else None
            scale, disp = o.mem.scale, o.mem.disp
            target = DATA_MID + 16 * rng.randrange(-64, 64)
            if index and index != base: regs[index] = rng.randrange(0, 8)
            if base and base == index: regs[base] = (target - disp) // (1 + scale)
            elif base: regs[base] = (target - disp - (regs[index] * scale if index else 0)) & 0xffffffffffffffff
            elif index: regs[index] = ((target - disp) // scale) & 0xffffffffffffffff
        flags = 0x202 | (rng.getrandbits(12) & 0x8d5)
        ymm = b''.join(float_lanes(rng, kind % 3) for _ in range(16))
        if vector_index:  # small indices, so every element lands in the scratch area
            r, size = vector_index
            lanes = b''.join(struct.pack('<q' if size == 8 else '<i', rng.randrange(-32, 32)) for _ in range(32 // size))
            ymm = ymm[:32 * r] + lanes + ymm[32 * (r + 1):]
        rec = struct.pack('<B', len(raw)) + bytes(raw).ljust(15, b'\x90')
        rec += b''.join(struct.pack('<Q', regs[r] & 0xffffffffffffffff) for r in ORDER)
        rec += struct.pack('<Q', flags) + ymm + struct.pack('<Q', rng.getrandbits(64))
        cases.append(rec)
    return cases

def main():
    out, forms_out, paths = sys.argv[1], sys.argv[2], sys.argv[3:]
    md, forms = sample_forms(paths)
    rng = random.Random(1234)
    count = 0
    with open(out, 'wb') as f, open(forms_out, 'w') as ff:
        for key, variants in forms.items():
            for ins in variants.values():
                for rec in make_cases(md, ins, rng):
                    f.write(rec)
                ff.write(f'{key}\t{bytes(ins.bytes).hex()}\t{ins.mnemonic} {ins.op_str}\n')
                count += 1
    print(f'{len(forms)} forms in {count} encodings, {count * CASES_PER_FORM} cases')

main()
