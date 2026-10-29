# Compares real.bin and interp.bin case by case; prints each instruction form with mismatches.
# Usage: compare.py <cases.bin> <forms.txt> <real.bin> <interp.bin> [verbose-form-substring]
import struct, sys
from collections import defaultdict
CASE, RES = 672, 1168
GPR = 'rax rcx rdx rbx rsp rbp rsi rdi r8 r9 r10 r11 r12 r13 r14 r15'.split()
CF, PF, AF, ZF, SF, OF = 1, 4, 0x10, 0x40, 0x80, 0x800
ALL = CF | PF | AF | ZF | SF | OF
# flags the architecture leaves undefined, by mnemonic prefix
UNDEF = {'and': AF, 'or': AF, 'xor': AF, 'test': AF, 'mul': SF | ZF | AF | PF, 'imul': SF | ZF | AF | PF,
         'div': ALL, 'idiv': ALL, 'bsf': CF | OF | SF | AF | PF, 'bsr': CF | OF | SF | AF | PF,
         'bt': OF | SF | AF | PF, 'bts': OF | SF | AF | PF, 'btr': OF | SF | AF | PF, 'btc': OF | SF | AF | PF,
         'shl': OF | AF, 'sal': OF | AF, 'shr': OF | AF, 'sar': OF | AF, 'rol': OF, 'ror': OF, 'rcl': OF, 'rcr': OF,
         'shld': OF | AF | CF, 'shrd': OF | AF | CF, 'lzcnt': OF | SF | AF | PF, 'tzcnt': OF | SF | AF | PF,
         'popcnt': 0, 'bextr': SF | AF | PF, 'blsr': AF | PF, 'blsi': AF | PF, 'blsmsk': AF | PF,
         'andn': AF | PF, 'bzhi': AF | PF, 'aaa': ALL, 'daa': OF, 'neg': 0}
def undef_mask(mn):
    return UNDEF.get(mn, 0)

forms = [l.rstrip('\n').split('\t') for l in open(sys.argv[2])]
cases = open(sys.argv[1], 'rb').read()
real = open(sys.argv[3], 'rb').read()
interp = open(sys.argv[4], 'rb').read()
verbose = sys.argv[5] if len(sys.argv) > 5 else None
n = len(cases) // CASE
per_form = n // len(forms)
bad = defaultdict(list)
unimpl, notrun = [], 0
for i in range(n):
    key, hexb, text = forms[i // per_form]
    a = real[i * RES:(i + 1) * RES]
    b = interp[i * RES:(i + 1) * RES]
    sa, = struct.unpack_from('<I', a, 0)
    sb, = struct.unpack_from('<I', b, 0)
    if sa == 2: notrun += 1; continue
    if sb == 3:
        if not unimpl or unimpl[-1] != text: unimpl.append(text)
        continue
    if sa != sb:
        bad[(key, text)].append(f'status real {sa} interp {sb}')
        continue
    if sa == 1: continue
    mn = key.split()[0]
    msgs = []
    ga = struct.unpack_from('<16Q', a, 8); gb = struct.unpack_from('<16Q', b, 8)
    for k in range(16):
        if ga[k] != gb[k]: msgs.append(f'{GPR[k]}: real {ga[k]:#x} interp {gb[k]:#x}')
    fa, = struct.unpack_from('<Q', a, 136); fb, = struct.unpack_from('<Q', b, 136)
    m = ALL & ~undef_mask(mn)
    if (fa ^ fb) & m: msgs.append(f'flags: real {fa & ALL:#x} interp {fb & ALL:#x} (compared {m:#x})')
    ya = a[144:144 + 512]; yb = b[144:144 + 512]
    for k in range(16):
        if ya[k * 32:(k + 1) * 32] != yb[k * 32:(k + 1) * 32]:
            msgs.append(f'ymm{k}: real {ya[k*32:(k+1)*32][::-1].hex()} interp {yb[k*32:(k+1)*32][::-1].hex()}')
    na, = struct.unpack_from('<I', a, 4); nb, = struct.unpack_from('<I', b, 4)
    ca = [struct.unpack_from('<II', a, 656 + 8 * j) for j in range(na)]
    cb = [struct.unpack_from('<II', b, 656 + 8 * j) for j in range(nb)]
    if ca != cb: msgs.append(f'memory: real {[(hex(o), hex(v)) for o, v in ca][:8]} interp {[(hex(o), hex(v)) for o, v in cb][:8]}')
    if msgs:
        bad[(key, text)].append('; '.join(msgs))
print(f'{len(forms)} forms, {n} cases, {notrun} not runnable on the host, {len(unimpl)} forms unimplemented in the interpreter')
print('unimplemented:', '; '.join(unimpl))
print(f'{len(bad)} forms with mismatches:')
for (key, text), msgs in bad.items():
    print(f'  {text}  [{key}]  x{len(msgs)}: {msgs[0][:300]}')
    if verbose and verbose in text:
        for m_ in msgs: print('     ', m_[:600])
