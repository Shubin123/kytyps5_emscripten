# Differential instruction tests

Checks the x86 interpreter against a real x86-64 CPU, one instruction at a time, using the
instruction forms a game actually contains.

1. `cases.py` disassembles the executable segments of the given ELFs (decrypted `eboot.bin` and
   `.prx` files), keeps one example of every instruction form (mnemonic plus operand kinds), and
   writes several cases per form, each with random registers, flags, `ymm0-15` and memory. Memory
   operands, including RIP-relative ones, are pointed into a 1 MB scratch area. `VARIANTS=n` keeps
   up to n encodings of each form (other registers, REX/VEX bits, addressing modes), which catches
   bugs like a destination that is also a source.
2. `real` runs every case on the host CPU: natively on x86-64 Linux, or under Rosetta 2 on Apple
   silicon. `interp` runs the same cases in `src/web/x86_interpreter.cpp`.
3. `compare.py` reports every form whose registers, flags, vector registers, scratch memory or
   fault status differ. Flags the architecture leaves undefined are not compared.

```bash
VARIANTS=8 CASES=6 python3 tools/difftest/cases.py /tmp/cases.bin /tmp/forms.txt eboot.bin.esbak Module.prx.esbak
clang -arch x86_64 -O1 tools/difftest/real.c tools/difftest/real_stub.S -o /tmp/real   # Linux: cc -O1 ...
c++ -std=c++17 -O2 -Isrc/web/include tools/difftest/interp.cpp src/web/x86_interpreter.cpp -o /tmp/interp
/tmp/real /tmp/cases.bin /tmp/real.bin
/tmp/interp /tmp/cases.bin /tmp/interp.bin
python3 tools/difftest/compare.py /tmp/cases.bin /tmp/forms.txt /tmp/real.bin /tmp/interp.bin
```

Known differences that are not interpreter bugs:

- `rcpps`/`rsqrtps` and their scalar forms return hardware approximations; the interpreter
  computes exact results.
- Rosetta computes `lock cmpxchg` flags as `destination - accumulator` (hardware uses
  `accumulator - destination`), keeps OF/AF/PF on `vptest`, and uses only the low 32 bits of
  `vpsllvq`/`vpsrlvq` counts.
- MMX forms come from data in the code segments that happens to decode as instructions.
- Instructions Rosetta can't run (SSE4a `extrq`/`insertq`, a few others) are listed as "not
  runnable on the host"; after a SIGILL the harness re-executes itself to continue.
