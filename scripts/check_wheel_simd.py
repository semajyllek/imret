"""Fail if the FAISS bundled in an installed Linux imret wheel lacks hardware popcount.

imret's search is Hamming distance on binary ORB descriptors, i.e. popcount. A FAISS
built for generic x86-64 (no -mpopcnt) falls back to a software bit count, which made
search ~20x slower in 0.1.1. Run after installing the wheel (cibuildwheel test-command):

    python scripts/check_wheel_simd.py
"""
import glob
import os
import platform
import subprocess
import sys

import imret

if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
    print(f"skipped: {platform.system()} {platform.machine()}")
    sys.exit(0)

pkg_dir = os.path.dirname(imret.__file__)
libs = glob.glob(os.path.join(pkg_dir, "..", "imret.libs", "libfaiss*.so*"))
if not libs:
    sys.exit(f"no bundled libfaiss found next to {pkg_dir}")

for lib in libs:
    asm = subprocess.run(["objdump", "-d", "--no-show-raw-insn", lib],
                         capture_output=True, text=True, check=True).stdout
    popcnt = asm.count("popcnt")
    avx2 = sum(asm.count(op) for op in ("vpshufb", "vpcmpeqb", "vpaddq"))
    print(f"{os.path.basename(lib)}: popcnt={popcnt} avx2-ops={avx2}")
    if popcnt == 0:
        sys.exit("FAIL: bundled FAISS has no hardware popcnt; build it with FAISS_OPT_LEVEL=dd")
print("ok")
