import platform


def _check_x86_baseline():
    # Linux wheels bundle FAISS built for x86-64-v2 (SSE4.2 + popcnt) with runtime
    # AVX2/AVX-512 dispatch. On older CPUs, fail with a message instead of SIGILL.
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        return
    try:
        with open("/proc/cpuinfo") as f:
            flags = next((line.split(":", 1)[1].split() for line in f if line.startswith("flags")), None)
    except OSError:
        return
    if flags is not None and not {"popcnt", "sse4_2"} <= set(flags):
        raise ImportError("imret's Linux wheel requires an x86-64 CPU with SSE4.2 and POPCNT (x86-64-v2).")


_check_x86_baseline()

from .core import Vault  # noqa: E402
from ._core import OrbConfig, MatchResult  # noqa: E402

__all__ = ["Vault", "OrbConfig", "MatchResult"]
