#!/usr/bin/env python3
"""
Benchmark imret against OpenCV BFMatcher (exact Hamming) and imagehash (perceptual).

All three use the same synthetic image collection. Queries are re-searches of
ingested images, so accuracy reflects how well each method retrieves what it indexed.

Usage:
    pip install imret opencv-python-headless numpy
    pip install imagehash Pillow        # optional, enables the imagehash column
    pip install matplotlib              # optional, enables --plot

    python benchmark.py
    python benchmark.py --sizes 100,500,1000,5000 --queries 20 --plot
"""

import argparse
import time
import tracemalloc

import cv2
import numpy as np

import imret


# ── synthetic image generation ─────────────────────────────────────────────────

def _make_image(seed: int, size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 30, dtype=np.uint8)
    for _ in range(rng.integers(20, 40)):
        kind = rng.integers(3)
        color = int(rng.integers(80, 255))
        if kind == 0:
            p1 = tuple(rng.integers(0, size, 2).tolist())
            p2 = tuple(rng.integers(0, size, 2).tolist())
            cv2.line(img, p1, p2, color, int(rng.integers(1, 4)))
        elif kind == 1:
            cx, cy = rng.integers(10, size - 10, 2).tolist()
            r = int(rng.integers(5, size // 5))
            cv2.circle(img, (cx, cy), r, color, -1)
        else:
            x1, y1 = rng.integers(0, size - 20, 2).tolist()
            x2 = min(x1 + int(rng.integers(10, size // 3)), size - 1)
            y2 = min(y1 + int(rng.integers(10, size // 3)), size - 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    noise = rng.integers(0, 15, (size, size), dtype=np.uint8)
    return cv2.add(img, noise)


# ── BFMatcher baseline ─────────────────────────────────────────────────────────

_BF_CHUNK = 32767  # OpenCV BFMatcher internal row limit per matrix


class _BFMatcherBaseline:
    """Exact Hamming matching over a chunked flat descriptor pool."""

    def __init__(self, cfg: imret.OrbConfig):
        self._orb = cv2.ORB_create(nfeatures=cfg.max_features)
        self._max_hamming = cfg.max_hamming_distance
        self._all_descs: list = []
        self._all_labels: list = []
        self._chunks: list = []  # list of (desc_matrix, labels_slice)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def add(self, image: np.ndarray, label: str) -> None:
        _, desc = self._orb.detectAndCompute(image, None)
        if desc is not None:
            self._all_descs.append(desc)
            self._all_labels.extend([label] * len(desc))

    def build(self) -> None:
        if not self._all_descs:
            return
        flat = np.vstack(self._all_descs)
        for start in range(0, len(flat), _BF_CHUNK):
            end = start + _BF_CHUNK
            self._chunks.append((flat[start:end], self._all_labels[start:end]))

    def search(self, image: np.ndarray) -> tuple[str, float]:
        _, desc = self._orb.detectAndCompute(image, None)
        if desc is None or not self._chunks:
            return "Unknown", 0.0
        votes: dict[str, int] = {}
        for chunk_mat, chunk_labels in self._chunks:
            for m in self._bf.match(desc, chunk_mat):
                if m.distance <= self._max_hamming:
                    lbl = chunk_labels[m.trainIdx]
                    votes[lbl] = votes.get(lbl, 0) + 1
        if not votes:
            return "Unknown", 0.0
        total = sum(votes.values())
        best = max(votes, key=votes.get)
        return best, votes[best] / total


# ── imagehash baseline ─────────────────────────────────────────────────────────

try:
    import imagehash
    from PIL import Image as _PILImage

    class _ImageHashBaseline:
        def __init__(self, hash_size: int = 16):
            self._hash_size = hash_size
            self._db: list = []

        def add(self, image: np.ndarray, label: str) -> None:
            pil = _PILImage.fromarray(image)
            self._db.append((label, imagehash.phash(pil, hash_size=self._hash_size)))

        def build(self) -> None:
            pass

        def search(self, image: np.ndarray) -> tuple[str, float]:
            if not self._db:
                return "Unknown", 0.0
            pil = _PILImage.fromarray(image)
            qh = imagehash.phash(pil, hash_size=self._hash_size)
            best_label, best_dist = None, float("inf")
            for label, h in self._db:
                d = qh - h
                if d < best_dist:
                    best_dist, best_label = d, label
            confidence = max(0.0, 1.0 - best_dist / self._hash_size ** 2)
            return best_label, confidence

    _IMAGEHASH_AVAILABLE = True

except ImportError:
    _IMAGEHASH_AVAILABLE = False


# ── helpers ────────────────────────────────────────────────────────────────────

def _pct(data: list[float], p: float) -> float:
    if not data:
        return float("nan")
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _timed_build(engine, images, labels) -> float:
    t0 = time.perf_counter()
    for img, lbl in zip(images, labels):
        engine.add(img, lbl)
    engine.build()
    return time.perf_counter() - t0


def _timed_queries(engine, images, labels, n_queries) -> tuple[float, float, float]:
    q_images = images[:n_queries]
    q_labels = labels[:n_queries]
    latencies = []
    correct = 0
    for img, lbl in zip(q_images, q_labels):
        t0 = time.perf_counter()
        result_label, _ = engine.search(img)
        latencies.append((time.perf_counter() - t0) * 1000)
        if result_label == lbl:
            correct += 1
    return correct / n_queries, _pct(latencies, 50), _pct(latencies, 95)


# ── imret wrappers ─────────────────────────────────────────────────────────────

class _ImretWrapper:
    def __init__(self, cfg):
        self._vault = imret.Vault(cfg)

    def add(self, image, label):
        self._vault.add(image, label)

    def build(self):
        self._vault.build()

    def search(self, image):
        r = self._vault.search(image)
        return r.label, r.confidence


class _ImretBatchWrapper:
    """Uses add_batch() — parallel ORB extraction via OpenMP."""

    def __init__(self, cfg):
        self._vault = imret.Vault(cfg)
        self._images: list = []
        self._labels: list = []

    def add(self, image, label):
        self._images.append(image)
        self._labels.append(label)

    def build(self):
        self._vault.add_batch(self._images, self._labels)
        self._vault.build()

    def search(self, image):
        r = self._vault.search(image)
        return r.label, r.confidence


# ── benchmark ──────────────────────────────────────────────────────────────────

def run_one(n: int, n_queries: int, cfg: imret.OrbConfig) -> dict:
    n_queries = min(n_queries, n)
    images = [_make_image(i) for i in range(n)]
    labels = [f"img_{i:06d}" for i in range(n)]

    row: dict = {"n": n}

    # imret (single add)
    tracemalloc.start()
    engine = _ImretWrapper(cfg)
    build_s = _timed_build(engine, images, labels)
    mem_mb = tracemalloc.get_traced_memory()[1] / 1024 / 1024
    tracemalloc.stop()
    acc, p50, p95 = _timed_queries(engine, images, labels, n_queries)
    row["imret"] = {"build_s": build_s, "acc": acc, "p50_ms": p50, "p95_ms": p95, "mem_mb": mem_mb}

    # imret (add_batch — parallel ORB via OpenMP)
    tracemalloc.start()
    batch_engine = _ImretBatchWrapper(cfg)
    batch_build_s = _timed_build(batch_engine, images, labels)
    batch_mem_mb = tracemalloc.get_traced_memory()[1] / 1024 / 1024
    tracemalloc.stop()
    batch_acc, batch_p50, batch_p95 = _timed_queries(batch_engine, images, labels, n_queries)
    row["imret-batch"] = {"build_s": batch_build_s, "acc": batch_acc, "p50_ms": batch_p50, "p95_ms": batch_p95, "mem_mb": batch_mem_mb}

    # BFMatcher
    tracemalloc.start()
    bf = _BFMatcherBaseline(cfg)
    bf_build_s = _timed_build(bf, images, labels)
    bf_mem_mb = tracemalloc.get_traced_memory()[1] / 1024 / 1024
    tracemalloc.stop()
    bf_acc, bf_p50, bf_p95 = _timed_queries(bf, images, labels, n_queries)
    row["bfmatcher"] = {"build_s": bf_build_s, "acc": bf_acc, "p50_ms": bf_p50, "p95_ms": bf_p95, "mem_mb": bf_mem_mb}

    # imagehash (optional)
    if _IMAGEHASH_AVAILABLE:
        tracemalloc.start()
        ih = _ImageHashBaseline()
        ih_build_s = _timed_build(ih, images, labels)
        ih_mem_mb = tracemalloc.get_traced_memory()[1] / 1024 / 1024
        tracemalloc.stop()
        ih_acc, ih_p50, ih_p95 = _timed_queries(ih, images, labels, n_queries)
        row["imagehash"] = {"build_s": ih_build_s, "acc": ih_acc, "p50_ms": ih_p50, "p95_ms": ih_p95, "mem_mb": ih_mem_mb}

    return row


# ── output ─────────────────────────────────────────────────────────────────────

_COL = 14


def _header(methods):
    cols = ["N"] + [f"{m:>{_COL}}" for m in methods]
    print("".join(cols))
    print("-" * (_COL + len(methods) * _COL))


def _row(n, data, methods, key, fmt):
    vals = [f"{n:>{_COL},d}"] + [f"{fmt(data[m][key]):>{_COL}}" for m in methods if m in data]
    print("".join(vals))


def print_results(rows: list[dict]) -> None:
    methods = ["imret", "imret-batch", "bfmatcher"] + (["imagehash"] if _IMAGEHASH_AVAILABLE else [])

    for metric, key, fmt, label in [
        ("Accuracy (%)", "acc", lambda v: f"{v * 100:.1f}%", None),
        ("Search p50 (ms)", "p50_ms", lambda v: f"{v:.2f}", None),
        ("Search p95 (ms)", "p95_ms", lambda v: f"{v:.2f}", None),
        ("Build time (s)", "build_s", lambda v: f"{v:.2f}", None),
        ("Peak memory (MB)", "mem_mb", lambda v: f"{v:.1f}", None),
    ]:
        print(f"\n{metric}")
        _header(methods)
        for row in rows:
            _row(row["n"], row, methods, key, fmt)


def plot_results(rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed — skipping plot.")
        return

    methods = ["imret", "imret-batch", "bfmatcher"] + (["imagehash"] if _IMAGEHASH_AVAILABLE else [])
    colors = {"imret": "#1f77b4", "imret-batch": "#17becf", "bfmatcher": "#ff7f0e", "imagehash": "#2ca02c"}
    ns = [r["n"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("imret vs alternatives — synthetic image collection")

    for ax, key, ylabel in [
        (axes[0], "p50_ms", "Search latency p50 (ms)"),
        (axes[1], "acc", "Accuracy (re-query)"),
        (axes[2], "build_s", "Build time (s)"),
    ]:
        for m in methods:
            vals = [r[m][key] for r in rows if m in r]
            scale = 100 if key == "acc" else 1
            ax.plot(ns, [v * scale for v in vals], marker="o", label=m, color=colors.get(m))
        ax.set_xlabel("Collection size")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = "benchmark_results.png"
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", default="100,500,1000,2000",
                        help="Comma-separated collection sizes (default: 100,500,1000,2000)")
    parser.add_argument("--queries", type=int, default=20,
                        help="Number of queries per size (default: 20)")
    parser.add_argument("--max-features", type=int, default=500)
    parser.add_argument("--max-hamming", type=int, default=45)
    parser.add_argument("--plot", action="store_true", help="Save a matplotlib chart")
    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(",")]

    cfg = imret.OrbConfig()
    cfg.max_features = args.max_features
    cfg.max_hamming_distance = args.max_hamming

    print(f"imret benchmark — {len(sizes)} sizes, {args.queries} queries each")
    if not _IMAGEHASH_AVAILABLE:
        print("(imagehash not installed — skipping that column; pip install imagehash Pillow to include it)")

    rows = []
    for n in sizes:
        print(f"\nRunning N={n:,}...", end="", flush=True)
        rows.append(run_one(n, args.queries, cfg))
        print(" done")

    print_results(rows)

    if args.plot:
        plot_results(rows)


if __name__ == "__main__":
    main()
