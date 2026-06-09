#!/usr/bin/env python3
"""
Benchmark imret against OpenCV BFMatcher (exact Hamming) and imagehash (perceptual).

Two datasets:
  synthetic  — random geometric images; all methods score ~100% accuracy (speed comparison only)
  wikiart    — real paintings streamed from HuggingFace; exposes accuracy differences
               when combined with a query transform

Five query transforms (applied to the query image only; indexed images are originals):
  none         — exact re-query; all methods trivially accurate
  affine       — random rotation ±15° and scale 0.80–1.0
  perspective  — random keystone warp up to 10% corner displacement
  book         — perspective warp + cream page border (photo from a book)
  wall         — perspective warp + grey wall border + slight blur (gallery photo)

The meaningful comparison is --dataset wikiart --transform wall (or book).
imagehash collapses because the border pixels dominate the DCT; imret's interior
ORB keypoints are unaffected.

Usage:
    pip install imret opencv-python-headless numpy
    pip install imagehash Pillow datasets   # optional / for wikiart
    pip install matplotlib                  # optional, for --plot

    python benchmark.py
    python benchmark.py --dataset wikiart --transform wall --sizes 100,500,1000 --plot
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


# ── WikiArt loader ─────────────────────────────────────────────────────────────

def _load_wikiart(n: int) -> tuple[list[np.ndarray], list[str]]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets to use --dataset wikiart")
    print(f"  Streaming {n:,} WikiArt images from HuggingFace...", flush=True)
    ds = load_dataset("huggan/wikiart", split="train", streaming=True)
    images: list[np.ndarray] = []
    labels: list[str] = []
    for i, item in enumerate(ds):
        if i >= n:
            break
        rgb  = np.array(item["image"].convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        images.append(gray)
        labels.append(f"wikiart_{i:06d}")
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{n}", flush=True)
    print(f"  Loaded {len(images):,} images.", flush=True)
    return images, labels


# ── query transforms ───────────────────────────────────────────────────────────

def _affine_jitter(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = gray.shape
    angle = float(rng.uniform(-15, 15))
    scale = float(rng.uniform(0.80, 1.0))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    return cv2.warpAffine(gray, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _perspective_warp(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = gray.shape
    lim = int(min(h, w) * 0.10)
    def jitter(): return int(rng.integers(0, lim + 1))
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32([
        [jitter(),         jitter()],
        [w - 1 - jitter(), jitter()],
        [w - 1 - jitter(), h - 1 - jitter()],
        [jitter(),         h - 1 - jitter()],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(gray, M, (w, h))


def _book_page(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    warped = _perspective_warp(gray, rng)
    h, w = warped.shape
    bw = int(w * float(rng.uniform(0.10, 0.20)))
    bh = int(h * float(rng.uniform(0.10, 0.20)))
    canvas = np.full((h + 2 * bh, w + 2 * bw), int(rng.integers(210, 246)), dtype=np.uint8)
    canvas[bh:bh + h, bw:bw + w] = warped
    return canvas


def _wall_photo(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    warped = _perspective_warp(gray, rng)
    h, w = warped.shape
    bw = int(w * float(rng.uniform(0.05, 0.15)))
    bh = int(h * float(rng.uniform(0.05, 0.15)))
    canvas = np.full((h + 2 * bh, w + 2 * bw), int(rng.integers(100, 181)), dtype=np.uint8)
    canvas[bh:bh + h, bw:bw + w] = warped
    k = int(rng.integers(0, 2)) * 2 + 1
    if k > 1:
        canvas = cv2.GaussianBlur(canvas, (k, k), 0)
    return canvas


_TRANSFORMS = {
    "none":        lambda g, rng: g,
    "affine":      _affine_jitter,
    "perspective": _perspective_warp,
    "book":        _book_page,
    "wall":        _wall_photo,
}


# ── BFMatcher baseline ─────────────────────────────────────────────────────────

_BF_CHUNK = 32767


class _BFMatcherBaseline:
    """Exact Hamming matching over a chunked flat descriptor pool."""

    def __init__(self, cfg: imret.OrbConfig):
        self._orb = cv2.ORB_create(nfeatures=cfg.max_features)
        self._max_hamming = cfg.max_hamming_distance
        self._all_descs: list = []
        self._all_labels: list = []
        self._chunks: list = []
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
            self._db.append((label, imagehash.phash(_PILImage.fromarray(image), hash_size=self._hash_size)))

        def build(self) -> None:
            pass

        def search(self, image: np.ndarray) -> tuple[str, float]:
            if not self._db:
                return "Unknown", 0.0
            qh = imagehash.phash(_PILImage.fromarray(image), hash_size=self._hash_size)
            best_label, best_dist = None, float("inf")
            for label, h in self._db:
                d = qh - h
                if d < best_dist:
                    best_dist, best_label = d, label
            return best_label, max(0.0, 1.0 - best_dist / self._hash_size ** 2)

    _IMAGEHASH_AVAILABLE = True

except ImportError:
    _IMAGEHASH_AVAILABLE = False


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
        self._vault  = imret.Vault(cfg)
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


def _timed_queries(engine, images, labels, n_queries, transform_fn) -> tuple[float, float, float]:
    rng       = np.random.default_rng(42)
    q_images  = images[:n_queries]
    q_labels  = labels[:n_queries]
    latencies = []
    correct   = 0
    for img, lbl in zip(q_images, q_labels):
        query = transform_fn(img, rng)
        t0 = time.perf_counter()
        result_label, _ = engine.search(query)
        latencies.append((time.perf_counter() - t0) * 1000)
        if result_label == lbl:
            correct += 1
    return correct / n_queries, _pct(latencies, 50), _pct(latencies, 95)


# ── benchmark ──────────────────────────────────────────────────────────────────

def run_one(n: int, n_queries: int, cfg: imret.OrbConfig,
            images: list, labels: list, transform_fn) -> dict:
    n_queries = min(n_queries, n)
    imgs = images[:n]
    lbls = labels[:n]
    row: dict = {"n": n}

    for key, factory in [
        ("imret",       lambda: _ImretWrapper(cfg)),
        ("imret-batch", lambda: _ImretBatchWrapper(cfg)),
        ("bfmatcher",   lambda: _BFMatcherBaseline(cfg)),
    ] + ([("imagehash", lambda: _ImageHashBaseline())] if _IMAGEHASH_AVAILABLE else []):
        tracemalloc.start()
        engine    = factory()
        build_s   = _timed_build(engine, imgs, lbls)
        mem_mb    = tracemalloc.get_traced_memory()[1] / 1024 / 1024
        tracemalloc.stop()
        acc, p50, p95 = _timed_queries(engine, imgs, lbls, n_queries, transform_fn)
        row[key] = {"build_s": build_s, "acc": acc, "p50_ms": p50, "p95_ms": p95, "mem_mb": mem_mb}

    return row


# ── output ─────────────────────────────────────────────────────────────────────

_COL = 14


def _header(methods):
    cols = ["N"] + [f"{m:>{_COL}}" for m in methods]
    print("".join(cols))
    print("-" * (_COL + len(methods) * _COL))


def _row(n, data, methods, key, fmt):
    vals = [f"{n:>{_COL},d}"] + [f"{fmt(data[m][key]):>{_COL}}" if m in data else f"{'—':>{_COL}}" for m in methods]
    print("".join(vals))


def print_results(rows: list[dict], dataset: str, transform: str) -> None:
    methods = ["imret", "imret-batch", "bfmatcher"] + (["imagehash"] if _IMAGEHASH_AVAILABLE else [])
    print(f"\ndataset={dataset}  transform={transform}")
    for metric, key, fmt in [
        ("Accuracy (%)",    "acc",     lambda v: f"{v * 100:.1f}%"),
        ("Search p50 (ms)", "p50_ms",  lambda v: f"{v:.2f}"),
        ("Search p95 (ms)", "p95_ms",  lambda v: f"{v:.2f}"),
        ("Build time (s)",  "build_s", lambda v: f"{v:.2f}"),
        ("Peak mem (MB)",   "mem_mb",  lambda v: f"{v:.1f}"),
    ]:
        print(f"\n{metric}")
        _header(methods)
        for row in rows:
            _row(row["n"], row, methods, key, fmt)


def plot_results(rows: list[dict], dataset: str, transform: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed — skipping plot.")
        return

    methods = ["imret", "imret-batch", "bfmatcher"] + (["imagehash"] if _IMAGEHASH_AVAILABLE else [])
    colors  = {"imret": "#1f77b4", "imret-batch": "#17becf", "bfmatcher": "#ff7f0e", "imagehash": "#2ca02c"}
    ns      = [r["n"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"imret vs alternatives — dataset={dataset}  transform={transform}", fontsize=11)

    for ax, key, ylabel in [
        (axes[0], "p50_ms", "Search latency p50 (ms)"),
        (axes[1], "acc",    "Accuracy (re-query)"),
        (axes[2], "build_s","Build time (s)"),
    ]:
        for m in methods:
            vals  = [r[m][key] for r in rows if m in r]
            scale = 100 if key == "acc" else 1
            ax.plot(ns, [v * scale for v in vals], marker="o", label=m, color=colors.get(m))
        ax.set_xlabel("Collection size")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = f"benchmark_{dataset}_{transform}.png"
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes",       default="100,500,1000,2000")
    parser.add_argument("--queries",     type=int, default=20)
    parser.add_argument("--max-features",type=int, default=500)
    parser.add_argument("--max-hamming", type=int, default=45)
    parser.add_argument("--dataset",     default="synthetic", choices=["synthetic", "wikiart"])
    parser.add_argument("--transform",   default="none",      choices=list(_TRANSFORMS))
    parser.add_argument("--plot",        action="store_true")
    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(",")]
    max_n = max(sizes)

    cfg = imret.OrbConfig()
    cfg.max_features        = args.max_features
    cfg.max_hamming_distance = args.max_hamming

    transform_fn = _TRANSFORMS[args.transform]

    print(f"imret benchmark  dataset={args.dataset}  transform={args.transform}  "
          f"{len(sizes)} sizes  {args.queries} queries each")
    if not _IMAGEHASH_AVAILABLE:
        print("(imagehash not installed — pip install imagehash Pillow to include it)")

    if args.dataset == "wikiart":
        all_images, all_labels = _load_wikiart(max_n)
    else:
        all_images = [_make_image(i) for i in range(max_n)]
        all_labels = [f"img_{i:06d}" for i in range(max_n)]

    rows = []
    for n in sizes:
        print(f"\nN={n:,}...", end="", flush=True)
        rows.append(run_one(n, args.queries, cfg, all_images, all_labels, transform_fn))
        print(" done")

    print_results(rows, args.dataset, args.transform)

    if args.plot:
        plot_results(rows, args.dataset, args.transform)


if __name__ == "__main__":
    main()
