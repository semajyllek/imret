# imret

imret is a C++ image retrieval library. Given a query image, it finds the closest matching image from a previously ingested collection. It uses ORB binary feature descriptors and a FAISS inverted-file index (IVF) for approximate nearest-neighbour search, with a two-tier search strategy and per-keypoint voting to produce a confidence score.

## How it works

1. **Feature extraction** — ORB extracts up to `max_features` keypoints per image, each producing a 256-bit (32-byte) binary descriptor.
2. **Indexing** — `build()` runs k-means on all accumulated descriptors to partition them into Voronoi cells (`IndexBinaryIVF`). The number of cells scales with the total feature count.
3. **Search** — A query image is described with ORB. Each descriptor votes for the image it matches (filtered by Hamming distance). Tier 1 searches `fast_cells` cells; if the top vote fraction falls below `confidence_threshold`, Tier 2 searches `deep_cells` cells.
4. **Result** — The image with the most votes wins, returning its label and a confidence value in `[0, 1]`.

## Dependencies

- CMake >= 3.15
- C++17 compiler
- OpenCV 4.x
- FAISS
- OpenMP (macOS: `brew install libomp`)

## Building (C++)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

This produces `build/imret_cli` and `build/libimret_core.a`.

### Running the tests

C++ tests:

```bash
ctest --test-dir build --output-on-failure
```

Python binding tests (requires imret installed and pytest):

```bash
pip install pytest
python -m pytest bindings/python/tests/ -v
```

## C++ API

Include `vault.hpp` and link against `libimret_core.a` and its dependencies (`OpenCV`, `faiss`, `omp`).

### OrbConfig

```cpp
#include "imret.hpp"

OrbConfig cfg;
cfg.max_features         = 500;   // ORB keypoints per image
cfg.resize_dim           = 0;     // 0 = no resize; >0 = resize to (N x N) before extraction
cfg.fast_cells           = 8;     // IVF cells probed in tier-1 search
cfg.deep_cells           = 64;    // IVF cells probed in tier-2 fallback
cfg.max_hamming_distance = 45;    // maximum Hamming distance for a keypoint to count as a match
cfg.confidence_threshold = 0.15f; // confidence below this triggers the tier-2 fallback
```

### Ingest, build, search

```cpp
#include "vault.hpp"

OrbConfig cfg;
Vault vault(cfg);

// Ingest images (grayscale cv::Mat)
vault.add(image_a, "label_a");
vault.add(image_b, "label_b");

// Bulk ingest with OpenMP parallelism — preferred for large collections
vault.add_batch({image_a, image_b, image_c}, {"label_a", "label_b", "label_c"});

// Build the index — required before searching
vault.build();

// Query
MatchResult result = vault.search(query_image);
// result.label        — label of the best match, or "Unknown"
// result.confidence   — fraction of keypoints that voted for the winner [0, 1]
// result.fallback_used — true if the tier-2 search was triggered
```

`add()` and `add_batch()` can be called after `build()`. Call `build()` again afterwards to retrain the index over all accumulated data.

### Stats

```cpp
Vault::Stats s = vault.stats();
// s.n_images    — number of unique images in the vault
// s.n_features  — total feature vectors accumulated
// s.nlist       — number of IVF clusters (0 if not yet built)
// s.is_built    — whether the index has been trained
```

All fields are O(1) reads from in-memory structures.

### Persistence

```cpp
vault.save("/path/to/prefix");   // writes prefix.faiss and prefix.meta
vault.load("/path/to/prefix");   // restores the index and label map; no rebuild needed
```

The `.meta` file stores the `OrbConfig` alongside the label map, so a loaded vault always uses the config it was originally built with.

## Python

### Install

```bash
pip install imret
```

Pre-built binary wheels are available for Linux x86_64 and macOS arm64, covering Python 3.9–3.13. Google Colab is supported without any additional setup.

The Linux wheel needs an x86-64-v2 CPU (SSE4.2 + POPCNT, any x86 CPU from roughly 2009 on). Its bundled FAISS is built with runtime SIMD dispatch, so AVX2/AVX-512 are used when present. 0.1.1's Linux wheel shipped a generic FAISS without hardware popcount, which made search ~20x slower; upgrade to 0.1.4 or later.

### Usage

```python
import cv2
import imret

cfg = imret.OrbConfig()
cfg.max_features         = 500
cfg.resize_dim           = 800   # resize images to 800x800 before extraction
cfg.fast_cells           = 8
cfg.deep_cells           = 64
cfg.max_hamming_distance = 45
cfg.confidence_threshold = 0.15

vault = imret.Vault(cfg)

# Ingest — images must be grayscale uint8 numpy arrays
gray = cv2.imread("painting.jpg", cv2.IMREAD_GRAYSCALE)
vault.add(gray, "my_label")

# Bulk ingest (parallel via OpenMP)
vault.add_batch([gray_a, gray_b, gray_c], ["label_a", "label_b", "label_c"])

# Build the index
vault.build()

# Search
result = vault.search(query_gray)
print(result.label, result.confidence, result.fallback_used)

# Vault stats (all O(1))
s = vault.stats()
print(s["n_images"])    # images in the vault
print(s["n_features"])  # total ORB descriptor vectors stored
print(s["nlist"])       # IVF Voronoi clusters (0 before build)
print(s["is_built"])    # whether the index has been trained

# Save and load
vault.save("/tmp/my_vault")
vault2 = imret.Vault.load_from_disk("/tmp/my_vault", cfg)
```

`add()` expects a 2-D `numpy.ndarray` with dtype `uint8`. If `resize_dim > 0`, resizing is applied internally before extraction.

### Building from source

Requirements: Python >= 3.9, pybind11, scikit-build-core, OpenCV, FAISS, OpenMP.

On macOS, install OpenMP first:

```bash
brew install libomp
```

Then build and install the Python package:

```bash
cd bindings/python
pip install scikit-build-core pybind11
pip install .
```

The CMakeLists detects the Homebrew libomp prefix automatically.

## CLI

```bash
./build/imret_cli <vault_prefix> <path_to_image>
```

Loads the vault at `<vault_prefix>.faiss` / `<vault_prefix>.meta`, searches with the given image, and prints the matched label to stdout. Exits with code 1 and prints `UNKNOWN` if confidence is below `confidence_threshold`.

## OrbConfig reference

| Field | Default | Description |
|---|---|---|
| `max_features` | 500 | Maximum ORB keypoints extracted per image |
| `resize_dim` | 0 | If > 0, resize each image to `resize_dim x resize_dim` before extraction |
| `fast_cells` | 8 | IVF cells probed during tier-1 search |
| `deep_cells` | 64 | IVF cells probed during tier-2 fallback |
| `max_hamming_distance` | 45 | Keypoints with Hamming distance above this threshold are excluded from voting |
| `confidence_threshold` | 0.15 | Vote fraction below this triggers the tier-2 fallback |

## Benchmarks

Generated by `benchmark.py` with imret 0.1.4 on macOS arm64, `max_features=500`, 20 queries per size. Linux x86_64 numbers and a large-scale run are further down.

`imret` and `imret-batch` differ only at ingest: the batch variant uses `add_batch()` which extracts ORB features in parallel via OpenMP, reducing build time by up to 34% (WikiArt, N=1,000). Search latency is identical.

### Speed comparison — synthetic images, no transform

All methods score 100% accuracy on structurally distinct synthetic images, so this measures speed only.

**Search latency p50 (ms)**

| N | imret | imret-batch | bfmatcher | imagehash |
|---|---|---|---|---|
| 100 | 3.08 | 3.04 | 8.28 | 0.26 |
| 500 | 3.24 | 3.26 | 37.10 | 0.50 |
| 1,000 | 3.35 | 3.37 | 84.34 | 0.95 |
| 2,000 | 3.61 | 3.44 | 162.72 | 1.63 |

**Build time (s)**

| N | imret | imret-batch | bfmatcher | imagehash |
|---|---|---|---|---|
| 100 | 1.02 | 0.91 | 0.11 | 1.95 |
| 500 | 15.23 | 14.73 | 0.56 | 0.08 |
| 1,000 | 30.56 | 29.84 | 1.14 | 0.23 |
| 2,000 | 61.61 | 59.20 | 2.24 | 0.35 |

imret search latency is nearly flat (3.08ms → 3.61ms across 20× more images). BFMatcher grows linearly. Build time is higher because imret trains a FAISS IVF index upfront — a one-time cost at index time.

### Accuracy comparison — WikiArt paintings, wall-photo transform

The wall transform adds a grey background border and light Gaussian blur, simulating a gallery photo. This is where the methods diverge. Perceptual hashing (imagehash) is designed for near-duplicate detection; adding a border shifts the global DCT hash enough to confuse it with unrelated paintings. imret's ORB keypoints are local to the painting interior and are unaffected.

**Accuracy (%)**

| N | imret | imret-batch | bfmatcher | imagehash |
|---|---|---|---|---|
| 100 | 100.0 | 100.0 | 100.0 | 15.0 |
| 500 | 100.0 | 100.0 | 100.0 | 15.0 |
| 1,000 | 100.0 | 100.0 | 100.0 | 10.0 |

**Search latency p50 (ms)**

| N | imret | imret-batch | bfmatcher | imagehash |
|---|---|---|---|---|
| 100 | 15.23 | 15.29 | 23.06 | 1.81 |
| 500 | 15.15 | 16.04 | 64.94 | 2.13 |
| 1,000 | 16.14 | 15.82 | 122.45 | 2.49 |

imagehash is fast but effectively random on real images under this transform (10–15% ≈ 1-in-10 chance). imret maintains 100% accuracy. BFMatcher matches imret accuracy but latency grows linearly.

### Linux x86_64 — 0.1.1 vs 0.1.4

Synthetic images, Google Cloud Build `E2_HIGHCPU_8` (8 vCPU, AMD EPYC 7B12). imret 0.1.1's Linux wheel bundled a FAISS built for generic x86-64 with no hardware popcount, so every Hamming distance used a software bit count. 0.1.4 builds FAISS with runtime SIMD dispatch (`FAISS_OPT_LEVEL=dd`).

**Search latency p50 (ms)**

| N | imret 0.1.1 | imret 0.1.4 | speedup | bfmatcher | imagehash |
|---|---|---|---|---|---|
| 1,000 | 7.51 | 3.35 | 2.2× | 370.22 | 3.18 |
| 5,000 | 11.75 | 4.41 | 2.7× | 1,847.78 | 14.88 |
| 20,000 | 24.38 | 7.66 | 3.2× | 7,413.86 | 57.14 |

**Build time (s)**

| N | imret 0.1.1 | imret 0.1.4 | imret-batch 0.1.4 |
|---|---|---|---|
| 1,000 | 151.30 | 107.39 | 103.40 |
| 5,000 | 426.76 | 271.01 | 266.34 |
| 20,000 | 509.06 | 317.62 | 294.51 |

All methods score 100% here, in both versions. The gap grows with N because the index scan, where popcount matters, grows while ORB extraction stays fixed. bfmatcher and imagehash are from the 0.1.4 run; they don't use FAISS, and in the 0.1.1 run they were within 5%.

### At scale — 78k album covers

imret as an album-cover recognizer. One vault holds 77,893 distinct covers (38.4M ORB descriptors, 1.54 GB on disk, 1.74 GB RSS when loaded), built in about 11 minutes on an M-series Mac. It was queried with synthetic phone photos of the covers: perspective, rotation, a cluttered background, a glare band, blur and JPEG compression, at three severities.

| photo | top-1 accuracy |
|---|---|
| clean | 99.2% |
| easy | 99.8% |
| medium | 98.5% |
| hard (cover at 40–65% of frame, ±15°) | 94.5% |

Search latency at this scale depends mostly on CPU. Server-side search time, same photos (easy/medium/hard) and `confidence_threshold=0.05` in every row:

| where | p50 | p95 |
|---|---|---|
| M-series Mac, 2 threads | 24 ms | 132 ms |
| Cloud Run, 2 vCPU, imret 0.1.1 | 348 ms | 2,823 ms |
| Cloud Run, 2 vCPU, imret 0.1.4 | 158 ms | 1,236 ms |

The p95 tail is queries that fall back to the 64-cell search. For photos, `confidence_threshold=0.05` (instead of the default 0.15) kept the same accuracy, cut fallbacks from 60% to 42% of queries, and cut p50 from 78 ms to 22 ms (Mac, 14 threads).

To reproduce:

```bash
python benchmark.py --dataset wikiart --transform wall --sizes 100,500,1000 --plot
python benchmark.py --sizes 100,500,1000,2000 --plot   # synthetic speed comparison
python benchmark.py --sizes 1000,5000,20000            # the Linux table
```

## Publishing a release

Wheels for Linux x86_64 and macOS arm64 are built automatically via GitHub Actions using cibuildwheel. To publish a new version to PyPI:

1. Update `version` in `pyproject.toml`.
2. Push a version tag:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```
3. The workflow builds wheels for all supported platforms and Python versions, then uploads to PyPI using trusted publishing (no API token required — configure the PyPI project to trust this repository's Actions environment named `pypi`).
