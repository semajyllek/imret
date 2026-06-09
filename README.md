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

```bash
ctest --test-dir build --output-on-failure
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

Pre-built binary wheels are available for Linux x86_64 and macOS arm64, covering Python 3.9–3.12. Google Colab is supported without any additional setup.

Until imret is published to PyPI, install from source — see [Building from source (Python)](#building-from-source-python).

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

Requirements: Python >= 3.8, pybind11, scikit-build-core, OpenCV, FAISS, OpenMP.

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

Generated by `benchmark.py` on macOS arm64, `max_features=500`, 20 queries per size.

`imret` and `imret-batch` differ only at ingest: the batch variant uses `add_batch()` which extracts ORB features in parallel via OpenMP, reducing build time by up to 37% at large N. Search latency is identical.

### Speed comparison — synthetic images, no transform

All methods score 100% accuracy on structurally distinct synthetic images, so this measures speed only.

**Search latency p50 (ms)**

| N | imret | imret-batch | bfmatcher | imagehash |
|---|---|---|---|---|
| 100 | 3.19 | 3.11 | 8.60 | 0.20 |
| 500 | 3.26 | 3.27 | 39.97 | 0.52 |
| 1,000 | 3.30 | 3.33 | 85.14 | 0.90 |
| 2,000 | 3.36 | 3.33 | 166.58 | 1.67 |

**Build time (s)**

| N | imret | imret-batch | bfmatcher | imagehash |
|---|---|---|---|---|
| 100 | 1.11 | 0.98 | 0.11 | 0.44 |
| 500 | 15.64 | 15.12 | 0.56 | 0.07 |
| 1,000 | 31.19 | 30.18 | 1.11 | 0.17 |
| 2,000 | 62.20 | 60.27 | 2.25 | 0.33 |

imret search latency is nearly flat (3.19ms → 3.36ms across 20× more images). BFMatcher grows linearly. Build time is higher because imret trains a FAISS IVF index upfront — a one-time cost at index time.

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
| 100 | 21.64 | 21.54 | 29.74 | 1.69 |
| 500 | 22.53 | 22.27 | 71.64 | 2.07 |
| 1,000 | 22.21 | 22.33 | 130.68 | 2.42 |

imagehash is fast but effectively random on real images under this transform (10–15% ≈ 1-in-10 chance). imret maintains 100% accuracy. BFMatcher matches imret accuracy but latency grows linearly.

To reproduce:

```bash
python benchmark.py --dataset wikiart --transform wall --sizes 100,500,1000 --plot
python benchmark.py --sizes 100,500,1000,2000 --plot   # synthetic speed comparison
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
