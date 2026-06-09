import os
import numpy as np
from typing import Union

# Import the compiled C++ binary
# (The try/except handles errors gracefully if the user didn't compile it)
try:
    from ._core import OrbConfig, MatchResult, Vault as _VaultCore
except ImportError:
    raise ImportError(
        "Could not load the compiled C++ engine. "
        "Ensure you have installed imret correctly (e.g., pip install .)"
    )

class Vault:
    """
    The main orchestrator for the imret Image Retrieval engine.
    This wraps the high-performance C++ backend.
    """
    def __init__(self, config: OrbConfig = None):
        if config is None:
            config = OrbConfig()
        self._engine = _VaultCore(config)
        self._is_built = False

    def add(self, image_matrix: np.ndarray, label: str) -> None:
        """Adds an image to the RAM buffer."""
        if not isinstance(image_matrix, np.ndarray):
            raise TypeError("Image must be a numpy ndarray (cv2 image).")
        if image_matrix.dtype != np.uint8:
            raise ValueError("Image matrix must be 8-bit unsigned integers (uint8).")
        self._engine.add(image_matrix, label)

    def add_batch(self, images: list, labels: list) -> None:
        """Adds multiple images in parallel using OpenMP. Prefer over repeated add() calls."""
        self._engine.add_batch(images, labels)

    def build(self) -> None:
        """Compiles the FAISS Voronoi cells. Required before searching."""
        self._engine.build()
        self._is_built = True

    def search(self, query_matrix: np.ndarray) -> MatchResult:
        """Searches the built vault for a matching image."""
        if not self._is_built:
            raise RuntimeError("You must call .build() before searching!")

        return self._engine.search(query_matrix)

    def save(self, prefix: str) -> None:
        """Saves the built vault to <prefix>.faiss and <prefix>.meta."""
        self._engine.save(prefix)

    def stats(self) -> dict:
        """Returns n_images, n_features, nlist, is_built."""
        return self._engine.stats()

    def load(self, prefix: str) -> None:
        """Loads a vault from <prefix>.faiss and <prefix>.meta."""
        self._engine.load(prefix)
        self._is_built = True

    @classmethod
    def load_from_disk(cls, prefix: str, config=None) -> "Vault":
        """Convenience: creates a Vault and loads from disk in one call."""
        v = cls(config)
        v.load(prefix)
        return v
