import tempfile
import numpy as np
import pytest
import imret


def _textured(seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (400, 400), dtype=np.uint8)


def _blank() -> np.ndarray:
    return np.zeros((400, 400), dtype=np.uint8)


@pytest.fixture
def built_vault():
    cfg = imret.OrbConfig()
    v = imret.Vault(cfg)
    v.add(_textured(0), "img_0")
    v.add(_textured(1), "img_1")
    v.add(_textured(2), "img_2")
    v.build()
    return v


# ── OrbConfig ─────────────────────────────────────────────────────────

def test_orb_config_default_fields():
    cfg = imret.OrbConfig()
    assert cfg.max_features > 0
    assert cfg.fast_cells > 0
    assert cfg.deep_cells >= cfg.fast_cells
    assert 0 < cfg.confidence_threshold < 1
    assert cfg.max_hamming_distance > 0


# ── Stats ─────────────────────────────────────────────────────────────

def test_stats_before_build():
    v = imret.Vault(imret.OrbConfig())
    v.add(_textured(), "a")
    s = v.stats()
    assert s["n_images"] == 1
    assert s["n_features"] > 0
    assert s["nlist"] == 0
    assert s["is_built"] is False


def test_stats_after_build(built_vault):
    s = built_vault.stats()
    assert s["n_images"] == 3
    assert s["n_features"] > 0
    assert s["nlist"] > 0
    assert s["is_built"] is True


# ── Search ────────────────────────────────────────────────────────────

def test_result_attributes(built_vault):
    result = built_vault.search(_textured(0))
    assert hasattr(result, "label")
    assert hasattr(result, "confidence")
    assert hasattr(result, "fallback_used")
    assert isinstance(result.label, str)
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.fallback_used, bool)


def test_search_correct_label(built_vault):
    assert built_vault.search(_textured(0)).label == "img_0"
    assert built_vault.search(_textured(1)).label == "img_1"
    assert built_vault.search(_textured(2)).label == "img_2"


def test_blank_query_returns_unknown(built_vault):
    assert built_vault.search(_blank()).label == "Unknown"


# ── add_batch ─────────────────────────────────────────────────────────

def test_add_batch_labels_correctly():
    cfg = imret.OrbConfig()
    v = imret.Vault(cfg)
    images = [_textured(i) for i in range(4)]
    labels = [f"batch_{i}" for i in range(4)]
    v.add_batch(images, labels)
    v.build()
    for i, img in enumerate(images):
        assert v.search(img).label == f"batch_{i}"


# ── Save / load ───────────────────────────────────────────────────────

def test_save_load_roundtrip(built_vault):
    with tempfile.TemporaryDirectory() as d:
        prefix = f"{d}/vault"
        built_vault.save(prefix)
        loaded = imret.Vault.load_from_disk(prefix, imret.OrbConfig())
        assert loaded.search(_textured(0)).label == "img_0"
        assert loaded.search(_textured(1)).label == "img_1"

def test_save_load_preserves_stats(built_vault):
    with tempfile.TemporaryDirectory() as d:
        prefix = f"{d}/vault"
        built_vault.save(prefix)
        loaded   = imret.Vault.load_from_disk(prefix, imret.OrbConfig())
        orig     = built_vault.stats()
        loaded_s = loaded.stats()
        assert loaded_s["n_images"]   == orig["n_images"]
        assert loaded_s["n_features"] == orig["n_features"]
        assert loaded_s["nlist"]      == orig["nlist"]
        assert loaded_s["is_built"]   is True


def test_incremental_add_after_load():
    cfg = imret.OrbConfig()
    with tempfile.TemporaryDirectory() as d:
        prefix = f"{d}/vault"
        v = imret.Vault(cfg)
        v.add(_textured(0), "img_0")
        v.build()
        v.save(prefix)

        v2 = imret.Vault.load_from_disk(prefix, cfg)
        v2.add(_textured(1), "img_1")
        v2.build()

        assert v2.search(_textured(0)).label == "img_0"
        assert v2.search(_textured(1)).label == "img_1"
        assert v2.stats()["n_images"] == 2
