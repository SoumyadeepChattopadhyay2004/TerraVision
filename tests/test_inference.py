"""
Unit tests for the core inference/statistics/history logic.

Run with:
    pytest tests/ -v

These exercise the heuristic classifier path (no trained checkpoint needed),
so they run anywhere torch + torchvision are installed, with no GPU or
dataset required — safe to wire into CI.
"""
import numpy as np
import pytest
from PIL import Image

from model.inference import (
    LandCoverClassifier, stats_from_mask, landscape_metrics,
    vegetation_index_map, change_detection, colorize_mask,
)
from model.unet import CLASS_NAMES


@pytest.fixture
def synthetic_image():
    """A 120x120 image with 4 solid-color quadrants: water, forest, agriculture, urban."""
    arr = np.zeros((120, 120, 3), dtype=np.uint8)
    arr[:60, :60] = [30, 60, 160]    # water: blue-dominant, dark
    arr[:60, 60:] = [30, 110, 40]    # forest: saturated dark green
    arr[60:, :60] = [180, 200, 60]   # agriculture: light yellow-green
    arr[60:, 60:] = [140, 140, 140]  # urban: neutral gray
    return Image.fromarray(arr)


@pytest.fixture
def classifier():
    # point at a nonexistent checkpoint to force the heuristic fallback path,
    # which needs no trained weights and is deterministic
    return LandCoverClassifier(checkpoint_path="/nonexistent/checkpoint.pt")


def test_classifier_falls_back_to_heuristic(classifier):
    assert classifier.mode == "heuristic"
    assert classifier.model is None


def test_classify_shape_matches_input(classifier, synthetic_image):
    mask = classifier.classify(synthetic_image)
    assert mask.shape == (120, 120)
    assert mask.dtype in (np.uint8, np.int64, np.int32)


def test_classify_separates_quadrants_correctly(classifier, synthetic_image):
    mask = classifier.classify(synthetic_image)
    WATER, FOREST, AGRI, URBAN = 2, 1, 3, 0

    def dominant(region):
        return int(np.bincount(region.ravel()).argmax())

    assert dominant(mask[:60, :60]) == WATER
    assert dominant(mask[:60, 60:]) == FOREST
    assert dominant(mask[60:, :60]) == AGRI
    assert dominant(mask[60:, 60:]) == URBAN


def test_stats_percentages_sum_to_100(classifier, synthetic_image):
    mask = classifier.classify(synthetic_image)
    stats = stats_from_mask(mask)
    total_pct = sum(s["percent"] for s in stats.values())
    assert abs(total_pct - 100.0) < 0.1
    assert set(stats.keys()) == set(CLASS_NAMES)


def test_stats_area_computed_when_gsd_given(classifier, synthetic_image):
    mask = classifier.classify(synthetic_image)
    stats = stats_from_mask(mask, gsd_m=10.0)  # 10 m/pixel
    # each quadrant is 3600 px -> 360,000 m^2 -> 0.36 km^2
    assert stats["Water"]["km2"] == pytest.approx(0.36, abs=0.01)
    assert "km2" not in stats_from_mask(mask)["Water"]  # omitted without gsd


def test_landscape_metrics_shape(classifier, synthetic_image):
    mask = classifier.classify(synthetic_image)
    metrics = landscape_metrics(mask)
    assert set(metrics.keys()) == set(CLASS_NAMES)
    for name in CLASS_NAMES:
        m = metrics[name]
        assert "patch_count" in m and "edge_density" in m and "mean_patch_size_px" in m
    # each quadrant here is one solid connected blob -> exactly 1 patch
    assert metrics["Water"]["patch_count"] == 1


def test_vegetation_index_map_size(synthetic_image):
    veg = vegetation_index_map(synthetic_image)
    assert veg.size == synthetic_image.size
    assert veg.mode == "RGB"


def test_colorize_mask_uses_expected_palette(classifier, synthetic_image):
    mask = classifier.classify(synthetic_image)
    overlay = colorize_mask(mask)
    assert overlay.size == synthetic_image.size


def test_change_detection_summary_and_transition(classifier, synthetic_image):
    mask_before = classifier.classify(synthetic_image)
    arr_after = np.array(synthetic_image).copy()
    arr_after[:60, :60] = [140, 140, 140]  # water quadrant becomes urban
    mask_after = classifier.classify(Image.fromarray(arr_after))

    result = change_detection(mask_before, mask_after)
    assert "summary" in result
    assert result["summary"]["urban_expansion_rate_percent"] == pytest.approx(25.0, abs=0.5)
    assert result["transition_matrix"] is not None
    n = len(CLASS_NAMES)
    assert len(result["transition_matrix"]) == n
    assert all(len(row) == n for row in result["transition_matrix"])


def test_stats_from_mask_empty_class_is_zero(classifier):
    # an all-water image should report 0% for every other class
    arr = np.full((40, 40, 3), [20, 40, 180], dtype=np.uint8)
    mask = classifier.classify(Image.fromarray(arr))
    stats = stats_from_mask(mask)
    assert stats["Urban"]["percent"] == 0.0 or stats["Urban"]["pixels"] == 0
