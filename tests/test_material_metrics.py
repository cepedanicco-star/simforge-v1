from pathlib import Path

import numpy as np
import pytest

from pipeline.simforge.materials.metrics import delta_e2000, srgb_to_lab, ssim_map


def test_all_34_published_sharma_ciede2000_pairs():
    # Author-provided numerical test data, not third-party implementation code.
    data = np.loadtxt(Path(__file__).with_name("fixtures") / "ciede2000-sharma.txt")
    assert len(data) == 34
    assert np.max(np.abs(delta_e2000(data[:,:3], data[:,3:6])-data[:,6])) <= 5e-5
    assert np.allclose(delta_e2000(data[:,:3],data[:,3:6]),delta_e2000(data[:,3:6],data[:,:3]), atol=1e-12)


def test_neutral_color_and_identical_image():
    lab = srgb_to_lab(np.array([[0,0,0],[255,255,255]], dtype=np.uint8))
    assert lab[0] == pytest.approx([0,0,0], abs=1e-12)
    assert lab[1] == pytest.approx([100,0,0], abs=2e-5)
    rgb = np.random.default_rng(0).integers(0,256,(31,35,3),dtype=np.uint8)
    assert np.all(delta_e2000(srgb_to_lab(rgb), srgb_to_lab(rgb)) == 0)
    assert np.allclose(ssim_map(rgb,rgb),1,atol=1e-12)
    assert ssim_map(rgb,255-rgb)[5:-5,5:-5].mean() < .1
