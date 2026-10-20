import numpy as np
import pytest

from pipeline.simforge.materials.graycard import acceptance, linear_to_srgb, ratios, srgb_to_linear


def test_luminance_transfer_and_single_scalar_ratios():
    values=np.array([0,1,10,11,64,128,254,255])
    assert np.allclose(linear_to_srgb(srgb_to_linear(values)),values,atol=1e-12)
    reference=dict(gray18=dict(linear_luminance_mean=.2),white90=dict(linear_luminance_mean=.8))
    measured=dict(gray18=dict(linear_luminance_mean=.02),white90=dict(linear_luminance_mean=.08))
    assert ratios(measured,reference)==pytest.approx(dict(gray18=.1,white90=.1))
    assert not acceptance(ratios(measured,reference))["gray_pass"]


def test_white_is_independent_failure_not_hidden_by_gray_fit():
    result=acceptance(dict(gray18=1.01,white90=.8))
    assert result["gray_pass"] and not result["white_pass"]
    assert result["gray_residual"]==pytest.approx(.01)
    assert result["white_residual"]==pytest.approx(-.2)
    assert acceptance(dict(gray18=1,white90=1))["gray_pass"]
    assert acceptance(dict(gray18=1,white90=1))["white_pass"]
