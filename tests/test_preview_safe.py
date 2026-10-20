"""Safety checks for a root-layer-unit regression and invalid native buffers."""
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pxr import Sdf, Usd, UsdGeom

spec = importlib.util.spec_from_file_location("preview_safe", Path(__file__).parents[1]/"pipeline/native/preview_safe.py")
safe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(safe)


def test_composition_copies_units_and_never_rewrites_source(tmp_path):
    source=Usd.Stage.CreateNew(str(tmp_path/"source.usda"))
    UsdGeom.SetStageMetersPerUnit(source,.01)
    UsdGeom.SetStageUpAxis(source,"Z")
    source.SetTimeCodesPerSecond(50)
    source.SetEndTimeCode(42)
    source.DefinePrim("/World").CreateAttribute("answer",Sdf.ValueTypeNames.Int).Set(1)
    source.GetRootLayer().Save()
    digest=hashlib.sha256((tmp_path/"source.usda").read_bytes()).hexdigest()
    override=Usd.Stage.CreateNew(str(tmp_path/"override.usda"))
    override.OverridePrim("/World").CreateAttribute("answer",Sdf.ValueTypeNames.Int).Set(2)
    override.GetRootLayer().Save()
    target=safe.compose_scene(tmp_path/"source.usda",tmp_path/"override.usda",tmp_path/"composed.usda")
    result=Usd.Stage.Open(target)
    assert result.GetPrimAtPath("/World").GetAttribute("answer").Get()==2
    assert UsdGeom.GetStageMetersPerUnit(result)==.01 and UsdGeom.GetStageUpAxis(result)=="Z"
    assert result.GetTimeCodesPerSecond()==50 and result.GetEndTimeCode()==42
    assert hashlib.sha256((tmp_path/"source.usda").read_bytes()).hexdigest()==digest


def mock_product(tmp_path, buffers):
    product=safe.SafeProduct.__new__(safe.SafeProduct)
    product.rep=SimpleNamespace(orchestrator=SimpleNamespace(step=lambda **kw:None))
    product.res=(6,4)
    product.diagnostics=tmp_path
    product.history=[]
    iterator=iter(buffers)
    product.annotators={"rgb":SimpleNamespace(get_data=lambda:next(iterator))}
    product.close=lambda:None
    product._create=lambda:None
    return product


def test_invalid_buffer_is_retained_then_retry_recovers(tmp_path):
    product=mock_product(tmp_path,[np.empty((0,0,4),np.uint8),np.full((4,6,4),31,np.uint8)])
    image=product.read("frame",warmup=0,retries=3)
    assert image.shape==(4,6,3) and image.mean()==31
    assert (tmp_path/"frame-failure-1.npy").exists()
    assert not product.history[0]["is_valid"] and product.history[1]["is_valid"]


def test_three_bad_buffers_raise_without_saving_fake_frame(tmp_path):
    product=mock_product(tmp_path,[np.zeros((4,6,4),np.uint8) for _ in range(3)])
    with pytest.raises(RuntimeError,match="invalid after 3"):
        product.read("frame",warmup=0,retries=3)
    assert len(list(tmp_path.glob("frame-failure-*.npy")))==3
    assert not list(tmp_path.glob("*.png"))
