"""Precompose once, open in a fresh worker, and retain every failed RGB buffer."""
import json
from pathlib import Path

import numpy as np


def compose_scene(original_usd, override_usd, out_usd):
    """Only call before native open/annotator attachment. Preserve SI metadata."""
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    original = Path(original_usd).resolve(strict=True)
    overrides = [] if not override_usd else [Path(override_usd).resolve(strict=True)]
    target = Path(out_usd).resolve()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    source = Usd.Stage.Open(str(original))
    layer = Sdf.Layer.CreateNew(str(target))
    layer.subLayerPaths = [p.as_posix() for p in overrides] + [original.as_posix()]
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(source))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(source))
    UsdPhysics.SetStageKilogramsPerUnit(stage, UsdPhysics.GetStageKilogramsPerUnit(source))
    stage.SetTimeCodesPerSecond(source.GetTimeCodesPerSecond())
    stage.SetFramesPerSecond(source.GetFramesPerSecond())
    stage.SetStartTimeCode(source.GetStartTimeCode())
    stage.SetEndTimeCode(source.GetEndTimeCode())
    if source.GetDefaultPrim():
        stage.SetDefaultPrim(stage.GetPrimAtPath(source.GetDefaultPrim().GetPath()))
    layer.Save()
    return str(target)


def frame_statistics(raw, res=(2048, 1152)):
    a = np.asarray(raw)
    numeric = np.issubdtype(a.dtype, np.number)
    finite = bool(numeric and np.isfinite(a).all())
    has_rgb = a.ndim == 3 and a.shape[:2] == (res[1], res[0]) and a.shape[2] >= 3
    rgb = a[..., :3] if has_rgb else a
    return dict(shape=list(a.shape), dtype=str(a.dtype), finite=finite,
                mean=float(rgb.mean()) if rgb.size and finite else None,
                min=float(rgb.min()) if rgb.size and finite else None,
                max=float(rgb.max()) if rgb.size and finite else None,
                is_valid=bool(has_rgb and finite and 1 < rgb.mean() < 254))


class SafeProduct:
    """One fixed stage stack; rebuild only this product on a bad buffer."""
    def __init__(self, camera_path, diagnostics, res=(2048, 1152), channels=("rgb",)):
        import omni.replicator.core as rep

        self.rep, self.camera_path, self.res = rep, str(camera_path), tuple(res)
        self.diagnostics = Path(diagnostics)
        self.diagnostics.mkdir(parents=True, exist_ok=True)
        self.channels = tuple(channels)
        self.history = []
        self._create()

    def _create(self):
        self.product = self.rep.create.render_product(self.camera_path, self.res)
        self.annotators = {name: self.rep.AnnotatorRegistry.get_annotator(name) for name in self.channels}
        for annotator in self.annotators.values():
            annotator.attach(self.product)

    def close(self):
        for annotator in self.annotators.values():
            annotator.detach(self.product)
        self.product.destroy()

    def read(self, tag, warmup=16, retries=3):
        for attempt in range(retries):
            for _ in range(warmup if attempt == 0 else max(warmup, 16)):
                self.rep.orchestrator.step(rt_subframes=1, delta_time=0, pause_timeline=True)
            raw = np.asarray(self.annotators["rgb"].get_data()).copy()
            row = dict(tag=tag, attempt=attempt+1, **frame_statistics(raw, self.res))
            row["channels_attached"] = list(self.annotators)
            self.history.append(row)
            with (self.diagnostics / "capture-history.jsonl").open("a", encoding="utf8") as stream:
                stream.write(json.dumps(row)+"\n")
            if row["is_valid"]:
                return np.ascontiguousarray(raw[..., :3], dtype=np.uint8)
            # Preserve even zero-size/non-finite buffers, never just the exception.
            np.save(self.diagnostics / f"{tag}-failure-{attempt+1}.npy", raw)
            print("SAFE CAPTURE RETRY", json.dumps(row), flush=True)
            if attempt+1 < retries:
                self.close()
                self._create()
        raise RuntimeError(f"RGB invalid after {retries} attempts: {self.history[-1]}")


def capture(scene_usd, camera_path, camera_matrix, out_png, res=(2048, 1152),
            warmup=16, retries=3, prepare=None, lens=None, channels=("rgb",)):
    """Run inside one already initialized, dedicated SimulationApp worker."""
    import omni.usd
    from PIL import Image
    from pxr import Gf, UsdGeom

    context = omni.usd.get_context()
    if not context.open_stage(str(Path(scene_usd).resolve(strict=True))):
        raise RuntimeError("Precomposed stage could not open")
    stage = context.get_stage()
    stack = list(stage.GetRootLayer().subLayerPaths)
    if prepare:
        prepare(stage)
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.SetResetXformStack(True)
    m = np.asarray(camera_matrix, dtype=np.float64)
    UsdGeom.Xformable(camera).MakeMatrixXform().Set(Gf.Matrix4d(*m.ravel().tolist()))
    camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 100))
    camera.CreateFStopAttr(0)
    for name, value in (lens or {}).items():
        getattr(camera, "Get"+name+"Attr")().Set(value)
    out = Path(out_png)
    product = SafeProduct(camera_path, out.parent / out.stem, res, channels)
    try:
        pixels = product.read(out.stem, warmup, retries)
        Image.fromarray(pixels).save(out)
        np.save(out.with_suffix(".npy"), pixels)
        for name, annotator in product.annotators.items():
            if name == "rgb":
                continue
            data = annotator.get_data()
            if isinstance(data, dict):
                np.save(out.parent / (out.stem+"-"+name+".npy"), np.asarray(data["data"]))
                (out.parent / (out.stem+"-"+name+"-info.json")).write_text(json.dumps(data.get("info", {})), encoding="utf8")
            else:
                np.save(out.parent / (out.stem+"-"+name+".npy"), np.asarray(data))
        assert stack == list(stage.GetRootLayer().subLayerPaths), "Live layer stack changed"
        (out.parent / (out.stem+"-capture.json")).write_text(json.dumps(dict(status="PASS", scene=scene_usd,
            camera_matrix=m.tolist(), lens=lens, layer_stack=stack, stack_unchanged=True,
            attempts=product.history), indent=2), encoding="utf8")
        stage.GetRootLayer().Save()
        return True
    finally:
        product.close()
