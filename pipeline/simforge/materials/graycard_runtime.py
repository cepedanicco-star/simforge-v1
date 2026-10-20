"""Native-only card rendering and bounded one-knob exposure calibration."""
import json
import math
from pathlib import Path

import numpy as np

from .camera_pairing import require_camera_pairing
from .graycard import acceptance, measure_cards, ratios
from ..sensors.geometry import camera_contract
from ..sensors.export import plain


def calibrate(stage, camera, transform, rgb, semantic, camera_params, rep, settings, config_path):
    from PIL import Image
    from pxr import Gf, Sdf, UsdGeom, UsdShade
    from isaacsim.core.utils.semantics import add_update_semantics
    from pipeline.ue_isaac_pipeline.camera_projection import equivalent_native_lens

    config=json.loads(Path(config_path).read_text(encoding="utf8"))
    root=Path(config["output"])
    output=root/"isaac"
    output.mkdir(exist_ok=False)
    reference=json.loads((root/"ue-gray-measurements.json").read_text(encoding="utf8"))
    if reference["status"]!="PASS":
        raise ValueError("UE graycard capture must pass before calibration")
    reference=reference["cards"]
    group=UsdGeom.Xform.Define(stage,"/World/GrayCardCalibration")
    construction=[]
    for card in config["calibration_cards"]:
        cube=UsdGeom.Cube.Define(stage,str(group.GetPath())+"/"+card["name"])
        cube.CreateSizeAttr(1.)
        cube.AddTranslateOp().Set(Gf.Vec3d(*card["position_m"]))
        cube.AddScaleOp().Set(Gf.Vec3f(*card["dimensions_m"]))
        material=UsdShade.Material.Define(stage,str(group.GetPath())+"/M_"+card["name"])
        shader=UsdShade.Shader.Define(stage,str(material.GetPath())+"/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor",Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(card["reflectance"]))
        shader.CreateInput("roughness",Sdf.ValueTypeNames.Float).Set(.5)
        shader.CreateInput("metallic",Sdf.ValueTypeNames.Float).Set(0.)
        shader.CreateInput("useSpecularWorkflow",Sdf.ValueTypeNames.Int).Set(1)
        shader.CreateInput("specularColor",Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),"surface")
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
        add_update_semantics(cube.GetPrim(),"calibration_"+card["name"])
        construction.append(dict(**card,prim=str(cube.GetPath()),roughness=.5,metallic=0,specular=0))
    (root/"isaac-card-construction.json").write_text(json.dumps(construction,indent=2),encoding="utf8")
    view=config["views"][0]
    lens,_=equivalent_native_lens(view["lens"])
    transform.Set(Gf.Matrix4d(*np.asarray(view["usd_camera_matrix"], dtype=np.float64).ravel().tolist()))
    for name,value in lens.items():
        getattr(camera,"Get"+name+"Attr")().Set(value)
    knob=config["exposure"]["knob"]
    if knob!="/rtx/post/tonemap/exposureTime":
        raise ValueError("This run is locked to exposureTime as the only degree of freedom")
    invariant={key:settings.get(key) for key in ["/rtx/post/tonemap/filmIso","/rtx/post/tonemap/fNumber","/rtx/post/tonemap/op","/rtx/post/histogram/enabled"]}
    trials=[]

    def save_report(status,best=None):
        document=dict(status=status,knob=knob,baseline_exposure_seconds=config["exposure"]["baseline_seconds"],
            reference_ue=reference,unchanged_settings=invariant,fixture=config["lights"],trials=trials,
            locked_exposure_seconds=best["exposure_seconds"] if best else None,
            exposure_factor=best["exposure_seconds"]/config["exposure"]["baseline_seconds"] if best else None,
            baseline_ratios=trials[0]["ratios"] if trials else None,final_ratios=best["ratios"] if best else None,
            acceptance=acceptance(best["ratios"]) if best else None,
            only_engine_exposure_changed=True,light_units_normalize_angle_unchanged=True,
            normalization_fallback_authorized_by_user=True,
            normalization_rule="If white fails, separate diagnostic: each engine display-linear RGB * (0.18 / its measured gray-card luminance). No image-wise fit, no replacement of raw renders.")
        path=root/"calibration-report.json"
        # Append-only progress snapshots; final report is created only once.
        if status=="RUNNING":
            path=output/f"progress-{len(trials):02d}.json"
        elif path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(plain(document),indent=2),encoding="utf8")
        return document

    def sample(exposure):
        if len(trials)>=config["exposure"]["max_trials"]:
            raise RuntimeError("Bounded exposure calibration exhausted")
        settings.set(knob,float(exposure))
        actual=float(settings.get(knob))
        if actual!=float(exposure):
            raise ValueError("Runtime clamped requested exposure")
        if any(settings.get(k)!=v for k,v in invariant.items()):
            raise ValueError("A second exposure/tonemap setting changed")
        for _ in range(4):
            rep.orchestrator.step(rt_subframes=4,delta_time=0,pause_timeline=True)
        image=np.asarray(rgb.get_data())[...,:3].astype(np.uint8)
        native=camera_params.get_data()
        intrinsics=camera_contract(native)
        world=np.asarray(UsdGeom.XformCache(0).GetLocalToWorldTransform(camera.GetPrim()))
        measured=dict(position_cm=(world[3,:3]*[100,-100,100]).tolist(),
            forward=(-world[2,:3]*[1,-1,1]).tolist(),up=(world[1,:3]*[1,-1,1]).tolist(),
            horizontal_fov_degrees=math.degrees(2*math.atan(2048/(2*intrinsics["fx"]))),
            vertical_fov_degrees=math.degrees(2*math.atan(1152/(2*intrinsics["fy"]))))
        camera_gate=require_camera_pairing(view,measured)
        cards=measure_cards(image,config,intrinsics)
        seg=semantic.get_data()
        ids=np.asarray(seg["data"]).squeeze()
        coverage={}
        for name in ["gray18","white90"]:
            label_ids=[int(k) for k,v in seg["info"]["idToLabels"].items() if v.get("class")=="calibration_"+name]
            card_mask=np.isin(ids,label_ids)
            coverage[name]=float(card_mask.mean())
            x0,y0,x1,y1=cards[name]["roi_xyxy"]
            if card_mask[y0:y1,x0:x1].mean()<.999:
                raise ValueError("Card centre ROI not inside its actual rendered semantic mask")
        if coverage["gray18"]<.30:
            raise ValueError("Native rendered gray card coverage below 30 percent")
        index=len(trials)
        filename=f"trial-{index:02d}.png"
        Image.fromarray(image).save(output/filename)
        record=dict(index=index,image=filename,exposure_seconds=actual,cards=cards,ratios=ratios(cards,reference),
                    actual_semantic_coverage=coverage,camera_gate=camera_gate,native_camera=plain(native))
        (output/f"trial-{index:02d}.json").write_text(json.dumps(plain(record),indent=2),encoding="utf8")
        trials.append(record)
        save_report("RUNNING")
        print("GRAYCARD TRIAL",index,"exposure",actual,"ratios",record["ratios"],flush=True)
        return record

    try:
        current=sample(config["exposure"]["baseline_seconds"])
        lo,hi=None,None
        if current["ratios"]["gray18"]<1:
            lo=current
            while hi is None:
                current=sample(current["exposure_seconds"]*4)
                if current["ratios"]["gray18"]>=1:
                    hi=current
                else:
                    lo=current
        else:
            hi=current
            while lo is None:
                current=sample(current["exposure_seconds"]/4)
                if current["ratios"]["gray18"]<=1:
                    lo=current
                else:
                    hi=current
        best=min(trials,key=lambda x:abs(x["ratios"]["gray18"]-1))
        # Target the centre of the user's ±3% interval, not a relaxed criterion.
        while abs(best["ratios"]["gray18"]-1)>.005 and len(trials)<config["exposure"]["max_trials"]:
            current=sample(math.sqrt(lo["exposure_seconds"]*hi["exposure_seconds"]))
            if current["ratios"]["gray18"]<1:
                lo=current
            else:
                hi=current
            best=min(trials,key=lambda x:abs(x["ratios"]["gray18"]-1))
        check=acceptance(best["ratios"])
        if not check["gray_pass"]:
            save_report("FAIL_STOPPED",best)
            raise ValueError("Exposure-only gray-card fit failed the unchanged ±3% limit")
        if best["cards"]["gray18"]["linear_luminance_mean"]<=0:
            raise ValueError("Cannot normalize a black calibration ROI")
        status="EXPOSURE_ALIGNED" if check["white_pass"] else "FALLBACK_LINEAR_NORMALIZATION"
        settings.set(knob,best["exposure_seconds"])
        document=save_report(status,best)
        print("GRAYCARD LOCKED",document["locked_exposure_seconds"],status,flush=True)
        return document
    except Exception:
        if not (root/"calibration-report.json").exists():
            save_report("FAIL_STOPPED",min(trials,key=lambda x:abs(x["ratios"]["gray18"]-1)) if trials else None)
        raise
    finally:
        group.GetPrim().SetActive(False)
