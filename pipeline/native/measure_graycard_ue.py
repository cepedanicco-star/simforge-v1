"""Check the calibration-only UE view and measure its unmodified card PNG."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from pipeline.simforge.materials.camera_pairing import camera_pairing  # noqa: E402
from pipeline.simforge.materials.graycard import measure_cards  # noqa: E402


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text(encoding="utf8"))
    root=Path(config["output"])
    out=root/"ue-gray-measurements.json"
    if out.exists():
        raise FileExistsError(out)
    result=dict(status="FAIL_STOPPED",cards=None)
    try:
        if (root/"ue-card-failure.json").exists() or (root/"ue/failure.json").exists():
            raise ValueError("UE calibration worker failed; preserve logs")
        report=json.loads((root/"ue/capture-report.json").read_text(encoding="utf8"))
        assert report["status"]=="PASS" and len(report["records"])==len(config["views"])==1
        record=report["records"][0]
        gate=camera_pairing(config["views"][0],record)
        assert gate["status"]=="PASS"
        constructed=json.loads((root/"ue-card-construction.json").read_text(encoding="utf8"))
        for saved,card in zip(constructed,config["calibration_cards"],strict=True):
            x,y,z=card["position_m"]
            assert np.allclose(saved["actual_position_cm"],[x*100,-y*100,z*100],rtol=0,atol=1e-4)
            assert np.allclose(saved["actual_scale_m"],card["dimensions_m"],rtol=0,atol=1e-6)
        picture_path=root/"ue"/record["image"]
        with Image.open(picture_path) as im:
            rgb=np.asarray(im.convert("RGB"))
        cards=measure_cards(rgb,config)
        if cards["white90"]["linear_luminance_mean"]<=cards["gray18"]["linear_luminance_mean"] or cards["gray18"]["linear_luminance_mean"]<=0:
            raise ValueError("Gray/white ROI invalid or zero; do not fit an invalid reference")
        fixture=json.loads((root/"ue/fixture.json").read_text(encoding="utf8"))
        assert fixture["lighting"]==config["lights"] and fixture["iso"]==100 and fixture["fnumber"]==4 and fixture["shutter_s"]==1/16
        result=dict(status="PASS",cards=cards,camera_gate=gate,source_png=str(picture_path),
                    source_png_sha256=hashlib.sha256(picture_path.read_bytes()).hexdigest(),fixture=fixture,
                    measurement="sRGB inverse EOTF, Y=.2126729 R+.7151522 G+.0721750 B; centre ROI mean, no image correction")
        overlay=Image.fromarray(rgb)
        draw=ImageDraw.Draw(overlay)
        for name,card in cards.items():
            draw.rectangle(card["roi_xyxy"],outline="#ffbb44",width=4)
            draw.text((card["roi_xyxy"][0],card["roi_xyxy"][1]-22),name+" centre ROI",fill="#ffbb44")
        overlay.save(root/"ue-card-rois.png")
    except Exception as error:
        result["error"]=repr(error)
    out.write_text(json.dumps(result,indent=2),encoding="utf8")
    print(json.dumps(result,indent=2))
    return 0 if result["status"]=="PASS" else 1


if __name__=="__main__":
    raise SystemExit(main())
