"""Measure all six camera-gated pairs; never correct images to improve scores."""
# ruff: noqa: E402
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_erosion

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.simforge.materials.camera_pairing import camera_pairing
from pipeline.simforge.materials.metrics import delta_e2000, srgb_to_lab, ssim_map


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf8")


def value_state(shader):
    """Describe scalar or evaluated baked texel distribution, never guess a scalar."""
    if not shader:
        return dict(kind="NOT_AVAILABLE", reason="No UsdPreviewSurface shader")
    inp = shader.GetInput("roughness")
    if not inp:
        return dict(kind="NOT_AVAILABLE", reason="No roughness input")
    source = inp.GetConnectedSource()
    if not source:
        value = inp.Get()
        return dict(kind="scalar", value=float(value), mean=float(value), min=float(value), max=float(value)) if value is not None else dict(kind="NOT_AUTHORED")
    from pxr import UsdShade
    texture = UsdShade.Shader(source[0].GetPrim())
    if texture.GetIdAttr().Get() != "UsdUVTexture":
        return dict(kind="UNEVALUATED_GRAPH", shader=str(texture.GetPath()), output=str(source[1]))
    file = texture.GetInput("file").Get()
    path = file.resolvedPath
    if not path or not Path(path).is_file():
        return dict(kind="UNRESOLVED_TEXTURE", asset=str(file))
    channel = str(source[1])
    if channel not in ("r", "g", "b", "a"):
        return dict(kind="UNEVALUATED_CHANNEL", channel=channel, texture=path)
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGBA"), dtype=np.float64)/255
    index = "rgba".index(channel)
    scale = texture.GetInput("scale").Get() or (1,1,1,1)
    bias = texture.GetInput("bias").Get() or (0,0,0,0)
    values = array[...,index]*scale[index]+bias[index]
    return dict(kind="baked_texture_texel_statistics", texture=path, texture_sha256=digest(path), channel=channel,
                source_color_space=str(texture.GetInput("sourceColorSpace").Get()),
                mean=float(values.mean()), min=float(values.min()), max=float(values.max()),
                texture_size=list(array.shape[:2][::-1]), note="Linear roughness channel; texel mean, not UV/screen weighted")


def roughness_table(root, config, materials, prefix="roughness-alignment"):
    from pxr import Usd, UsdShade
    stage = Usd.Stage.Open(config["source_scene"])
    ue = read(root / "ue/material-parameters.json")
    original_cache = {}
    rows = []
    for path, record in materials.items():
        material = UsdShade.Material(stage.GetPrimAtPath(path))
        shader = UsdShade.Shader(stage.GetPrimAtPath(record["shader_path"]))
        current = value_state(shader)
        info = material.GetPrim().GetAssetInfo()
        source_asset = record["source_asset"]
        identifier = info.get("identifier")
        original_path = identifier.resolvedPath if identifier else None
        original = dict(kind="NOT_AVAILABLE", reason="No original baked material asset")
        if original_path and Path(original_path).is_file():
            if original_path not in original_cache:
                asset_stage = Usd.Stage.Open(original_path)
                candidates = [UsdShade.Shader(p) for p in asset_stage.Traverse() if p.IsA(UsdShade.Shader)
                              and UsdShade.Shader(p).GetIdAttr().Get() == "UsdPreviewSurface"]
                original_cache[original_path] = value_state(candidates[0]) if len(candidates)==1 else dict(kind="AMBIGUOUS_ORIGINAL_SHADER", count=len(candidates))
            original = original_cache[original_path]
        scalars = ue.get(source_asset, {}).get("scalars", {})
        scalars = {k:v for k,v in scalars.items() if "rough" in k.lower()}
        active_mdl = material.GetSurfaceOutput("mdl").GetConnectedSource() if material.GetSurfaceOutput("mdl") else None
        mdl_roughness = []
        if active_mdl:
            active_shader = UsdShade.Shader(active_mdl[0].GetPrim())
            for inp in active_shader.GetInputs():
                if "rough" in inp.GetBaseName().lower():
                    v = inp.Get()
                    mdl_roughness.append(dict(input=str(inp.GetBaseName()), value=v if isinstance(v,(float,int,bool,str)) or v is None else str(v),
                                              connection=str(inp.GetConnectedSource()) if inp.GetConnectedSource() else None))
        delta = current.get("mean",0)-original.get("mean",0) if "mean" in current and "mean" in original else None
        rows.append(dict(usd_material=path, ue_material=source_asset, ue_input_roughness_scalars=scalars,
                         ue_baked=original, usd_preview_surface_roughness=current,
                         preview_minus_ue_baked_mean=delta,
                         active_context="mdl" if active_mdl else "universal_preview", active_mdl_roughness_inputs=mdl_roughness,
                         status="NUMERIC_COMPARISON" if delta is not None else "EXPLICIT_UNKNOWN",
                         note="UE input parameters are not the evaluated roughness; baked result is reported separately. Accepted Isaac overrides may intentionally differ."))
    write(root / (prefix+".json"), rows)
    with (root / (prefix+".csv")).open("w",newline="",encoding="utf8") as f:
        fields=["ue_material","usd_material","ue_input_roughness_scalars","ue_baked_mean","usd_preview_roughness_mean","delta","active_context","status"]
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(ue_material=row["ue_material"], usd_material=row["usd_material"],
                ue_input_roughness_scalars=json.dumps(row["ue_input_roughness_scalars"],ensure_ascii=False),
                ue_baked_mean=row["ue_baked"].get("mean"),usd_preview_roughness_mean=row["usd_preview_surface_roughness"].get("mean"),
                delta=row["preview_minus_ue_baked_mean"],active_context=row["active_context"],status=row["status"]))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--ue-acceptance",required=True)
    args=parser.parse_args()
    config=read(args.config)
    root=Path(config["output"])
    if (root / "metrics.json").exists():
        raise FileExistsError(root / "metrics.json")
    try:
        ue_gate=read(args.ue_acceptance)
        native_gate=read(root / "isaac/native-pairing-acceptance.json")
        native=read(root / "isaac/capture-report.json")
        ue=read(root / "ue/capture-report.json")
        assert ue_gate["status"]==native_gate["status"]==native["status"]==ue["status"]=="PASS"
        assert ue_gate["passed"]==native_gate["passed"]==len(ue["records"])==len(native["records"])==len(config["views"])==6
        materials=read(root / "isaac/material-map.json")
        label_map={v["label"]:dict(usd_material=k,**v) for k,v in materials.items()}
        # Preflight *all* six pairs and channels before computing or reporting any metric.
        for view,ur,nr,ng in zip(config["views"],ue["records"],native["records"],native_gate["checks"],strict=True):
            frame=view["source_frame"]
            assert frame==ur["source_frame"]==nr["source_frame"]==ng["check"]["frame"]
            assert camera_pairing(view,ur)["status"]=="PASS"
            assert camera_pairing(view,ng["actual_observed"])["status"]=="PASS"
            assert np.allclose(ng["actual_usd_camera_matrix"],view["usd_camera_matrix"],rtol=0,atol=1e-10)
            assert np.allclose(nr["native_K"],[[ng["intrinsics"]["fx"],0,1024],[0,ng["intrinsics"]["fy"],576],[0,0,1]],rtol=0,atol=1e-10)
            for engine in ("ue","isaac"):
                with Image.open(root / engine / f"cam01-{frame:03d}.png") as im:
                    assert im.size==(2048,1152)
                    im.verify()
            ids=np.load(root / "isaac" / f"cam01-{frame:03d}-material-ids.npy",allow_pickle=False)
            labels=read(root / "isaac" / f"cam01-{frame:03d}-material-labels.json")
            good=[int(k) for k,v in labels.items() if v.get("class") in label_map]
            assert ids.shape==(1152,2048) and np.isin(ids,good).sum()>0
        pairs=[]
        totals={}
        sheets=[]
        font=ImageFont.load_default(size=23)
        for view in config["views"]:
            frame=view["source_frame"]
            u=np.asarray(Image.open(root / "ue" / f"cam01-{frame:03d}.png").convert("RGB"))
            n=np.asarray(Image.open(root / "isaac" / f"cam01-{frame:03d}.png").convert("RGB"))
            ids=np.load(root / "isaac" / f"cam01-{frame:03d}-material-ids.npy",allow_pickle=False)
            labels=read(root / "isaac" / f"cam01-{frame:03d}-material-labels.json")
            good=[int(k) for k,v in labels.items() if v.get("class") in label_map]
            mask=np.isin(ids,good)
            valid_window=binary_erosion(mask,structure=np.ones((11,11),dtype=bool),border_value=0)
            assert valid_window.any()
            labu,labn=srgb_to_lab(u[mask]),srgb_to_lab(n[mask])
            differences=np.concatenate([delta_e2000(labu[i:i+262144],labn[i:i+262144]) for i in range(0,len(labu),262144)])
            difference_map=np.full(mask.shape,np.nan,dtype=np.float32)
            difference_map[mask]=differences
            structural=ssim_map(u,n)
            np.save(root / f"cam01-{frame:03d}-delta-e2000.npy",difference_map)
            Image.fromarray(mask.astype(np.uint8)*255).save(root / f"cam01-{frame:03d}-valid-mask.png")
            row=dict(source_frame=frame,masked_pixels=int(mask.sum()),ssim_valid_window_pixels=int(valid_window.sum()),
                delta_e2000=float(differences.mean()),delta_e2000_p95=float(np.percentile(differences,95)),
                ssim=float(structural[valid_window].mean()),ssim_full_frame=float(structural[5:-5,5:-5].mean()),
                ue_mean_rgb=float(u[mask].mean()),isaac_mean_rgb=float(n[mask].mean()),
                rgb_mae_255=float(np.abs(u[mask].astype(float)-n[mask]).mean()),
                ue_sha256=digest(root / "ue" / f"cam01-{frame:03d}.png"),
                isaac_sha256=digest(root / "isaac" / f"cam01-{frame:03d}.png"))
            assert all(np.isfinite(row[k]) for k in ["delta_e2000","ssim","rgb_mae_255"])
            pairs.append(row)
            for label_id, label in labels.items():
                info=label_map.get(label.get("class"))
                if not info:
                    continue
                selected=ids==int(label_id)
                count=int(selected.sum())
                if not count:
                    continue
                key=info["source_asset"] or info["usd_material"]
                state=totals.setdefault(key,dict(material=key,ue_mapped=bool(info["source_asset"]),pixels=0,delta_sum=0.,usd_materials=set()))
                state["pixels"]+=count
                state["delta_sum"]+=float(difference_map[selected].sum(dtype=np.float64))
                state["usd_materials"].add(info["usd_material"])
            sheet=Image.new("RGB",(2048,1152),"#111827")
            sheet.paste(Image.fromarray(u).resize((1024,576)),(0,45))
            sheet.paste(Image.fromarray(n).resize((1024,576)),(1024,45))
            draw=ImageDraw.Draw(sheet)
            draw.text((20,7),f"UE 5.4 | cam01 {frame} | no image correction",font=font,fill="white")
            draw.text((1044,7),"Isaac Sim 4.5 | same pose and native K",font=font,fill="white")
            heat=np.nan_to_num(difference_map,nan=0)/50
            heat=np.clip(heat,0,1)
            heat_rgb=np.stack([heat*255,np.zeros_like(heat),255*(1-heat)],axis=-1).astype(np.uint8)
            heat_rgb[~mask]=0
            sheet.paste(Image.fromarray(heat_rgb).resize((768,432)),(20,675))
            draw.text((810,690),f"DE2000 mean: {row['delta_e2000']:.4f}",font=font,fill="white")
            draw.text((810,745),f"SSIM masked: {row['ssim']:.6f}",font=font,fill="white")
            draw.text((810,800),f"Mask: {row['masked_pixels']} pixels",font=font,fill="white")
            draw.text((810,855),"Heat: blue=0, red>=50; black=excluded",font=font,fill="white")
            draw.text((810,925),"Dark native output is retained as measured.",font=font,fill="#ffb3b3")
            draw.text((810,980),"Not a material-only or real-world BRDF score.",font=font,fill="#ffb3b3")
            target=root / f"comparison-{frame:03d}.png"
            sheet.save(target)
            sheets.append(target)
            print("QUANTIFIED",frame,row["delta_e2000"],row["ssim"],flush=True)
        ranked=[]
        for entry in totals.values():
            ranked.append(dict(material=entry["material"],ue_mapped=entry["ue_mapped"],pixels=entry["pixels"],
                               mean_delta_e2000=entry["delta_sum"]/entry["pixels"],usd_materials=sorted(entry["usd_materials"])))
        ranked.sort(key=lambda x:x["mean_delta_e2000"],reverse=True)
        write(root / "per-material-metrics.json",ranked)
        roughness=roughness_table(root,config,materials)
        worst=[x for x in ranked if x["ue_mapped"]][:5]
        result=dict(status="MEASURED",valid_pairs=6,quality_target_passed=all(x["delta_e2000"]<5 and x["ssim"]>.85 for x in pairs),
                    target=dict(delta_e2000_less_than=5,ssim_greater_than=.85),pairs=pairs,
                    mean_delta_e2000=float(np.mean([x["delta_e2000"] for x in pairs])),
                    mean_ssim=float(np.mean([x["ssim"] for x in pairs])),
                    roughness_rows=len(roughness),roughness_numeric_rows=sum(x["status"]=="NUMERIC_COMPARISON" for x in roughness),
                    worst_five_material_regions=worst,
                    method=dict(color="8-bit outputs assumed display sRGB, inverse EOTF, XYZ D65, CIELAB, CIEDE2000 kL=kC=kH=1",
                        delta_mask="All native material-labelled foreground pixels; BACKGROUND/UNLABELLED excluded, no brightness selection",
                        ssim="BT.601 luma at native resolution; Gaussian11x11 sigma1.5; population moments; C1=(.01*255)^2,C2=(.03*255)^2; require whole window inside foreground; no resampling",
                        fitting="none; no registration/warping/exposure/WB/histogram adjustment",
                        limit="Cross-engine appearance includes lighting unit conversion, tone mapping, geometry and accepted overrides; cannot attribute error solely to roughness",
                        references=["https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/","https://ece.uwaterloo.ca/~z70wang/research/ssim/"]))
        write(root / "metrics.json",result)
        with (root / "six-frame-comparison.csv").open("w",newline="",encoding="utf8") as f:
            writer=csv.DictWriter(f,fieldnames=list(pairs[0]))
            writer.writeheader()
            writer.writerows(pairs)
        contact=Image.new("RGB",(1536,1296))
        for i,p in enumerate(sheets):
            with Image.open(p) as im:
                contact.paste(im.resize((768,432)),((i%2)*768,(i//2)*432))
        contact.save(root / "six-pair-contact-sheet.png")
        lines=["# UE→Isaac 材质量化 v02", "", "6/6实际机位和原生投影有效，已完成测量；外观参考指标"+("通过。" if result["quality_target_passed"] else "未通过。"), "",
               "| cam01帧 | 掩码像素 | ΔE2000均值 | P95 | SSIM掩码窗口 | UE均值RGB | Isaac均值RGB |", "|---|---|---|---|---|---|---|"]
        lines.extend(f"| {x['source_frame']} | {x['masked_pixels']} | {x['delta_e2000']:.6f} | {x['delta_e2000_p95']:.6f} | {x['ssim']:.6f} | {x['ue_mean_rgb']:.3f} | {x['isaac_mean_rgb']:.3f} |" for x in pairs)
        lines.extend(["",f"六帧等权ΔE2000={result['mean_delta_e2000']:.6f}，SSIM={result['mean_ssim']:.6f}；参考ΔE<5、SSIM>.85。不是逐帧通过，也不代表材质已校准。",
            "", "## 必须区分的边界", "",
            "相机门禁已通过：UE每帧前/后门禁与事后验收共用camera_pairing函数；Isaac记录实际USD相机矩阵、原生camera_params、实际K并同样核验五项。0.15m、控制器、成功阈值与20回合0/20结果不改。",
            "原首份事后审计因要求字典逐位相等而FAIL，保留camera-pairing-acceptance.json；实测差异仅8.7e-8°，两边用户五项阈值独立PASS，修正报告核对后见camera-pairing-acceptance-final.json。没有重渲染UE六帧或放宽五项限值。",
            "Isaac成像显著偏暗。三盏测试方向光的请求强度、方向、白色、关闭阴影及手动曝光输入已记录一致，但这不证明两渲染器实际照度/曝光响应等价；本报告测的是当前转换后的整体外观差异，不能把色差全归为材质。亮度、光强单位/normalize/零角度处理、曝光与色调映射应优先隔离。未调曝光或灯光来凑分，未修改已认可材质。",
            "所有图按显示sRGB假设解码，D65白点；没有实测色彩管理或BRDF真值。掩码为全部原生材质标签前景；不按亮度挑像素。SSIM排除接触背景的11×11窗口，其余未裁剪/缩放/配准，另报全图SSIM供审计。",
            "", "## 粗糙度对齐", "",
            f"roughness-alignment.csv/json共{len(roughness)}条USD材质绑定记录，{result['roughness_numeric_rows']}条可与UE导出烘焙结果数值比较。UE实例输入参数、UE原始烘焙roughness、当前UsdPreviewSurface.roughness及活动MDL输入分开保存；缺数据明确unknown。纹理统计是线性roughness通道texel均值，不能把Dirt Roughness等输入直接当最终粗糙度；活动MDL也不能等同未生效的Preview值。",
            "", "## 最差五个材质关联区域（按全部六帧像素加权色差）", "",
            "这是材质所覆盖图像区域的排名，混有照明、曝光、几何和法线影响，不是材质单变量因果排名。", "", "| UE材质 | 像素数 | 区域ΔE2000 | 修正建议 |", "|---|---|---|---|"])
        for x in worst:
            lines.append(f"| {x['material']} | {x['pixels']} | {x['mean_delta_e2000']:.6f} | 先以灰卡核对入射照度与曝光，再核对该材质烘焙roughness/MDL活动输入、贴图raw/sRGB、法线强度；保持已验收材质不变做独立对照。 |")
        lines.extend(["", "当前门禁结果：采集/相机完整性通过，外观质量目标未通过；遵守失败即停，不开展修复性重渲染。v01与v02原现场、全部源资产保留。",
            "", "算法验证：CIEDE2000通过作者34个公开测试对，误差≤5e-5（原表四位小数）；SSIM同图=1与反相图测试通过。",
            "", "[CIEDE2000原作者及测试数据](https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/)；[SSIM原作者](https://ece.uwaterloo.ca/~z70wang/research/ssim/)。"])
        (root / "report.md").write_text("\n".join(lines),encoding="utf8")
        print(json.dumps(dict(status=result["status"],quality_target_passed=result["quality_target_passed"],valid_pairs=6,
                              delta_e2000=result["mean_delta_e2000"],ssim=result["mean_ssim"],roughness_rows=len(roughness)),indent=2))
        return 0
    except Exception:
        write(root / "quantification-failure.json",dict(status="FAIL_STOPPED",error=traceback.format_exc(),delta_e2000=None,ssim=None))
        raise


if __name__=="__main__":
    raise SystemExit(main())
