"""Accept fresh six-variant collections, without a historical directory layout."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image,ImageDraw

from .simforge.randomize.sampler import compare_values


REQUIRED=('rgb.png','depth.npy','semantic.png','instance.png','boxes_2d.json','boxes_3d.json','camera.json','labels.json','native_annotations.json')


def load(path):return json.loads(Path(path).read_text(encoding='utf8'))


def compare_geometry(a,b):
    """Compare actual exported native geometry; never construct substitute boxes."""
    if len(a)!=len(b):return False
    for first,second in zip(a,b):
        if not first.get('native_prim_path') or first.get('native_prim_path')!=second.get('native_prim_path'):return False
        for key in ('center','extent','rotation'):
            if key not in first or key not in second or not np.allclose(first[key],second[key],atol=1e-7,rtol=0):return False
    return True


def validate_variant(root,number):
    folder=Path(root)/f'var_{number:02d}';frame=folder/'frames/frame_000000'
    meta=load(folder/'meta.json');report=load(folder/'frames/export-report.json')
    if report.get('status')!='completed' or report.get('frames')!=1:raise ValueError('Need one complete native frame per variant')
    if any(not (frame/n).is_file() or not (frame/n).stat().st_size for n in REQUIRED):raise ValueError('Missing native frame files')
    if not report.get('source_layers_unchanged') or report.get('maximum_box_edge_error_px',float('inf'))>5:
        raise ValueError('Source integrity/native box alignment failed')
    camera=load(frame/'camera.json');boxes=load(frame/'boxes_3d.json');native=load(frame/'native_annotations.json')
    if not boxes or not native.get('boxes_3d'):raise ValueError('Native 3D box channel must be nonempty')
    if camera.get('depth_mode')!='z' or camera.get('depth_unit')!='m':raise ValueError('Metric z depth required')
    for name in ('rgb.png','semantic.png','instance.png'):
        with Image.open(frame/name) as image:
            if list(image.size)!=[camera['width'],camera['height']]:raise ValueError('Image/camera resolution mismatch')
    depth=np.load(frame/'depth.npy',allow_pickle=False)
    if depth.dtype!=np.float32 or not np.any(np.isfinite(depth)&(depth>0)):raise ValueError('No positive float32 depth')
    if meta['dimensions']['D5']['enabled']:
        safety=meta['applied']['D5']['camera_safety']
        if safety.get('accepted') is not True or safety.get('radius_m')!=.15:raise ValueError('Camera clearance failed')
    return meta,camera,boxes,frame


def accept(root,repeat,ablation,out,ffmpeg=None):
    out=Path(out)
    if out.exists():raise FileExistsError('Fresh acceptance output required')
    out.mkdir(parents=True);rows=[]
    try:
        for number in range(6):
            meta,camera,boxes,frame=validate_variant(root,number)
            again,again_camera,again_boxes,_=validate_variant(repeat,number)
            off,_,_,_=validate_variant(ablation,number)
            if compare_values(meta,again):raise ValueError('Same-seed values differ')
            if compare_values(meta,off)!=['D6'] or off['dimensions']['D6']['enabled']:
                raise ValueError('Ablation must disable only D6')
            for key in ('fx','fy','cx','cy','extrinsic_world_to_cam'):
                if not np.allclose(camera[key],again_camera[key],atol=1e-7,rtol=0):raise ValueError('Same-seed camera differs')
            if not compare_geometry(boxes,again_boxes):raise ValueError('Same-seed native geometry differs')
            rows.append(dict(variant=number,seed=meta['seed'],native_boxes=len(boxes),same_seed_values=True,
                same_seed_camera_and_boxes=True,D6_only_ablation=True,camera_clearance_m=.15))
        board=Image.new('RGB',(1280,400),'#101827')
        for slot,index in enumerate((0,5)):
            meta,_,_,frame=validate_variant(root,index)
            with Image.open(frame/'rgb.png') as image:board.paste(image.convert('RGB').resize((640,360)),(640*slot,0))
            ImageDraw.Draw(board).text((640*slot+12,372),f'var_{index:02d} | seed {meta["seed"]}',fill='white')
        board.save(out/'two-variant-comparison.png')
        if ffmpeg:
            subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-n','-loop','1','-i',str(out/'two-variant-comparison.png'),
                '-t','5','-r','25','-c:v','libx264','-pix_fmt','yuv420p',str(out/'m3-comparison.mp4')],check=True)
        report=dict(status='PASS',variants=rows,native_boxes_estimated=False,
            same_seed_comparison='Sample values, camera and actual native geometry; stochastic RGB equality not claimed',
            visual_comparison_variants=[0,5],manual_visual_review_required=True)
    except Exception as error:
        report=dict(status='FAIL',variants=rows,error=str(error));raise
    finally:
        (out/'m3-acceptance.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--repeat-root',required=True);p.add_argument('--ablation-root',required=True)
    p.add_argument('--out',required=True);p.add_argument('--ffmpeg');a=p.parse_args()
    print(json.dumps(accept(a.root,a.repeat_root,a.ablation_root,a.out,a.ffmpeg),indent=2))


if __name__=='__main__':main()
