"""Asset-free contracts: real USD authoring, no simulated robot success claim."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np


def run(output):
    from .ue_isaac_pipeline.usd_pipeline import make_demo,prepare,validate_scene
    from .simforge.randomize.sampler import sample,compare_values
    from .simforge.sensors.geometry import backproject,camera_contract
    from training.safe_execute import Parameters,safety_filter
    from types import SimpleNamespace
    import yaml
    started=time.monotonic();out=Path(output)
    out.mkdir(parents=True,exist_ok=False)
    demo=make_demo(out/'synthetic-source');source=Path(demo['source'])
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    composed=prepare(source,out/'prepared',manifest=demo['manifest'],registry=demo['registry'],allow_provisional=True)
    validation=validate_scene(composed['scene'])
    assert validation['structural_passed'],validation
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    cfg=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/randomize_office_v03.yaml').read_text(encoding='utf8'))
    assert all(not compare_values(sample(cfg,i),sample(cfg,i)) for i in range(6))
    native=dict(renderProductResolution=[640,480],metersPerSceneUnit=1,cameraModel='pinhole',
        cameraApertureOffset=[0,0],cameraProjection=np.diag([2,2,1,1]).ravel(),cameraViewTransform=np.eye(4).ravel())
    camera=camera_contract(native)
    assert np.allclose(backproject([[319.5,239.5]],[2],camera),[[0,0,-2]])
    p=Parameters();scene=SimpleNamespace(params=p,nominal=np.zeros(14),limits=np.tile([-2.,2.],(14,1)),
        grasp_phase=True,tilt_rate=0.,closing_speed=lambda a,b:0.,check_path=lambda a,b,n:[])
    bounded,_=safety_filter(np.ones(14),np.zeros(14),scene)
    assert np.max(abs(bounded))<=p.slew_max_rad
    report=dict(status='PASS_OFFLINE_CONTRACT_SMOKE',checks=4,source_hash_unchanged=True,
        six_seed_reproducibility=True,camera_math=True,slew_bound=True,
        native_Isaac_M1_M4_executed=False,BC_policy_evaluated=False,robot_task_success=None,
        elapsed_seconds=time.monotonic()-started)
    (out/'smoke-summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out');args=p.parse_args(argv)
    if args.out:report=run(args.out)
    else:
        with tempfile.TemporaryDirectory(prefix='simforge-check-') as folder:
            report=run(Path(folder)/'run')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
