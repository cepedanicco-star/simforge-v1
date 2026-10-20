"""Frozen seed sampling and initialization only; native execution is in training.evaluate."""
import hashlib
import json
from pathlib import Path
import numpy as np

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

def write(path, data):
    target = Path(path)
    temporary = target.with_suffix(target.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf8')
    temporary.replace(target)

def draw_initialization(seed):
    rng = np.random.default_rng(seed)
    return dict(seed=seed, cup_xy_delta_m=rng.uniform(-0.15, 0.15, 2).tolist(), cup_yaw_delta_degrees=float(rng.uniform(-20, 20)), robot_xy_delta_m=rng.uniform(-0.1, 0.1, 2).tolist(), robot_yaw_delta_degrees=float(rng.uniform(-15, 15)))

def classify_failure(phase, error):
    if 'tipped' in error.lower() or 'Cup contacted' in error:
        return 'cup_topple_or_collision'
    if phase in {'settle_at_desk', 'move_above_desk', 'lower_to_desk', 'release_on_desk', 'withdraw_hand', 'verify_placement'} or 'desktop' in error:
        return 'placement_failure'
    if phase in {'turn_right_with_cup', 'navigate_apartment_with_cup', 'turn_left_to_desk', 'approach_desk_forward', 'turn_with_cup', 'walk_forward_with_cup'} or 'navigation' in error:
        return 'trajectory_deviation'
    return 'grasp_failure'

def prepare(config, seed, output, baseline_args):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    sample = draw_initialization(seed)
    output.mkdir(parents=True, exist_ok=False)
    stage = Usd.Stage.Open(baseline_args['scene'])
    stage.SetEditTarget(stage.GetSessionLayer())
    cup = stage.GetPrimAtPath('/World/Environment/Source/cup_V1')
    cache = UsdGeom.XformCache(0)
    old = cache.GetLocalToWorldTransform(cup)
    position = np.asarray(old.ExtractTranslation()) + [*sample['cup_xy_delta_m'], 0.0]
    matrix = old * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), sample['cup_yaw_delta_degrees']))
    matrix.SetTranslateOnly(Gf.Vec3d(*map(float, position)))
    UsdGeom.Xformable(cup).MakeMatrixXform().Set(matrix * cache.GetLocalToWorldTransform(cup.GetParent()).GetInverse())
    stage.GetSessionLayer().Export(str(output / 'initialization.usda'))
    layer = Sdf.Layer.CreateNew(str(output / 'Initial_scene.usda'))
    layer.subLayerPaths = ['initialization.usda', Path(baseline_args['scene']).as_posix()]
    layer.defaultPrim = 'World'
    initial = Usd.Stage.Open(layer)
    UsdGeom.SetStageMetersPerUnit(initial, 1)
    UsdGeom.SetStageUpAxis(initial, 'Z')
    UsdPhysics.SetStageKilogramsPerUnit(initial, 1)
    initial.SetTimeCodesPerSecond(stage.GetTimeCodesPerSecond())
    layer.Save()
    arguments = dict(baseline_args)
    arguments.update(scene=str(output / 'Initial_scene.usda'), out=str(output / 'physics'), x=baseline_args['x'] + sample['robot_xy_delta_m'][0], y=baseline_args['y'] + sample['robot_xy_delta_m'][1], yaw=baseline_args['yaw'] + sample['robot_yaw_delta_degrees'])
    sample.update(cup_initial_world_position_m=position.tolist(), robot_initial_world_position_m=[arguments['x'], arguments['y'], arguments['z']], robot_initial_yaw_degrees=arguments['yaw'], baseline_arguments=baseline_args, episode_arguments=arguments, controller_changed=False)
    write(output / 'initialization.json', sample)
    return sample
