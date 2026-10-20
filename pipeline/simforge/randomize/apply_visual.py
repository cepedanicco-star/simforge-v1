"""Apply a pre-sampled episode to a new USD opinion layer, never draw randomness."""

import colorsys
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import mesh_intersects_box, support_z, triangles


def world_matrix(stage, prim):
    from pxr import UsdGeom

    return UsdGeom.XformCache(0).GetLocalToWorldTransform(prim)


def set_world(stage, prim, matrix):
    from pxr import UsdGeom

    inverse_parent = world_matrix(stage, prim.GetParent()).GetInverse()
    UsdGeom.Xformable(prim).MakeMatrixXform().Set(matrix * inverse_parent)


def apply(stage, config, sample, camera_path):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, UsdPhysics

    dimensions = sample["dimensions"]
    applied = {}
    target = stage.GetPrimAtPath(config["dimensions"]["D4"]["target_prim"])
    if not target:
        raise ValueError("Missing target cup")
    initial_cup = np.asarray(world_matrix(stage, target).ExtractTranslation())
    d = dimensions["D4"]
    if d["enabled"]:
        values = d["values"]
        old = world_matrix(stage, target)
        rotated = old * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), values["yaw_degrees"]))
        position = initial_cup + values["position_m"]
        rotated.SetTranslateOnly(Gf.Vec3d(*map(float, position)))
        set_world(stage, target, rotated)
        applied["D4"] = dict(
            enabled=True,
            initial_world_position_m=initial_cup.tolist(),
            world_position_m=position.tolist(),
            yaw_degrees=values["yaw_degrees"],
            support_height_preserved=True,
        )
    cup_position = np.asarray(world_matrix(stage, target).ExtractTranslation())
    d = dimensions["D5"]
    if d["enabled"]:
        camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
        old = world_matrix(stage, camera.GetPrim())
        camera_cfg = config["dimensions"]["D5"]
        position = np.asarray(old.ExtractTranslation()) + d["values"]["position_m"]
        safety = None
        rotation = Gf.Rotation()
        for axis, angle in zip(((1, 0, 0), (0, 1, 0), (0, 0, 1)), d["values"]["euler_rad"]):
            rotation = rotation * Gf.Rotation(Gf.Vec3d(*axis), float(np.degrees(angle)))
        new = old * Gf.Matrix4d().SetRotate(rotation)
        new.SetTranslateOnly(
            Gf.Vec3d(*map(float, position))
        )
        set_world(stage, camera.GetPrim(), new)
        # A camera may have physically scaled, animated frame-0 lens values.
        # Default-time focal/aperture must not be mixed with those frame values.
        aperture = float(camera.GetHorizontalApertureAttr().Get(Usd.TimeCode(0)))
        original_focal = float(camera.GetFocalLengthAttr().Get(Usd.TimeCode(0)))
        fov0 = 2 * np.arctan(aperture / (2 * original_focal))
        fov1 = fov0 * d["values"]["fov_scale"]
        focal = aperture / (2 * np.tan(fov1 / 2))
        camera.GetFocalLengthAttr().Set(float(focal), Usd.TimeCode(0))
        applied["D5"] = dict(
            enabled=True,
            original_horizontal_fov_rad=float(fov0),
            horizontal_fov_rad=float(fov1),
            focal_length_mm=float(focal),
            horizontal_aperture_mm=aperture,
            projection_rule="Exact tangent relation; capture reads native camera_params",
            camera_safety=safety,
            world_position_m=position.tolist(),
            lens_time_code=0,
        )
    d = dimensions["D2"]
    if d["enabled"]:
        authored = []
        for prim in stage.Traverse():
            shader = UsdShade.Shader(prim)
            if not shader or shader.GetIdAttr().Get() != "UsdPreviewSurface":
                continue
            transmission = shader.GetInput("opacity")
            if "glass" in str(prim.GetPath()).lower() or (
                transmission
                and (
                    transmission.GetConnectedSource()
                    or (transmission.Get() is not None and transmission.Get() < 0.99)
                )
            ):
                continue
            roughness = shader.CreateInput("roughness", Sdf.ValueTypeNames.Float)
            roughness.DisconnectSource()
            roughness.Set(d["values"]["roughness"])
            color = shader.GetInput("diffuseColor")
            source = color.GetConnectedSource() if color else None
            hue = d["values"]["hue_degrees"] / 360
            if source:
                texture = UsdShade.Shader(source[0].GetPrim())
                if texture.GetIdAttr().Get() == "UsdUVTexture":
                    scale = texture.GetInput("scale")
                    before = scale.Get() if scale else Gf.Vec4f(1)
                    asset = texture.GetInput("file").Get()
                    if not asset or not asset.resolvedPath:
                        raise ValueError(
                            "Cannot resolve source texture for hue control: " + str(texture.GetPath())
                        )
                    with Image.open(asset.resolvedPath) as image:
                        mean = np.asarray(image.convert("RGB").resize((1, 1)), dtype=float).reshape(3) / 255
                    hsv = colorsys.rgb_to_hsv(*mean)
                    desired = np.asarray(colorsys.hsv_to_rgb((hsv[0] + hue) % 1, hsv[1], hsv[2]))
                    gain = np.divide(desired**2.2, mean**2.2, out=np.ones(3), where=mean > 1e-5)
                    texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
                        Gf.Vec4f(*map(float, gain * np.asarray(before)[:3]), float(before[3]))
                    )
            elif color and color.Get() is not None:
                hsv = colorsys.rgb_to_hsv(*map(float, color.Get()))
                color.Set(Gf.Vec3f(*colorsys.hsv_to_rgb((hsv[0] + hue) % 1, hsv[1], hsv[2])))
            authored.append(str(prim.GetPath()))
        if not authored:
            raise ValueError("No supported opaque PreviewSurface materials randomized")
        applied["D2"] = dict(
            enabled=True,
            roughness=d["values"]["roughness"],
            hue_degrees=d["values"]["hue_degrees"],
            shaders=authored,
            normal_maps_preserved=True,
            transmissive_materials_excluded=True,
            textured_hue_rule="Mean-texel hue rotation via linear RGB gains; patterns retained, not an exact per-texel HSV operation",
        )
    d = dimensions["D1"]
    if d["enabled"]:
        calibration = json.loads(Path(config["lighting_calibration"]).read_text(encoding="utf8"))
        rows = calibration["table"]
        temperature = d["values"]["temperature_K"]
        if not rows[0]["temperature_K"] <= temperature <= rows[-1]["temperature_K"]:
            raise ValueError("Temperature is outside the measured native calibration table")
        coefficient = float(
            np.interp(temperature, [r["temperature_K"] for r in rows], [r["lux_per_intensity"] for r in rows])
        )
        reference = config["lighting_reference"]
        distance = reference["distance_m"]
        if abs(calibration["fixture"]["reference_distance_m"] - distance) > 1e-9:
            raise ValueError("Light distance differs from the measured reference fixture")
        intensity = d["values"]["illuminance_lux"] / coefficient
        light = UsdLux.RectLight.Define(stage, "/World/SimForgeRandomization/ReferencePanel")
        light.CreateWidthAttr(reference["width_m"])
        light.CreateHeightAttr(reference["height_m"])
        light.CreateNormalizeAttr(True)
        light.CreateIntensityAttr(intensity)
        light.CreateEnableColorTemperatureAttr(True)
        light.CreateColorTemperatureAttr(temperature)
        light.AddTranslateOp().Set(Gf.Vec3d(*map(float, initial_cup + np.array([0, 0, distance]))))
        changed = []
        for prim in stage.Traverse():
            if prim.HasAPI(UsdLux.LightAPI) and str(prim.GetPath()) != str(light.GetPath()):
                api = UsdLux.LightAPI(prim)
                old = api.GetIntensityAttr().Get()
                if old is not None:
                    api.GetIntensityAttr().Set(
                        float(old)
                        * d["values"]["illuminance_lux"]
                        / config["dimensions"]["D1"]["center"]["illuminance_lux"]
                    )
                    api.CreateEnableColorTemperatureAttr(True)
                    api.CreateColorTemperatureAttr(temperature)
                    changed.append(str(prim.GetPath()))
        applied["D1"] = dict(
            enabled=True,
            reference_illuminance_lux=d["values"]["illuminance_lux"],
            temperature_K=temperature,
            authored_intensity=float(intensity),
            calibration_lux_per_intensity_at_reference_distance=coefficient,
            reference_distance_m=distance,
            source_lights_varied=changed,
            scene_lux_requires_native_aov=True,
            real_world_calibrated=False,
        )
    d = dimensions["D3"]
    if d["enabled"]:
        from pxr import Semantics
        cfg = config["dimensions"]["D3"]
        support = stage.GetPrimAtPath(cfg["support_prim"])
        surface = triangles(stage, support)
        if not len(surface):
            raise ValueError("No measured support geometry for clutter")
        bounds = np.asarray([surface.min((0, 1)), surface.max((0, 1))])
        record = np.load(Path(config["source_recording"]) / "physics-body-states.npz", allow_pickle=False)
        route = record["states"][:, record["body_names"].tolist().index("pelvis"), :2]
        goal = record["cup"][-1, :2]
        size = np.asarray(cfg["box_size_m"])
        plan = json.loads(Path(config["label_plan"]).read_text(encoding="utf8"))["objects"]
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode(0), ["default", "render"])
        obstacles = []
        for obj in plan:
            prim = stage.GetPrimAtPath(obj["prepared_prim_path"])
            if not prim or str(prim.GetPath()) == cfg["support_prim"]:
                continue
            bound = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            obstacles.append((prim, np.asarray(bound.GetMin()), np.asarray(bound.GetMax())))
        selected = []
        geometry_cache = {}
        for uv in d["values"]["proposal_xy_unit"]:
            if len(selected) == d["values"]["count"]:
                break
            xy = bounds[0, :2] + np.asarray(uv) * (bounds[1, :2] - bounds[0, :2])
            if (
                min(np.linalg.norm(xy - cup_position[:2]), np.linalg.norm(xy - goal))
                < cfg["target_clearance_m"] + np.linalg.norm(size[:2]) / 2
            ):
                continue
            route_distance = float(np.linalg.norm(route - xy, axis=1).min())
            if route_distance < cfg["route_clearance_m"] + np.linalg.norm(size[:2]) / 2:
                continue
            z = support_z(surface, xy, cfg["support_height_m"])
            if z is None:
                continue
            corners = xy + np.asarray([[-1, -1], [-1, 1], [1, -1], [1, 1]]) * size[:2] / 2
            heights = [support_z(surface, corner, cfg["support_height_m"]) for corner in corners]
            if any(h is None or abs(h - z) > 1e-4 for h in heights):
                continue
            low = np.r_[xy - size[:2] / 2, z + cfg["support_gap_m"]]
            high = low + size
            if mesh_intersects_box(surface, low, high):
                continue
            intersects = False
            for prim, other_low, other_high in obstacles:
                margin = cfg["other_object_clearance_m"]
                if (high + margin < other_low).any() or (low - margin > other_high).any():
                    continue
                path = str(prim.GetPath())
                if path not in geometry_cache:
                    geometry_cache[path] = triangles(stage, prim)
                if mesh_intersects_box(geometry_cache[path], low - margin, high + margin):
                    intersects = True
                    break
            if intersects or any(
                np.all(high + cfg["other_object_clearance_m"] >= r["low"])
                and np.all(low - cfg["other_object_clearance_m"] <= r["high"])
                for r in selected
            ):
                continue
            index = len(selected)
            cube = UsdGeom.Cube.Define(stage, f"/World/SimForgeRandomization/Clutter/box_{index}")
            cube.CreateSizeAttr(1)
            cube.AddTranslateOp().Set(Gf.Vec3d(*map(float, (low + high) / 2)))
            cube.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
            cube.CreateDisplayColorAttr([Gf.Vec3f(*d["values"]["colors_rgb"][index])])
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            semantics = Semantics.SemanticsAPI.Apply(cube.GetPrim(), "Semantics")
            semantics.CreateSemanticTypeAttr().Set("class")
            semantics.CreateSemanticDataAttr().Set("clutter")
            selected.append(
                dict(
                    prim_path=str(cube.GetPath()),
                    low=low.tolist(),
                    high=high.tolist(),
                    support_z_m=z,
                    route_center_clearance_m=route_distance,
                )
            )
        if len(selected) != d["values"]["count"]:
            raise ValueError(
                f"Only {len(selected)} safe clutter placements for requested {d['values']['count']}; no unsafe fallback"
            )
        applied["D3"] = dict(
            enabled=True,
            requested_count=d["values"]["count"],
            objects=selected,
            guards="Four supported corners; triangle/AABB overlap; measured route clearance; cup/desk target clearance; static colliders",
        )
    # Check after every episode-start edit, including the real clutter colliders.
    if dimensions["D5"]["enabled"] and config["dimensions"]["D5"].get("require_free_space"):
        from .camera_constraints import inspect_position, live_query, select_position
        camera_cfg = config["dimensions"]["D5"]
        camera_prim = stage.GetPrimAtPath(camera_path)
        original = np.asarray(world_matrix(stage, camera_prim).ExtractTranslation())
        query = live_query()
        check = inspect_position(original, query, camera_cfg["min_collider_clearance_m"])
        safety = dict(original_check=check, repaired=False, **check)
        if not check["accepted"]:
            # Used only for a demonstrably unsafe camera, before the render loop.
            rng = np.random.default_rng(np.random.SeedSequence([sample["seed"], 905]))
            proposals = rng.uniform(-camera_cfg["local_repair_extent_m"], camera_cfg["local_repair_extent_m"],
                                    (camera_cfg["repair_attempts"], 3))
            selected = select_position(original, proposals, query, camera_cfg["min_collider_clearance_m"])
            changed = world_matrix(stage, camera_prim)
            changed.SetTranslateOnly(Gf.Vec3d(*selected["position_m"]))
            set_world(stage, camera_prim, changed)
            safety = dict(original_check=check, repaired=True, **selected)
        applied["D5"]["camera_safety"] = safety
        applied["D5"]["world_position_m"] = list(world_matrix(stage, camera_prim).ExtractTranslation())
    return applied
