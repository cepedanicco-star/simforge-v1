"""UE 5.4 editor capture in a disposable project copy; never save source assets.

Uses Epic SceneCapture2D/RenderingLibrary APIs, asynchronous Slate warm-up.
AISIM_MATERIAL_CONFIG points to the common fixture JSON.
"""
import json
import math
import os
import sys
from pathlib import Path
import time
import traceback

import unreal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.simforge.materials.camera_pairing import (  # noqa: E402
    angle_degrees, camera_pairing, require_camera_pairing, vertical_fov,
)

CONFIG = json.loads(Path(os.environ["AISIM_MATERIAL_CONFIG"]).read_text(encoding="utf8"))
OUT = Path(CONFIG.get("ue_output", str(Path(CONFIG["output"]) / "ue")))
OUT.mkdir(exist_ok=False)
WORLD = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
ACTORS = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
STATE = {"index": 0, "ticks": 0, "started": time.monotonic(), "records": []}


def xyz(v):
    return [v.x, v.y, v.z]


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf8")


try:
    unreal.EditorPythonScripting.set_keep_python_script_alive(True)
    disabled = []
    for actor in ACTORS.get_all_level_actors():
        if isinstance(actor, (unreal.Light, unreal.SkyLight, unreal.PostProcessVolume)):
            actor.set_actor_hidden_in_game(True)
            if isinstance(actor, unreal.PostProcessVolume):
                actor.set_editor_property("enabled", False)
            for component in actor.get_components_by_class(unreal.LightComponentBase):
                component.set_visibility(False)
            disabled.append(actor.get_actor_label())
    actual_lights = []
    for fixture in CONFIG["lights"]:
        actor = ACTORS.spawn_actor_from_class(unreal.DirectionalLight, unreal.Vector(0, 0, 350))
        pitch, yaw, roll = fixture["ue_rotation_pitch_yaw_roll"]
        actor.set_actor_rotation(unreal.Rotator(pitch=pitch, yaw=yaw, roll=roll), False)
        light = actor.get_component_by_class(unreal.DirectionalLightComponent)
        light.set_mobility(unreal.ComponentMobility.MOVABLE)
        light.set_intensity(fixture["illuminance_lux"])
        light.set_light_color(unreal.LinearColor(1, 1, 1, 1))
        light.set_cast_shadows(False)
        p_rad, y_rad = math.radians(pitch), math.radians(yaw)
        expected_forward = [math.cos(p_rad)*math.cos(y_rad), math.cos(p_rad)*math.sin(y_rad), math.sin(p_rad)]
        actual_forward = xyz(actor.get_actor_forward_vector())
        direction_error = angle_degrees(expected_forward, actual_forward)
        if direction_error > 1e-4:
            raise ValueError("Fixture light direction mismatch: " + str(direction_error))
        actual_lights.append(dict(name=fixture["name"], forward_ue=actual_forward,
            direction_error_degrees=direction_error, intensity=light.get_editor_property("intensity"),
            cast_shadows=light.get_editor_property("cast_shadows")))
    unreal.SystemLibrary.execute_console_command(WORLD, "r.MotionBlurQuality 0")
    unreal.SystemLibrary.execute_console_command(WORLD, "r.Streaming.FullyLoadUsedTextures 1")
    capture = ACTORS.spawn_actor_from_class(unreal.SceneCapture2D, unreal.Vector())
    component = capture.get_component_by_class(unreal.SceneCaptureComponent2D)
    component.set_editor_property("capture_every_frame", False)
    component.set_editor_property("capture_on_movement", False)
    component.set_editor_property("always_persist_rendering_state", True)
    component.set_editor_property("override_custom_near_clipping_plane", True)
    component.set_editor_property("custom_near_clipping_plane", 1.0)
    component.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
    target = unreal.RenderingLibrary.create_render_target2d(WORLD, 2048, 1152, unreal.TextureRenderTargetFormat.RTF_RGBA8)
    component.set_editor_property("texture_target", target)
    pp = unreal.PostProcessSettings()
    for name, value in dict(auto_exposure_method=unreal.AutoExposureMethod.AEM_MANUAL,
                            auto_exposure_apply_physical_camera_exposure=True,
                            camera_shutter_speed=16.0, camera_iso=100.0,
                            depth_of_field_fstop=4.0, auto_exposure_bias=0.0,
                            motion_blur_amount=0.0, bloom_intensity=0.0,
                            vignette_intensity=0.0).items():
        pp.set_editor_property("override_" + name, True)
        pp.set_editor_property(name, value)
    component.set_editor_property("post_process_settings", pp)
    component.set_editor_property("post_process_blend_weight", 1.0)
    write("fixture.json", dict(engine=unreal.SystemLibrary.get_engine_version(), disabled=disabled,
                              lighting=CONFIG["lights"], actual_lights=actual_lights, shutter_s=1/16, iso=100, fnumber=4,
                              automatic_exposure=False, capture="SCS_FINAL_COLOR_LDR", resolution=[2048,1152]))

    # Scalar values are inputs, never assumed to be evaluated texture-driven roughness.
    materials = {}
    for actor in ACTORS.get_all_level_actors():
        for mesh in actor.get_components_by_class(unreal.StaticMeshComponent):
            for i in range(mesh.get_num_materials()):
                material = mesh.get_material(i)
                if not material or material.get_path_name() in materials:
                    continue
                names = unreal.MaterialEditingLibrary.get_scalar_parameter_names(material) if isinstance(material, unreal.MaterialInstanceConstant) else []
                materials[material.get_path_name()] = dict(
                    name=material.get_name(),
                    scalars={str(n): unreal.MaterialEditingLibrary.get_material_instance_scalar_parameter_value(material,n) for n in names})
    write("material-parameters.json", materials)

    def observe_and_gate(view, timing):
        fov = float(component.get_editor_property("fov_angle"))
        width, height = target.get_editor_property("size_x"), target.get_editor_property("size_y")
        observed = dict(position_cm=xyz(capture.get_actor_location()),
            forward=xyz(capture.get_actor_forward_vector()), up=xyz(capture.get_actor_up_vector()),
            horizontal_fov_degrees=fov, vertical_fov_degrees=vertical_fov(fov, width, height),
            resolution=[width, height])
        result = camera_pairing(view, observed)
        write(f"gate-{view['source_frame']:03d}-{timing}.json", result)
        require_camera_pairing(view, observed)
        return observed, result

    def tick(delta):
        try:
            STATE["ticks"] += 1
            i = STATE["index"]
            view = CONFIG["views"][i]
            if STATE["ticks"] == 1:
                capture.set_actor_location(unreal.Vector(*view["ue_position_cm"]), False, False)
                pitch, yaw, roll = view["ue_rotation_pitch_yaw_roll"]
                capture.set_actor_rotation(unreal.Rotator(pitch=pitch, yaw=yaw, roll=roll), False)
                component.set_editor_property("fov_angle", view["horizontal_fov_degrees"])
                observe_and_gate(view, "before-render")
            component.capture_scene()
            if STATE["ticks"] < 120 or (i == 0 and time.monotonic()-STATE["started"] < 30):
                return
            observed, gate = observe_and_gate(view, "before-write")
            filename = f"cam01-{view['source_frame']:03d}.png"
            unreal.RenderingLibrary.export_render_target(WORLD, target, str(OUT), filename)
            if not (OUT / filename).is_file():
                raise RuntimeError("UE render target export missing")
            STATE["records"].append(dict(source_frame=view["source_frame"], image=filename,
                                          **observed, pre_write_gate=gate))
            unreal.log("MATERIAL UE " + str(i+1) + "/" + str(len(CONFIG["views"])))
            STATE["index"] += 1
            STATE["ticks"] = 0
            if STATE["index"] == len(CONFIG["views"]):
                write("capture-report.json", dict(status="PASS", records=STATE["records"], saved_assets=False))
                unreal.unregister_slate_post_tick_callback(HANDLE)
                unreal.SystemLibrary.quit_editor()
        except Exception:
            write("failure.json", dict(error=traceback.format_exc(), saved_records=STATE["records"]))
            unreal.unregister_slate_post_tick_callback(HANDLE)
            unreal.SystemLibrary.quit_editor()

    HANDLE = unreal.register_slate_post_tick_callback(tick)
except Exception:
    write("failure.json", dict(error=traceback.format_exc(), saved_records=STATE["records"]))
    unreal.SystemLibrary.quit_editor()
    raise
