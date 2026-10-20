"""Add new calibration-only diffuse cards in the disposable UE project copy."""
import json
import os
from pathlib import Path
import runpy
import traceback

import unreal

config=json.loads(Path(os.environ["AISIM_MATERIAL_CONFIG"]).read_text(encoding="utf8"))
root=Path(config["output"])
try:
    actors=unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    tools=unreal.AssetToolsHelpers.get_asset_tools()
    mesh=unreal.load_asset("/Engine/BasicShapes/Cube.Cube")
    if not mesh:
        raise RuntimeError("Engine calibration cube asset missing")
    records=[]
    for card in config["calibration_cards"]:
        material=tools.create_asset("M_"+card["name"],"/Game/SimforgeCalibration",unreal.Material,unreal.MaterialFactoryNew())
        if not material:
            raise RuntimeError("Use fresh project copy for calibration-only materials")
        lib=unreal.MaterialEditingLibrary
        color=lib.create_material_expression(material,unreal.MaterialExpressionConstant3Vector)
        reflectance=card["reflectance"]
        color.set_editor_property("constant",unreal.LinearColor(reflectance,reflectance,reflectance,1))
        assert lib.connect_material_property(color,"",unreal.MaterialProperty.MP_BASE_COLOR)
        for value,prop in [(0.5,unreal.MaterialProperty.MP_ROUGHNESS),(0.0,unreal.MaterialProperty.MP_METALLIC),(0.0,unreal.MaterialProperty.MP_SPECULAR)]:
            scalar=lib.create_material_expression(material,unreal.MaterialExpressionConstant)
            scalar.set_editor_property("r",value)
            assert lib.connect_material_property(scalar,"",prop)
        lib.recompile_material(material)
        x,y,z=card["position_m"]
        actor=actors.spawn_actor_from_class(unreal.StaticMeshActor,unreal.Vector(x*100,-y*100,z*100))
        actor.set_actor_label("SimforgeCalibration_"+card["name"])
        component=actor.get_component_by_class(unreal.StaticMeshComponent)
        component.set_static_mesh(mesh)
        component.set_material(0,material)
        actor.set_actor_scale3d(unreal.Vector(*card["dimensions_m"]))
        actor.set_actor_rotation(unreal.Rotator(pitch=0,yaw=0,roll=0),False)
        records.append(dict(**card,ue_material=material.get_path_name(),ue_mesh=mesh.get_path_name(),
            actual_position_cm=[actor.get_actor_location().x,actor.get_actor_location().y,actor.get_actor_location().z],
            actual_scale_m=[actor.get_actor_scale3d().x,actor.get_actor_scale3d().y,actor.get_actor_scale3d().z],
            roughness=.5,metallic=0,specular=0,source_material_changed=False))
    (root/"ue-card-construction.json").write_text(json.dumps(records,indent=2),encoding="utf8")
    runpy.run_path(str(Path(__file__).with_name("capture_material_ue.py")),run_name="__main__")
except Exception:
    (root/"ue-card-failure.json").write_text(json.dumps(dict(status="FAIL_STOPPED",error=traceback.format_exc()),indent=2),encoding="utf8")
    unreal.SystemLibrary.quit_editor()
    raise
