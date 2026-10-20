"""Execute inside Unreal Editor after loading the intended training region.

    import sys
    sys.path.insert(0, r'<this-repo>/pipeline/ue')
    import export_scene
    export_scene.export_scene(r'<output-directory>')

Requires Epic USD Importer and Python Editor Script Plugin. Supports UE 5.4–5.8.
Only reads loaded objects; does not claim World Partition is completely loaded.
"""

import hashlib
import json
from pathlib import Path


def export_scene(output_directory):
    import unreal

    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    target = output / "scene.usda"
    if target.exists():
        raise FileExistsError("Choose a new export directory")
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    engine_version = unreal.SystemLibrary.get_engine_version()
    engine_minor = int(engine_version.split(".")[1])
    components = []
    unsupported = []
    for actor in actors:
        tags = [str(t) for t in actor.tags]
        semantic = {}
        for tag in tags:
            if tag.startswith("aisim:class="):
                semantic["class_name"] = tag.split("=", 1)[1]
            if tag.startswith("aisim:role="):
                semantic["role"] = tag.split("=", 1)[1]
        semantic["reviewed"] = "aisim:reviewed" in tags
        semantic["source"] = "ue_actor_tags"
        try:
            actor_guid = str(actor.get_editor_property("actor_guid"))
        except Exception:
            actor_guid = actor.get_path_name()
        for component in actor.get_components_by_class(unreal.SceneComponent):
            try:
                mesh = component.get_editor_property("static_mesh")
            except Exception:
                mesh = None
            if mesh is None:
                # Landscape/skeletal/other geometry is still exported and discovered in USD,
                # but is not falsely mapped using an invented component name.
                unsupported.append(component.get_path_name())
                continue
            key = actor_guid + ":" + component.get_name()
            if engine_minor >= 6:
                prim_path = unreal.UsdConversionLibrary.get_prim_path_for_object(
                    component, use_actor_folders=False, root_prim_name="World"
                )
            else:
                prim_path = unreal.UsdConversionLibrary.get_prim_path_for_object(
                    component, use_actor_folders=False
                )
            instances = 0
            if isinstance(component, unreal.InstancedStaticMeshComponent):
                instances = component.get_instance_count()
            components.append(
                dict(
                    object_id="ue-" + hashlib.sha256(key.encode()).hexdigest()[:20],
                    actor_path=actor.get_path_name(),
                    actor_label=actor.get_actor_label(),
                    actor_guid=actor_guid,
                    component_path=component.get_path_name(),
                    prim_path=prim_path,
                    asset_path=mesh.get_path_name(),
                    semantic=semantic,
                    instance_count=instances,
                )
            )
    # Epic 5.4's C++ exporter calls Python export_with_cdo_options(), which reads
    # the class default object even when AssetExportTask.options is supplied.
    options = unreal.get_default_object(unreal.LevelExporterUSDOptions)
    stage_options = options.get_editor_property("stage_options")
    stage_options.set_editor_property("meters_per_unit", 1.0)
    stage_options.set_editor_property("up_axis", unreal.UsdUpAxis.Z_AXIS)
    options.set_editor_property("stage_options", stage_options)
    inner = options.get_editor_property("inner")
    if engine_minor >= 6:
        inner.set_editor_property("root_prim_name", "World")
    inner.set_editor_property("selection_only", False)
    inner.set_editor_property("export_actor_folders", False)
    asset_options = inner.get_editor_property("asset_options")
    asset_options.set_editor_property("bake_materials", True)
    asset_options.set_editor_property("lowest_mesh_lod", 0)
    asset_options.set_editor_property("highest_mesh_lod", 0)
    baking = asset_options.get_editor_property("material_baking_options")
    baking.set_editor_property("default_texture_size", unreal.IntPoint(1024, 1024))
    asset_options.set_editor_property("material_baking_options", baking)
    inner.set_editor_property("asset_options", asset_options)
    options.set_editor_property("inner", inner)
    task = unreal.AssetExportTask()
    task.object = world
    task.filename = str(target)
    task.automated = True
    task.prompt = False
    task.replace_identical = False
    task.options = options
    task.exporter = unreal.LevelExporterUSD()
    result = unreal.Exporter.run_asset_export_task(task)
    if not result or not target.exists():
        raise RuntimeError("Epic USD export failed; inspect Unreal's export log")
    inventory = dict(
        schema_version=1,
        engine_version=engine_version,
        loaded_actor_count=len(actors),
        source_inventory_complete=False,
        scope="loaded_editor_objects_only",
        components=components,
        components_without_static_mesh_mapping=unsupported,
        next_step="aisim link-ue scene.usda --ue-inventory ue_inventory.json --out mapping",
    )
    (output / "ue_inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    unreal.log(f"Exported {target}; source scope remains unverified until cells/instances are audited")
    return inventory
