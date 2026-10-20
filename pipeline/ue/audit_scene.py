"""Read saved UE scene/material/sequence facts; export cam01 with Epic's USD exporter."""

import json
import os
from pathlib import Path

import unreal


def xyz(value):
    return [value.x, value.y, value.z]


def material_info(material):
    info = dict(path=material.get_path_name(), class_name=material.get_class().get_name())
    library = unreal.MaterialEditingLibrary
    try:
        info["scalar_parameters"] = {
            str(n): library.get_material_instance_scalar_parameter_value(material, n)
            for n in library.get_scalar_parameter_names(material)
        }
        textures = {}
        for name in library.get_texture_parameter_names(material):
            texture = library.get_material_instance_texture_parameter_value(material, name)
            if texture:
                record = dict(path=texture.get_path_name())
                for prop in ("srgb", "compression_settings", "flip_green_channel"):
                    try:
                        value = texture.get_editor_property(prop)
                        record[prop] = value if isinstance(value, bool) else str(value)
                    except Exception:
                        pass
                textures[str(name)] = record
        info["texture_parameters"] = textures
    except Exception as error:
        info["parameter_inspection_error"] = str(error)
    return info


def main():
    output = Path(os.environ["AISIM_AUDIT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
    report = dict(engine=unreal.SystemLibrary.get_engine_version(), actors=[], materials={}, sequences=[])
    for actor in actors:
        label = actor.get_actor_label()
        record = dict(
            label=label,
            path=actor.get_path_name(),
            class_name=actor.get_class().get_name(),
            position_cm=xyz(actor.get_actor_location()),
            rotation=str(actor.get_actor_rotation()),
        )
        origin, extent = actor.get_actor_bounds(False)
        record.update(bounds_origin_cm=xyz(origin), bounds_half_extent_cm=xyz(extent))
        components = []
        for component in actor.get_components_by_class(unreal.StaticMeshComponent):
            mesh = component.get_editor_property("static_mesh")
            mats = []
            for i in range(component.get_num_materials()):
                mat = component.get_material(i)
                if mat:
                    mats.append(mat.get_path_name())
                    report["materials"].setdefault(mat.get_path_name(), material_info(mat))
            components.append(
                dict(
                    name=component.get_name(),
                    mesh=mesh.get_path_name() if mesh else None,
                    materials=mats,
                    scale=xyz(component.get_world_scale()),
                )
            )
        record["components"] = components
        report["actors"].append(record)
    for asset in unreal.EditorAssetLibrary.list_assets(
        "/Game/Apartment_sim/Sequence", recursive=True, include_folder=False
    ):
        sequence = unreal.load_asset(asset)
        if not isinstance(sequence, unreal.LevelSequence):
            continue
        rate = sequence.get_display_rate()
        entry = dict(
            asset=asset,
            display_rate=[rate.numerator, rate.denominator],
            playback_start=sequence.get_playback_start(),
            playback_end=sequence.get_playback_end(),
            bindings=[
                dict(name=str(b.get_display_name()), guid=str(b.get_id())) for b in sequence.get_bindings()
            ],
        )
        if sequence.get_name().lower() == "cam01" and not (output / "cam01.usda").exists():
            options_class = next(
                getattr(unreal, n) for n in dir(unreal) if n.lower() == "levelsequenceexporterusdoptions"
            )
            exporter_class = next(
                getattr(unreal, n) for n in dir(unreal) if n.lower() == "levelsequenceexporterusd"
            )
            options = unreal.get_default_object(options_class)
            stage = options.get_editor_property("stage_options")
            stage.set_editor_property("meters_per_unit", 1.0)
            stage.set_editor_property("up_axis", unreal.UsdUpAxis.Z_AXIS)
            options.set_editor_property("stage_options", stage)
            options.set_editor_property("time_codes_per_second", float(rate.numerator / rate.denominator))
            options.set_editor_property("export_level", False)
            options.set_editor_property("export_subsequences_as_layers", False)
            task = unreal.AssetExportTask()
            task.object, task.filename, task.options = sequence, str(output / "cam01.usda"), options
            task.automated, task.prompt, task.replace_identical = True, False, False
            task.exporter = exporter_class()
            entry["usd_export_result"] = unreal.Exporter.run_asset_export_task(task)
        report["sequences"].append(entry)
    (output / "ue-audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    unreal.log("AISIM audit completed")
    if os.environ.get("AISIM_EXPORT_DIR"):
        import runpy

        runpy.run_path(str(Path(__file__).with_name("export_batch.py")), run_name="__main__")


if __name__ == "__main__":
    main()
