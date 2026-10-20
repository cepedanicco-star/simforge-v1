from pipeline.audit_release import inspect_file,inspect_text
import importlib


def test_coco_export_source_is_part_of_the_delivered_package():
    module=importlib.import_module('pipeline.simforge.dataset.coco')
    assert callable(module.export)


def test_host_path_and_credential_detectors_without_real_sensitive_values():
    candidate=chr(67)+':'+chr(92)+'Us'+'ers'+chr(92)+'example'+chr(92)+'file.txt'
    assert 'absolute_drive_path' in inspect_text(candidate)
    candidate='gh'+'p_'+'x'*32
    assert 'github_credential' in inspect_text(candidate)
    assert inspect_text('UsdGeom.Tokens.z')==[]


def test_raw_experiment_files_and_weights_are_excluded():
    assert 'raw_experiment_artifact' in inspect_file('results/bonus-report-example.json',b'{}')
    assert 'excluded_binary_or_ledger' in inspect_file('checkpoint.pt',b'example')
    assert inspect_file('configs/schema.json',b'{"type":"object"}')==[]
