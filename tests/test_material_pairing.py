import math

import pytest

from pipeline.simforge.materials.camera_pairing import camera_pairing, require_camera_pairing


def fixture():
    matrix = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [1,2,3,1]]
    view = dict(source_frame=0, usd_camera_matrix=matrix, ue_position_cm=[100,-200,300],
                horizontal_fov_degrees=50, vertical_fov_degrees=30)
    measured = dict(position_cm=[100,-200,300], forward=[0,0,-1], up=[0,-1,0],
                    horizontal_fov_degrees=50, vertical_fov_degrees=30)
    return view, measured


@pytest.mark.parametrize("field,value,error", [
    ("position_cm", [100.001,-200,300], "position_error_m"),
    ("forward", [math.sin(.01),0,-math.cos(.01)], "forward_error_degrees"),
    ("up", [0,-math.cos(.01),math.sin(.01)], "up_error_degrees"),
    ("horizontal_fov_degrees", 50.001, "horizontal_fov_error_degrees"),
    ("vertical_fov_degrees", 30.001, "vertical_fov_error_degrees")])
def test_every_component_can_independently_reject(field, value, error):
    view, measured = fixture()
    assert require_camera_pairing(view, measured)["status"] == "PASS"
    measured[field] = value
    assert camera_pairing(view, measured)["violations"] == [error]
    with pytest.raises(ValueError):
        require_camera_pairing(view, measured)


def test_v01_real_failure_formula_is_preserved():
    matrix = [[-.5735764363510459,.8191520442889921,0,0],
              [-.11400393016147588,-.07982641128459868,.9902680687415704,0],
              [.8111801129037697,.5679944299010226,.13917310096006547,0],
              [5.477235595347679,-.34988536052662056,1.5599999651312828,1]]
    view, observed = fixture()
    view["usd_camera_matrix"] = matrix
    observed.update(forward=[-.8191520442889912,9.71445146547012e-17,.5735764363510458],
                    up=[-.5679944299010224,-.13917310096006522,-.8111801129037688])
    result = camera_pairing(view, observed)
    assert result["forward_error_degrees"] == pytest.approx(54.22148933870065, abs=1e-10)
    assert result["up_error_degrees"] == pytest.approx(138.55936384465818, abs=1e-10)
