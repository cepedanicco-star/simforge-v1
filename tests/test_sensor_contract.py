import numpy as np
import pytest

from pipeline.simforge.sensors.geometry import backproject, camera_contract, native_box_world, require_labels, uint16_ids


def test_cv_center_depth_and_world_translation():
    native = dict(renderProductResolution=[640, 480], metersPerSceneUnit=1,
                  cameraModel="pinhole", cameraApertureOffset=[0, 0],
                  cameraProjection=np.diag([2, 2, 1, 1]).ravel(),
                  cameraViewTransform=np.eye(4).ravel())
    camera = camera_contract(native)
    assert camera["fx"] == 640 and camera["fy"] == 480
    # Pixel centre lies exactly on the optical axis. USD looks down -Z, CV down +Z.
    np.testing.assert_allclose(backproject([[319.5, 239.5]], [2], camera), [[0, 0, -2]])
    cv_from_world = np.asarray(camera["extrinsic_world_to_cam"])
    cv_from_world[:3, 3] = [-1, -2, -3]
    camera["extrinsic_world_to_cam"] = cv_from_world.tolist()
    np.testing.assert_allclose(backproject([[319.5, 239.5]], [2], camera), [[1, -2, -5]])


def test_scaled_rotated_box_known_world_result():
    row = dict(x_min=-1, y_min=-2, z_min=0, x_max=1, y_max=2, z_max=2,
               transform=[[0, 2, 0, 0], [-1, 0, 0, 0], [0, 0, 3, 0], [5, 6, 7, 1]])
    box = native_box_world(row)
    np.testing.assert_allclose(box["center"], [5, 6, 10])
    np.testing.assert_allclose(box["extent"], [4, 4, 6])
    np.testing.assert_allclose(np.asarray(box["rotation"])[:, 0], [0, 1, 0])


def test_sparse_large_ids_remain_reversible():
    ids = np.asarray([[0, 70001], [2, 70001]], dtype=np.uint32)
    image, mapping = uint16_ids(ids)
    assert image.dtype == np.uint16
    reverse = {png: native for native, png in mapping.items()}
    decoded = np.vectorize(reverse.__getitem__)(image)
    np.testing.assert_array_equal(decoded, ids)
    with pytest.raises(ValueError, match="65536"):
        uint16_ids(np.arange(65537, dtype=np.uint32).reshape(1, -1))


def test_missing_target_label_is_an_error():
    with pytest.raises(ValueError, match="/World/cup"):
        require_labels({"/World/table": "desk"}, {"/World/cup": "cup_v1"})


def test_noncentral_projection_is_not_silently_centered():
    native = dict(renderProductResolution=[640, 480], metersPerSceneUnit=1,
                  cameraModel="pinhole", cameraApertureOffset=[0.1, 0],
                  cameraProjection=np.eye(4).ravel(), cameraViewTransform=np.eye(4).ravel())
    with pytest.raises(ValueError, match="Non-centred"):
        camera_contract(native)
