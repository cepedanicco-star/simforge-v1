"""Hard alignment gates before combining two native renderer passes."""

import numpy as np


def verify(rtl_camera, pt_camera, rtl_poses, pt_poses):
    def intrinsic(camera):
        return np.array([[camera["fx"], 0, camera["cx"]],
                         [0, camera["fy"], camera["cy"]], [0, 0, 1]], dtype=float)

    k_equal = np.array_equal(intrinsic(rtl_camera), intrinsic(pt_camera))
    resolution_equal = all(rtl_camera[k] == pt_camera[k] for k in ("width", "height"))
    extrinsic_equal = np.array_equal(rtl_camera["extrinsic_world_to_cam"],
                                    pt_camera["extrinsic_world_to_cam"])
    paths_equal = rtl_poses.keys() == pt_poses.keys()
    differences = {path: float(np.max(np.abs(np.asarray(matrix) - pt_poses[path])))
                   for path, matrix in rtl_poses.items() if path in pt_poses}
    maximum = max(differences.values(), default=None)
    finite = maximum is not None and all(np.isfinite(v) for v in differences.values())
    passed = bool(k_equal and resolution_equal and extrinsic_equal and paths_equal
                  and finite and maximum < 1e-4)
    report = dict(passed=passed, K_elementwise_exact=bool(k_equal),
                  resolution_equal=resolution_equal, extrinsic_elementwise_exact=extrinsic_equal,
                  object_paths_equal=paths_equal, object_count=len(rtl_poses),
                  maximum_pose_matrix_difference=maximum, pose_tolerance_strict=1e-4)
    if not passed:
        raise ValueError("Dual-pass alignment failed: " + str(report))
    return report
