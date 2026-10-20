"""One camera criterion for UE pre-render gates and independent post-capture audits.

USD camera looks along local -Z with local +Y up. UE is centimetres and
opposite world Y. Keep the dot/acos formula used by the v01 failure receipt.
"""
import math

LIMITS = dict(position_error_m=1e-6, forward_error_degrees=1e-4,
              up_error_degrees=1e-4, horizontal_fov_error_degrees=1e-4,
              vertical_fov_error_degrees=1e-4)


def angle_degrees(expected, observed):
    return math.degrees(math.acos(max(-1.0, min(1.0, sum(a*b for a, b in zip(expected, observed))))))


def vertical_fov(horizontal_degrees, width, height):
    return math.degrees(2*math.atan(math.tan(math.radians(horizontal_degrees/2))*height/width))


def camera_pairing(view, observed):
    """Return raw errors and fixed tolerances; do not normalize away bad axes."""
    matrix = view["usd_camera_matrix"]
    forward = [-matrix[2][0], matrix[2][1], -matrix[2][2]]
    up = [matrix[1][0], -matrix[1][1], matrix[1][2]]
    result = dict(frame=view["source_frame"],
        position_error_m=math.dist(observed["position_cm"], view["ue_position_cm"])/100,
        expected_forward=forward, actual_forward=observed["forward"],
        expected_up=up, actual_up=observed["up"],
        forward_error_degrees=angle_degrees(forward, observed["forward"]),
        up_error_degrees=angle_degrees(up, observed["up"]),
        horizontal_fov_error_degrees=abs(observed["horizontal_fov_degrees"]-view["horizontal_fov_degrees"]),
        vertical_fov_error_degrees=abs(observed["vertical_fov_degrees"]-view["vertical_fov_degrees"]))
    result["limits"] = dict(LIMITS)
    result["violations"] = [key for key, limit in LIMITS.items()
                            if not math.isfinite(result[key]) or result[key] > limit]
    result["status"] = "FAIL" if result["violations"] else "PASS"
    return result


def require_camera_pairing(view, observed):
    result = camera_pairing(view, observed)
    if result["status"] != "PASS":
        raise ValueError("Same-camera gate failed: " + repr(result))
    return result
