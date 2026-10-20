"""Native names verified by M1's Isaac Sim 4.5 probe."""


def attach(product, channels):
    import omni.replicator.core as rep
    sensors = {}
    for name in channels:
        init = {"colorize": False} if name in ("semantic_segmentation", "instance_segmentation") else {}
        # No fallback to a different annotator: a failed channel fails capture.
        ann = rep.annotators.get(name, init_params=init)
        ann.attach([product])
        sensors[name] = ann
    return sensors


def read(sensors):
    import numpy as np
    values = {}
    for name, ann in sensors.items():
        value = ann.get_data()
        if value is None:
            raise ValueError("Native annotator returned None: " + name)
        array = value.get("data") if isinstance(value, dict) else value
        if isinstance(array, np.ndarray) and array.size == 0:
            raise ValueError("Native annotator returned an empty channel: " + name)
        values[name] = value
    return values


def bbox_state(sensor):
    """Describe native data without replacing an empty channel with invented boxes."""
    import numpy as np
    value = sensor.get_data()
    data = value.get("data") if isinstance(value, dict) else value
    return dict(top_type=type(value).__name__, data_type=type(data).__name__,
                is_none=value is None or data is None,
                shape=list(data.shape) if isinstance(data, np.ndarray) else None,
                size=int(data.size) if isinstance(data, np.ndarray) else None,
                nonempty=isinstance(data, np.ndarray) and data.size > 0)


def wait_for_bbox(sensors, step, record, retries=10):
    """Discard each empty render buffer; never write a frame before this gate."""
    if retries != 10:
        raise ValueError("M3 requires exactly ten consecutive empty attempts before failure")
    for attempt in range(1, retries + 1):
        step()
        state = dict(attempt=attempt, **bbox_state(sensors["bounding_box_3d"]))
        record(state)
        if state["nonempty"]:
            return read(sensors)
    raise ValueError("Native bounding_box_3d empty for 10 consecutive rendered buffers")
