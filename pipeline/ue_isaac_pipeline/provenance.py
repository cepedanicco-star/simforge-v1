"""Parameter evidence describes what is known, not just where a number was copied."""

import math

METHODS = {
    "measured",
    "manufacturer",
    "official_asset",
    "source_authored",
    "reference",
    "estimated",
    "example",
}
TRUSTED_METHODS = {"measured", "manufacturer", "official_asset"}


def evidence(value, *, method, source_url, note, unit="1", applies_to=""):
    record = dict(
        value=value, method=method, source_url=source_url, note=note, unit=unit, applies_to=applies_to
    )
    validate_evidence(record)
    return record


def validate_evidence(record, *, parameter=""):
    if not isinstance(record, dict) or record.get("method") not in METHODS:
        raise ValueError(f"{parameter}: evidence must have a supported method")
    v = record.get("value")
    if not isinstance(v, (float, int)) or isinstance(v, bool) or not math.isfinite(v):
        raise ValueError(f"{parameter}: parameter value must be finite")
    if v < 0 or (parameter in ("mass_kg", "density_kg_m3") and v <= 0):
        raise ValueError(f"{parameter}: invalid negative or zero value")
    if parameter == "restitution" and v > 1:
        raise ValueError("restitution must be between 0 and 1")
    if not isinstance(record.get("source_url"), str) or not record["source_url"]:
        raise ValueError(f"{parameter}: source_url is required (asset path allowed)")
    if not record.get("note") or not record.get("unit"):
        raise ValueError(f"{parameter}: note and unit are required")
    if record["method"] in TRUSTED_METHODS and not record.get("applies_to"):
        raise ValueError(f"{parameter}: trusted evidence needs exact asset/contact-pair applicability")
    return record


def is_verified(record):
    return record.get("method") in TRUSTED_METHODS and bool(record.get("applies_to"))


def validate_profile(profile):
    required = {"static_friction", "dynamic_friction", "restitution"}
    if not required <= set(profile):
        raise ValueError(f"Physical material requires {sorted(required)}")
    for name, record in profile.items():
        if name not in required | {"mass_kg", "density_kg_m3"}:
            raise ValueError(f"Unknown physical parameter {name}")
        validate_evidence(record, parameter=name)
    if profile["dynamic_friction"]["value"] > profile["static_friction"]["value"]:
        raise ValueError("dynamic_friction must not exceed static_friction in V1 profiles")
    return profile
