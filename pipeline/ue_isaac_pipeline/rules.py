"""Small, auditable classifier. External agents can supply reviewed manifest overrides."""

import re

# This is a rule dictionary, not a claim that a neural model recognizes every object.
CLASS_PATTERNS = [
    ("water_surface", "visual_only", r"(?:toilet|bathtub)_water"),
    ("floor_drain", "static", r"floor_drain"),
    ("lamp", "static", r"lamp|halogen|light_fixture|灯"),
    ("shelf", "static", r"shelf|书架|搁板"),
    ("floor", "static", r"floor|ground|landscape|地面|地板|terrain"),
    ("wall", "static", r"wall|ceiling|building|墙|天花|建筑"),
    ("door", "articulated", r"door|drawer|门|抽屉"),
    ("table", "static", r"table|desk|桌"),
    ("chair", "static", r"chair|seat|椅"),
    ("tree", "static", r"tree|trunk|树"),
    ("mug", "dynamic", r"mug|cup|杯"),
    ("bottle", "dynamic", r"bottle|瓶"),
    ("box", "dynamic", r"box|crate|箱|盒"),
    ("ball", "dynamic", r"ball|sphere|球"),
    ("window", "static", r"window|窗"),
    ("sofa", "static", r"sofa|couch|沙发"),
    ("bed", "static", r"(?:^|_)bed(?:_|\b)|床"),
    ("cabinet", "static", r"cabinet|wardrobe|kitchen_furniture|柜"),
    ("sink", "static", r"sink|洗手盆"),
    ("toilet", "static", r"toilet$|toilet\.|马桶"),
    ("bathtub", "static", r"bathtub|浴缸"),
    ("shower", "static", r"shower|showehandle|淋浴"),
    ("curtain", "static", r"curtain|窗帘"),
    ("radiator", "static", r"radiator|暖气"),
    ("kitchen_appliance", "static", r"microwave|microvawe|cooker|hood|灶"),
    ("computer", "static", r"computer|keyboard|mouse_|电脑|键盘"),
    ("television", "static", r"(?:_|\b)tv(?:_|\b)|tuner|电视"),
    ("book", "dynamic", r"books?|album|folder|书"),
    ("picture", "static", r"picture|paint_|画"),
    ("plant", "static", r"plant|盆栽"),
    ("tableware", "dynamic", r"plate|pan_|pot_|knife|tray|餐具"),
    ("bathroom_accessory", "static", r"shampoo|soap|tooth|brush|hair|nailpolish|towel|paperholder|wax|deo"),
    ("pipe", "static", r"pipe|管"),
    ("electrical_fixture", "static", r"socket|switch|插座|开关"),
    ("trashbin", "static", r"trashbin|垃圾桶"),
    ("food", "dynamic", r"cheese|cake|food|食物"),
    ("basket", "dynamic", r"basket|篮"),
    ("glassware", "dynamic", r"glass_v|玻璃杯"),
    ("textile", "static", r"wool|fabric|织物"),
]


def classify(name, explicit=None):
    if explicit:
        label = explicit.get("class_name", "unknown")
        role = explicit.get("role", "static")
        if role not in {"static", "dynamic", "articulated", "visual_only", "preserve"}:
            raise ValueError(f"Invalid physics role: {role}")
        if not isinstance(label, str) or not label:
            raise ValueError("class_name must be nonempty")
        return dict(
            class_name=label,
            role=role,
            class_source=explicit.get("source", "manifest"),
            review_required=not explicit.get("reviewed", False),
        )
    for label, role, pattern in CLASS_PATTERNS:
        if re.search(pattern, name, re.IGNORECASE):
            return dict(class_name=label, role=role, class_source="name_rule_estimate", review_required=True)
    return dict(class_name="unknown", role="static", class_source="unclassified", review_required=True)
