"""Frozen arm order, phase order and USD target paths."""
PHASES = ("settle_open_grasp", "close", "lift", "hold", "withdraw_with_cup", "turn_right_with_cup",
          "navigate_apartment_with_cup", "turn_left_to_desk", "approach_desk_forward", "settle_at_desk",
          "move_above_desk", "lower_to_desk", "release_on_desk", "withdraw_hand", "verify_placement", "unknown")
ARM_NAMES = tuple(side+"_"+joint+"_joint" for side in ("left", "right") for joint in (
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"))
PALMS = ("left_hand_palm_link", "right_hand_palm_link")
CUP = "/World/Environment/Source/cup_V1"
