from pipeline.simforge.eval.batch import classify_failure, draw_initialization


def test_all_twenty_holdouts_repeat_and_stay_inside_authorized_pose_ranges():
    samples = [draw_initialization(seed) for seed in range(20)]
    assert samples == [draw_initialization(seed) for seed in range(20)]
    assert len({tuple(row["cup_xy_delta_m"]) for row in samples}) == 20
    for row in samples:
        assert all(-.15 <= value <= .15 for value in row["cup_xy_delta_m"])
        assert all(-.10 <= value <= .10 for value in row["robot_xy_delta_m"])
        assert -20 <= row["cup_yaw_delta_degrees"] <= 20
        assert -15 <= row["robot_yaw_delta_degrees"] <= 15


def test_extracting_cup_is_distinct_from_withdrawing_after_desk_placement():
    assert classify_failure("withdraw_with_cup", "Unexpected robot/environment contact") == "grasp_failure"
    assert classify_failure("withdraw_hand", "") == "placement_failure"
    assert classify_failure("turn_left_to_desk", "") == "trajectory_deviation"
    assert classify_failure("settle_open_grasp", "Cup tipped to 29 degrees") == "cup_topple_or_collision"
