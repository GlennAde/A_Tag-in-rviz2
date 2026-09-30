"""位姿求解与滤波的测试。"""

from __future__ import annotations

import numpy as np
import pytest

from a_tag.board import BoardLayout
from a_tag.detector import AprilTagDetector
from a_tag.filter import OneEuroFilter, PoseFilter
from a_tag.pose import build_correspondences, solve_board_pose
from a_tag.se3 import invert, make_T, rotation_angle

from .synthetic import board_pose, camera_matrix, default_layout, project_board

K = camera_matrix()
DIST = np.zeros(5)


def solve_from_render(layout, T_true, **kw):
    img = project_board(layout, T_true, K, DIST)
    corners = AprilTagDetector(expected_ids=layout.tag_ids).detect(img)
    return solve_board_pose(corners, layout, K, DIST, **kw), corners


def test_board_cam_is_inverse_of_cam_board():
    layout = default_layout()
    res, _ = solve_from_render(layout, board_pose(10, 20))
    assert res.ok
    assert np.allclose(res.T_board_cam @ res.T_cam_board, np.eye(4), atol=1e-9)
    assert np.allclose(res.T_board_cam, invert(res.T_cam_board), atol=1e-12)


def test_camera_position_and_distance():
    layout = default_layout()
    T_true = board_pose(0, 0, t=(0.0, 0.0, 0.7))
    res, _ = solve_from_render(layout, T_true)
    assert res.ok
    # 相机在板坐标系下的位置 = 真值平移取逆
    expected = invert(T_true)[:3, 3]
    assert np.allclose(res.camera_position, expected, atol=0.002)
    assert res.distance == pytest.approx(0.7, abs=0.003)


def test_pose_dict_has_all_six_dof():
    layout = default_layout()
    res, _ = solve_from_render(layout, board_pose(5, 10))
    p = res.pose()
    assert set(p) == {"x", "y", "z", "roll", "pitch", "yaw"}


def test_min_tags_enforced():
    layout = default_layout()
    img = project_board(layout, board_pose(0, 0), K, DIST)
    corners = AprilTagDetector(expected_ids=layout.tag_ids).detect(img)

    one = {layout.tag_ids[0]: corners[layout.tag_ids[0]]}
    assert not solve_board_pose(one, layout, K, DIST, min_tags=2).ok

    two = {t: corners[t] for t in layout.tag_ids[:2]}
    assert solve_board_pose(two, layout, K, DIST, min_tags=2).ok
    assert not solve_board_pose(two, layout, K, DIST, min_tags=3).ok


def test_unknown_tag_ids_are_ignored():
    layout = default_layout()
    img = project_board(layout, board_pose(0, 0), K, DIST)
    corners = AprilTagDetector(expected_ids=layout.tag_ids).detect(img)
    corners[999] = corners[layout.tag_ids[0]].copy()          # 板上没有 999
    res = solve_board_pose(corners, layout, K, DIST)
    assert res.ok
    assert 999 not in res.tag_ids


def test_empty_detection_is_not_ok():
    layout = default_layout()
    res = solve_board_pose({}, layout, K, DIST)
    assert not res.ok and res.n_tags == 0
    assert not bool(res)


def test_reprojection_threshold_rejects_bad_pose():
    """把 tag 尺寸写错，重投影误差应超标并被拒绝。"""
    layout = default_layout()
    img = project_board(layout, board_pose(10, 0), K, DIST)
    corners = AprilTagDetector(expected_ids=layout.tag_ids).detect(img)

    wrong = BoardLayout(tag_size=0.04 * 1.3,
                        tag_centers=default_layout().tag_centers)
    res = solve_board_pose(corners, wrong, K, DIST, max_reproj_px=1.0)
    assert not res.ok
    assert res.reproj_rmse_px > 1.0
    assert "重投影" in res.reason


def test_more_tags_give_better_rotation():
    """4 个 tag 应优于 1 个 tag —— 这是「用四个 tag」的核心收益。"""
    layout = default_layout()
    T_true = board_pose(12, 0)
    img = project_board(layout, T_true, K, DIST)
    corners = AprilTagDetector(expected_ids=layout.tag_ids).detect(img)

    full = solve_board_pose(corners, layout, K, DIST)
    single = solve_board_pose({layout.tag_ids[0]: corners[layout.tag_ids[0]]},
                              layout, K, DIST, min_tags=1)
    e_full = rotation_angle(full.T_cam_board[:3, :3], T_true[:3, :3])
    e_single = rotation_angle(single.T_cam_board[:3, :3], T_true[:3, :3])
    assert e_full <= e_single


def test_build_correspondences():
    layout = default_layout()
    img = project_board(layout, board_pose(0, 0), K, DIST)
    corners = AprilTagDetector(expected_ids=layout.tag_ids).detect(img)
    obj, ip, ids = build_correspondences(corners, layout)
    assert obj.shape == (16, 3) and ip.shape == (16, 2)
    assert ids == layout.tag_ids
    assert np.allclose(obj[:, 2], 0.0)


# ------------------------------------------------------------------ 滤波
def test_one_euro_first_sample_passthrough():
    f = OneEuroFilter()
    assert f(3.0, 1 / 30) == pytest.approx(3.0)


def test_one_euro_smooths_noise():
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0, d_cutoff=1.0)
    out = [f(1.0 + rng.normal(0, 0.1), 1 / 30) for _ in range(300)]
    assert np.std(out[100:]) < np.std([1.0 + rng.normal(0, 0.1) for _ in range(200)])


def test_pose_filter_converges_and_holds():
    layout = default_layout()
    res, _ = solve_from_render(layout, board_pose(8, 15))
    f = PoseFilter()
    for _ in range(50):
        T = f(res.T_board_cam, 1 / 30)
    assert np.allclose(T, res.T_board_cam, atol=1e-3)

    held = f(None, 1 / 30)
    assert np.allclose(held, T)
    assert f.stale_frames == 1


def test_pose_filter_first_frame_is_exact():
    """首帧必须直接采纳，不能从单位矩阵慢慢收敛。"""
    layout = default_layout()
    res, _ = solve_from_render(layout, board_pose(0, 0))
    f = PoseFilter()
    T = f(res.T_board_cam, 1 / 30)
    assert np.allclose(T, res.T_board_cam)


def test_pose_filter_reduces_jitter():
    layout = default_layout()
    rng = np.random.default_rng(1)
    T = board_pose(10, 0).copy()
    f = PoseFilter()
    raw, smooth = [], []
    for _ in range(200):
        noisy = T.copy()
        noisy[:3, 3] += rng.normal(0, 0.001, size=3)
        raw.append(noisy[:3, 3].copy())
        smooth.append(f(noisy, 1 / 30)[:3, 3].copy())

    # 必须逐轴算抖动：直接用全局 std 会被 x/y/z 三轴之间的固有差异
    # (z 约 0.8m，x/y 约 0.03m) 淹没，量到的就不是噪声了。
    raw_jitter = np.array(raw)[50:].std(axis=0).mean()
    smooth_jitter = np.array(smooth)[50:].std(axis=0).mean()
    assert smooth_jitter < 0.5 * raw_jitter, (
        f"滤波后抖动 {smooth_jitter:.5f} 未明显小于输入 {raw_jitter:.5f}")


def test_pose_filter_preserves_mean():
    """滤波只能压抖动，不能把位置拉偏。"""
    layout = default_layout()
    rng = np.random.default_rng(2)
    T = board_pose(10, 0, t=(0.05, 0.02, 0.6)).copy()
    f = PoseFilter()
    for _ in range(400):
        noisy = T.copy()
        noisy[:3, 3] += rng.normal(0, 0.001, size=3)
        f(noisy, 1 / 30)
    assert np.allclose(f.T[:3, 3], T[:3, 3], atol=2e-3)
