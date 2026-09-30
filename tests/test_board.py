"""板几何与角点顺序的回归测试。

CORNER_SIGNS 是本库最容易写错、且错了以后症状很隐蔽的地方：
位姿会镜像翻转，重投影误差从 <0.5px 涨到十几个 px。这里用真实 tag 图钉死它。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from a_tag.board import CORNER_SIGNS, BoardLayout
from a_tag.detector import AprilTagDetector
from a_tag.pose import solve_board_pose
from a_tag.se3 import rotation_angle

from .synthetic import board_pose, camera_matrix, default_layout, project_board


def test_corner_signs_shape_and_uniqueness():
    assert CORNER_SIGNS.shape == (4, 2)
    assert len({tuple(row) for row in CORNER_SIGNS}) == 4, "四个角点不能重复"
    assert np.allclose(np.abs(CORNER_SIGNS), 1.0)


def test_tag_corners_are_on_board_plane():
    layout = default_layout()
    for tid in layout.tag_ids:
        c = layout.tag_corners(tid)
        assert c.shape == (4, 3)
        assert np.allclose(c[:, 2], 0.0), "板面必须是 Z=0"


def test_board_size():
    layout = default_layout(tag_size=0.04, hw=0.15, hh=0.10)
    assert layout.board_size == pytest.approx((0.34, 0.24))


def test_object_points_order_matches_tag_corners():
    layout = default_layout()
    pts, owner = layout.object_points([2, 0])
    assert owner.tolist() == [2, 2, 2, 2, 0, 0, 0, 0]
    assert np.allclose(pts[:4], layout.tag_corners(2))
    assert np.allclose(pts[4:], layout.tag_corners(0))


def test_config_roundtrip():
    layout = default_layout()
    assert BoardLayout.from_config(layout.to_config()) == layout


@pytest.mark.parametrize("tilt,spin", [(0.0, 0.0), (15.0, 0.0), (0.0, 25.0), (18.0, 35.0)])
def test_corner_order_recovers_true_pose(tilt, spin):
    """角点顺序正确时，端到端解出的位姿应几乎等于真值。"""
    layout = default_layout()
    K = camera_matrix()
    dist = np.zeros(5)
    T_true = board_pose(tilt, spin)

    img = project_board(layout, T_true, K, dist)
    corners = AprilTagDetector(expected_ids=layout.tag_ids).detect(img)
    assert len(corners) == 4, f"应检测到 4 个 tag，实际 {len(corners)}"

    res = solve_board_pose(corners, layout, K, dist)
    assert res.ok, res.reason
    assert res.reproj_rmse_px < 0.5

    ang = np.degrees(rotation_angle(res.T_cam_board[:3, :3], T_true[:3, :3]))
    dt = np.linalg.norm(res.T_cam_board[:3, 3] - T_true[:3, 3])
    assert ang < 0.5, f"角度误差 {ang:.3f} 度过大，角点顺序可能写反了"
    assert dt < 0.002, f"平移误差 {dt * 1000:.2f} mm 过大"


def test_wrong_corner_order_is_detected():
    """故意打乱角点顺序，重投影误差必须明显变大 —— 证明这个测试真的有鉴别力。"""
    layout = default_layout()
    K = camera_matrix()
    dist = np.zeros(5)
    img = project_board(layout, board_pose(15, 0), K, dist)
    corners = AprilTagDetector(expected_ids=layout.tag_ids).detect(img)

    good = solve_board_pose(corners, layout, K, dist)

    swapped = dict(corners)
    swapped[0] = corners[0][[1, 0, 3, 2]]        # 交换角点顺序
    bad = solve_board_pose(swapped, layout, K, dist)

    assert good.reproj_rmse_px < 0.5
    assert bad.reproj_rmse_px > 10.0 * max(good.reproj_rmse_px, 0.05)


def test_board_facing_camera_has_expected_normal():
    """板正对相机时，板坐标系 Z 轴(指向观察者)在相机系下应为 [0, 0, -1]。"""
    layout = default_layout()
    K = camera_matrix()
    res = solve_board_pose(
        AprilTagDetector(expected_ids=layout.tag_ids).detect(
            project_board(layout, board_pose(0, 0), K, np.zeros(5))),
        layout, K, np.zeros(5))
    assert res.ok
    assert np.allclose(res.T_cam_board[:3, 2], [0, 0, -1], atol=0.02)
