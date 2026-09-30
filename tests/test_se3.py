"""SE(3) 与欧拉角工具的测试。"""

from __future__ import annotations

import numpy as np
import pytest

from a_tag.se3 import (R_from_euler_zyx, R_from_quat, euler_zyx_from_R, invert,
                       make_T, pose_from_T, quat_from_R, rotation_angle, slerp,
                       transform_points)


def random_rotation(rng):
    q = rng.normal(size=4)
    return R_from_quat(q / np.linalg.norm(q))


def test_invert_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        T = make_T(random_rotation(rng), rng.normal(size=3))
        assert np.allclose(invert(T) @ T, np.eye(4), atol=1e-12)


def test_invert_matches_numpy():
    rng = np.random.default_rng(1)
    T = make_T(random_rotation(rng), rng.normal(size=3))
    assert np.allclose(invert(T), np.linalg.inv(T), atol=1e-10)


def test_transform_points_matches_matrix():
    rng = np.random.default_rng(2)
    T = make_T(random_rotation(rng), rng.normal(size=3))
    pts = rng.normal(size=(7, 3))
    homo = np.hstack([pts, np.ones((7, 1))]) @ T.T
    assert np.allclose(transform_points(T, pts), homo[:, :3])


def test_quat_roundtrip():
    rng = np.random.default_rng(3)
    for _ in range(50):
        R = random_rotation(rng)
        assert np.allclose(R_from_quat(quat_from_R(R)), R, atol=1e-10)


def test_euler_roundtrip_away_from_gimbal_lock():
    rng = np.random.default_rng(4)
    for _ in range(200):
        rpy = rng.uniform(-1.2, 1.2, size=3)      # 避开 pitch = +-90 度
        R = R_from_euler_zyx(*rpy)
        assert np.allclose(R_from_euler_zyx(*euler_zyx_from_R(R)), R, atol=1e-10)


def test_euler_matches_convention():
    """约定 R = Rz(yaw) @ Ry(pitch) @ Rx(roll)。"""
    roll, pitch, yaw = 0.3, -0.4, 0.7

    def Rx(x):
        return np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])

    def Ry(x):
        return np.array([[np.cos(x), 0, np.sin(x)], [0, 1, 0], [-np.sin(x), 0, np.cos(x)]])

    def Rz(x):
        return np.array([[np.cos(x), -np.sin(x), 0], [np.sin(x), np.cos(x), 0], [0, 0, 1]])

    assert np.allclose(R_from_euler_zyx(roll, pitch, yaw), Rz(yaw) @ Ry(pitch) @ Rx(roll))


def test_rotation_angle():
    rng = np.random.default_rng(5)
    R = random_rotation(rng)
    assert rotation_angle(R, R) == pytest.approx(0.0, abs=1e-9)
    # 绕 Z 轴 90 度
    Rz90 = R_from_euler_zyx(0, 0, np.pi / 2)
    assert rotation_angle(Rz90, np.eye(3)) == pytest.approx(np.pi / 2)


def test_slerp_endpoints_and_midpoint():
    R0, R1 = np.eye(3), R_from_euler_zyx(0, 0, np.pi / 2)
    q0, q1 = quat_from_R(R0), quat_from_R(R1)
    assert np.allclose(R_from_quat(slerp(q0, q1, 0.0)), R0, atol=1e-10)
    assert np.allclose(R_from_quat(slerp(q0, q1, 1.0)), R1, atol=1e-10)
    assert rotation_angle(R_from_quat(slerp(q0, q1, 0.5)), R0) == pytest.approx(np.pi / 4)


def test_slerp_takes_short_arc():
    """四元数反号时也必须走短弧，否则旋转会绕远路。"""
    R = R_from_euler_zyx(0, 0, 0.1)
    q0, q1 = quat_from_R(np.eye(3)), -quat_from_R(R)
    mid = R_from_quat(slerp(q0, q1, 0.5))
    assert rotation_angle(mid, np.eye(3)) == pytest.approx(0.05, abs=1e-6)


def test_to_camera_body_axis_mapping():
    """相机正对板且摆平时，机体轴在板系下应为 X=-Z, Y=-X, Z=+Y。"""
    from a_tag.se3 import to_camera_body

    T_board_optical = np.eye(4)
    T_board_optical[:3, :3] = R_from_euler_zyx(np.pi, 0.0, 0.0)   # 正对板
    T_board_optical[:3, 3] = [0.0, 0.0, 0.8]                      # 相机在板系 +Z 上

    T = to_camera_body(T_board_optical)
    bx, by, bz = T[:3, 0], T[:3, 1], T[:3, 2]
    assert np.allclose(bx, [0, 0, -1], atol=1e-9), "机体 X(前) 应指向板"
    assert np.allclose(by, [-1, 0, 0], atol=1e-9), "机体 Y(左) 应为板 -X"
    assert np.allclose(bz, [0, 1, 0], atol=1e-9), "机体 Z(上) 应为板 +Y"
    assert np.allclose(T[:3, 3], T_board_optical[:3, 3]), "位置不应改变"
    assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-9), "必须仍是正交阵"


def test_pose_from_T_units():
    T = make_T(R_from_euler_zyx(0, 0, np.pi / 2), [1.0, 2.0, 3.0])
    p = pose_from_T(T, degrees=True)
    assert (p["x"], p["y"], p["z"]) == (1.0, 2.0, 3.0)
    assert p["yaw"] == pytest.approx(90.0)
    assert pose_from_T(T, degrees=False)["yaw"] == pytest.approx(np.pi / 2)
