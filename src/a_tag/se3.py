"""SE(3) 刚体变换与欧拉角工具。

约定
----
* 变换矩阵 T_a_b 表示把 **b** 坐标系中的点变换到 **a** 坐标系::

      X_a = T_a_b @ X_b

  因此 T_board_cam 就是「相机在板坐标系下的位姿」，即本库的对外输出。
* 欧拉角采用 ZYX 内旋 (intrinsic)，等价于外旋 XYZ::

      R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

  该约定在 pitch = +-90 度处存在万向锁。内部请始终以 SE(3) 为准，
  只在展示或对外输出时才转欧拉角。
* 内部计算一律用弧度，对外输出一律用度。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "make_T", "invert", "transform_points", "rotation_angle",
    "euler_zyx_from_R", "R_from_euler_zyx", "pose_from_T",
    "quat_from_R", "R_from_quat", "slerp", "to_camera_body",
]


#: OpenCV 光学系 -> 相机机体系 的旋转。
#:
#: 列向量是机体轴在光学系下的表示:
#:   机体 X(前) = 光学 +Z     机体 Y(左) = 光学 -X     机体 Z(上) = 光学 -Y
#:
#: 用途: 对接 ROS / 机器人栈时，它们用 X 前 / Y 左 / Z 上。
#:
#: .. warning::
#:   转成机体系后**不要**再取 ZYX 欧拉角。相机正对板时其光轴沿板系 Z 轴，
#:   机体系下 pitch 会正好落在 +-90 度（实测 89.99 度），即万向锁，
#:   roll 与 yaw 不可分离。请以 4x4 矩阵形式继续传递。
_OPTICAL_TO_BODY = np.array([
    [0.0, -1.0, 0.0],
    [0.0,  0.0, -1.0],
    [1.0,  0.0, 0.0],
])


def to_camera_body(T_board_optical: np.ndarray) -> np.ndarray:
    """把「相机在板系下的位姿」从光学系改写为机体系。

    Parameters
    ----------
    T_board_optical:
        相机光学系在板坐标系下的位姿，即 PoseResult.T_board_cam。

    Returns
    -------
    T_board_body:
        相机机体系(X 前 / Y 左 / Z 上)在板坐标系下的位姿。位置不变，只换姿态约定。
    """
    return np.asarray(T_board_optical, dtype=np.float64) @ make_T(_OPTICAL_TO_BODY,
                                                                  np.zeros(3))


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """由旋转 (3,3) 和平移 (3,) 组装 4x4 齐次变换矩阵。"""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert(T: np.ndarray) -> np.ndarray:
    """刚体变换求逆（利用 R 的正交性，比 np.linalg.inv 更稳更快）。"""
    T = np.asarray(T, dtype=np.float64)
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """把 (N,3) 点集按 4x4 变换。"""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    T = np.asarray(T, dtype=np.float64)
    return pts @ T[:3, :3].T + T[:3, 3]


def rotation_angle(R1: np.ndarray, R2: np.ndarray) -> float:
    """两个旋转矩阵之间的测地夹角（弧度）。"""
    d = np.asarray(R1) @ np.asarray(R2).T
    return float(np.arccos(np.clip((np.trace(d) - 1.0) / 2.0, -1.0, 1.0)))


def euler_zyx_from_R(R: np.ndarray) -> tuple[float, float, float]:
    """旋转矩阵 -> (roll, pitch, yaw)，单位弧度，约定 R = Rz(y)Ry(p)Rx(r)。

    pitch 接近 +-90 度时 roll/yaw 不可分离（万向锁），此时把 roll 归零。
    """
    R = np.asarray(R, dtype=np.float64)
    sp = float(np.clip(-R[2, 0], -1.0, 1.0))
    pitch = float(np.arcsin(sp))
    if abs(sp) > 1.0 - 1e-9:                     # 万向锁
        roll = 0.0
        yaw = float(np.arctan2(-R[0, 1], R[1, 1]))
    else:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return roll, pitch, yaw


def R_from_euler_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """(roll, pitch, yaw) 弧度 -> 旋转矩阵，euler_zyx_from_R 的逆。"""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def pose_from_T(T: np.ndarray, degrees: bool = True) -> dict:
    """4x4 变换 -> {x, y, z, roll, pitch, yaw}，角度默认输出「度」。

    注意：欧拉角只适合展示，不要拿它做插值或滤波。
    """
    T = np.asarray(T, dtype=np.float64)
    roll, pitch, yaw = euler_zyx_from_R(T[:3, :3])
    if degrees:
        roll, pitch, yaw = np.degrees([roll, pitch, yaw])
    return {
        "x": float(T[0, 3]), "y": float(T[1, 3]), "z": float(T[2, 3]),
        "roll": float(roll), "pitch": float(pitch), "yaw": float(yaw),
    }


def quat_from_R(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 -> 单位四元数 (w, x, y, z)。"""
    R = np.asarray(R, dtype=np.float64)
    tr = float(np.trace(R))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                      (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                      0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    q = q / np.linalg.norm(q)
    return q if q[0] >= 0.0 else -q


def R_from_quat(q: np.ndarray) -> np.ndarray:
    """单位四元数 (w, x, y, z) -> 旋转矩阵。"""
    w, x, y, z = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    """四元数球面插值。滤波旋转请用它，不要直接平滑 rvec 或欧拉角。"""
    q0 = np.asarray(q0, dtype=np.float64).reshape(4)
    q1 = np.asarray(q1, dtype=np.float64).reshape(4)
    d = float(np.dot(q0, q1))
    if d < 0.0:                       # 走短弧
        q1, d = -q1, -d
    if d > 1.0 - 1e-9:                # 几乎重合，退化为线性插值
        out = q0 + u * (q1 - q0)
        return out / np.linalg.norm(out)
    th = float(np.arccos(np.clip(d, -1.0, 1.0)))
    return (np.sin((1 - u) * th) * q0 + np.sin(u * th) * q1) / np.sin(th)
