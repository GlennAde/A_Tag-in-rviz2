"""合成 AprilTag 板图像，供无相机环境下的回归测试使用。

用 cv2.aruco 生成真实的 tag36h11 图案（和 pupil_apriltags 同族），再按已知位姿
做透视投影贴到画布上，于是**真值位姿已知**，可以端到端校验整条链路。
"""

from __future__ import annotations

import cv2
import numpy as np
from pupil_apriltags import Detector

from a_tag.board import BoardLayout

_ARUCO = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
_PX = 800
_src_cache: dict = {}


def marker_and_src(tag_id: int, px: int = _PX):
    """返回 (marker 图像, tag 四角在 marker 图像中的像素位置)。

    注意: generateImageMarker 输出的其实是 tag 本体(8 模块)，
    外侧白边不一定按 borderBits 加。所以这里不猜模块数，而是把 marker
    平铺检测一次，用**检测到的角点**作为映射源，自动和 CORNER_SIGNS 同序。
    """
    key = (tag_id, px)
    if key in _src_cache:
        mk = cv2.aruco.generateImageMarker(_ARUCO, tag_id, px, borderBits=1)
        return mk, _src_cache[key]

    mk = cv2.aruco.generateImageMarker(_ARUCO, tag_id, px, borderBits=1)
    pad = px // 2
    canvas = np.full((px + 2 * pad, px + 2 * pad), 255, np.uint8)
    canvas[pad:pad + px, pad:pad + px] = mk
    found = Detector(families="tag36h11", nthreads=1, quad_decimate=1.0).detect(canvas)
    if not found:
        raise RuntimeError(f"合成失败: 无法在平铺图中检测到 tag {tag_id}")
    src = (found[0].corners - pad).astype(np.float32)
    _src_cache[key] = src
    return mk, src


def project_board(layout: BoardLayout, T_cam_board: np.ndarray,
                  camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                  size: tuple = (1280, 720)) -> np.ndarray:
    """把整块板按 T_cam_board 投影渲染成灰度图。"""
    W, H = size
    img = np.full((H, W), 255, np.uint8)
    T = np.asarray(T_cam_board, dtype=np.float64)
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    tvec = T[:3, 3].reshape(3, 1)
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)

    for tid in layout.tag_ids:
        quad = layout.tag_corners(tid)                      # (4,3) 板坐标系
        pts, _ = cv2.projectPoints(quad, rvec, tvec, camera_matrix, dist_coeffs)
        mk, src = marker_and_src(tid)
        Hm = cv2.getPerspectiveTransform(src, pts.reshape(-1, 2).astype(np.float32))
        warped = cv2.warpPerspective(mk, Hm, (W, H),
                                     flags=cv2.INTER_AREA, borderValue=255)
        img = np.minimum(img, warped)
    return img


def camera_matrix(fx: float = 1000.0, fy: float = 1000.0,
                  cx: float = 640.0, cy: float = 360.0) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def board_pose(tilt_deg: float = 0.0, spin_deg: float = 0.0,
               t=(0.03, -0.02, 0.8)) -> np.ndarray:
    """构造一个「板大致正对相机」的真值位姿 T_cam_board。

    tilt_deg 绕相机 X 轴倾斜，spin_deg 绕相机 Z 轴面内旋转。板坐标系 Z 指向相机，
    所以正对相机时 R = Rx(180 度)。
    """
    a, b = np.deg2rad(tilt_deg), np.deg2rad(spin_deg)

    def Rx(x):
        return np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])

    def Rz(x):
        return np.array([[np.cos(x), -np.sin(x), 0], [np.sin(x), np.cos(x), 0], [0, 0, 1]])

    T = np.eye(4)
    T[:3, :3] = Rz(b) @ Rx(np.pi) @ Rx(-a)
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


def default_layout(tag_size: float = 0.04, hw: float = 0.15, hh: float = 0.10) -> BoardLayout:
    return BoardLayout(tag_size=tag_size, tag_centers={
        0: (-hw, -hh), 1: (hw, -hh), 2: (hw, hh), 3: (-hw, hh)})
