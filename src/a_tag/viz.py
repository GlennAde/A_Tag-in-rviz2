"""可视化：把检测、板框和位姿画到图像上。"""

from __future__ import annotations

import cv2
import numpy as np

from .board import BoardLayout
from .pose import PoseResult
from .se3 import pose_from_T

__all__ = ["draw_detections", "draw_board", "draw_pose_axes", "draw_hud"]

_COLOR_OK = (0, 220, 0)
_COLOR_BAD = (0, 0, 255)
_COLOR_TAG = (0, 200, 255)
_COLOR_BOARD = (255, 160, 0)


def draw_detections(bgr: np.ndarray, corners_by_id: dict,
                    layout: BoardLayout | None = None) -> np.ndarray:
    """画出每个 tag 的四边形、角点编号和 id。"""
    for tid, corners in corners_by_id.items():
        pts = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(bgr, [pts], True, _COLOR_TAG, 2, cv2.LINE_AA)
        for i, (x, y) in enumerate(np.asarray(corners).reshape(4, 2)):
            cv2.circle(bgr, (int(x), int(y)), 3, _COLOR_TAG, -1, cv2.LINE_AA)
            # 角点序号：0=右上 1=左上 2=左下 3=右下，排查顺序问题很有用
            cv2.putText(bgr, str(i), (int(x) + 4, int(y) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, _COLOR_TAG, 1, cv2.LINE_AA)
        cx, cy = np.asarray(corners).reshape(4, 2).mean(axis=0)
        known = layout is not None and tid in layout.tag_centers
        cv2.putText(bgr, f"id={tid}" + ("?" if not known else ""),
                    (int(cx) - 20, int(cy) + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    _COLOR_TAG if known else _COLOR_BAD, 2, cv2.LINE_AA)
    return bgr


def draw_board(bgr: np.ndarray, result: PoseResult, layout: BoardLayout,
               camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
    """把标定板外框投影回图像。"""
    outline = result.project_board_outline(camera_matrix, dist_coeffs, layout)
    if len(outline) == 0:
        return bgr
    cv2.polylines(bgr, [outline.astype(np.int32).reshape(-1, 1, 2)],
                  True, _COLOR_BOARD, 2, cv2.LINE_AA)
    return bgr


def draw_pose_axes(bgr: np.ndarray, T_cam_board: np.ndarray,
                   camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                   length: float = 0.05) -> np.ndarray:
    """在板坐标系原点画出 X(红) / Y(绿) / Z(蓝) 三轴。

    T_cam_board 是「板在相机系下的位姿」，正是 drawFrameAxes 需要的 rvec/tvec。
    Z 轴指向相机（因为板坐标系的 Z 朝外指向观察者）。
    """
    if T_cam_board is None:
        return bgr
    rvec, _ = cv2.Rodrigues(np.asarray(T_cam_board)[:3, :3])
    cv2.drawFrameAxes(bgr, camera_matrix, dist_coeffs, rvec,
                      np.asarray(T_cam_board)[:3, 3], length, 2)
    return bgr


def draw_hud(bgr: np.ndarray, result: PoseResult | None,
             T_filtered: np.ndarray | None = None,
             fps: float = 0.0, stale: int = 0) -> np.ndarray:
    """左上角叠加状态文字。

    显示的是**相机在板坐标系下的位姿**（也就是本库的主输出）。
    """
    lines: list[str] = []
    if result is None:
        lines.append("no detection")
    else:
        lines.append(f"tags: {result.n_tags} {list(result.tag_ids)}")
        lines.append(f"reproj: {result.reproj_rmse_px:.3f} px")

    if T_filtered is not None:
        p = pose_from_T(T_filtered, degrees=True)
        lines.append(f"cam in board (mm, deg)")
        lines.append(f"  x={p['x'] * 1000:8.1f}  y={p['y'] * 1000:8.1f}  z={p['z'] * 1000:8.1f}")
        lines.append(f"  r={p['roll']:7.2f}  p={p['pitch']:7.2f}  y={p['yaw']:7.2f}")
        if stale:
            lines.append(f"  STALE x{stale}")

    if result is not None and result.reason:
        lines.append(result.reason)

    lines.append(f"fps: {fps:5.1f}")

    ok = result is not None and result.ok
    y = 22
    for i, text in enumerate(lines):
        color = _COLOR_OK if ok else _COLOR_BAD
        if text.startswith("  "):
            color = (255, 255, 255)
        cv2.putText(bgr, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(bgr, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)
        y += 20
    return bgr
