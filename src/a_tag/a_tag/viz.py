"""把检测结果、板框投影和状态文字画到图像上（节点发 debug 图用）。

和以前那个窗口版 viz.py 不同：这里只做"叠加"，不碰窗口/键盘，结果交给 ROS 话题发布。
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["draw_overlay"]

_COLOR_TAG = (0, 200, 255)      # 期望的 tag：橙黄
_COLOR_UNKNOWN = (0, 0, 255)    # 不在 target.yaml 里的 id：红
_COLOR_BOARD = (255, 160, 0)    # 板框：蓝
_COLOR_PROJ = (255, 0, 255)     # 按解出的位姿反投影的 tag 角点：品红
_COLOR_TEXT = (255, 255, 255)
_COLOR_SHADOW = (0, 0, 0)


def _put_text(bgr, text, x, y, scale=0.6, color=_COLOR_TEXT, thickness=1) -> None:
    """带黑描边的文字，亮背景上也看得清。"""
    cv2.putText(bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                _COLOR_SHADOW, thickness + 2, cv2.LINE_AA)
    cv2.putText(bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def draw_overlay(
    bgr: np.ndarray,
    corners_by_id: dict,
    result=None,
    layout=None,
    camera_matrix=None,
    dist_coeffs=None,
    lines=(),
    expected_ids=None,
) -> np.ndarray:
    """在图上叠加：tag 框+编号、板框与角点反投影、左上角状态文字。

    Parameters
    ----------
    bgr:
        uint8 HxWx3 BGR 图（会被复制，不改原图）。
    corners_by_id:
        {tag_id: (4,2)} 检测到的角点（本库顺序：右上/左上/左下/右下）。
    result:
        PoseResult 或 None；有 T_cam_board 时画板框和 tag 角点反投影。
    layout, camera_matrix, dist_coeffs:
        画投影需要；缺一个就跳过投影部分。
    lines:
        左上角逐行显示的文字（就是节点日志里那几行）。
    expected_ids:
        target.yaml 里期望的 id；不在这里的画红。
    """
    img = bgr.copy()
    known = None if expected_ids is None else {int(i) for i in expected_ids}

    for tid, corners in corners_by_id.items():
        pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
        color = _COLOR_TAG if (known is None or tid in known) else _COLOR_UNKNOWN
        cv2.polylines(img, [pts.astype(np.int32).reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA)
        for i, (x, y) in enumerate(pts):
            cv2.circle(img, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)
            _put_text(img, str(i), int(x) + 4, int(y) - 4, 0.45, color, 1)
        cx, cy = pts.mean(axis=0)
        _put_text(img, f"id={tid}", int(cx) - 26, int(cy) + 6, 0.6, color, 2)

    # 按解出的位姿反投影：板外框 + 每个 tag 的角点。对不齐就说明布局/内参不对。
    if result is not None and result.T_cam_board is not None and layout is not None:
        K = camera_matrix
        D = dist_coeffs if dist_coeffs is not None else np.zeros(5)
        T = np.asarray(result.T_cam_board)
        rvec, _ = cv2.Rodrigues(T[:3, :3])
        tvec = T[:3, 3]
        w, h = layout.board_size
        outline = np.array([[-w / 2, -h / 2, 0.0], [w / 2, -h / 2, 0.0],
                            [w / 2, h / 2, 0.0], [-w / 2, h / 2, 0.0]])
        pts, _ = cv2.projectPoints(outline, rvec, tvec, K, D)
        cv2.polylines(img, [pts.reshape(-1, 1, 2).astype(np.int32)], True,
                      _COLOR_BOARD, 2, cv2.LINE_AA)
        for tid in corners_by_id:
            if tid in layout.tag_centers:
                tp = layout.tag_corners(tid)[:, :2]
                proj, _ = cv2.projectPoints(np.column_stack([tp, np.zeros(4)]),
                                            rvec, tvec, K, D)
                for p in proj.reshape(4, 2):
                    cv2.circle(img, (int(p[0]), int(p[1])), 2, _COLOR_PROJ, -1, cv2.LINE_AA)

    # 左上角文字块
    y = 24
    for text in lines:
        _put_text(img, text, 10, y, 0.55)
        y += 22
    return img
