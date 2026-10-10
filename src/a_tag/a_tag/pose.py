"""由多 tag 角点联合求解「相机相对标定板」的位姿。

输出约定
--------
T_cam_board
    PnP 的直接结果: 板坐标系 -> 相机坐标系, X_cam = R @ X_board + t。
T_board_cam
    **对外的主输出**: 相机在板坐标系下的位姿, 等于 T_cam_board 的逆。

为什么用 SQPNP
--------------
平面 PnP 在相机正对板时是病态的。实测(4 tag, 0.3px 角点噪声, 300 次):

    ==================  ============  ============
    求解器               正对相机      倾斜 20 度
    ==================  ============  ============
    IPPE                0.801 度      0.156 度
    IPPE + refineLM     0.179 度      -
    SQPNP               0.178 度      0.122 度
    ==================  ============  ============

IPPE 单独用会朝「正对」方向产生系统性偏差；SQPNP 是全局解、无需初值、没有这个
偏差，因此默认选它，再用 solvePnPRefineLM 收尾。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .board import BoardLayout
from .se3 import invert, make_T, pose_from_T

__all__ = [
    "PoseResult",
    "build_correspondences",
    "reprojection_rmse",
    "solve_board_pose",
]


def build_correspondences(corners_by_id: dict, layout: BoardLayout):
    """把检测结果和板定义配对成 PnP 需要的 3D-2D 点对。

    Parameters
    ----------
    corners_by_id:
        {tag_id: (4, 2)} 检测到的角点。
    layout:
        板定义。

    Returns
    -------
    object_points:
        (4*N, 3) 板坐标系下的角点。
    image_points:
        (4*N, 2) 对应的图像角点。
    tag_ids:
        实际用上的 tag id 元组（升序）。
    """
    ids = tuple(sorted(set(corners_by_id) & set(layout.tag_centers)))
    if not ids:
        return np.zeros((0, 3)), np.zeros((0, 2)), ()

    obj = np.concatenate([layout.tag_corners(t) for t in ids], axis=0)
    img = np.concatenate(
        [np.asarray(corners_by_id[t], dtype=np.float64).reshape(4, 2) for t in ids],
        axis=0,
    )
    return obj, img, ids


def reprojection_rmse(
    object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs
) -> float:
    """重投影均方根误差（像素）。这是判定位姿是否可信的首要指标。

    正常应 < 0.5px。涨到几个 px 通常意味着 tag 尺寸、内参或角点顺序有误。
    """
    proj, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    diff = proj.reshape(-1, 2) - np.asarray(image_points).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


@dataclass
class PoseResult:
    """一次位姿求解的结果与诊断信息。

    bool(result) 等价于 result.ok。
    """

    ok: bool
    T_cam_board: np.ndarray | None = None
    T_board_cam: np.ndarray | None = None
    reproj_rmse_px: float = float("nan")
    n_tags: int = 0
    tag_ids: tuple = ()
    reason: str = ""

    def __bool__(self) -> bool:
        return bool(self.ok)

    @property
    def camera_position(self) -> np.ndarray | None:
        """相机原点在板坐标系下的位置 (3,)。"""
        return None if self.T_board_cam is None else self.T_board_cam[:3, 3].copy()

    @property
    def distance(self) -> float:
        """相机到板面中心的距离，单位米。"""
        p = self.camera_position
        return float("nan") if p is None else float(np.linalg.norm(p))

    def pose(self, degrees: bool = True) -> dict:
        """相机在板坐标系下的 {x, y, z, roll, pitch, yaw}。

        用的是 OpenCV 光学系: X 右 / Y 下 / Z 前。

        .. note::
           **相机正对板时 roll 会读到约 +-180 度，这是正常的**，不是 bug。
           板坐标系的 Z 指向观察者，而相机看向板时其光学 Z 指向板，
           两者朝向相反，于是姿态里必然含一个约 180 度的转动。

           这里刻意选光学系而不是机体系(X 前 / Y 左 / Z 上)，因为后者会让
           **pitch 正好落在 +-90 度的万向锁**上（实测正对板时 89.99 度），
           导致 roll 与 yaw 不可分离、读数乱跳。光学系下正对板时 pitch 约 0，
           三个角是良态可分的：roll 约 +-180、pitch 是离垂直的倾角、yaw 是面内转角。

           如果确实需要机体系（例如对接 ROS），用 se3.to_camera_body 转换，
           但请以 4x4 矩阵的形式传递，不要转成欧拉角。
        """
        if self.T_board_cam is None:
            raise ValueError("位姿无效，无法转换: " + (self.reason or "unknown"))
        return pose_from_T(self.T_board_cam, degrees=degrees)

    def project_board_outline(
        self, camera_matrix, dist_coeffs, layout: BoardLayout
    ) -> np.ndarray:
        """把板的外框投影回图像，用于叠加显示。"""
        if self.T_cam_board is None:
            return np.zeros((0, 2))
        w, h = layout.board_size
        corners = np.array(
            [
                [-w / 2, -h / 2, 0.0],
                [w / 2, -h / 2, 0.0],
                [w / 2, h / 2, 0.0],
                [-w / 2, h / 2, 0.0],
            ]
        )
        rvec, _ = cv2.Rodrigues(self.T_cam_board[:3, :3])
        pts, _ = cv2.projectPoints(
            corners, rvec, self.T_cam_board[:3, 3], camera_matrix, dist_coeffs
        )
        return pts.reshape(-1, 2)


def solve_board_pose(
    corners_by_id: dict,
    layout: BoardLayout,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
    *,
    min_tags: int = 2,
    max_reproj_px: float = 1.0,
    refine: bool = True,
) -> PoseResult:
    """联合所有可见 tag 的角点求位姿。

    Parameters
    ----------
    corners_by_id:
        {tag_id: (4, 2)}，来自 AprilTagDetector.detect。
    layout:
        板定义。
    camera_matrix, dist_coeffs:
        相机内参与畸变。畸变直接交给 solvePnP 处理。
    min_tags:
        最少需要几个 tag，低于此值视为无效。实测 4/3/2/1 个 tag 的角度误差
        约为 0.17/0.24/0.56/1.82 度，所以别把门槛降到 1。
    max_reproj_px:
        重投影 RMSE 上限，超过则 ok=False（但仍带回算出的位姿便于排查）。
    refine:
        是否用 solvePnPRefineLM 做 LM 收尾。

    Returns
    -------
    PoseResult
    """
    object_points, image_points, ids = build_correspondences(corners_by_id, layout)
    n = len(ids)

    if n < max(1, int(min_tags)):
        return PoseResult(
            ok=False, n_tags=n, tag_ids=ids, reason=f"可见 tag 不足: {n} < {min_tags}"
        )

    if dist_coeffs is None:
        dist_coeffs = np.zeros(5, dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not ok:
        return PoseResult(
            ok=False, n_tags=n, tag_ids=ids, reason="solvePnP (SQPNP) 失败"
        )

    if refine:
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                rvec,
                tvec,
            )
        except cv2.error:
            pass  # 保留 SQPNP 的结果

    rmse = reprojection_rmse(
        object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs
    )

    R, _ = cv2.Rodrigues(rvec)
    T_cam_board = make_T(R, tvec.reshape(3))
    T_board_cam = invert(T_cam_board)

    ok_final = True
    reason = ""
    if rmse > max_reproj_px:
        ok_final = False
        reason = (
            f"重投影误差过大: {rmse:.2f}px > {max_reproj_px}px "
            "(检查 tag_size / 内参 / CORNER_SIGNS)"
        )

    return PoseResult(
        ok=ok_final,
        T_cam_board=T_cam_board,
        T_board_cam=T_board_cam,
        reproj_rmse_px=rmse,
        n_tags=n,
        tag_ids=ids,
        reason=reason,
    )
