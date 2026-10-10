"""a_tag: 用 AprilTag 标定板估计相机相对板的 6DoF 位姿。

板坐标系: 原点在板面中心, X 向右, Y 向上, Z 垂直板面向外(指向相机), 板面为 Z=0。

主要输出是 T_board_cam —— 相机在板坐标系下的位姿。

快速开始
--------
>>> from a_tag import AppConfig, AprilTagDetector, solve_board_pose
>>> cfg = AppConfig.load()                                    # doctest: +SKIP
>>> det = AprilTagDetector(expected_ids=cfg.board.tag_ids)    # doctest: +SKIP
>>> corners = det.detect(gray)                                # doctest: +SKIP
>>> K, D = cfg.require_intrinsics()                            # doctest: +SKIP
>>> r = solve_board_pose(corners, cfg.board, K, D)             # doctest: +SKIP
>>> r.pose()                                                  # doctest: +SKIP
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .board import CORNER_SIGNS, BoardLayout
from .camera import FrameSource
from .config import AppConfig, load_yaml
from .detector import AprilTagDetector
from .filter import OneEuroFilter, PoseFilter
from .main import main
from .pose import PoseResult, reprojection_rmse, solve_board_pose
from .se3 import invert, make_T, pose_from_T, to_camera_body

try:
    __version__ = _version("a-tag")  # 单一来源: package.xml / setup.py
except PackageNotFoundError:  # 未安装时(直接跑源码)
    __version__ = "0.0.0+unknown"

__all__ = [
    "CORNER_SIGNS",
    "AppConfig",
    "AprilTagDetector",
    "BoardLayout",
    "FrameSource",
    "OneEuroFilter",
    "PoseFilter",
    "PoseResult",
    "__version__",
    "invert",
    "load_yaml",
    "main",
    "make_T",
    "pose_from_T",
    "reprojection_rmse",
    "solve_board_pose",
    "to_camera_body",
]
