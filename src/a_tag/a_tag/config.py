"""配置加载。

a_tag 只拥有**目标**配置；相机侧的一切（内参、曝光、增益）归相机驱动包
（本工作空间里是 src/rm_vision_ros2_hik_camera）所有：

* config/target.yaml        目标：board（tag 尺寸/位置）+ tracking（检测与求解参数）
* config/a_tag_params.yaml  ROS 2 节点参数，由 launch 加载，本模块不读

内参从哪来？
------------
1. ROS 流程：订阅 <camera_info_topic>（相机驱动每帧跟着图像发）；
2. 离线：显式传 camera_info_path（camera_info.yaml 文件，或含它的目录）；
3. 单文件合并写法（path 指向一个 .yaml）：文件里有 camera: / board: / tracking: 三段。

找不到内参就**没有**内参：命令行报错退出，节点等话题并告警，绝不使用占位值 ——
平面 PnP 里焦距与距离是尺度等价的，焦距写错位姿会整体缩放，而重投影误差可能照样很小。
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .board import BoardLayout

__all__ = [
    "CAMERA_INFO_FILENAME",
    "TARGET_FILENAME",
    "AppConfig",
    "MissingIntrinsicsError",
    "load_yaml",
]

_ENV_VAR = "A_TAG_CONFIG"
_DEFAULT_DIRNAME = "config"
_PACKAGE_NAME = "a_tag"

TARGET_FILENAME = "target.yaml"
CAMERA_INFO_FILENAME = "camera_info.yaml"

#: 允许出现在内参文件里的字段（就是 ROS CameraInfo / camera_calibration 的输出）。
_CAMERA_KEYS = frozenset(
    {
        "image_width",
        "image_height",
        "camera_name",
        "camera_matrix",
        "distortion_model",
        "distortion_coefficients",
        "rectification_matrix",
        "projection_matrix",
    }
)
_BOARD_KEYS = frozenset({"tag_size", "tag_centers", "tag_rotations"})
_TRACKING_KEYS = frozenset(
    {
        "families",
        "detector",          # "pupil"(默认) 或 "aruco"（tag16h5 / 真 ArUco 字典用它）
        "nthreads",
        "quad_decimate",
        "min_tags",
        "max_reproj_px",
    }
)
_TARGET_TOP_KEYS = frozenset({"board", "tracking"})

INTRINSICS_HINT = (
    "没有相机内参。内参归相机驱动包所有，a_tag 不自带占位值：\n"
    "  ROS 流程: 由 /camera_info 话题提供（hik_camera 的 camera_info_url 指向它自己的"
    " config/camera_info.yaml）\n"
    "  离线跑:   a-tag --camera-info src/rm_vision_ros2_hik_camera/config"
    " --source board.png"
)


class MissingIntrinsicsError(ValueError):
    """没有任何可用的相机内参来源。"""


def load_yaml(path) -> dict:
    """读取单个 YAML 文件，返回 dict。"""
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"配置文件不存在: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{p}: 顶层必须是映射(字典)，收到 {type(data).__name__}")
    return data


def _warn_unknown(where: str, data: dict, allowed: frozenset) -> None:
    """未识别字段告警。写错字段名是最难查的一类问题，不能静默忽略。"""
    extra = sorted(k for k in data if k not in allowed)
    if extra:
        warnings.warn(
            f"{where}: 未识别的字段 {extra}（已忽略）。可用字段: {sorted(allowed)}",
            UserWarning,
            stacklevel=3,
        )


def _split_target(raw: dict, where: str, extra_top_keys: frozenset = frozenset()) -> tuple:
    """把 target 文件切成 (board, tracking)。extra_top_keys 供合并写法放行 camera 段。"""
    if "board" in raw:
        _warn_unknown(where, raw, _TARGET_TOP_KEYS | extra_top_keys)
        board = raw.get("board") or {}
    else:
        board = {
            k: v for k, v in raw.items() if k not in ("tracking",) + tuple(extra_top_keys)
        }
    tracking = raw.get("tracking") or {}
    if not isinstance(board, dict) or not isinstance(tracking, dict):
        raise ValueError(f"{where}: board/tracking 段必须是映射(字典)")
    _warn_unknown(where, board, _BOARD_KEYS)
    _warn_unknown(where, tracking, _TRACKING_KEYS)
    return board, tracking


def _read_target_file(target_file: Path) -> tuple:
    return _split_target(load_yaml(target_file), str(target_file))


def _read_camera_info(camera_info_path) -> tuple:
    """读内参：camera_info.yaml 文件，或含它的目录。"""
    p = Path(camera_info_path).expanduser()
    if p.is_dir():
        p = p / CAMERA_INFO_FILENAME
    info = load_yaml(p)
    _warn_unknown(str(p), info, _CAMERA_KEYS)
    return info, str(p)


def _ros_share_config_dir() -> Path | None:
    """colcon install 之后的 share/a_tag/config；非 ROS 环境或还没 install 则 None。"""
    try:
        from ament_index_python.packages import get_package_share_directory
    except ImportError:
        return None
    try:
        return Path(get_package_share_directory(_PACKAGE_NAME)) / _DEFAULT_DIRNAME
    except Exception:  # PackageNotFoundError: 还没 colcon install
        return None


def _find_target(path=None, target_path=None) -> tuple:
    """定位 target：显式 -> $A_TAG_CONFIG -> ./config -> share/a_tag/config -> ./config.yaml。

    Returns
    -------
    (target_file, merged_raw):
        merged_raw 非 None 表示 path 指向的是单文件合并写法，board/tracking 要从它里面取。
    """
    if target_path is not None:
        tf = Path(target_path).expanduser()
        return (tf / TARGET_FILENAME if tf.is_dir() else tf), None

    explicit = path if path is not None else (os.environ.get(_ENV_VAR) or None)
    if explicit is not None:
        p = Path(explicit).expanduser()
        if p.is_dir():
            return p / TARGET_FILENAME, None
        if p.is_file():
            return p, load_yaml(p)
        raise FileNotFoundError(
            f"配置路径不存在: {p}\n"
            f"  --config 可以指向一个 config 目录（内含 {TARGET_FILENAME}），"
            "或指向单个合并的 YAML 文件。"
        )

    tried = []
    local = Path.cwd() / _DEFAULT_DIRNAME / TARGET_FILENAME
    if local.is_file():
        return local, None
    tried.append(f"{local.parent}/ （需要 {TARGET_FILENAME}）")

    share = _ros_share_config_dir()
    if share is not None:
        sf = share / TARGET_FILENAME
        if sf.is_file():
            return sf, None
        tried.append(f"{share}/ （share/a_tag/config，缺 {TARGET_FILENAME}）")

    legacy = Path.cwd() / "config.yaml"
    if legacy.is_file():
        return legacy, load_yaml(legacy)
    tried.append(str(legacy))

    raise FileNotFoundError(
        "找不到配置。已尝试:\n  - " + "\n  - ".join(tried) + "\n"
        "用 --config 指定 config 目录/文件，设置环境变量 A_TAG_CONFIG，"
        "或先 colcon build 再 source install/setup.bash。"
    )


def _to_matrix(node, rows: int, cols: int, where: str) -> np.ndarray:
    """把 {rows, cols, data} 形式的 YAML 节点转成 ndarray。"""
    try:
        return np.asarray(node["data"], dtype=np.float64).reshape(rows, cols)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{where}: 需要 {{rows, cols, data}} 形式的 {rows}x{cols} 矩阵，收到 {node!r}"
        ) from exc


@dataclass
class AppConfig:
    """一次运行的配置：目标（board/tracking）+ 可选内参。

    内参可能为空（ROS 流程由话题提供）：用 has_intrinsics 判断，要用了再 require_intrinsics()。
    """

    board: BoardLayout
    camera_matrix: np.ndarray | None = None
    dist_coeffs: np.ndarray | None = None
    image_width: int = 1280
    image_height: int = 720
    camera_name: str = ""
    min_tags: int = 2
    max_reproj_px: float = 1.0
    quad_decimate: float = 1.0
    nthreads: int = 2
    families: str = "tag36h11"
    #: 检测后端："pupil"(libapriltag) 或 "aruco"(OpenCV)
    detector: str = "pupil"
    target_path: str = ""
    camera_info_path: str = ""

    @property
    def has_intrinsics(self) -> bool:
        return self.camera_matrix is not None

    def require_intrinsics(self) -> tuple:
        """返回 (K, D)；没有内参时抛 MissingIntrinsicsError（带可照抄的命令）。"""
        if self.camera_matrix is None:
            raise MissingIntrinsicsError(INTRINSICS_HINT)
        D = self.dist_coeffs
        if D is None:
            D = np.zeros((1, 5), dtype=np.float64)
        return self.camera_matrix, D

    @classmethod
    def load(cls, path=None, *, camera_info_path=None, target_path=None) -> AppConfig:
        """解析配置。

        Parameters
        ----------
        path:
            a_tag 的 config 目录（含 target.yaml），或单个合并的 YAML 文件。None = 自动查找。
        camera_info_path:
            内参来源（camera_info.yaml 文件或含它的目录）。ROS 流程不用给，走话题。
        target_path:
            单独覆盖 target.yaml。
        """
        info, info_path = ({}, "")
        if camera_info_path is not None:
            info, info_path = _read_camera_info(camera_info_path)

        target_file, merged = _find_target(path, target_path)
        if merged is not None:
            board_raw, tracking = _split_target(
                merged, str(target_file), frozenset({"camera"})
            )
            cam_raw = merged.get("camera") or {}
            if not isinstance(cam_raw, dict):
                raise ValueError(f"{target_file}: camera 段必须是映射(字典)")
            _warn_unknown(str(target_file), cam_raw, _CAMERA_KEYS)
            if not info:  # 显式给的内参优先于文件里的
                info, info_path = cam_raw, str(target_file)
        else:
            board_raw, tracking = _read_target_file(target_file)

        if not board_raw:
            raise ValueError(
                f"{target_file}: 缺少 board 段，需要 tag_size 和 tag_centers"
            )
        try:
            layout = BoardLayout.from_config(board_raw)
        except KeyError as exc:
            raise ValueError(f"{target_file}: board 段缺少字段 {exc}") from None

        where = info_path or "(camera_info)"
        K = D = None
        if info.get("camera_matrix"):
            K = _to_matrix(info["camera_matrix"], 3, 3, where)
            D = (
                _to_matrix(info["distortion_coefficients"], 1, 5, where)
                if info.get("distortion_coefficients")
                else np.zeros((1, 5), dtype=np.float64)
            )

        return cls(
            board=layout,
            camera_matrix=K,
            dist_coeffs=D,
            image_width=int(info.get("image_width", 1280)),
            image_height=int(info.get("image_height", 720)),
            camera_name=str(info.get("camera_name", "")),
            min_tags=int(tracking.get("min_tags", 2)),
            max_reproj_px=float(tracking.get("max_reproj_px", 1.0)),
            quad_decimate=float(tracking.get("quad_decimate", 1.0)),
            nthreads=int(tracking.get("nthreads", 2)),
            families=str(tracking.get("families", "tag36h11")),
            detector=str(tracking.get("detector", "pupil")),
            target_path=str(target_file),
            camera_info_path=info_path,
        )
