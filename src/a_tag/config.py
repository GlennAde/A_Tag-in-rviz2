"""配置加载。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np
import yaml

from .board import BoardLayout

__all__ = ["AppConfig", "load_yaml", "default_config_path"]

_ENV_VAR = "A_TAG_CONFIG"


def default_config_path() -> Path | None:
    """随包分发的默认配置路径，找不到返回 None。"""
    try:
        return Path(str(resources.files("a_tag").joinpath("default_config.yaml")))
    except (ModuleNotFoundError, FileNotFoundError):
        return None


def load_yaml(path=None) -> dict:
    """读取 YAML 配置。

    查找顺序: 显式 path -> 环境变量 A_TAG_CONFIG -> ./config.yaml
    -> 随包默认配置。
    """
    candidates = []
    if path is not None:
        candidates.append(Path(path).expanduser())
    elif os.environ.get(_ENV_VAR):
        candidates.append(Path(os.environ[_ENV_VAR]).expanduser())
    else:
        candidates.append(Path.cwd() / "config.yaml")
        d = default_config_path()
        if d is not None:
            candidates.append(d)

    for p in candidates:
        if p.is_file():
            with open(p, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            data["_source"] = str(p)
            return data

    raise FileNotFoundError(
        "找不到配置文件，已尝试: " + ", ".join(str(c) for c in candidates)
    )


def _to_matrix(node, rows: int, cols: int, default: np.ndarray) -> np.ndarray:
    """把 {rows, cols, data} 形式的 YAML 节点转成 ndarray。"""
    if not node:
        return default
    return np.asarray(node["data"], dtype=np.float64).reshape(rows, cols)


@dataclass
class AppConfig:
    """一次运行的完整配置。"""

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    board: BoardLayout
    image_width: int = 1280
    image_height: int = 720
    min_tags: int = 2
    max_reproj_px: float = 1.0
    quad_decimate: float = 1.0
    nthreads: int = 2
    families: str = "tag36h11"
    source: object = 0
    fps: int = 30
    source_path: str = field(default="", compare=False)

    @property
    def camera_params(self) -> tuple:
        """(fx, fy, cx, cy)，传给 pupil_apriltags 用。"""
        K = self.camera_matrix
        return (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))

    @classmethod
    def load(cls, path=None) -> "AppConfig":
        """从 YAML 构造配置。"""
        raw = load_yaml(path)
        cam = raw.get("camera", raw)          # 兼容扁平写法与嵌套写法
        K = _to_matrix(cam.get("camera_matrix"), 3, 3,
                       np.array([[1000.0, 0.0, 640.0],
                                 [0.0, 1000.0, 360.0],
                                 [0.0, 0.0, 1.0]]))
        dist = _to_matrix(cam.get("distortion_coefficients"), 1, 5, np.zeros((1, 5)))

        board_raw = raw.get("board")
        if not board_raw:
            raise ValueError(
                "配置缺少 board 段 ("
                + str(raw.get("_source", path))
                + ")，需要 tag_size 和 tag_centers"
            )
        layout = BoardLayout.from_config(board_raw)

        tracking = raw.get("tracking", {}) or {}
        return cls(
            camera_matrix=K,
            dist_coeffs=dist,
            board=layout,
            image_width=int(cam.get("image_width", 1280)),
            image_height=int(cam.get("image_height", 720)),
            min_tags=int(tracking.get("min_tags", 2)),
            max_reproj_px=float(tracking.get("max_reproj_px", 1.0)),
            quad_decimate=float(tracking.get("quad_decimate", 1.0)),
            nthreads=int(tracking.get("nthreads", 2)),
            families=str(tracking.get("families", "tag36h11")),
            source=raw.get("source", 0),
            fps=int(raw.get("fps", 30)),
            source_path=str(raw.get("_source", "")),
        )
