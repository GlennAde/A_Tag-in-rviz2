"""相机采集封装。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import AppConfig

__all__ = ["Camera"]


class Camera:
    """封装 cv2.VideoCapture。

    Parameters
    ----------
    source:
        摄像头索引 (int)，或视频/图片路径 (str / Path)。
    width, height, fps:
        请求的分辨率与帧率。**必须和相机内参标定时一致**，否则内参失效。
    """

    def __init__(self, source=0, width: int = 1280, height: int = 720,
                 fps: int = 30) -> None:
        self.source = source
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.cap: cv2.VideoCapture | None = None
        self.is_image = isinstance(source, (str, Path)) and Path(str(source)).suffix.lower() in {
            ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        self._image_cache: np.ndarray | None = None

    # ------------------------------------------------------------- 生命周期
    def open(self) -> "Camera":
        """打开设备或文件。"""
        if self.cap is not None:
            return self
        self.cap = cv2.VideoCapture(int(self.source)
                                    if isinstance(self.source, (int, np.integer))
                                    else str(self.source))
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {self.source}")
        if not self.is_image:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        return self

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.release()

    # ----------------------------------------------------------------- 读取
    @property
    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    @property
    def actual_size(self) -> tuple:
        """(宽, 高)，来自实际拿到的帧。"""
        if self.cap is None:
            return (0, 0)
        return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def read(self) -> tuple:
        """抓一帧。

        Returns
        -------
        (ok, bgr)
            bgr 是 uint8 的 HxWx3 BGR 图。图片源只返回一次。
        """
        if self.cap is None:
            self.open()
        if self.is_image:
            if self._image_cache is None:
                img = cv2.imread(str(self.source), cv2.IMREAD_COLOR)
                if img is None:
                    return False, None
                self._image_cache = img
            return True, self._image_cache
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return False, None
        return True, frame

    @staticmethod
    def to_gray(bgr: np.ndarray) -> np.ndarray:
        """BGR -> 单通道灰度，供 AprilTag 检测使用。"""
        if bgr.ndim == 2:
            return bgr
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    @classmethod
    def from_config(cls, cfg: AppConfig) -> "Camera":
        """按配置构造。"""
        return cls(source=cfg.source, width=cfg.image_width,
                   height=cfg.image_height, fps=cfg.fps)
