"""AprilTag 检测封装（围绕 pupil_apriltags）。"""

from __future__ import annotations

import numpy as np
from pupil_apriltags import Detector

__all__ = ["AprilTagDetector"]


class AprilTagDetector:
    """检测 AprilTag，返回 {tag_id: corners}。

    Parameters
    ----------
    families:
        传给 libapriltag 的家族字符串，如 "tag36h11"。多个家族用空格分隔。
    nthreads:
        检测线程数。多核机器上设成 2~4 通常有接近线性的加速。
    quad_decimate:
        **精度关键参数**。libapriltag 默认 2.0，会先把图像降采样一半再找四边形，
        角点精度大约减半。追求精度请保持 1.0（代价是变慢）。
    refine_edges:
        对四边形边缘做亚像素细化，保持 1。
    expected_ids:
        只保留这些 id 的检测结果，用来滤掉环境里的干扰 tag。

    Notes
    -----
    返回的 corners 是 (4, 2) float64，顺序见 a_tag.board.CORNER_SIGNS。
    本类**不做**去畸变，畸变系数交给 cv2.solvePnP 处理。
    """

    def __init__(
        self,
        families: str = "tag36h11",
        *,
        nthreads: int = 1,
        quad_decimate: float = 1.0,
        refine_edges: int = 1,
        quad_sigma: float = 0.0,
        decode_sharpening: float = 0.25,
        expected_ids=None,
    ) -> None:
        self.families = families
        self.expected_ids = (None if expected_ids is None
                             else {int(i) for i in expected_ids})
        self._detector = Detector(
            families=families,
            nthreads=nthreads,
            quad_decimate=quad_decimate,
            refine_edges=refine_edges,
            quad_sigma=quad_sigma,
            decode_sharpening=decode_sharpening,
        )

    def detect(self, gray: np.ndarray) -> dict:
        """在灰度图上检测。

        Parameters
        ----------
        gray:
            uint8 单通道灰度图。

        Returns
        -------
        dict
            {tag_id: (4, 2) float64 角点}，只包含期望的 id。
        """
        if gray.ndim != 2:
            raise ValueError(f"需要单通道灰度图，收到 shape={gray.shape}")
        if gray.dtype != np.uint8:
            raise ValueError(f"需要 uint8 灰度图，收到 dtype={gray.dtype}")

        out: dict = {}
        for det in self._detector.detect(gray):
            tid = int(det.tag_id)
            if self.expected_ids is not None and tid not in self.expected_ids:
                continue
            out[tid] = np.asarray(det.corners, dtype=np.float64).reshape(4, 2)
        return out
