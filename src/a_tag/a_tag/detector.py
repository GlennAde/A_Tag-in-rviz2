"""AprilTag / ArUco 检测封装。

两个后端
--------
* backend="pupil"（默认）: 用 pupil_apriltags（libapriltag）。tag36h11 用这个最稳。
* backend="aruco": 用 OpenCV 的 ArUco 模块。**tag16h5 建议用这个** ——
  libapriltag 的 16h5 在同一个进程里跑会把堆弄坏（实测退出时段错误），而且误检多
  （hamming 1~2、margin 个位数的假 id）。OpenCV 这边干净，还能顺带支持真正的
  ArUco 字典（4x4/5x5/6x6），换板不用改代码。

两个后端返回的 corners 顺序都是本库的约定（见 board.CORNER_SIGNS）。
"""

from __future__ import annotations

import numpy as np

try:
    from pupil_apriltags import Detector
except ImportError as exc:  # pragma: no cover - 只在缺依赖时触发
    raise ImportError(
        "缺少 pupil-apriltags。它不在 rosdep 里，而且 ROS 节点是被**系统 python3.10**"
        "拉起来的（colcon 生成的入口点写死 /usr/bin/python3），所以要装进系统 python：\n"
        "  python3 -m pip install --user --no-deps pupil_apriltags\n"
        "  --no-deps 别去掉：让 pip 顺手把 numpy 升到 2.x 会弄坏 Ubuntu 自带的 python3-opencv。\n"
        "  如果 ~/.local 不可写（只读挂载 / 没有管理员），就装到工作区里再指路：\n"
        "    python3 -m pip install --target <工作区>/.deps --no-deps pupil_apriltags\n"
        "    PYTHONPATH=<工作区>/.deps:$PYTHONPATH ros2 launch a_tag a_tag.launch.py"
    ) from exc

import cv2

__all__ = ["ARUCO_DICTS", "AprilTagDetector"]

#: 家族名 -> OpenCV 预定义字典（backend="aruco" 用）。
#: 前四个是 AprilTag 家族（名字和 libapriltag 一致），后面是真正的 ArUco 字典。
ARUCO_DICTS = {
    "tag16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
    "tag36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}

#: OpenCV 返回的角点顺序(左上,右上,右下,左下) -> 本库的(右上,左上,左下,右下)
_ARUCO_TO_OURS = [1, 0, 3, 2]

BACKENDS = ("pupil", "aruco")


class AprilTagDetector:
    """检测 tag，返回 {tag_id: (4,2) 角点}。

    Parameters
    ----------
    families:
        家族/字典名。backend="pupil" 时是 libapriltag 家族（一次只能给一个）；
        backend="aruco" 时是 ARUCO_DICTS 的键。
    backend:
        "pupil"（默认）或 "aruco"（见模块文档：tag16h5 用 aruco）。
    nthreads:
        检测线程数（只对 pupil 后端有效）。
    quad_decimate:
        **精度关键参数**（只对 pupil 后端有效）。libapriltag 默认 2.0，会把图像降采样
        一半再找四边形，角点精度大约减半。追求精度请保持 1.0（代价是变慢）。
    expected_ids:
        只保留这些 id，用来滤掉干扰 tag 和误检。
    """

    def __init__(
        self,
        families: str = "tag36h11",
        *,
        backend: str = "pupil",
        nthreads: int = 1,
        quad_decimate: float = 1.0,
        refine_edges: int = 1,
        quad_sigma: float = 0.0,
        decode_sharpening: float = 0.25,
        expected_ids=None,
    ) -> None:
        if backend not in BACKENDS:
            raise ValueError(f"backend 必须是 {BACKENDS} 之一，收到 {backend!r}")
        self.families = families
        self.backend = backend
        self.expected_ids = (
            None if expected_ids is None else {int(i) for i in expected_ids}
        )
        self._detector = None

        if backend == "pupil":
            self._detector = Detector(
                families=families,
                nthreads=nthreads,
                quad_decimate=quad_decimate,
                refine_edges=refine_edges,
                quad_sigma=quad_sigma,
                decode_sharpening=decode_sharpening,
            )
        else:
            try:
                dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[families])
            except KeyError:
                raise ValueError(
                    f"OpenCV 里没有这个字典: {families!r}。可用: {sorted(ARUCO_DICTS)}"
                ) from None
            params = cv2.aruco.DetectorParameters()
            # 亚像素角点细化：和 libapriltag 的 refine_edges 对应，直接影响精度
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self._detector = cv2.aruco.ArucoDetector(dictionary, params)

    # ------------------------------------------------------------------ 检测
    def detect(self, gray: np.ndarray) -> dict:
        """在灰度图上检测。

        Parameters
        ----------
        gray:
            uint8 单通道灰度图。

        Returns
        -------
        dict
            {tag_id: (4, 2) float64 角点}，只包含 expected_ids 里的 id。
        """
        if gray.ndim != 2:
            raise ValueError(f"需要单通道灰度图，收到 shape={gray.shape}")
        if gray.dtype != np.uint8:
            raise ValueError(f"需要 uint8 灰度图，收到 dtype={gray.dtype}")

        if self.backend == "pupil":
            raw = [
                (int(d.tag_id), np.asarray(d.corners, dtype=np.float64).reshape(4, 2))
                for d in self._detector.detect(gray)
            ]
        else:
            corners, ids, _ = self._detector.detectMarkers(gray)
            raw = []
            if ids is not None:
                for tid, quad in zip(ids.ravel().tolist(), corners):
                    pts = np.asarray(quad, dtype=np.float64).reshape(4, 2)
                    raw.append((int(tid), pts[_ARUCO_TO_OURS]))

        out: dict = {}
        for tid, pts in raw:
            if self.expected_ids is not None and tid not in self.expected_ids:
                continue
            out[tid] = pts
        return out
