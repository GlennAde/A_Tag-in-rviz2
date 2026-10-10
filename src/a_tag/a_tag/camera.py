"""帧源：图片/视频文件，或 ROS 图像话题。只给命令行用（节点有自己的订阅）。

两种源提供同一个 read() -> (ok, bgr) 接口，main.py 的主循环不用分叉。
"""

from __future__ import annotations

import array
from pathlib import Path

import cv2
import numpy as np

__all__ = ["FrameSource", "RosFrameSource", "bgr_from_ros_image", "bgr_to_ros_image"]

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

#: sensor_msgs/Image 的编码 -> 通道数
_ENCODING_CHANNELS = {
    "mono8": 1,
    "8uc1": 1,
    "rgb8": 3,
    "bgr8": 3,
    "rgba8": 4,
    "bgra8": 4,
}
_ENCODING_TO_BGR = {
    "rgb8": cv2.COLOR_RGB2BGR,
    "rgba8": cv2.COLOR_RGBA2BGR,
    "bgra8": cv2.COLOR_BGRA2BGR,
}


def bgr_from_ros_image(msg) -> np.ndarray:
    """sensor_msgs/Image -> uint8 的 HxWx3 BGR 图（单通道转成三通道）。

    支持 mono8 / rgb8 / bgr8 / rgba8 / bgra8（hik_camera 发的是 rgb8）。
    """
    encoding = (getattr(msg, "encoding", "") or "").lower()
    channels = _ENCODING_CHANNELS.get(encoding)
    if channels is None:
        raise ValueError(
            f"不支持的图像编码: {getattr(msg, 'encoding', None)}"
            "（支持 mono8/rgb8/bgr8/rgba8/bgra8）"
        )

    height, width, step = int(msg.height), int(msg.width), int(msg.step)
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if buf.size < height * step:
        raise ValueError(f"图像数据长度不足: {buf.size} < {height * step}（step={step}）")
    # step 可能含行尾填充，先按 step 铺开再裁掉
    img = buf[: height * step].reshape(height, step)[:, : width * channels]
    img = img.reshape(height, width, channels)

    if channels == 1:
        return cv2.cvtColor(np.ascontiguousarray(img[:, :, 0]), cv2.COLOR_GRAY2BGR)
    if encoding == "bgr8":          # 已经是 BGR，别再做一次转换
        return np.ascontiguousarray(img)
    return cv2.cvtColor(img, _ENCODING_TO_BGR[encoding])


def bgr_to_ros_image(bgr: np.ndarray, stamp=None, frame_id: str = ""):
    """uint8 HxWx3 BGR 图 -> sensor_msgs/Image（encoding=bgr8）。"""
    from sensor_msgs.msg import Image

    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"需要 HxWx3 的 BGR 图，收到 shape={bgr.shape}")
    img = np.ascontiguousarray(bgr, dtype=np.uint8)
    msg = Image()
    if stamp is not None:
        msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = img.shape[:2]
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = img.shape[1] * 3
    # ⚠️ 必须给 array.array('B')，**不能**直接给 bytes！
    # rclpy 生成的 setter 对 bytes 会走 Python 级逐元素校验：1440x1080x3 一帧 4.6MB
    # 要 229ms（这曾经就是 debug 图把整条链路拖到几 fps 的元凶）；
    # array.array('B') 走 setter 的快路径直接接管 -> 0.45ms。
    msg.data = array.array("B", img.tobytes())
    return msg


class FrameSource:
    """图片或视频文件 -> 逐帧 BGR。图片只返回一帧。

    Examples
    --------
    >>> with FrameSource("board.png") as src:       # doctest: +SKIP
    ...     ok, bgr = src.read()
    """

    def __init__(self, source) -> None:
        self.path = Path(str(source)).expanduser()
        self.is_image = self.path.suffix.lower() in _IMAGE_SUFFIXES
        self.cap: cv2.VideoCapture | None = None
        self._consumed = False

    def open(self) -> FrameSource:
        if self.is_image:
            if not self.path.is_file():
                raise FileNotFoundError(f"读不到图片: {self.path}")
            return self
        if self.cap is None:
            self.cap = cv2.VideoCapture(str(self.path))
            if not self.cap.isOpened():
                raise RuntimeError(f"打不开视频: {self.path}")
        return self

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> FrameSource:
        return self.open()

    def __exit__(self, *exc) -> None:
        self.release()

    def read(self) -> tuple:
        """抓一帧。图片源只成功返回一次，之后是 (False, None)。"""
        if self.cap is None and not self.is_image:
            self.open()
        if self.is_image:
            if self._consumed:
                return False, None
            img = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"读不到图片: {self.path}")
            self._consumed = True
            return True, img
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


class RosFrameSource:
    """订阅一个 sensor_msgs/Image 话题，每次 read() 等一帧。

    用来在没起节点的情况下，直接拿相机驱动的画面做诊断/单帧求解。
    rclpy 只在 open() 里 import，非 ROS 环境 import 本模块不受影响。
    """

    def __init__(self, topic: str = "/image_raw", timeout: float = 10.0) -> None:
        self.topic = topic
        self.timeout = float(timeout)
        self.node = None
        self._msg = None
        self._rclpy = None
        self._owns_context = False

    def open(self) -> RosFrameSource:
        if self.node is not None:
            return self
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image

        self._rclpy = rclpy
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init()
        self.node = Node("a_tag_framesource")
        # 工业相机高帧率用 sensor-data QoS，与相机驱动发布端一致
        self.node.create_subscription(
            Image, self.topic, self._on_image, qos_profile_sensor_data
        )
        return self

    def _on_image(self, msg) -> None:
        self._msg = msg          # 只留最新一帧，不排队

    def release(self) -> None:
        if self.node is not None:
            self.node.destroy_node()
            self.node = None
        if self._rclpy is not None and self._owns_context and self._rclpy.ok():
            self._rclpy.shutdown()
        self._rclpy = None

    def __enter__(self) -> RosFrameSource:
        return self.open()

    def __exit__(self, *exc) -> None:
        self.release()

    def read(self) -> tuple:
        """等一帧（最多 self.timeout 秒）。超时返回 (False, None)。"""
        if self.node is None:
            self.open()
        import time

        self._msg = None
        deadline = time.perf_counter() + self.timeout
        while self._msg is None and time.perf_counter() < deadline:
            self._rclpy.spin_once(self.node, timeout_sec=0.1)
        if self._msg is None:
            return False, None
        msg, self._msg = self._msg, None
        return True, bgr_from_ros_image(msg)

    to_gray = staticmethod(FrameSource.to_gray)
