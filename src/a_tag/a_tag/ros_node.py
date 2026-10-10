"""ROS 2 节点：订阅相机图像，发布标定板位姿。

设计要点
--------
* 只有本模块 import rclpy。核心算法（board/detector/pose/filter/se3）不依赖 ROS，
  所以 a_tag 作为库在任何 Python 环境都能用，节点只是薄薄一层适配。
* 内参来自 <camera_info_topic>（ROS 标准做法：相机驱动每帧跟着图像发，内容就是
  camera_calibration 写进驱动包的那份标定）。也可以显式用 camera_info_file 参数指定
  一个 camera_info.yaml；两者都没有时**不解算**并告警，绝不使用占位内参。
* 图像 -> numpy 手工转换（mono8/rgb8/bgr8/rgba8/bgra8），不引入 cv_bridge。
* 发布 PoseStamped + TF：**板在相机坐标系下的位姿**（T_cam_board = 本库主输出
  T_board_cam 的逆），frame_id = camera_frame_id，TF 父子也是 camera -> board。
  相机是参考系、板是被观测物；这样"固定相机看板动"，和 RViz / aruco_ros 的习惯一致。

话题接口
--------
====  ======================  ============================
方向  话题                    类型
====  ======================  ============================
订阅  image_topic             sensor_msgs/Image
订阅  camera_info_topic       sensor_msgs/CameraInfo
发布  pose_topic              geometry_msgs/PoseStamped
====  ======================  ============================
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from visualization_msgs.msg import Marker, MarkerArray

from .camera import FrameSource, bgr_from_ros_image, bgr_to_ros_image
from .config import AppConfig
from .detector import AprilTagDetector
from .filter import PoseFilter
from .pose import PoseResult, solve_board_pose
from .se3 import invert, pose_from_T, quat_from_R
from .viz import draw_overlay

__all__ = ["ATagNode", "main"]


def _pt_xyz(x, y, z: float = 0.0) -> Point:
    """Marker 的一个点（板坐标系，单位米）。"""
    p = Point()
    p.x, p.y, p.z = float(x), float(y), float(z)
    return p


def _pt_xy(x, y) -> Point:
    return _pt_xyz(x, y, 0.0)


class ATagNode(Node):
    """把 A_tag 的检测+求解链路接到 ROS 2 图上。"""

    def __init__(self) -> None:
        super().__init__("a_tag")

        self.declare_parameter("image_topic", "/image_raw")
        self.declare_parameter("camera_info_topic", "/camera_info")
        self.declare_parameter("pose_topic", "/a_tag/pose")
        self.declare_parameter("use_sensor_data_qos", True)
        self.declare_parameter("image_queue_depth", 1)   # 图像订阅队列：1 = 只留最新帧
        self.declare_parameter("board_frame_id", "tag_board")
        self.declare_parameter("config_dir", "")
        self.declare_parameter("camera_info_file", "")
        self.declare_parameter("min_tags", 0)
        self.declare_parameter("max_reproj_px", 0.0)
        self.declare_parameter("use_filter", True)
        self.declare_parameter("print_every", 30)
        self.declare_parameter("publish_debug_image", False)
        self.declare_parameter("debug_image_topic", "/a_tag/debug_image")
        self.declare_parameter("debug_period", 0.2)  # debug 图最短间隔（秒），<=0 = 每帧都发
        self.declare_parameter("debug_adaptive", True)  # 观看端跟不上时自动放慢
        self.declare_parameter("debug_scale", 1.0)  # debug 图缩放（0.5 = 一半边长，带宽 1/4）
        self.declare_parameter("debug_use_sensor_data_qos", False)
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("camera_frame_id", "camera_optical_frame")
        self.declare_parameter("publish_markers", False)
        self.declare_parameter("marker_topic", "/a_tag/board_markers")

        def param(name):
            return self.get_parameter(name).value

        cfg_dir = str(param("config_dir") or "") or None
        info_file = str(param("camera_info_file") or "") or None
        self.cfg = AppConfig.load(cfg_dir, camera_info_path=info_file)
        self.get_logger().info(f"目标配置: {self.cfg.target_path}")
        if self.cfg.has_intrinsics:
            self.get_logger().info(f"内参来自文件: {self.cfg.camera_info_path}")
        else:
            self.get_logger().warning(
                f"还没有内参，等待 {param('camera_info_topic')} 话题；"
                "收到之前不会输出位姿（也可以用 camera_info_file 参数直接给标定文件）"
            )

        self.detector = AprilTagDetector(
            families=self.cfg.families,
            backend=self.cfg.detector,
            nthreads=self.cfg.nthreads,
            quad_decimate=self.cfg.quad_decimate,
            expected_ids=self.cfg.board.tag_ids,
        )

        # 0 / 0.0 表示「用配置文件里的值」
        self.min_tags = int(param("min_tags")) or self.cfg.min_tags
        self.max_reproj_px = float(param("max_reproj_px")) or self.cfg.max_reproj_px
        self.print_every = int(param("print_every"))
        self.board_frame_id = str(param("board_frame_id"))
        self.pose_filter = PoseFilter() if bool(param("use_filter")) else None

        # 内参可能为空（等话题），收到 CameraInfo 后填入
        self.K = self.cfg.camera_matrix
        self.D = self.cfg.dist_coeffs
        self.have_camera_info = self.cfg.has_intrinsics
        self._warned_no_intrinsics = False

        # 队列深度默认 1：相机比处理快的时候，宁可丢掉旧帧也不要背着一队旧数据在算
        # （best-effort + depth=1 => 永远处理"刚到的"那一帧，延迟上界 = 一个相机周期；
        #  depth=5 在卡一下之后会先算 5 帧旧的，位姿就会"晚一拍"）
        qos = QoSProfile(
            depth=max(1, int(param("image_queue_depth"))),
            history=HistoryPolicy.KEEP_LAST,
            reliability=(
                ReliabilityPolicy.BEST_EFFORT
                if bool(param("use_sensor_data_qos"))
                else ReliabilityPolicy.RELIABLE
            ),
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            Image, str(param("image_topic")), self.on_image, qos
        )
        self.create_subscription(
            CameraInfo, str(param("camera_info_topic")), self.on_camera_info, qos
        )
        self.pub_pose = self.create_publisher(PoseStamped, str(param("pose_topic")), 10)

        # TF：位姿广播成 <camera_frame_id> -> <board_frame_id>（板挂在相机下面）。
        # 相机当父系有两个好处：
        #   1) 机器人自己的树（base_link -> ... -> camera_optical_frame）不用动，
        #      板是本包独有的叶子 frame，不会和别的 broadcasters 抢父子关系；
        #   2) RViz 里 Fixed Frame 用相机（或 base_link）就能看到板在动。
        self.publish_tf = bool(param("publish_tf"))
        self.camera_frame_id = str(param("camera_frame_id"))
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None
        if self.publish_tf:
            self.get_logger().info(
                f"TF: {self.camera_frame_id} -> {param('board_frame_id')}"
                f"（板在相机系；只看板动就把 RViz 的 Fixed Frame 设成 {self.camera_frame_id}）"
            )

        # 可视化：检测框 + 板框投影 + 日志文字画进图像再发出去。
        # publisher 一直建好，开关是**每帧读参数** —— 所以可以用
        #   ros2 param set /a_tag publish_debug_image true/false   运行时开关。
        # QoS 默认 reliable（rqt_image_view / showimage 默认就是这么订的）；
        # 想改成 best-effort 用参数 debug_use_sensor_data_qos。
        self.pub_debug = self.create_publisher(
            Image,
            str(param("debug_image_topic")),
            qos_profile_sensor_data if bool(param("debug_use_sensor_data_qos")) else 10,
        )
        qos_name = (
            "best-effort" if bool(param("debug_use_sensor_data_qos")) else "reliable"
        )
        if bool(param("publish_debug_image")):
            self.get_logger().info(
                f"可视化已开启: {param('debug_image_topic')}（bgr8, {qos_name}）"
                "；rqt_image_view 里 transport 要选 raw —— 本话题没有 compressed 子话题"
            )

        # 板的三维示意（板框 + 四个 tag 方框 + 板坐标系箭头），在**板坐标系**里是静止的，
        # 所以只发一次；用 TRANSIENT_LOCAL（latched）让后开 RViz 也能收到。
        # pose 靠 TF 表达，这里只是让 RViz 里能看见"板"这个物体。
        self.pub_markers = None
        self._marker_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        if bool(param("publish_markers")):
            self._ensure_marker_publisher()

        # debug 图**不在图像回调里画/发**：1440x1080 一帧 4.6MB，慢订阅者（rqt 渲染）
        # 会把 publish 堵住，连带把位姿链路拖到几 fps。这里只暂存，交给另一个回调组的
        # 定时器去画+发（main() 用的是 MultiThreadedExecutor，所以真的在另一个线程）。
        self._debug_group = MutuallyExclusiveCallbackGroup()
        self._debug_pending = None
        self._t_debug = 0.0
        self._debug_cost = 0.0      # 上一次"画+发"花了多久（自适应限速用）
        self._debug_warned = 0.0
        self.create_timer(0.05, self._on_debug_timer, callback_group=self._debug_group)

        self.n_frames = 0
        self.n_seen = 0        # 收到的图像帧数（含没解出位姿的）
        self.n_ok = 0
        self._t_prev = time.perf_counter()
        self._bad_encodings: set[str] = set()

    # ------------------------------------------------------------- 回调
    def on_camera_info(self, msg: CameraInfo) -> None:
        """用话题里的实时内参覆盖配置文件里的值。"""
        K = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        if not np.any(K):
            return
        self.K = K
        d = np.asarray(msg.d, dtype=np.float64)
        if d.size:
            self.D = d.reshape(1, -1)
        if not self.have_camera_info:
            self.have_camera_info = True
            self.get_logger().info(
                f"使用 {msg.header.frame_id or 'camera_info'} 的内参: "
                f"{msg.width}x{msg.height} fx={K[0, 0]:.2f} fy={K[1, 1]:.2f}"
            )

    def on_image(self, msg: Image) -> None:
        now = time.perf_counter()
        dt = now - self._t_prev
        self._t_prev = now
        # 采集时刻 -> 现在：包含传输、排队、处理。判断"有没有积压"看这个数
        delay_ms = (self.get_clock().now() - Time.from_msg(msg.header.stamp)).nanoseconds * 1e-6

        # 板的三维示意：运行中打开也能立刻生效（只发一次，静态）
        if bool(self.get_parameter("publish_markers").value) and self.pub_markers is None:
            self._ensure_marker_publisher()

        # debug 图：开关是每帧读的，但发布按 debug_period 限速（默认 5Hz），
        # 而且只在这里暂存，真正画+发在 _on_debug_timer 里（另一个线程）
        self.n_seen += 1
        period = float(self.get_parameter("debug_period").value)
        # 自适应：发一帧要是很慢（观看端/网络跟不上，reliable 发布会堵住），
        # 就自动把间隔拉开到它的 3 倍，别让 debug 图拖着整个节点；
        # 想固定速率就把 debug_adaptive 关掉。
        if period > 0 and bool(self.get_parameter("debug_adaptive").value):
            period = max(period, min(5.0, self._debug_cost * 3.0))
            if self._debug_cost > 0.05 and now - self._debug_warned > 5.0:
                self._debug_warned = now
                self.get_logger().warning(
                    f"debug 图发一帧要 {self._debug_cost * 1000:.0f} ms（观看端跟不上），"
                    f"已自动放慢到 {1.0 / period:.1f} Hz；"
                    "嫌慢就调大 debug_period 或关掉 publish_debug_image"
                )
        want_debug = bool(self.get_parameter("publish_debug_image").value) and (
            period <= 0 or self.n_seen == 1 or (now - self._t_debug) >= period
        )
        try:
            if not want_debug:
                gray, bgr = self._to_gray(msg), None
            else:  # 要发可视化才解成 BGR（否则省掉一次颜色转换）
                bgr = bgr_from_ros_image(msg)
                gray = FrameSource.to_gray(bgr)
        except ValueError as exc:
            if str(exc) not in self._bad_encodings:
                self._bad_encodings.add(str(exc))
                self.get_logger().warning(str(exc))
            return

        if self.K is None:  # 没有内参就不解算，避免用错焦距算出整体缩放的位姿
            if not self._warned_no_intrinsics:
                self._warned_no_intrinsics = True
                self.get_logger().warning(
                    "收到图像但没有内参：请确认相机驱动在发 "
                    f"{self.get_parameter('camera_info_topic').value}"
                    "，或用 camera_info_file 参数指定标定文件"
                )
            if want_debug and bgr is not None:
                self._stage_debug(
                    bgr,
                    {},
                    PoseResult(ok=False, reason="还没有内参（等 camera_info）"),
                    None,
                    dt,
                    msg.header.stamp,
                    now,
                    delay_ms,
                )
            return

        corners = self.detector.detect(gray)
        result = solve_board_pose(
            corners,
            self.cfg.board,
            self.K,
            self.D,
            min_tags=self.min_tags,
            max_reproj_px=self.max_reproj_px,
        )
        self.n_frames += 1
        self.n_ok += int(result.ok)

        # 无有效观测时 T_meas=None，滤波器会沿用上一帧并标记 stale
        # （滤波器内部继续用 T_board_cam：相机在板系，抖动最小的一支）
        T_board_cam = result.T_board_cam if result.ok else None
        if self.pose_filter is not None:
            T_board_cam = self.pose_filter(T_board_cam, dt)

        # ROS 出口统一用 **T_cam_board（板在相机系）**：相机是参考系、板是被观测物，
        # 这样 TF 的父子方向（camera -> board）和 /a_tag/pose 的 frame_id 是同一个约定，
        # 也才和 RViz 里"固定相机看板动"、aruCo/apriltag 那些包一致。
        T_cam_board = invert(T_board_cam) if T_board_cam is not None else None

        if T_cam_board is not None:
            self._publish(T_cam_board, msg.header.stamp)

        # 有 pose 没 pose 都发 debug —— 恰恰是"没位姿"的时候最需要看画面
        if want_debug and bgr is not None:
            self._stage_debug(
                bgr, corners, result, T_cam_board, dt, msg.header.stamp, now, delay_ms
            )

        if self.print_every and self.n_frames % self.print_every == 0:
            self._log_status(result, T_cam_board, dt, delay_ms)

    # ------------------------------------------------------------- 三维示意
    def _ensure_marker_publisher(self) -> None:
        """建 marker 话题并发一次（latched）。"""
        if self.pub_markers is None:
            self.pub_markers = self.create_publisher(
                MarkerArray,
                str(self.get_parameter("marker_topic").value),
                self._marker_qos,
            )
            self.get_logger().info(
                f"板示意已开启: {self.get_parameter('marker_topic').value}"
                "（RViz: Add -> Marker + TF；想看板在动就把 Fixed Frame 设成 "
                f"{self.get_parameter('camera_frame_id').value}，"
                f"想看相机在动就设成 {self.get_parameter('board_frame_id').value}）"
            )
        self.pub_markers.publish(self._build_markers())

    def _build_markers(self) -> MarkerArray:
        """板外框 + 每个 tag 方框 + 板坐标系三根箭头，坐标都在板坐标系里。

        几何直接取 BoardLayout（和求解用的是同一份数据），所以画出来的框和
        真实板子对不上的话，就说明 target.yaml 里的 tag_centers 不对。
        """
        layout = self.cfg.board
        frame = str(self.get_parameter("board_frame_id").value)
        arr = MarkerArray()

        def base(mid, ns):
            m = Marker()
            m.header.frame_id = frame
            # stamp=0 = "用最新的 TF"，不要写当前时间：这个 marker 是静态的（只在板系里画一次），
            # 写死时间戳以后 RViz 会去查"那一刻"的 TF，晚开 RViz 就查不到、什么都不显示。
            m.header.stamp.sec = 0
            m.header.stamp.nanosec = 0
            m.ns = ns
            m.id = mid
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.lifetime = Duration(seconds=0).to_msg()   # 0 = 永久
            return m

        def color(m, r, g, b, a=1.0):
            m.color.r, m.color.g, m.color.b, m.color.a = (float(r), float(g), float(b), float(a))

        # 1) 板外廓（蓝色闭合折线，画在 z=0 平面上）
        w, h = layout.board_size
        outline = base(0, "board")
        outline.type = Marker.LINE_STRIP
        outline.scale.x = 0.002
        color(outline, 0.1, 0.4, 1.0)
        outline.points = [
            _pt_xy(x, y) for x, y in (
                (-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2), (-w / 2, -h / 2)
            )
        ]
        arr.markers.append(outline)

        # 2) 每个 tag 的方框（绿色，角点顺序和求解完全一致）
        for i, tid in enumerate(layout.tag_ids):
            corners = layout.tag_corners(tid)
            m = base(10 + i, "tags")
            m.type = Marker.LINE_STRIP
            m.scale.x = 0.001
            color(m, 0.1, 0.9, 0.2)
            m.points = [_pt_xyz(*c) for c in list(corners) + [corners[0]]]
            arr.markers.append(m)

        # 3) 板坐标系三根箭头（x 红 / y 绿 / z 蓝，长度 4cm 便于看清方向）
        for j, (axis, col) in enumerate((((1, 0, 0), (1, 0.2, 0.2)), ((0, 1, 0), (0.2, 1, 0.2)),
                                         ((0, 0, 1), (0.2, 0.4, 1)))):
            m = base(20 + j, "board_axes")
            m.type = Marker.ARROW
            m.scale.x = 0.004   # 杆粗
            m.scale.y = 0.008   # 箭头粗
            color(m, *col)
            m.points = [_pt_xyz(0, 0, 0), _pt_xyz(*(0.04 * c for c in axis))]
            arr.markers.append(m)

        return arr

    # ------------------------------------------------------------- 工具
    @staticmethod
    def _to_gray(msg: Image) -> np.ndarray:
        """sensor_msgs/Image -> uint8 单通道灰度图（编码转换在 camera.py 里，命令行也用同一份）。"""
        return FrameSource.to_gray(bgr_from_ros_image(msg))

    def _publish(self, T_cam_board: np.ndarray, stamp) -> None:
        """发 `/a_tag/pose` + TF：**板在相机坐标系下的位姿**（T_cam_board）。

        时间戳用图像 header.stamp（不是本机时间），pose 和 TF 是同一份数值：
          PoseStamped.frame_id = camera_frame_id
          TF: camera_frame_id -> board_frame_id
        """
        q = quat_from_R(T_cam_board[:3, :3])  # (w, x, y, z)
        out = PoseStamped()
        out.header.stamp = stamp
        out.header.frame_id = self.camera_frame_id
        out.pose.position.x = float(T_cam_board[0, 3])
        out.pose.position.y = float(T_cam_board[1, 3])
        out.pose.position.z = float(T_cam_board[2, 3])
        out.pose.orientation.x = float(q[1])
        out.pose.orientation.y = float(q[2])
        out.pose.orientation.z = float(q[3])
        out.pose.orientation.w = float(q[0])
        self.pub_pose.publish(out)

        if self.tf_broadcaster is not None:
            tf = TransformStamped()
            tf.header.stamp = stamp          # 用图像的时间戳，别用本机时间
            tf.header.frame_id = self.camera_frame_id    # 父：相机（参考系）
            tf.child_frame_id = self.board_frame_id      # 子：板（被观测物，树的叶子）
            tf.transform.translation.x = out.pose.position.x
            tf.transform.translation.y = out.pose.position.y
            tf.transform.translation.z = out.pose.position.z
            tf.transform.rotation = out.pose.orientation
            self.tf_broadcaster.sendTransform(tf)

    def _status_lines(self, result, T_cam_board, dt: float, delay_ms: float = 0.0) -> list:
        """状态文字：日志和控制台用同一份，debug 图上也叠这几行。"""
        fps = 1.0 / dt if dt > 0 else 0.0
        ok_rate = 100.0 * self.n_ok / max(1, self.n_frames)
        stale = self.pose_filter.stale_frames if self.pose_filter is not None else 0
        lines = [
            f"tags: {result.n_tags} {list(result.tag_ids)}",
            f"reproj: {result.reproj_rmse_px:.3f} px   有效 {ok_rate:.0f}%   fps {fps:.1f}",
            f"延迟(采集->现在): {delay_ms:.1f} ms",
        ]
        if T_cam_board is not None:
            p = pose_from_T(T_cam_board, degrees=True)
            lines.append("board in camera (mm, deg)")
            lines.append(
                f"  x={p['x'] * 1000:7.1f}  y={p['y'] * 1000:7.1f}  z={p['z'] * 1000:7.1f}"
            )
            lines.append(
                f"  r={p['roll']:6.2f}  p={p['pitch']:6.2f}  y={p['yaw']:6.2f}"
            )
            if stale:
                lines.append(f"  STALE x{stale}")
        elif result.reason:
            lines.append(result.reason)
        return lines

    def _stage_debug(self, bgr, corners, result, T_cam_board, dt, stamp, now, delay_ms=0.0) -> None:
        """暂存这一帧要画的东西（只在图像线程里拷引用，不画不发）。

        下一帧会覆盖上一帧 —— debug 图是"看个大概"，不需要一帧不落。
        """
        self._debug_pending = (bgr, corners, result, T_cam_board, dt, stamp, delay_ms)
        self._t_debug = now

    def _on_debug_timer(self) -> None:
        """独立线程：把暂存的帧画成叠加图发出去（发布可能被慢订阅者堵住，只堵这里）。"""
        item = self._debug_pending
        if item is None:
            return
        self._debug_pending = None
        bgr, corners, result, T_cam_board, dt, stamp, delay_ms = item
        t0 = time.perf_counter()
        self._publish_debug(bgr, corners, result, T_cam_board, dt, stamp, delay_ms)
        cost = time.perf_counter() - t0
        # 滑动平均，避免单帧抖动就缩限速
        self._debug_cost = 0.5 * self._debug_cost + 0.5 * cost

    def _publish_debug(self, bgr, corners, result, T_cam_board, dt, stamp, delay_ms=0.0) -> None:
        """把叠加图发到 debug_image_topic（RViz 的 Image display / rqt_image_view 能直接看）。"""
        lines = self._status_lines(result, T_cam_board, dt, delay_ms)
        out = draw_overlay(
            bgr,
            corners,
            result=result,
            layout=self.cfg.board,
            camera_matrix=self.K,
            dist_coeffs=self.D,
            lines=lines,
            expected_ids=self.cfg.board.tag_ids,
        )
        # 想看流畅的全帧率视频（debug_period: 0）时，缩一半边长能把带宽降到 1/4
        scale = float(self.get_parameter("debug_scale").value)
        if 0.0 < scale < 1.0:
            out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        msg = bgr_to_ros_image(out, stamp=stamp, frame_id=self.camera_frame_id)
        self.pub_debug.publish(msg)

    def _log_status(self, result, T_cam_board, dt: float, delay_ms: float = 0.0) -> None:
        fps = 1.0 / dt if dt > 0 else 0.0
        ok_rate = 100.0 * self.n_ok / max(1, self.n_frames)
        if not result.ok:
            self.get_logger().info(
                f"[{self.n_frames:6d}] 无位姿 tags={result.n_tags} {result.reason} "
                f"有效 {ok_rate:.1f}% fps={fps:4.1f} 延迟={delay_ms:.1f}ms"
            )
            return
        p = pose_from_T(T_cam_board, degrees=True)
        stale = self.pose_filter.stale_frames if self.pose_filter is not None else 0
        self.get_logger().info(
            f"[{self.n_frames:6d}] tags={result.n_tags} "
            f"reproj={result.reproj_rmse_px:5.3f}px "
            f"x={p['x'] * 1000:8.1f} y={p['y'] * 1000:8.1f} z={p['z'] * 1000:8.1f} mm "
            f"rpy=({p['roll']:6.2f},{p['pitch']:6.2f},{p['yaw']:6.2f}) deg "
            f"stale={stale} 有效 {ok_rate:.1f}% fps={fps:4.1f} 延迟={delay_ms:.1f}ms"
        )


def main(args=None) -> int:
    """a_tag_node 的入口。"""
    rclpy.init(args=args)
    node = ATagNode()
    # 多线程：debug 图的画+发在另一个回调组里，慢订阅者拖不动位姿链路
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Ctrl-C 或被 SIGTERM 停掉，都算正常退出
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
