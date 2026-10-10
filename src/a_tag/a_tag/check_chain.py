"""链路自检：相机 -> /camera_info -> a_tag 节点 -> /a_tag/pose + TF，逐跳查。

位姿约定：**板在相机系**（PoseStamped.frame_id = camera_frame_id，TF 父 = 相机、子 = 板）。

用法（另开一个终端，相机和 a_tag 节点都在跑的时候）：
    ros2 run a_tag a-tag-check
    ros2 run a_tag a-tag-check --seconds 15
"""

from __future__ import annotations

import argparse
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_msgs.msg import TFMessage

OK = "✓"
BAD = "✗"
SKIP = "⊘"


def _stamp_key(h):
    return (h.sec, h.nanosec)


class ChainChecker(Node):
    def __init__(self, args):
        super().__init__("a_tag_check")
        self.args = args
        self.n_image = 0
        self.n_info = 0
        self.n_pose = 0
        self.n_tf = 0
        self.image_desc = ""
        self.info_desc = ""
        self.pose = {}
        self.tf = {}
        self.t0 = time.time()

        img_qos = qos_profile_sensor_data if args.sensor_data_qos else 10
        self.create_subscription(Image, args.image_topic, self.on_image, img_qos)
        self.create_subscription(CameraInfo, args.camera_info_topic, self.on_info, img_qos)
        self.create_subscription(PoseStamped, args.pose_topic, self.on_pose, 50)
        self.create_subscription(TFMessage, "/tf", self.on_tf, 100)

    # ------------------------------------------------------------- 回调
    def on_image(self, m):
        self.n_image += 1
        if not self.image_desc:
            self.image_desc = f"{m.width}x{m.height} {m.encoding} frame_id={m.header.frame_id!r}"

    def on_info(self, m):
        self.n_info += 1
        if not self.info_desc:
            k = m.k
            self.info_desc = (
                f"{m.width}x{m.height} fx={k[0]:.1f} fy={k[4]:.1f} cx={k[2]:.1f} cy={k[5]:.1f}"
            )

    def on_pose(self, m):
        self.n_pose += 1
        self.pose[_stamp_key(m.header.stamp)] = m.header.frame_id

    def on_tf(self, m):
        for tr in m.transforms:
            # 板在相机系：父 = camera_frame_id，子 = board_frame_id
            if (
                tr.child_frame_id == self.args.board_frame_id
                and tr.header.frame_id == self.args.camera_frame_id
            ):
                self.n_tf += 1
                self.tf[_stamp_key(tr.header.stamp)] = tr

    # ------------------------------------------------------------- 报告
    def publishers(self, topic):
        try:
            return self.count_publishers(topic)
        except Exception:
            return -1

    def known(self, topic):
        return any(t == topic for t, _ in self.get_topic_names_and_types())

    def run(self):
        t0 = time.time()
        while time.time() - t0 < self.args.seconds:
            rclpy.spin_once(self, timeout_sec=0.1)
        dur = time.time() - t0
        a = self.args
        lines = []
        hints = []

        def hop(name, ok, text, hint=""):
            lines.append(f"  [{name}] {OK if ok else BAD} {text}")
            if not ok and hint:
                hints.append(f"  - {name}: {hint}")

        # 1) 图像
        img_ok = self.n_image > 0
        hop("1/4 图像", img_ok,
            f"{a.image_topic}: {self.n_image} 帧 / {dur:.1f}s = {self.n_image/dur:.1f} Hz"
            + (f"  {self.image_desc}" if self.image_desc else "")
            + f"  发布者 {self.publishers(a.image_topic)}",
            "相机驱动没在发图：ros2 launch hik_camera hik_camera.launch.py；"
            "查 USB3 线 / MVS SDK / udev 权限，或话题名不对（用 ros2 topic list 看一眼）")

        # 2) 内参
        info_ok = self.n_info > 0
        hop("2/4 内参", info_ok,
            f"{a.camera_info_topic}: {self.n_info} 条" + (f"  {self.info_desc}" if self.info_desc else ""),
            "驱动没发 CameraInfo（a_tag 节点会一直等它、不发位姿）；"
            "或话题名不对，或用 camera_info_file 参数直接给标定文件")

        # 3) 位姿
        pose_ok = self.n_pose > 0
        extra = ""
        if img_ok:
            ratio = self.n_pose / max(1, self.n_image)
            extra = f"（图像的 {100*ratio:.0f}%；板不全在画面里时本来就出不了位姿）"
        hop("3/4 位姿", pose_ok,
            f"{a.pose_topic}: {self.n_pose} 帧 = {self.n_pose/dur:.1f} Hz {extra}"
            + f"  发布者 {self.publishers(a.pose_topic)}",
            "a_tag 节点没在跑（ros2 node list | grep a_tag），或它在等内参/等图像，"
            "或画面里看不到 ≥min_tags 个 tag（ros2 run a_tag a-tag --inspect --topic " + a.image_topic + "）")

        # 4) TF
        tf_ok = self.n_tf > 0
        hop("4/4 TF", tf_ok,
            f"/tf {a.camera_frame_id} -> {a.board_frame_id}: {self.n_tf} 条 = {self.n_tf/dur:.1f} Hz",
            "publish_tf 被设成 false 了，或 board_frame_id / camera_frame_id 和节点参数不一致"
            "（ros2 param get /a_tag board_frame_id）")

        # TF 与 pose 是否逐帧一致
        verdict = BAD
        if pose_ok and tf_ok:
            both = sorted(set(self.pose) & set(self.tf))
            only_pose = len(set(self.pose) - set(self.tf))
            if not both:
                lines.append(f"  [比对] {BAD} pose 和 TF 的时间戳对不上（各 {len(self.pose)}/{len(self.tf)} 帧）")
                hints.append("  - 比对: 两个出口的时间戳应该完全一样；检查是不是别的节点也在发同名 TF")
            else:
                same_frame = all(self.pose[k] == self.args.camera_frame_id for k in both)
                lines.append(
                    f"  [比对] {OK if same_frame else BAD} 共同时间戳 {len(both)} 帧，"
                    f"只有 pose 的 {only_pose} 帧，只有 TF 的 {len(set(self.tf) - set(self.pose))} 帧；"
                    f"pose.frame_id = {self.args.camera_frame_id!r}？{same_frame}"
                )
                verdict = OK if (same_frame and only_pose == 0) else BAD
        elif pose_ok != tf_ok:
            lines.append(f"  [比对] {BAD} 只有一边有数据，两个出口应该同时出现")
        else:
            lines.append(f"  [比对] {SKIP} 两边都没数据，先修上面")

        print("\n=== a_tag 链路自检（" + f"{dur:.1f}s" + "）===")
        for ln in lines:
            print(ln)
        if hints:
            print("\n下一步看这里：")
            for h in hints:
                print(h)
        all_ok = img_ok and info_ok and pose_ok and tf_ok and verdict == OK
        print("\n结论:", "链路通 ✓" if all_ok else "链路有断点 ↑")
        return 0 if all_ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(description="a_tag 链路自检：相机 -> 内参 -> 位姿 -> TF")
    p.add_argument("--seconds", type=float, default=10.0, help="观察窗口（秒）")
    p.add_argument("--image-topic", default="/image_raw")
    p.add_argument("--camera-info-topic", default="/camera_info")
    p.add_argument("--pose-topic", default="/a_tag/pose")
    p.add_argument("--board-frame-id", default="tag_board")
    p.add_argument("--camera-frame-id", default="camera_optical_frame")
    p.add_argument("--sensor-data-qos", action="store_true", default=True,
                   help="图像/内参按 sensor-data QoS 订阅（默认开，和节点一致）")
    p.add_argument("--no-sensor-data-qos", dest="sensor_data_qos", action="store_false")
    args, _ = p.parse_known_args(argv)

    rclpy.init()
    node = ChainChecker(args)
    try:
        return node.run()
    except KeyboardInterrupt:
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
