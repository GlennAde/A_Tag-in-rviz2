# A_tag 工作空间

colcon 工作空间，装两件事：

| 包 | 说明 |
|---|---|
| `src/a_tag` | 用共面四 AprilTag 标定板估计相机 6DoF 位姿（Python，`ament_python`）。文档见 [src/a_tag/README.md](src/a_tag/README.md) |
| `src/rm_vision_ros2_hik_camera` | 海康 USB3 工业相机 ROS 2 驱动（C++，包名 `hik_camera`），通过 MVS SDK 取流并发布 `/image_raw` + `/camera_info` |

A_tag 默认订阅相机驱动发布的 `/image_raw` + `/camera_info`，把「**板在相机坐标系下的位姿**」
发到 `/a_tag/pose`（`frame_id = camera_optical_frame`），同时用同一份数据广播 TF
（`camera_optical_frame -> tag_board`，`publish_tf:=false` 可关）。相机是参考系、板是被观测物，
所以 RViz 里 Fixed Frame 用相机就能看到板在动。

## 构建

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install                       # 全工作空间
colcon build --packages-select a_tag --symlink-install   # 只编 A_tag
source install/setup.bash
```

## 跑起来

```bash
# 终端 1: 相机出图（需要海康相机 + MVS SDK）
ros2 launch hik_camera hik_camera.launch.py

# 终端 2: 位姿估计
ros2 launch a_tag a_tag.launch.py
ros2 topic echo /a_tag/pose
ros2 run tf2_ros tf2_echo camera_optical_frame tag_board   # 同一份位姿的 TF

# 终端 3: 链路自检（图像 -> 内参 -> 位姿 -> TF 逐跳查，断在哪跳会告诉你）
ros2 run a_tag a-tag-check
```

想在 RViz 里看板的三维位姿（板框 + tag 方框随板移动）：

```bash
ros2 launch a_tag a_tag.launch.py publish_markers:=true
rviz2 -d "$(ros2 pkg prefix a_tag)/share/a_tag/config/a_tag.rviz"   # 配好的，Fixed Frame=相机
```

没接相机时，可以用图片/视频跑命令行验证整条链路：

```bash
# 命令行只做「图片/视频 -> 打印位姿」，不看窗口、不存视频（内参归相机包，要显式指）
ros2 run a_tag a-tag --camera-info src/rm_vision_ros2_hik_camera/config \
    --source .scratch/board.png
```

## 目录约定

包内一律 `config/` + `launch/` + 源码目录 + `package.xml`，与 `hik_camera` 保持一致；
A_tag 的源码目录是 `a_tag/a_tag/`（Python 包必须同名嵌套，不能像 C++ 那样放 `src/`）。