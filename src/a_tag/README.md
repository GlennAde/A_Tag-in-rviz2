# a_tag

用一块**共面四 AprilTag 标定板**估计相机相对板的 6DoF 位姿。

库的主输出是 `T_board_cam` —— **相机在板坐标系下的位姿**（x, y, z, roll, pitch, yaw）。

ROS 两个出口（`/a_tag/pose` 和 TF）发的是它的逆 **`T_cam_board`（板在相机系）**：
相机当参考系、板当被观测物，这样 RViz 里"固定相机看板动"，也和 aruco_ros / apriltag_ros
的习惯一致。两者只差一个求逆，见下「坐标系约定」。

## 坐标系约定

板坐标系（board frame）：

| 轴 | 方向 |
|---|---|
| 原点 | 板面中心 |
| X | 板面内向右 |
| Y | 板面内向上 |
| Z | 垂直板面向外，指向观察者（相机） |

板面即 **Z = 0** 平面，四个 tag 的 16 个角点 3D 坐标都落在这个平面上。

变换记号：`T_a_b` 表示把 b 坐标系的点变换到 a 坐标系，即 `X_a = T_a_b @ X_b`。

- `T_cam_board`：PnP 的直接结果，板 → 相机
- `T_cam_board`：PnP 的直接结果，板 → 相机；**ROS 出口（pose + TF）用这个**
- `T_board_cam`：**库/命令行的主输出**，等于 `T_cam_board` 的逆

欧拉角约定为 ZYX 内旋：`R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`，
在 `pitch = ±90°` 处存在万向锁。**内部一律以 SE(3) 为准，只在输出时转欧拉角。**

### 为什么正对板时 roll 读到约 ±180°

**这是正常的，不是 bug。** 板坐标系的 Z 指向观察者，而相机看向板时其光学 Z
指向板，两者朝向相反，所以姿态里必然含一个约 180° 的转动。相机正对板时：

| 角度 | 读数 | 含义 |
|---|---|---|
| roll | ≈ ±180° | 相机在看着板 |
| pitch | ≈ 0° | 离垂直的倾角 |
| yaw | 面内转角 | 相机绕板法线的转动 |

这里**刻意**不用机体系（X 前 / Y 左 / Z 上）。实测机体系下正对板时
`pitch = 89.99°`，正好落在万向锁上，roll 与 yaw 不可分离、读数乱跳。
光学系下三个角是良态可分的。

如果确实需要机体系对接 ROS，用 `to_camera_body()` 转换，
但请以 4×4 矩阵继续传递，不要再转欧拉角。

## 目录结构

本仓库是一个 colcon 工作空间，A_tag 自己是 `ament_python` 包，目录格式跟配套的
海康相机包（`src/rm_vision_ros2_hik_camera`）对齐：

```
A_tag/                             # 工作空间根
├── build/ install/ log/           # colcon 产物（已 gitignore）
└── src/
    ├── a_tag/                     # 本包（包根，与参考包同一层）
    │   ├── package.xml            # 元数据/依赖，对应参考包的 package.xml
    │   ├── setup.py setup.cfg     # ament_python 构建入口（对应参考包的 CMakeLists.txt）
    │   ├── resource/a_tag         # ament 资源索引标记（缺了 ros2 run 找不到本包）
    │   ├── config/                # 安装到 share/a_tag/config
    │   │   ├── target.yaml        #   目标：板定义 + 检测求解参数（a_tag 唯一拥有的配置）
    │   │   └── a_tag_params.yaml  #   ROS 2 节点参数（话题/坐标系/QoS/滤波）
    │   ├── launch/a_tag.launch.py
    │   └── a_tag/                 # Python 源码（对应参考包的 src/*.cpp）
    └── rm_vision_ros2_hik_camera/ # 配套的海康相机驱动（C++ 包，出图给本包用）
```

注意 Python 与 C++ 的区别：参考包的 `src/` 放 `.cpp`，而 Python 包的源码目录必须**同名**
（`a_tag/a_tag/*.py`），再往里套一层 `src/` 就不再是可导入的包了。

## 安装

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select a_tag --symlink-install
source install/setup.bash
```

### 依赖：pupil-apriltags 要装进**系统 python**

ROS Humble 的 `rclpy` / `sensor_msgs` / `geometry_msgs`，加上 rosdep 里的
`python3-numpy`、`python3-opencv`、`python3-yaml` 都在 apt 里。剩下一个 pip 侧的
`pupil-apriltags`（rosdep 里没有）：

```bash
python3 -m pip install --user --no-deps pupil-apriltags
```

三个容易踩的点：

1. **要装进 `/usr/bin/python3`**，不是你随手激活的某个 venv —— colcon 生成的入口点
   (`install/a_tag/lib/a_tag/a_tag_node`) shebang 写死 `/usr/bin/python3`，所以
   `ros2 launch a_tag a_tag.launch.py` 用的永远是系统 python。装错地方的表现就是
   `ModuleNotFoundError: No module named 'pupil_apriltags'`。
2. **别去掉 `--no-deps`**：Ubuntu 自带的 `python3-opencv` 是对 numpy 1.x ABI 编的，
   让 pip 顺手把 numpy 升到 2.x 会把 `cv2` 弄坏（实测报 `numpy.core.multiarray failed to import`）。
3. 如果 `~/.local` 不可写（只读挂载 / 没管理员），就装到工作区里，跑的时候指一下：

   ```bash
   python3 -m pip install --target <工作区>/.deps --no-deps pupil-apriltags
   PYTHONPATH=<工作区>/.deps:$PYTHONPATH ros2 launch a_tag a_tag.launch.py
   PYTHONPATH=<工作区>/.deps:$PYTHONPATH ros2 run a_tag a-tag --inspect --topic /image_raw
   ```

   工作区里现成的 `.venv`（3.10 + `--system-site-packages`，已带 pupil-apriltags）也能当这个目录用：
   `PYTHONPATH=$PWD/.venv/lib/python3.10/site-packages:$PYTHONPATH ros2 launch a_tag a_tag.launch.py`。

## 快速开始

ROS 2 里（图像来自相机驱动）：

```bash
ros2 launch a_tag a_tag.launch.py                      # 订阅 /image_raw + /camera_info
ros2 launch a_tag a_tag.launch.py image_topic:=/hik_camera/image_raw
ros2 topic echo /a_tag/pose                            # 板在相机坐标系下的位姿
ros2 run a_tag a-tag-check                             # 链路自检（见「链路自检」）
ros2 run tf2_ros tf2_echo camera_optical_frame tag_board   # 同一份位姿的 TF
```

命令行（不写节点也能直接拿相机画面或文件验证）：

```bash
# 诊断：画面里到底有哪些 tag（换相机/换板第一步，见「换相机 / 换板之后怎么验证」）
ros2 run a_tag a-tag --inspect --topic /image_raw
ros2 run a_tag a-tag --inspect --source board.png

# 求解：图片/视频/话题 -> 逐帧打印位姿。内参归相机包，离线跑必须显式指一份（见「配置」）
ros2 run a_tag a-tag --camera-info src/rm_vision_ros2_hik_camera/config --source board.png
ros2 run a_tag a-tag --camera-info src/rm_vision_ros2_hik_camera/config \
    --topic /image_raw --frames 60 --print-every 10
```

这个命令行刻意做得很窄：**不带窗口、不存视频、不直连相机**。想看画面就装
`rqt_image_view` 看 ROS 话题，或拿看图软件看文件。

作为库（不依赖 rclpy）：

```python
from a_tag import AppConfig, AprilTagDetector, solve_board_pose

# 目标配置自动找 ./config 或 share/a_tag/config；内参显式给（离线没有话题可订阅）
cfg = AppConfig.load(camera_info_path="src/rm_vision_ros2_hik_camera/config")
K, D = cfg.require_intrinsics()                # 没内参会抛 MissingIntrinsicsError
det = AprilTagDetector(expected_ids=cfg.board.tag_ids)

corners = det.detect(gray)                     # {tag_id: (4,2) 角点}
result = solve_board_pose(corners, cfg.board, K, D)

if result.ok:
    print(result.pose())                       # 相机在板系下的 x,y,z,rpy
    print(result.distance, result.reproj_rmse_px)
else:
    print("无效:", result.reason)
```

## 链路自检（不确定哪一段断了就跑这个）

三个终端：

```bash
# 1) 相机
ros2 launch hik_camera hik_camera.launch.py
# 2) 位姿
ros2 launch a_tag a_tag.launch.py
# 3) 自检（随时可跑，跑完自己退出；不加参数看 10 秒）
ros2 run a_tag a-tag-check
ros2 run a_tag a-tag-check --seconds 45
```

它一路查下去：图像 → 内参 → 位姿 → TF，最后逐帧比对 pose 和 TF。真数据的样子：

```
=== a_tag 链路自检（45.1s）===
  [1/4 图像] ✓ /image_raw: 138 帧 / 45.1s = 3.1 Hz  1440x1080 rgb8 frame_id='camera_optical_frame'  发布者 1
  [2/4 内参] ✓ /camera_info: 418 条  1440x1080 fx=1804.0 fy=1803.2 cx=683.0 cy=545.0
  [3/4 位姿] ✓ /a_tag/pose: 97 帧 = 2.2 Hz （图像的 70%；板不全在画面里时本来就出不了位姿）  发布者 1
  [4/4 TF] ✓ /tf camera_optical_frame -> tag_board: 97 条 = 2.2 Hz
  [比对] ✓ 共同时间戳 97 帧，只有 pose 的 0 帧，只有 TF 的 0 帧；pose.frame_id = 'camera_optical_frame'？True

结论: 链路通 ✓
```

断在哪一跳，它就把那一跳的排查方向打出来：

| 断点 | 先查什么 |
|---|---|
| 1/4 图像 | 相机没插/没供电、USB3 口不对、**MVS Viewer 或别的进程正占着相机**（同一时刻只允许一个进程打开）、驱动没起来 |
| 2/4 内参 | 驱动没发 `/camera_info`、`camera_info_url` 指错、话题名和节点参数不一致 |
| 3/4 位姿 | a_tag 节点没跑；节点在等内参（启动日志有 WARN）；画面里看不到 ≥`min_tags` 个 tag（用 `--inspect` 看）；或 `tags=4` 但 `reproj` 超大（tag_size / 内参 / 板对不上） |
| 4/4 TF | `publish_tf:=false` 了，或 `board_frame_id` / `camera_frame_id` 和自检参数不一致 |

两个别误判的地方：

* **位姿帧率必然低于图像帧率**，板不全在画面里（或 tag 不够 `min_tags`）就没有位姿。上面这个 70% 是那段 bag 里板大部分时间在画面内的运气，换一段只有 20% 也正常。
* 图像那行的 Hz 是**订阅侧下限**（含节点发现握手时间），不是相机真实帧率；要看真实帧率用 `ros2 topic hz /image_raw`。

**手上没有相机时**，把第一跳换成 bag 也能把后半段打通（工作区 `.scratch/board_bag` 有之前录的 929 帧）：

```bash
ros2 bag play .scratch/board_bag --loop
ros2 launch a_tag a_tag.launch.py
ros2 run a_tag a-tag-check --seconds 45
```

## 话题接口

| 方向 | 话题 | 类型 | 说明 |
|---|---|---|---|
| 订阅 | `/image_raw` | `sensor_msgs/Image` | `mono8` / `rgb8` / `bgr8` / `rgba8` / `bgra8`；默认 best-effort + **队列深度 1**（见下） |
| 订阅 | `/camera_info` | `sensor_msgs/CameraInfo` | **内参的唯一来源**（相机驱动发布）；收不到就不输出位姿 |
| 发布 | `/a_tag/pose` | `geometry_msgs/PoseStamped` | **板在相机坐标系下的位姿**（`T_cam_board`），`frame_id = camera_frame_id` |
| 发布 | `/a_tag/debug_image` | `sensor_msgs/Image` | 可视化叠加图（bgr8）。默认**关**，要用加 `publish_debug_image:=true`，见「可视化 / 排障」 |
| 发布 | `/tf` | `tf2_msgs/TFMessage` | 同一份位姿的 TF：`camera_frame_id -> board_frame_id`（父=相机、子=板）。默认**开**，`publish_tf:=false` 关，见「TF 广播」 |
| 发布 | `/a_tag/board_markers` | `visualization_msgs/MarkerArray` | 板的三维示意（外框 + tag 方框 + 板系箭头）。默认**关**，`publish_markers:=true` 开，见「在 RViz 里看板」 |

### 队列深度：只留最新帧（`image_queue_depth: 1`）

图像订阅是 **best-effort + KEEP_LAST depth 1**：相机比处理快的时候，宁可丢掉旧帧，
也要保证算的是**刚到的这一帧**。depth 默认 5 的坏处是"卡一下之后先补算 5 帧旧的"，
位姿会晚一拍（65Hz 下最多多背 77ms 延迟）；对瞄准/控制这类用途，延迟比丢几帧更难受。

配套的诊断：日志和 debug 图上都会打 **`延迟=x.x ms`**（图像时间戳 → 当前时刻，
含传输+排队+处理）。正常情况下应该是**个位数 ms 且稳定**；持续变大就说明在处理不过来
（看 fps、或者相机帧率是不是超过处理能力了），这时该裁 ROI 而不是加大队列。

节点参数（`config/a_tag_params.yaml`，都能被 launch 参数覆盖）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `image_topic` / `camera_info_topic` / `pose_topic` | `/image_raw` / `/camera_info` / `/a_tag/pose` | 话题名 |
| `use_sensor_data_qos` | `true` | 工业相机高帧率必须 best-effort，用 reliable 会积压、延迟越跑越大 |
| `image_queue_depth` | `1` | 图像订阅队列深度：**只留最新帧**（见下「队列深度」） |
| `board_frame_id` | `tag_board` | 板的坐标系名（TF 的 **child**，树的叶子） |
| `config_dir` | `""` | 留空 = 用 `share/a_tag/config`（只放 target.yaml） |
| `camera_info_file` | `""` | 可选：直接指定 `camera_info.yaml`；留空 = 只用话题内参 |
| `min_tags` / `max_reproj_px` | `0` / `0.0` | 0 表示用 `target.yaml` 里的值 |
| `use_filter` | `true` | one-euro 平移 + 四元数 slerp 旋转滤波 |
| `print_every` | `30` | 每 N 帧打一条日志，0 = 静默 |
| `publish_debug_image` | `false` | 发可视化叠加图（检测框 + 板框投影 + 日志文字） |
| `debug_image_topic` | `/a_tag/debug_image` | 可视化话题名 |
| `debug_period` | `0.2` | debug 图最短间隔（秒）= 5Hz；`0` = 每帧都发（见「可视化 / 排障」） |
| `debug_adaptive` | `true` | 观看端跟不上时自动放慢 debug 图，别拖慢位姿 |
| `debug_scale` | `1.0` | debug 图缩放（`0.5` = 边长一半，带宽 1/4）；配 `debug_period: 0` 看全帧率流畅视频 |
| `debug_use_sensor_data_qos` | `false` | debug 图默认 reliable 发（`rqt_image_view` 好订） |
| `publish_tf` | `true` | 广播 TF（和 `/a_tag/pose` 同一份数值、同一个时间戳） |
| `publish_markers` | `false` | 发板的三维示意 `MarkerArray`（RViz 用，见「在 RViz 里看板」） |
| `marker_topic` | `/a_tag/board_markers` | 板示意的话题名 |
| `camera_frame_id` | `camera_optical_frame` | 相机光学坐标系名（TF 的**父**，也是 pose 的 `frame_id`）；要和相机驱动 `Image.header.frame_id` 一致 |

## 在 RViz 里看板（`publish_markers:=true`）

### 一条命令（用包里的 RViz 配置）

**全新的 `rviz2` 默认只有一个 Grid** —— 没有 TF、没有 Marker，Fixed Frame 还是 `map`，
所以直接开就是"什么都没有"。包里带了一份配好的：

```bash
ros2 launch a_tag a_tag.launch.py publish_markers:=true     # marker 是开关，默认关
rviz2 -d "$(ros2 pkg prefix a_tag)/share/a_tag/config/a_tag.rviz"
```

这份配置已经设好：Fixed Frame = `camera_optical_frame`、TF 显示、`/a_tag/board_markers`
的 Marker 显示（**Durability = Transient Local**，不然收不到 latched 的板示意）、
`/a_tag/debug_image` 的 Image 显示（默认没勾，想看勾上）。

### 或者手动配（三步）

1. **Fixed Frame 填 `camera_optical_frame`**（左上 Global Options；默认是 `map`，
   不改的话满屏报错、连 Grid 都不显示）。
2. `Add` → `By topic` → `/a_tag/board_markers` → `Marker`：蓝色板外框 + 4 个绿色 tag 方框
   + 板坐标系三根箭头（x 红 / y 绿 / z 蓝）。**Durability 要选 Transient Local**。
3. `Add` → `TF`：`tag_board` 三脚架就是实时位姿（板是相机的子系，所以它随板动）；
   也可以 `Add → By topic → /a_tag/pose → Pose` 加个箭头。

### 什么都没有时按这个顺序查

| 现象 | 原因 |
|---|---|
| 满屏红字 `Fixed Frame [map] does not exist` | Fixed Frame 没改 |
| `Fixed Frame [camera_optical_frame] does not exist` | **还没有任何位姿**：这个 frame 是本包发 TF 时才出现的（相机驱动只发图、不发 TF），板不在画面里就没有它。先看 `ros2 run a_tag a-tag-check` |
| 只有两个三脚架、没有板 | 没开 `publish_markers:=true`（或者 Marker 显示的 Durability 不是 Transient Local） |
| 板出现在原点、之后不动 | 最后一次有效位姿之后就没有新位姿了（板出画面 / tag 不够 `min_tags`） |

打开画面里的板示意用的是 latched（TRANSIENT_LOCAL）发布，所以**后开 RViz 也收得到**；
几何直接取自 `target.yaml`，**画出来的框应该正好套在你板子上** —— 套不上就是 `tag_centers` 写错了。
运行中开关：`ros2 param set /a_tag publish_markers true`。

板不在画面里（tag 不够 `min_tags`）时节点不发位姿 → 也不发 TF，RViz 里板就停在最后一次有效位姿。

## TF 广播

节点默认把位姿同时以 TF 发出去，**数值和时间戳与 `/a_tag/pose` 完全一致**
（时间戳取图像 `header.stamp`，不是本机时间），父子关系是：

```
parent = camera_frame_id (camera_optical_frame)   <- 相机是参考系
child  = board_frame_id  (tag_board)              <- 板是被观测物，树的叶子
```

也就是「**板在相机坐标系下的位姿**」，和 `/a_tag/pose` 逐位相同 —— 两个出口一份数据，
不用互相换算。**相机当父系是刻意的**：

1. 板是本包独有的 frame，当叶子不会和别人的 broadcaster 抢父子关系（机器人自己的
   `base_link -> ... -> camera_optical_frame` 那棵树照旧，本包只是往相机下面挂一块板）；
2. RViz 里 Fixed Frame 用相机（或 `base_link`）就能看到**板在动**；想看相机在板系里动，
   把 Fixed Frame 设成 `tag_board` 也一样能看（TF 是树，方向只决定谁挂在谁下面）。

```bash
ros2 run tf2_ros tf2_echo camera_optical_frame tag_board
ros2 run tf2_ros tf2_monitor camera_optical_frame tag_board
```

⚠️ 反过来说，如果你是"相机装在动的东西上、板固定在场地里"（比如云台追板），
那么"谁动"正好相反，但 **frame 的父子不用跟着改** —— 它只影响可视化时谁看起来静止，
不影响任何 `lookupTransform` 的结果。

**要注意的坑**：一个坐标系在 TF 树里只能有一个父坐标系。`board_frame_id`（`tag_board`）
是本包独有的，正常不会冲突；但你要是也在别处发过同名 frame，或者想反过来自己发
板 → 相机的 TF，就得二选一（`publish_tf:=false` 只用 `/a_tag/pose`，或者把
`board_frame_id` 改成别的名字）。

## 配置

a_tag 只管**目标**。相机侧的一切（内参、曝光、增益）归相机驱动包所有，本包不再存一份：

| 文件 | 在哪 | 管什么 |
|---|---|---|
| `config/target.yaml` | **a_tag** | `board`（tag_size、tag_centers）+ `tracking`（`detector`、`families`、检测/求解参数） |
| `config/a_tag_params.yaml` | **a_tag** | ROS 节点参数（话题名、坐标系、QoS、滤波） |
| `config/camera_info.yaml` | 相机包 `hik_camera` | 内参、畸变、分辨率（标定产物） |
| `config/camera_params.yaml` | 相机包 `hik_camera` | `exposure_time`、`gain`（设备/采集参数） |

内参的来源，按优先级：

1. **ROS 流程**：订阅 `<camera_info_topic>`。相机驱动每帧跟着图像发 `CameraInfo`，内容就是
   它 `camera_info_url` 指向的那份标定文件（默认 `package://hik_camera/config/camera_info.yaml`）；
2. **离线 / 无 ROS**：`a-tag --camera-info <文件或目录>`，例如
   `--camera-info src/rm_vision_ros2_hik_camera/config`；
3. **单文件合并写法**：`--config my.yaml`，一个文件里同时有 `camera:` / `board:` / `tracking:`。

三者都没有 = 没有内参：命令行**直接报错退出**，节点等待话题并告警。这里刻意不做占位值兜底，
因为平面 PnP 里**焦距与距离是尺度等价的**：`u = fx·X/Z + cx`，把 Z 按 `fx'/fx` 等比缩放就能
投影出同样的像。所以：
* 若两份内参**只有焦距不同、主点相同**，位姿会整体缩放而重投影照样 <0.5px —— 健康指标
  完全抓不到，只能靠"内参唯一来源"来防；
* 实测拿 fx=1000 / cx=640 的占位值去跑 fx=1804 / cx=683 的海康标定，位姿缩到 **0.55 倍**
  （z 偏 −45%），重投影涨到 3.9px —— 这次能报出来，靠的是**主点差了 43px**，不是焦距。

查找顺序：`--config`（config 目录，或单个合并的 YAML）→ `$A_TAG_CONFIG` → `./config/`
→ `share/a_tag/config/`（colcon install 之后）→ `./config.yaml`。

`board` 段必须填**真实尺寸**，否则位姿会整体缩放或偏斜。

`config/target.yaml`：

```yaml
board:
  tag_size: 0.04          # tag 黑框边长，米（不含外侧白边）—— 卡尺量，别猜
  tag_centers:            # 各 tag 中心在板坐标系下的 [x, y]，米；**键就是 tag id**
    0: [-0.15, -0.10]
    1: [ 0.15, -0.10]
    2: [ 0.15,  0.10]
    3: [-0.15,  0.10]

tracking:
  detector: aruco         # pupil=libapriltag（默认）| aruco=OpenCV，见下面「两个后端」
  families: tag16h5       # 家族名 / OpenCV 字典名
  quad_decimate: 1.0      # 精度关键，别用 libapriltag 默认的 2.0（仅 pupil 后端有效）
  min_tags: 2             # 别降到 1（1 个 tag 的角度误差差一个数量级）
  max_reproj_px: 1.7      # 见「6. 重投影误差有个物理地板」：印刷板面不会绝对平
```

**布局写错的后果**：位姿会整体缩放/旋转，而**重投影误差照样很小** —— 重投影只验"自洽"，
不验真值（和焦距写错是同一类陷阱）。所以 `tag_size` 和四个坐标必须来自你板子的真实图纸或卡尺。

### 两个检测后端

| `tracking.detector` | 用的是什么 | 什么时候选它 |
|---|---|---|
| `pupil`（默认） | pupil_apriltags / libapriltag | **tag36h11** 最稳，参考工作流默认 |
| `aruco` | OpenCV 的 ArUco 模块 | **tag16h5**（libapriltag 的 16h5 会弄坏进程堆、误检多）、以及真正的 ArUco 字典 |

`aruco` 后端的 `families` 可以填：`tag16h5` / `tag25h9` / `tag36h11` / `tag36h10`，
或真正的 ArUco 字典 `DICT_4X4_50` / `DICT_5X5_100` / `DICT_6X6_250` …（完整列表见 `detector.ARUCO_DICTS`）。
换板不用改代码，改这两行即可。

相机包的 `camera_info.yaml` 就是 `camera_calibration` 的输出格式，标定完直接覆盖过去
（文件里的 `camera_name` 要跟 launch 的参数一致）：

```bash
ros2 run camera_calibration cameracalibrator --camera_name hero_hik_camera \
    --ros-args -r image:=/image_raw -r camera:=/camera_info
# 生成的文件覆盖 src/rm_vision_ros2_hik_camera/config/camera_info.yaml
```

任何 YAML 里未识别的字段都会打 `UserWarning` 而不是静默忽略 —— 写错字段名是最难查的
一类问题。

## 实测精度

4cm tag、34×24cm 板、fx=1000、1280×720。误差主导项是**角点定位精度**，
不是求解器。

| 距离 | tag 像素 | 角度误差 | 位置误差 |
|---|---|---|---|
| 0.4 m | 105 px | 0.02° | 0.1 mm |
| 0.8 m | 51 px | 0.01° | 0.4 mm |
| 1.2 m | 34 px | 0.17° | 4.2 mm |
| 2.0 m | 20 px | 1.10° | 38 mm |

**相机位置误差 ≈ 距离 × 旋转误差**，所以离得越远位置精度掉得越快。
建议让 tag 在图像里保持 **≥ 50px**。

可见 tag 数对精度的影响（0.8m）：

| 可见 tag | 角度误差 |
|---|---|
| 4 | 0.17° |
| 3 | 0.24° |
| 2 | 0.56° |
| 1 | 1.82° |

所以 `min_tags` 默认取 2，别降到 1。

### 真板回放实测（1440×1080 海康 + 本库当前这块板）

用工作区里录的 929 帧 bag（56 s）每 5 帧抽一帧跑完整管道（tag16h5 / `aruco` /
`tag_size=38.5mm` / `max_reproj_px=1.7`）：

| 可见 tag | 帧数 | 出位姿 | 出位姿帧的中位重投影 |
|---|---|---|---|
| 4 | 33 | 31 | 0.98 px |
| 3 | 20 | 17 | 1.04 px |
| 2 | 9 | 8 | 1.05 px |
| 0~1 | 123 | 0 | — |

两点结论：

* **板不全在画面里就没有位姿**（`min_tags=2`）。这段 bag 里只有约 1/3 的帧能看到
  ≥2 个 tag，所以「有效帧比例」低是取景问题，不是检测问题 —— 看 `/a_tag/pose`
  的帧率时要先确认画面里板在不在。
* **速度不是瓶颈**：同一批帧，节点实测 **60 fps**（含发布），命令行管道 **56 fps**；
  上面那 4 个超过 1.7px 被拒的帧基本都是运动模糊。

### 每帧耗时拆解（1440×1080，`aruco`，单线程）

| 环节 | 中位耗时 |
|---|---|
| 反序列化 | 0.97 ms |
| 转 BGR | 0.21 ms |
| 转灰度 | 0.08 ms |
| 检测（aruco） | **3.30 ms** |
| PnP 求解 | 0.03 ms |
| 画叠加图（debug） | 0.46 ms |
| 转回 ROS 消息（debug） | 0.66 ms |
| **主链路合计** | **4.6 ms/帧 → 218 fps 上限** |

相机 1440×1080@65Hz 的预算是 15.4 ms/帧，**还有 3 倍余量**，所以这个包没做多线程/流水线：
瓶颈在检测（占 72%），而且 Python 下加线程也绕不开 GIL（只有 cv2 调用期间释放）。
真到单帧超过相机周期时再考虑（例如 4K@60Hz ≈ 25 ms/帧），那时先裁 ROI 更划算。

## 几个必须知道的坑

### 1. 角点顺序

`pupil_apriltags` 的 tag 坐标系是 **X 朝左 / Y 朝上 / Z 背离相机**
（其自带 `pose_R = diag(-1,-1,1)`）。在本库的板坐标系下：

| `corners[i]` | 图像中位置 | 板坐标系 |
|---|---|---|
| `[0]` | 右上 | (+X, +Y) |
| `[1]` | 左上 | (-X, +Y) |
| `[2]` | 左下 | (-X, -Y) |
| `[3]` | 右下 | (+X, -Y) |

该顺序**锁在 tag 自身的物理角上**，把 tag 在图像中旋转 90/180/270° 不会换序。

顺序写错会让位姿镜像翻转，且重投影误差从 <0.5px 暴涨到十几个 px。
这是最容易自查的一项：重投影 >1px 就先怀疑角点顺序。

### 2. 平面 PnP 在正对相机时是病态的

实测（4 tag，0.3px 角点噪声，300 次）：

| 求解器 | 正对相机 | 倾斜 20° |
|---|---|---|
| IPPE | 0.801° | 0.156° |
| IPPE + refineLM | 0.179° | — |
| **SQPNP** | **0.178°** | **0.122°** |

IPPE 单独用会朝「正对」方向产生系统性偏差。本库默认 `SOLVEPNP_SQPNP`
（全局解、无需初值）再跟 `solvePnPRefineLM`。

**弱自由度是 Z / pitch / roll**，X / Y / yaw 一向稳定。

### 3. 正对板时 roll 约 ±180°

见上文「为什么正对板时 roll 读到约 ±180°」。别把它当成标定错误去"修"。

### 4. 用重投影误差当健康指标

理想印刷板应 **< 0.5px**；本库当前这块板实测约 **1.0px**，因为板面本身有 ~0.26mm
翘曲、有个 tag 粘歪了 0.72°（见「6. 重投影误差有个物理地板」）。持续远超这个量级
才说明 tag_size、相机内参或角点顺序有问题。
超过 `max_reproj_px` 时 `result.ok` 为 `False`。

### 5. 检测降采样

libapriltag 的 `quad_decimate` 默认是 2.0，会先把图像降采样一半再找四边形，
角点精度大约减半。本库默认设为 **1.0**。

### 6. 重投影误差有个"物理地板"

配置都对、检测也稳，`reproj` 仍然下不去（比如一直 5~8px），并且 `--check-layout` 报
"平面拟合残差"也一样大 —— 那不是配置问题，是板子本身：

* **tag 是分别贴的**，各自有 1~3° 的翘曲/倾斜：平面模型对不上，1° ≈ 每角点偏 2~3px；
* **各 tag 实际大小/深度有 1~2% 的差**（手工激光打印+手贴常见），表现为 tag 之间像"各自缩放了一点"；
* **曝光太暗/运动模糊**：角点噪声 1~2px 直接进误差。

判别方法（同一位姿静止拍 20 帧）：

```bash
ros2 run a_tag a-tag --topic /image_raw --camera-info src/rm_vision_ros2_hik_camera/config \
    --frames 20 --print-every 1
```

`reproj` 稳定在某个值附近 → 几何地板；抖动很大 → 噪声/曝光问题。
处理：物理上把 tag 贴平、把曝光调到画面通透（关自动曝光）、避免运动模糊；
数字上把 `max_reproj_px` 放宽到地板之上（例如 10px），别让所有帧都被判无效。
6~8px 的地板在 0.5m 处大约相当于 **2mm 的位置误差**，很多场合够用。

### 7. tag16h5 别用 libapriltag（用 `detector: aruco`）

实测（pupil-apriltags 1.0.4.post11）：`families="tag16h5"` 单次检测能出结果（连续 200 次也没事），
但**同一进程里再建 Detector 就会把堆弄坏**（`malloc(): mismatching next->prev_size`），退出时直接段错误；
而且误检多 —— 一块干净的 16h5 板会额外报出 hamming 1~2、margin 个位数（1.6~2.8）的假 id。
OpenCV 后端同一个画面只报真实的 4 个、零误检。所以 **tag16h5 请配 `tracking.detector: aruco`**。

### 8. 发大图别写 `msg.data = 数组.tobytes()`

rclpy 生成的 setter 对 **`bytes`** 会走 Python 级逐元素校验（`all(isinstance(v, int) ...)`），
1440×1080×3 一帧 4.6MB 要 **229ms** —— 这曾经就是"开了 debug 图整个人掉到 4fps"的元凶
（多线程救不了它，因为它是在占 GIL）。给 `array.array('B')` 就走 setter 的快路径：

```python
msg.data = img.tobytes()                    # 229 ms  ✗
msg.data = array.array("B", img.tobytes())  # 0.45 ms ✓ 差 500 倍
```

## 换相机 / 换板之后怎么验证

按顺序来，每步都有明确的成功判据（不用写测试、不用开窗口）。

**1. 先确认相机在出图**

```bash
ros2 topic hz /image_raw                  # 帧率稳定
ros2 topic echo /camera_info --once       # 分辨率/K 是你要用的那份标定
```

**2. 用 `--inspect` 看画面里到底有哪些 tag**（不读 target.yaml、不做 ID 过滤）

```bash
ros2 run a_tag a-tag --inspect --topic /image_raw        # 直接从相机驱动拿一帧
ros2 run a_tag a-tag --inspect --source frame.png        # 或跑存下来的图
ros2 run a_tag a-tag --inspect --topic /image_raw --family all   # 家族 + OpenCV 字典全扫一遍
ros2 run a_tag a-tag --inspect --topic /image_raw --save-annotated /tmp/a.png
```

不知道板是什么家族/字典时，先用 `--family all`：它会把常见家族逐个起进程试，并把
OpenCV 的十几个字典（含 `tag16h5`、`DICT_4X4_50`…）一次扫完，直接告诉你命中的是哪个。

```
图像: 1280x720   检测到 4 个 tag
    id  family     hamming  margin             中心(px)    边长(px)
     0  tag36h11         0   120.0     (487.2, 460.4)      51.1
     1  tag36h11         0   120.0     (870.6, 460.4)      51.1
ID 对得上：target.yaml 期望 [0, 1, 2, 3]，实测 [0, 1, 2, 3] ✓
```

判据：**边长 ≥ 50px**（低于这个精度开始崩）、`hamming = 0`、`margin` 越大越好。
它还会顺手按检测结果推一份 `tag_centers`，可以直接抄进 target.yaml。

**2.5 如果 `tags` 有了但 `reproj` 很大（几十上百 px）→ 布局对不上**

这说明检测没问题，是**板面坐标与 id 的对应关系**错了（id 贴错角、上下镜像、某个 tag 贴反）。
别手改试了，直接让它反推：

```bash
ros2 run a_tag a-tag --check-layout --topic /image_raw \
    --camera-info src/rm_vision_ros2_hik_camera/config
```

它会枚举「4 个 id 在四个角的分配 × 每个 tag 的贴装旋转」，按重投影误差排序，
并把**当前配置最该改成什么样**直接打印成可粘贴的 YAML：

```
=== 前 5 名（reproj 越小越对）===
  reproj     2.23px   左上=14  右上=4  左下=22  右下=6   贴装旋转 无
  ...
把下面这段覆盖进 config/target.yaml 的 board 段：
board:
  tag_size: 0.04
  tag_centers:
    14: [-0.0491, +0.0500]    # 左上
    ...
```

若最好的一组仍有几十 px，它会提示先查 `tag_size` / 内参 / 分辨率，而不是布局。

**3. ID 或家族对不上怎么办**

* **ID 不一样**：`target.yaml` 里 `tag_centers` 的**键就是 tag id**，改成实测值即可。
  a_tag 只保留列在 `tag_centers` 里的 ID（用来滤掉环境里的干扰 tag），所以 ID 写错时
  的症状是**一个 tag 都认不到**，而不是报错 —— `--inspect` 会把差在哪、新布局长什么样
  直接打出来。
* **家族不一样**：默认 `tag36h11`。别的家族**单独再跑一次**：`--family tag25h9`。
  这个版本的 libapriltag 一次只能安全地用一个家族（同一进程里轮几个家族会把堆弄坏，
  实测 glibc `malloc(): mismatching next->prev_size`）；`tag16h5` 更危险，要试就单独起进程。
* **板上是 ArUco 而不是 AprilTag**：`pupil_apriltags` 不认 ArUco，要换探测器（`cv2.aruco`）。
  症状就是 `--inspect` 一个都检不出来，而图上明明有方块图案。

**4. 单帧/离线跑一遍，看重投影误差**

```bash
ros2 run a_tag a-tag --camera-info src/rm_vision_ros2_hik_camera/config \
    --source frame.png --print-every 1
```

`reproj` 理想印刷板应 **< 0.5px**（本库这块板约 1.0px，见「6. 重投影误差有个物理地板」）。
涨到几个 px 先怀疑三件事：tag_size 量错、内参不是这份相机的、
输入分辨率和标定分辨率不一致（这一条命令行会直接告警）。

**5. 全链路 + 精度抽查**

```bash
ros2 launch a_tag a_tag.launch.py      # 另一个终端 ros2 topic echo /a_tag/pose
```

节点日志里 `有效 xx%` 应在 95% 以上；`stale` 长期不为 0 说明丢帧。要验绝对精度，
拿尺子量板心到相机的真实距离和打印的 `z` 比 —— 合成链路在 0.8m 处的位置误差约 0.4mm，
真实数据主要看重投影误差和**静止时的抖动**（同一位姿连续多帧的标准差）。
注意正对相机时 roll/pitch 是弱自由度，别拿它当精度指标。

## 可视化 / 排障

节点可以把自己"看到"的东西发成一个图像话题（默认关）。**没有位姿时也发** —— 画面里
一个 tag 都没有、或者 tag 不够 `min_tags` 时照样能看到图和原因，这正是最需要看图的时候：

```bash
ros2 launch a_tag a_tag.launch.py publish_debug_image:=true
# 运行中也可以随时开关（不用重启）：
ros2 param set /a_tag publish_debug_image true

# 看图（三选一）：
ros2 run image_tools showimage --ros-args -r image:=/a_tag/debug_image
ros2 run rqt_image_view rqt_image_view          # 选 /a_tag/debug_image，transport 必须是 raw
# RViz2: Add -> By topic -> /a_tag/debug_image -> Image
```

⚠️ 四个容易白折腾半天的点：

1. **本话题是裸 `sensor_msgs/Image`，没有 `/compressed` 子话题**（我们没用 image_transport
   发布）。rqt_image_view 的 transport 下拉框如果停在 `compressed` / `compressedDepth`，
   就永远看不到图（切到别的相机话题时那个下拉框是有效的，所以很容易被带偏）—— **选 `raw`**。
2. **QoS**：debug 图默认用 reliable 发（`debug_use_sensor_data_qos: false`），因为
   rqt 和 showimage 默认都按 reliable 订，best-effort 发它们收不到。
   要改成 best-effort，观看端也得跟着改（见下面第 4 条）。
3. **带宽**：debug 图是裸 bgr8，1440×1080 一帧 **4.6MB**，60fps 全发就是 ~290MB/s；
   订阅端跟不上时 reliable 发布还会反过来堵住发布者（DDS 的 max_blocking_time）。
   所以做了三层保护：
   * **限速**：默认 `debug_period: 0.2`（5Hz）——**所以默认看图会一顿一顿的**，
     想要流畅就 `debug_period:=0`（每帧）或 `0.05`（20Hz），再配 `debug_scale:=0.5` 省带宽
     （1440×1080：4.7MB/帧 → 1.2MB/帧；60fps 时 300MB/s → 75MB/s）；
   * **不在图像回调里发**：画图+发布在独立回调组/线程（`MultiThreadedExecutor`）；
   * **自适应**：`debug_adaptive: true` 时，如果"发一帧"超过 50ms（观看端跟不上），
     自动把间隔拉到耗时的 3 倍（最多 5s 一张），并在日志里告警。
   三个开关都是每帧读的，运行中可调：`ros2 param set /a_tag publish_debug_image true`、
   `ros2 param set /a_tag debug_period 1.0`。
4. **想看又不掉帧**：把 debug 图改成 best-effort 发，发布者**永远不会被堵**
   （代价是过载时丢帧，反正是给人看的）：
   ```bash
   ros2 launch a_tag a_tag.launch.py publish_debug_image:=true debug_use_sensor_data_qos:=true
   ros2 run image_tools showimage --ros-args -p reliability:=best_effort -r image:=/a_tag/debug_image
   # RViz 的 Image display：Reliability Policy 选 Best effort
   ```

图上画了四样东西：

| 画的东西 | 颜色 | 含义 |
|---|---|---|
| tag 四边形 + 角点序号 + `id=` | 橙黄（不在 target.yaml 里的 id 是**红**） | 检测结果 |
| 板子外框 | 蓝 | 用解出的位姿反投影出来的板框 |
| 每个 tag 的四个角点 | 品红小点 | 按位姿反投影的角点 |
| 左上角文字 | 白 | 和日志同一份：tags / reproj / 有效帧率 / fps / x,y,z / rpy / STALE |

**怎么用它判断问题**：品红反投影点应该**正好落在**橙黄检测角点上，板框应该贴合实际板子边缘。
如果品红整体偏出去一大截 → 就是上一条说的布局/内参问题（用 `--check-layout` 反推）；
如果个别 tag 的品红点转了 90/180° → 那个 tag 贴反了，改 `tag_rotations`。

开关在 `config/a_tag_params.yaml` 的 `publish_debug_image`（也可以像上面那样用 launch 参数覆盖）。

## 重新做一块板（`tools/make_board.py`）

手工把四个 tag 分别贴上去，很容易贴转 —— a_tag 的角点对应关系是按"所有 tag 朝向一致"写的，
贴转一个就会 reproj 暴涨。所以直接打印一整张板面，朝向由程序保证：

```bash
cd src/a_tag
# 输出到工作区里**可见**的 board/ 目录（别用 .scratch，它是以点开头的隐藏目录）
python3 tools/make_board.py --out ../board/board_4tag_40mm_98x100              # 复刻原布局
python3 tools/make_board.py --out ../board/board_4tag_40mm_120x120 \\
    --dx 120 --dy 120                                                          # 铺得更开
```

输出三件：`<out>.pdf`（A4 页面、板面按实际尺寸居中，**直接打印，缩放选 100%/实际大小**）、
`<out>.svg`（毫米单位的矢量，浏览器/Inkscape 按 100% 打印）、`<out>.png`（300dpi 预览）。
图上带：四个 tag、`id` 标号、每个 tag 的"↑朝向"提示、+X/+Y 轴、原点十字，以及一根
**100mm 校验条**。

打印后必做：

1. 卡尺量那根 100mm 校验条 —— 不是 **100.0 ± 0.5 mm** 就重打（打印机默认缩放常见 1~2%，
   会直接让位姿整体缩放）；
2. 把纸**贴平**（别起翘）：tag 各自倾斜 1~3° 就是 5~8px 的重proj 地板，这是手工板的通病；
3. 换上这块板后，`target.yaml` 里的 `tag_rotations` **整段删掉**（标准贴法不需要旋转修正）。

铺得越开精度越好：角度误差大致随「中心距 / tag 边长」这个比例改善（实测 34x24cm 布局比
14x11cm 好 4~5 倍）。不过 A4 上建议别超过 `--dx 120 --dy 120`。

## 板面自标定（`--self-calibrate`）

把"板面几何"和"每个视角的位姿"放进**同一个最小二乘**联合优化，用一串多视角画面把
`tag_centers` / `tag_rotations` 解出来，并给出体检报告。合成数据实测（真值已知）：
12 视角、角点噪声 0.3px、板还有 1.6~2.4° 的贴装翘曲 —— 中心距恢复到 **0.06~0.09 mm**，
贴装旋转正确取整（180° 那个也能认出来），残差 RMS 0.24px。

**唯一的硬前提**：必须给一个真实长度做尺度锚（单目尺度不可观测）：

```bash
source install/setup.bash
ros2 run a_tag a-tag --self-calibrate --topic /image_raw \
    --camera-info src/rm_vision_ros2_hik_camera/config \
    --tag-size 0.0400 \            # ← 卡尺量**打印出来的**黑框外沿，单位米
    --out src/a_tag/config/target.yaml
```

运行时会让你"举着板换姿势"，它自动挑姿势分散的帧（4 个 tag 都在、和已收的差 >35px）：

| 要覆盖 | 建议 |
|---|---|
| 视角数 | 15~30 个（最少 6 个，少于 10 个别太当真） |
| 距离 | 0.3 / 0.5 / 0.8 / 1.0 m 各来几档 |
| 倾角 | 板面与像面夹角 0°、±15°、±30°、±45° |
| 面内旋转 | 绕光轴 ±45° 转几档 |
| 画面位置 | 板出现过在画面中心、四角（主点/畸变才被约束） |
| tag 边长 | ≥ 80 px（`--inspect` 会报） |
| 曝光 | **关自动曝光**、别过曝、别糊（动的时候停一下再采） |

输出：

* 可直接覆盖的 `board:` 段（`tag_centers` + 非零的 `tag_rotations`），加 `--out` 直接写文件，
  并给出建议的 `max_reproj_px = 3 × 残差`；
* **体检报告**：每个 tag 的平面外偏移 `dz`、自身倾斜角、面内旋转（含离 90° 的残差）、
  位置 1σ；整体角点残差；初值异常的帧会被自动剔除并报告；
* `--refine-intrinsics`：顺便优化 fx/fy/cx/cy —— 会直接告诉你那份 `camera_info.yaml`
  偏了多少（偏很多就说明标定不是这台相机的，或者分辨率不一致）。

注意：`--tag-size` 只能来自卡尺。图像**推不出绝对尺度**（把板、tag、相机距离同时放大
k 倍产生的图像完全一样，重投影也一样），所以量错 tag_size = 位姿整体缩放，且不会报错。

## 模块

| 文件 | 职责 |
|---|---|
| `se3.py` | SE(3) / 欧拉角 / 四元数工具 |
| `board.py` | 板定义与 `CORNER_SIGNS`（角点顺序） |
| `detector.py` | AprilTag 检测封装 |
| `pose.py` | PnP 求解、重投影健康检查、求逆 |
| `filter.py` | one-euro 平移滤波 + 四元数 slerp 旋转滤波 |
| `config.py` | 配置加载（target.yaml + 可选内参） |
| `viz.py` | 把检测框 / 板框投影 / 日志文字叠加到图像上（节点 debug 图用） |
| `camera.py` | `FrameSource`（图片/视频）/ `RosFrameSource`（ROS 话题）+ 编码转换 |
| `inspect.py` | tag 诊断：列 ID/家族/质量/像素尺寸，推 `tag_centers`（`--inspect` 用它） |
| `ros_node.py` | ROS 2 节点（**唯一**常驻 import rclpy 的模块） |
| `main.py` | 命令行：诊断 / 求解（`ros2 run a_tag a-tag`） |
| `check_chain.py` | 链路自检：图像 → 内参 → 位姿 → TF 逐跳查（`ros2 run a_tag a-tag-check`） |

可视化分两层：节点能发一张叠加图（`/a_tag/debug_image`，默认关，见「可视化 / 排障」）；
命令行本身**不带窗口**，看画面走 `rqt_image_view`，看文件用看图软件。

## 已知限制 / 下一步

* 没有实时预览器：本包不带窗口、不直连相机、不写视频。要看画面请走 ROS（`rqt_image_view`
  或 RViz 的 Image display）或看图软件；回归精度目前靠实测 bag 重放（本包不带测试）。
* `a_tag` 自己直连海康相机（MVS SDK）还没实现：图像由
  `src/rm_vision_ros2_hik_camera`（MVS SDK 取流 → `image_transport` 发布）提供，
  这也顺便让内参走 `/camera_info` 话题，不用在本包重复存一份。将来真要做直连后端，
  再给 `AppConfig` 加设备参数（曝光/增益）即可，位置在命令行侧而不是 ROS 侧。
* `detector: aruco`（当前配置）在 1440×1080 tag16h5 上实测节点 60 fps、命令行 56 fps，
  实时不是瓶颈。`detector: pupil` 走 libapriltag，慢得多，而且 16h5 本来就不该走它
  （见「7. tag16h5 别用 libapriltag」）。真要再提速就调 `tracking.nthreads`、把
  `quad_decimate` 放到 2.0（角点精度约减半，仅 pupil 后端有效）或在相机侧裁 ROI。
