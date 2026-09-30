# a_tag

用一块**共面四 AprilTag 标定板**估计相机相对板的 6DoF 位姿。

主输出是 `T_board_cam` —— **相机在板坐标系下的位姿**（x, y, z, roll, pitch, yaw）。

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
- `T_board_cam`：**主输出**，等于 `T_cam_board` 的逆

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

## 安装

```bash
uv sync
```

## 快速开始

命令行：

```bash
uv run a-tag --config config.yaml              # 摄像头实时预览
uv run a-tag --source board.png --no-preview   # 单张图片
uv run a-tag --source video.mp4 --save-video out.mp4
```

作为库：

```python
from a_tag import AppConfig, AprilTagDetector, solve_board_pose

cfg = AppConfig.load("config.yaml")
det = AprilTagDetector(expected_ids=cfg.board.tag_ids)

corners = det.detect(gray)                     # {tag_id: (4,2) 角点}
result = solve_board_pose(corners, cfg.board,
                          cfg.camera_matrix, cfg.dist_coeffs)

if result.ok:
    print(result.pose())                       # 相机在板系下的 x,y,z,rpy
    print(result.distance, result.reproj_rmse_px)
else:
    print("无效:", result.reason)
```

## 配置

`board` 段必须填**真实尺寸**，否则位姿会整体缩放或偏斜。见
`src/a_tag/default_config.yaml`。

```yaml
board:
  tag_size: 0.04          # tag 黑框边长，米（不含外侧白边）
  tag_centers:            # 各 tag 中心在板坐标系下的 [x, y]，米
    0: [-0.15, -0.10]
    1: [ 0.15, -0.10]
    2: [ 0.15,  0.10]
    3: [-0.15,  0.10]
```

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
这条由 `tests/test_board.py` 钉死。

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

正常应 **< 0.5px**。持续超标说明 tag_size、相机内参或角点顺序有问题。
超过 `max_reproj_px` 时 `result.ok` 为 `False`。

### 5. 检测降采样

libapriltag 的 `quad_decimate` 默认是 2.0，会先把图像降采样一半再找四边形，
角点精度大约减半。本库默认设为 **1.0**。

## 测试

本机装了 ROS 2，其 pytest 插件在加载期会因缺 `lark` 报错，需要禁用插件自动加载：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
```

测试用 `cv2.aruco` 渲染**真实 tag36h11 图案**做端到端校验（`tests/synthetic.py`），
真值位姿已知，无需相机。

## 模块

| 文件 | 职责 |
|---|---|
| `se3.py` | SE(3) / 欧拉角 / 四元数工具 |
| `board.py` | 板定义与 `CORNER_SIGNS`（角点顺序） |
| `detector.py` | AprilTag 检测封装 |
| `pose.py` | PnP 求解、重投影健康检查、求逆 |
| `filter.py` | one-euro 平移滤波 + 四元数 slerp 旋转滤波 |
| `camera.py` | 采集封装 |
| `config.py` | YAML 配置 |
| `viz.py` | 叠加显示 |
| `app.py` | 命令行主循环 |
