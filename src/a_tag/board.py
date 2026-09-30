"""AprilTag 标定板定义。

板坐标系 (board frame)
----------------------
* 原点: 板面中心
* X 轴: 板面内向右
* Y 轴: 板面内向上
* Z 轴: 垂直板面向外，指向观察者(相机)

板面即 Z = 0 平面，所以 4 个 tag 的 16 个角点 3D 坐标都落在这个平面上。

.. warning::
   平面 PnP 的病态方向正是 Z / pitch / roll，X / Y / yaw 一向稳定。
   实测: 正对相机时角度误差比倾斜 20 度时大 10~20 倍。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["CORNER_SIGNS", "BoardLayout"]


#: pupil_apriltags 返回的 corners[i] 在**本板坐标系**下的方向符号。
#:
#: 这是实测确定的（用 libapriltag 自带 pose 反投影校验；其 tag 坐标系为
#: X 朝左 / Y 朝上 / Z 背离相机，即 pose_R = diag(-1,-1,1)）:
#:
#:     corners[0]  图像右上  ->  (+X, +Y)
#:     corners[1]  图像左上  ->  (-X, +Y)
#:     corners[2]  图像左下  ->  (-X, -Y)
#:     corners[3]  图像右下  ->  (+X, -Y)
#:
#: 好消息: 该顺序**锁在 tag 自身的物理角上**，把 tag 在图像中旋转 90/180/270 度
#: 不会改变 corners[0] 对应的物理角。
#:
#: 顺序写错会导致解出的位姿镜像翻转，且重投影误差会从 <0.5px 暴涨到十几个 px。
#: tests/test_board.py 对此有回归测试。
CORNER_SIGNS = np.array([
    [+1.0, +1.0],
    [-1.0, +1.0],
    [-1.0, -1.0],
    [+1.0, -1.0],
], dtype=np.float64)


@dataclass(frozen=True)
class BoardLayout:
    """一块共面 AprilTag 标定板。

    Parameters
    ----------
    tag_size:
        tag 黑色边框的边长，单位**米**。AprilTag 的 tag_size 指黑色方框外沿，
        **不含**外侧白边。
    tag_centers:
        {tag_id: (x, y)}，各 tag 中心在板坐标系下的位置，单位米。

    Examples
    --------
    >>> layout = BoardLayout(tag_size=0.04,
    ...                      tag_centers={0: (-0.15, -0.10), 1: (0.15, -0.10),
    ...                                   2: (0.15, 0.10),   3: (-0.15, 0.10)})
    >>> layout.board_size
    (0.34, 0.24)
    """

    tag_size: float
    tag_centers: dict
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.tag_size <= 0:
            raise ValueError(f"tag_size 必须为正，收到 {self.tag_size}")
        if not self.tag_centers:
            raise ValueError("至少需要一个 tag")
        centers = {int(k): (float(v[0]), float(v[1]))
                   for k, v in self.tag_centers.items()}
        object.__setattr__(self, "tag_centers", centers)

    # ------------------------------------------------------------- 几何属性
    @property
    def tag_ids(self) -> tuple:
        """按 id 升序排列的 tag 列表。"""
        return tuple(sorted(self.tag_centers))

    @property
    def half(self) -> float:
        """tag 半边长，单位米。"""
        return self.tag_size / 2.0

    @property
    def board_size(self) -> tuple:
        """整块板的物理外廓 (宽, 高)，单位米。"""
        xs = [c[0] for c in self.tag_centers.values()]
        ys = [c[1] for c in self.tag_centers.values()]
        return (max(xs) - min(xs) + self.tag_size,
                max(ys) - min(ys) + self.tag_size)

    def tag_corners(self, tag_id: int) -> np.ndarray:
        """单个 tag 的 4 个角点在板坐标系下的 (4, 3) 坐标，顺序同 corners。"""
        cx, cy = self.tag_centers[tag_id]
        xy = CORNER_SIGNS * self.half + np.array([cx, cy])
        return np.column_stack([xy, np.zeros(4)])

    def object_points(self, tag_ids=None) -> tuple:
        """拼接多个 tag 的角点。

        Returns
        -------
        pts:
            (4*N, 3) 板坐标系下的角点。
        owner:
            (4*N,) 每个点所属的 tag id。
        """
        ids = self.tag_ids if tag_ids is None else tuple(tag_ids)
        pts = np.concatenate([self.tag_corners(t) for t in ids], axis=0)
        owner = np.repeat(np.array(ids, dtype=int), 4)
        return pts, owner

    # ----------------------------------------------------------- 序列化
    def to_config(self) -> dict:
        """导出为可写入 YAML 的字典。"""
        return {
            "tag_size": float(self.tag_size),
            "tag_centers": {int(k): [float(v[0]), float(v[1])]
                            for k, v in sorted(self.tag_centers.items())},
        }

    @classmethod
    def from_config(cls, data: dict) -> "BoardLayout":
        """从 YAML 解析出的字典构造。"""
        return cls(
            tag_size=float(data["tag_size"]),
            tag_centers={int(k): (float(v[0]), float(v[1]))
                         for k, v in data["tag_centers"].items()},
        )
