"""位姿时间滤波。

要点
----
* 平移用 one-euro 滤波: 低速时强平滑、高速时低延迟，比固定 alpha 的 EMA 好。
* **旋转用四元数 slerp，绝不要直接平滑 rvec 或欧拉角** —— 欧拉角在万向锁附近
  会跳变，rvec 在 +-pi 处会绕远路。
* 检测丢失时保持上一帧并标记 stale，不要突然跳回零位。
"""

from __future__ import annotations

import numpy as np

from .se3 import R_from_quat, make_T, quat_from_R, slerp

__all__ = ["OneEuroFilter", "PoseFilter"]


class OneEuroFilter:
    """一维 one-euro 滤波器。

    Parameters
    ----------
    min_cutoff:
        最小截止频率 (Hz)。越小越平滑、延迟越大。
    beta:
        速度系数。越大则快速运动时延迟越小、抖动越大。
    d_cutoff:
        速度信号的截止频率 (Hz)。
    """

    def __init__(
        self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0
    ) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x: float | None = None
        self._dx = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._x = None
        self._dx = 0.0

    def __call__(self, x: float, dt: float) -> float:
        if dt <= 0.0:
            return float(x)
        if self._x is None:
            self._x = float(x)
            return self._x
        dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = self._alpha(cutoff, dt)
        self._x = a * float(x) + (1.0 - a) * self._x
        return self._x


class PoseFilter:
    """对 4x4 位姿做时间平滑。

    用法::

        f = PoseFilter()
        T_smooth = f(T_meas, dt)        # T_meas=None 表示本帧无有效检测
    """

    def __init__(
        self, min_cutoff: float = 1.0, beta: float = 0.007, rot_cutoff: float = 1.0
    ) -> None:
        self._tx = OneEuroFilter(min_cutoff, beta)
        self._ty = OneEuroFilter(min_cutoff, beta)
        self._tz = OneEuroFilter(min_cutoff, beta)
        self.rot_cutoff = float(rot_cutoff)
        self._q: np.ndarray | None = None
        self._T: np.ndarray | None = None
        self._stale_frames = 0

    def reset(self) -> None:
        for f in (self._tx, self._ty, self._tz):
            f.reset()
        self._q = None
        self._T = None
        self._stale_frames = 0

    @property
    def stale_frames(self) -> int:
        """连续多少帧没有新观测。"""
        return self._stale_frames

    @property
    def T(self) -> np.ndarray | None:
        """最近一次平滑后的位姿，4x4。"""
        return None if self._T is None else self._T.copy()

    def __call__(self, T_meas: np.ndarray | None, dt: float) -> np.ndarray | None:
        """喂入新观测；T_meas=None 表示本帧无有效检测，沿用上一帧。"""
        if T_meas is None:
            if self._T is not None:
                self._stale_frames += 1
            return self.T

        T_meas = np.asarray(T_meas, dtype=np.float64)
        if self._T is None:  # 首帧直接采纳，避免从零收敛
            self._T = T_meas.copy()
            self._q = quat_from_R(T_meas[:3, :3])
            step = dt if dt > 0 else 1.0
            for f, v in zip((self._tx, self._ty, self._tz), T_meas[:3, 3]):
                f(float(v), step)
            self._stale_frames = 0
            return self.T

        tx = self._tx(T_meas[0, 3], dt)
        ty = self._ty(T_meas[1, 3], dt)
        tz = self._tz(T_meas[2, 3], dt)

        q_meas = quat_from_R(T_meas[:3, :3])
        a = 1.0 if dt <= 0.0 else 1.0 - np.exp(-2.0 * np.pi * self.rot_cutoff * dt)
        self._q = slerp(self._q, q_meas, a)

        self._T = make_T(R_from_quat(self._q), np.array([tx, ty, tz]))
        self._stale_frames = 0
        return self.T
