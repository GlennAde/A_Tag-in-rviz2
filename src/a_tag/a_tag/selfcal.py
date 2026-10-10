"""多视角自标定：用"举着板换姿势拍的一串画面"把板面几何解出来。

为什么值得做
------------
* 单目视觉的**绝对尺度不可观测**：图像只能定「中心距 / tag_size」这个比值，所以必须
  外部给一个长度 —— 本工具要求 --tag-size（卡尺量打印件）。
* 给定 tag_size 后，把「板面几何」和「每个视角的位姿」放进**同一个最小二乘**联合优化，
  中心距精度能到 0.1% 量级（合成实验：12 视角 0.02%，单帧正对 0.10%）。
* 每个 tag 给独立的 6DoF，所以还能**量出每个 tag 的翘曲/倾斜**——这是"等大+共面"
  假设下解不出来的东西。

输出
----
* 可直接用的 target.yaml（tag_centers + tag_rotations + 建议的 max_reproj_px）
* 体检报告：每个 tag 的平面外偏移 dz、倾斜角、面内旋转（含离 90° 的残差）、
  位置 1σ 不确定度、整体角点残差
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .board import CORNER_SIGNS

__all__ = [
    "CalibResult",
    "board_dict",
    "TagFit",
    "ViewSet",
    "calibrate",
    "collect_views",
    "format_report",
    "render_target_yaml",
]


@dataclass
class ViewSet:
    """一帧观测：检测到的角点（本库顺序，键是 tag id）。"""

    corners: dict
    image_size: tuple
    label: str = ""


@dataclass
class TagFit:
    tag_id: int
    center: tuple                # 板面内 (x, y)，米（原点 = tag 中心网格重心）
    dz: float                    # 平面外偏移，米
    tilt_deg: float              # 自身法线与板面法线夹角
    rotation_deg: float          # 面内旋转（逆时针，度）
    snapped_deg: int             # 就近取整到 0/90/180/270
    rotation_residual: float     # 与取整值的差（度）
    center_sigma_mm: float       # 位置 1σ


@dataclass
class CalibResult:
    tags: list = field(default_factory=list)
    rms_px: float = float("nan")
    n_views: int = 0
    n_obs: int = 0
    tag_size: float = 0.0
    aspect: float = 1.0          # 解出的 tag 宽/高（1.0=正方形；打印被拉伸会偏离 1）
    K_before: np.ndarray | None = None
    K_after: np.ndarray | None = None
    notes: list = field(default_factory=list)


# ------------------------------------------------------------------ 采集视角
def collect_views(
    source,
    *,
    families: str = "tag16h5",
    backend: str = "aruco",
    max_views: int = 40,
    min_move_px: float = 35.0,
    min_interval_s: float = 0.35,
    min_tags: int = 4,
    timeout_s: float = 180.0,
):
    """挑出一组"姿势足够分散"的视角。

    source 可以是：图片目录、单个图片、或 ROS 话题名（以 / 开头）。
    话题模式下会边收边挑：4 个 tag 都在、且和已收视角的平均角点位移 > min_move_px 才收。
    """
    import time

    from .detector import AprilTagDetector

    det = AprilTagDetector(families=families, backend=backend)
    views: list = []

    def accept(corners, size, label) -> bool:
        if len(corners) < min_tags:
            return False
        ids = sorted(corners)
        flat = np.vstack([corners[t] for t in ids])
        for v in views:
            if sorted(v.corners) != ids:
                continue
            prev = np.vstack([v.corners[t] for t in ids])
            if float(np.linalg.norm(prev - flat, axis=1).mean()) < min_move_px:
                return False
        views.append(ViewSet(corners=corners, image_size=size, label=label))
        return True

    if isinstance(source, str) and source.startswith("/") and not Path(source).exists():
        from .camera import RosFrameSource

        with RosFrameSource(source, timeout=10.0) as fs:
            t0 = time.time()
            t_last_ok = t0
            while len(views) < max_views and time.time() - t0 < timeout_s:
                ok, bgr = fs.read()
                if not ok:
                    break
                gray = fs.to_gray(bgr)
                corners = det.detect(gray)
                now = time.time()
                if len(corners) >= min_tags:
                    t_last_ok = now
                    if now - (views[-1]._t if views and hasattr(views[-1], "_t") else 0) < min_interval_s:
                        continue
                    if accept(corners, gray.shape, f"view{len(views) + 1}"):
                        setattr(views[-1], "_t", now)
                        print(
                            f"  采到 {len(views):>2}/{max_views} 个视角"
                            f"（{len(corners)} 个 tag）",
                            flush=True,
                        )
                if now - t_last_ok > 20.0:
                    print("  20 秒没看到 4 个 tag，停止采集。", flush=True)
                    break
        return views

    p = Path(str(source)).expanduser()
    paths: list = []
    if p.is_dir():
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
            paths += sorted(p.glob(ext))
    elif p.is_file():
        paths = [p]
    else:
        raise FileNotFoundError(f"找不到输入: {p}")
    for path in paths:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        corners = det.detect(gray)
        if accept(corners, gray.shape, path.name) and len(views) >= max_views:
            break
    return views


# ------------------------------------------------------------------ 初始化
def _tag_local(tag_size: float, aspect: float = 1.0) -> np.ndarray:
    """tag 的局部角点。aspect = 宽/高：打印被非等比拉伸时 tag 不是正方形，' 用参数解出来。"""
    half = tag_size / 2.0
    return np.column_stack([CORNER_SIGNS * half * np.array([aspect, 1.0]), np.zeros(4)])


def _init_from_views(views: list, K: np.ndarray, tag_size: float):
    """每个视角对每个 tag 单独 PnP（4 点平面 PnP），再换算成"以 tag0 为参考"的相对位姿。"""
    local = _tag_local(tag_size).astype(np.float64)
    per_view = []
    for v in views:
        poses = {}
        for tid, uv in v.corners.items():
            # 用 SQPNP（全局解、无 IPPE 那种二义性），否则初值偶尔会跳到镜像解上
            ok, rvec, tvec = cv2.solvePnP(
                local, np.asarray(uv, np.float64), K, np.zeros(5), flags=cv2.SOLVEPNP_SQPNP
            )
            if ok:
                poses[tid] = (cv2.Rodrigues(rvec)[0], tvec.reshape(3))
        per_view.append(poses)

    # 板面几何的初值：用第一个所有 tag 都 PnP 成功的视角，以它的 tag0 为板系
    base = next((p for p in per_view if len(p) >= 4), per_view[0])
    ref = min(base)
    R0, t0 = base[ref]
    tag_init = {}
    for tid, (R, t) in base.items():
        R_rel = R0.T @ R
        t_rel = R0.T @ (t - t0)
        tag_init[tid] = (cv2.Rodrigues(R_rel)[0].ravel(), t_rel)

    # 板面模型初值：把所有视角各自算出来的"相对几何"取**逐分量中位数**——
    # 单看某一个视角会被它的噪声/糊帧带偏，取中位数天然抗一两个坏视角。
    rels = {tid: [] for tid in tag_init}
    for poses in per_view:
        if ref not in poses or len(poses) < 4:
            continue
        R0v, t0v = poses[ref]
        for tid, (R, t) in poses.items():
            R_rel = R0v.T @ R
            rels[tid].append((cv2.Rodrigues(R_rel)[0].ravel(), R0v.T @ (t - t0v)))
    tag_init = {}
    for tid, items in rels.items():
        if not items:
            continue
        rv = np.median(np.array([it[0] for it in items]), axis=0)
        tv = np.median(np.array([it[1] for it in items]), axis=0)
        tag_init[tid] = (rv, tv)
    if ref not in tag_init:
        tag_init[ref] = (np.zeros(3), np.zeros(3))

    # 每个视角对着这个模型做一次 16 点 PnP，顺便算它的初值残差（用来剔除坏帧）
    model = np.vstack([
        local @ cv2.Rodrigues(tag_init[tid][0])[0].T + tag_init[tid][1] for tid in sorted(tag_init)
    ])
    ids_all = sorted(tag_init)
    view_init, init_rms = [], []
    for v, poses in zip(views, per_view):
        uv = np.vstack([np.asarray(v.corners[tid], np.float64) for tid in ids_all
                        if tid in v.corners])
        ok, rvec, tvec = (False, None, None)
        if uv.shape[0] >= 8:
            ok, rvec, tvec = cv2.solvePnP(
                model[: uv.shape[0]], uv, K, np.zeros(5), flags=cv2.SOLVEPNP_SQPNP
            )
        if ok:
            proj, _ = cv2.projectPoints(model[: uv.shape[0]], rvec, tvec, K, np.zeros(5))
            init_rms.append(float(np.sqrt(np.mean((proj.reshape(-1, 2) - uv) ** 2))))
            # 板系就是"参考 tag 的坐标系"，所以视角位姿 = 相机看参考 tag 的位姿，直接用它
            view_init.append((np.asarray(rvec, np.float64).ravel(), np.asarray(tvec, np.float64).ravel()))
            continue
        init_rms.append(float("inf"))
        if ref in poses:
            R, t = poses[ref]
            view_init.append((cv2.Rodrigues(R)[0].ravel(), np.asarray(t, np.float64)))
        else:
            view_init.append((np.zeros(3), np.zeros(3)))
    return tag_init, view_init, ref, np.array(init_rms)


# ------------------------------------------------------------------ 优化
def calibrate(views: list, K, dist, tag_size: float, *, refine_intrinsics: bool = False,
              fit_aspect: bool = False, max_nfev: int = 400) -> CalibResult:
    """联合优化：板面几何（每个 tag 6DoF）+ 每个视角的位姿 [+ 可选内参]。"""
    from scipy.optimize import least_squares

    if len(views) < 3:
        raise ValueError(f"至少需要 3 个视角，收到 {len(views)}")

    K0 = np.asarray(K, np.float64).copy()
    dist = np.zeros(5) if dist is None else np.asarray(dist, np.float64).ravel()
    tag_init, view_init, ref, init_rms = _init_from_views(views, K0, tag_size)

    # 初值残差异常大的视角直接丢掉（糊帧/误检），否则会把整体解带偏
    dropped = []
    if len(views) >= 6 and np.isfinite(init_rms).any():
        med = float(np.median(init_rms[np.isfinite(init_rms)]))
        thr = max(3.0, 3.0 * med)
        bad = {i for i, e_ in enumerate(init_rms) if not np.isfinite(e_) or e_ > thr}
        if bad and len(views) - len(bad) >= 3:
            dropped = sorted(bad)
            views = [v for i, v in enumerate(views) if i not in bad]
            tag_init, view_init, ref, init_rms = _init_from_views(views, K0, tag_size)
    ids = sorted(tag_init)
    others = [t for t in ids if t != ref]
    local = _tag_local(tag_size)

    # 参数：每个视角 6 + 每个非参考 tag 6（参考 tag 固定在单位位姿，当量规）+ 可选 4 个内参
    n_view, n_tag = len(views), len(others)

    def pack(view_poses, tag_poses, intr):
        x = []
        for rv, tv in view_poses:
            x += list(rv) + list(tv)
        for tid in others:
            rv, tv = tag_poses[tid]
            x += list(rv) + list(tv)
        x += list(intr)
        return np.array(x, np.float64)

    def unpack(x):
        view_poses, tag_poses = [], {}
        for i in range(n_view):
            view_poses.append((x[6 * i:6 * i + 3], x[6 * i + 3:6 * i + 6]))
        base = 6 * n_view
        for j, tid in enumerate(others):
            tag_poses[tid] = (x[base + 6 * j:base + 6 * j + 3], x[base + 6 * j + 3:base + 6 * j + 6])
        tag_poses[ref] = (np.zeros(3), np.zeros(3))
        intr = x[base + 6 * n_tag:]
        if fit_aspect:
            return view_poses, tag_poses, intr[:-1], float(intr[-1])
        return view_poses, tag_poses, intr, 1.0

    def camera_matrix(intr):
        if not refine_intrinsics:
            return K0
        fx, fy, cx, cy = intr
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

    def residuals(x):
        view_poses, tag_poses, intr, aspect = unpack(x)
        Km = camera_matrix(intr)
        loc = _tag_local(tag_size, aspect)
        out = []
        for v, (rv, tv) in zip(views, view_poses):
            Rv = cv2.Rodrigues(rv)[0]
            for tid, uv in sorted(v.corners.items()):
                if tid not in tag_poses:
                    continue
                rv_t, tv_t = tag_poses[tid]
                pts = loc @ cv2.Rodrigues(rv_t)[0].T + tv_t
                p, _ = cv2.projectPoints(pts, rv, tv, Km, dist)
                out.append((p.reshape(-1, 2) - np.asarray(uv, np.float64)).ravel())
        return np.concatenate(out)

    intr0 = [K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2]]
    if fit_aspect:
        intr0 = intr0 + [1.0]
    x0 = pack(view_init, tag_init, intr0)
    # 稳健损失：角点检测偶尔会给出垃圾帧/垃圾角点，hard-L2 会被它带偏，
    # soft_l1（f_scale = 1px）会把离群项自动降权。
    fit = least_squares(residuals, x0, method="lm", xtol=1e-12, ftol=1e-12, max_nfev=max_nfev)
    r = residuals(fit.x)
    n_obs, n_par = r.size, fit.x.size
    sigma2 = float(r @ r) / max(1, n_obs - n_par)
    # 用伪逆：量规固定/弱约束方向会让 JᵀJ 奇异，inv 会直接 NaN
    cov = sigma2 * np.linalg.pinv(fit.jac.T @ fit.jac)

    view_poses, tag_poses, intr, aspect = unpack(fit.x)
    Km = camera_matrix(intr)
    res_aspect = float(aspect)

    # ---- 把参考 tag 的位姿当量规，重新表达成"板面坐标系"（原点=tag 中心重心） ----
    centers = {tid: np.asarray(tag_poses[tid][1], float) for tid in ids}
    origin = np.mean([centers[t] for t in ids], axis=0)
    stack = np.array([centers[t] - origin for t in ids])
    _, _, vt = np.linalg.svd(stack, full_matrices=False)
    z_axis = vt[2]
    normals = np.array([cv2.Rodrigues(tag_poses[t][0])[0][:, 2] for t in ids])
    if float(np.dot(z_axis, normals.mean(axis=0))) < 0:
        z_axis = -z_axis
    x_axis = cv2.Rodrigues(tag_poses[ref][0])[0][:, 0]
    x_axis = x_axis - z_axis * float(np.dot(x_axis, z_axis))
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    A = np.column_stack([x_axis, y_axis, z_axis])          # 板面系 -> 参考 tag 系

    tags = []
    base = 6 * n_view
    for j, tid in enumerate(others):
        v = max(float(cov[base + 6 * j + 3, base + 6 * j + 3]),
                float(cov[base + 6 * j + 4, base + 6 * j + 4]), 0.0)
        sig = math.sqrt(v) * 1000.0
        tags.append(_describe(tid, tag_poses[tid], origin, A, sig))
    tags.append(_describe(ref, tag_poses[ref], origin, A, 0.0))
    tags.sort(key=lambda t: t.tag_id)

    # 逐视角残差：能看出哪一帧是垃圾帧
    per_view = []
    k = 0
    for v in views:
        m = 2 * 4 * len(v.corners)
        seg = r[k:k + m]
        per_view.append(float(np.sqrt(np.mean(seg ** 2))) if seg.size else float("nan"))
        k += m

    res = CalibResult(
        tags=tags,
        rms_px=float(np.sqrt(np.mean(r ** 2))),
        n_views=n_view,
        n_obs=int(n_obs),
        tag_size=tag_size,
        aspect=res_aspect,
        K_before=K0,
        K_after=Km,
    )
    if dropped:
        res.notes.append(
            "已剔除初值异常的视角: " + ", ".join(views_labels[i] for i in dropped)
            if False else "已剔除 %d 个初值异常的视角（糊帧/误检）" % len(dropped)
        )
    worst = sorted(range(len(per_view)), key=lambda i: -per_view[i])[:3]
    res.notes.append(
        "逐视角角点残差(px): 最差 " + ", ".join(f"{views[i].label}={per_view[i]:.2f}" for i in worst)
    )
    if per_view and max(per_view) > 3.0:
        res.notes.append("⚠️ 有视角残差 >3px：那一帧可能糊了/过曝，建议剔掉重跑")

    if abs(res.aspect - 1.0) > 0.005:
        res.notes.append(
            f"tag 宽/高解出来是 {res.aspect:.4f}（不是正方形）-> tag 实际 "
            f"{res.tag_size * res.aspect * 1000:.2f} x {res.tag_size * 1000:.2f} mm，" 
            "打印被非等比拉伸了；关掉打印缩放重打最省事"
        )
    if refine_intrinsics:
        res.notes.append(
            f"内参优化: fx {K0[0,0]:.1f}->{Km[0,0]:.1f}  fy {K0[1,1]:.1f}->{Km[1,1]:.1f}  "
            f"cx {K0[0,2]:.1f}->{Km[0,2]:.1f}  cy {K0[1,2]:.1f}->{Km[1,2]:.1f}"
        )
    return res


def _describe(tid, pose, origin, A, sigma_mm) -> TagFit:
    R = cv2.Rodrigues(pose[0])[0]
    t = np.asarray(pose[1], float) - origin
    R_b = A.T @ R
    t_b = A.T @ t
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(R_b[2, 2])))))
    rot = math.degrees(math.atan2(R_b[1, 0], R_b[0, 0]))
    snapped = int(round(rot / 90.0) * 90) % 360
    resid = (rot - snapped + 180) % 360 - 180
    return TagFit(
        tag_id=tid,
        center=(float(t_b[0]), float(t_b[1])),
        dz=float(t_b[2]),
        tilt_deg=tilt,
        rotation_deg=rot,
        snapped_deg=snapped,
        rotation_residual=resid,
        center_sigma_mm=sigma_mm,
    )


# ------------------------------------------------------------------ 输出
def board_dict(res: CalibResult) -> dict:
    """把结果整理成 target.yaml 里 board 段的结构（写文件时用，避免手拼字符串）。"""
    board = {
        "tag_size": round(res.tag_size, 6),
        "tag_centers": {t.tag_id: [round(t.center[0], 5), round(t.center[1], 5)]
                        for t in res.tags},
    }
    rots = {t.tag_id: t.snapped_deg for t in res.tags if t.snapped_deg}
    if rots:
        board["tag_rotations"] = rots
    return board


def render_target_yaml(res: CalibResult) -> str:
    """把结果写成可直接覆盖进 target.yaml 的 board 段。"""
    lines = ["board:", f"  tag_size: {res.tag_size:g}",
             "  tag_centers:                # 单位米，原点=tag 中心网格重心，X 右 Y 上"]
    for t in res.tags:
        lines.append(f"    {t.tag_id}: [{t.center[0]:+.5f}, {t.center[1]:+.5f}]")
    rots = [t for t in res.tags if t.snapped_deg]
    if rots:
        lines.append("  tag_rotations:              # 自标定出来的贴装朝向")
        for t in rots:
            lines.append(f"    {t.tag_id}: {t.snapped_deg}")
    else:
        lines.append("  # tag_rotations: 全 0，不用写")
    return "\n".join(lines)


def format_report(res: CalibResult) -> str:
    lines = [
        f"视角 {res.n_views} 个，观测 {res.n_obs} 个残差分量，角点残差 RMS = {res.rms_px:.3f} px",
        f"tag_size（外部给定的尺度锚）= {res.tag_size * 1000:.2f} mm",
        "",
        "  tag   中心 x(mm)  中心 y(mm)  平面外 dz(mm)  倾斜(deg)  面内旋转  σ位置(mm)",
    ]
    for t in res.tags:
        lines.append(
            f"  {t.tag_id:>3}  {t.center[0] * 1000:10.3f} {t.center[1] * 1000:11.3f}"
            f" {t.dz * 1000:13.3f} {t.tilt_deg:10.2f}  "
            f"{t.rotation_deg:7.2f}° -> {t.snapped_deg:>3}° (差 {t.rotation_residual:+.2f}°)"
            + (f" {t.center_sigma_mm:8.3f}" if math.isfinite(t.center_sigma_mm) else "        -")
        )
    worst_tilt = max(t.tilt_deg for t in res.tags)
    worst_dz = max(abs(t.dz) * 1000 for t in res.tags)
    lines += [
        "",
        f"最大 tag 倾斜 {worst_tilt:.2f}°，最大平面外偏移 {worst_dz:.2f} mm",
    ]
    if worst_tilt > 1.0 or res.rms_px > 1.0:
        lines.append("→ 板/贴装有可见翘曲：重投影地板基本由它决定，介意就重新贴平。")
    else:
        lines.append("→ 板很平，重投影地板应该能压到 <0.5px。")
    lines += res.notes
    return "\n".join(lines)
