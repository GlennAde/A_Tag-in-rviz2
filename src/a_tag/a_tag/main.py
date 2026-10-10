"""命令行：图片 / 视频 / ROS 图像话题 -> AprilTag 检测 -> 打印位姿，或做 tag 诊断。

两个模式
--------
* 默认：求解位姿。需要内参（--camera-info 指相机包的标定文件，或 --config 指一个
  带 camera 段的合并 YAML）。
* --inspect：只列画面里有哪些 tag —— 不做 ID 过滤、也不依赖 target.yaml 的 tag_centers，
  换相机 / 换板 / “一个 tag 都认不到”时的第一步诊断。

输入源二选一：--source <图片或视频>，或 --topic <ROS 图像话题>（不起节点也能拿一帧）。
刻意不做窗口预览、不存视频。
"""

from __future__ import annotations

import argparse
import re
import time

import cv2
import numpy as np

from .camera import FrameSource, RosFrameSource
from .config import AppConfig, MissingIntrinsicsError
from .detector import AprilTagDetector
from .filter import PoseFilter
from .inspect import (
    DEFAULT_FAMILY,
    SUPPORTED_FAMILIES,
    annotate,
    detect_all,
    format_report,
    scan_aruco_dicts,
)
from .viz import draw_overlay

#: --family 的特殊取值：所有常见家族各起一个进程试一遍
ALL_FAMILIES_FLAG = "all"
from .board import CORNER_SIGNS, BoardLayout
from .pose import solve_board_pose
from .se3 import pose_from_T

__all__ = ["build_parser", "main", "run", "run_inspect"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="a-tag",
        description="共面四 AprilTag 标定板 -> 相机相对板的 6DoF 位姿；或列出画面里的 tag",
    )
    p.add_argument("--source", default=None, help="图片或视频文件路径")
    p.add_argument(
        "--topic",
        default=None,
        help="改成从 ROS 图像话题取帧（例如 /image_raw），需要 ROS 环境",
    )
    p.add_argument(
        "--self-calibrate",
        action="store_true",
        help="多视角自标定：举着板换 10~30 个姿势，用一串画面解出板面几何"
        "（必须给 --tag-size；详见 README「板面自标定」）",
    )
    p.add_argument(
        "--out", default=None, help="自标定结果写到这个 YAML（默认只打印）"
    )
    p.add_argument("--max-views", type=int, default=40, help="自标定最多用多少个视角")
    p.add_argument(
        "--refine-intrinsics",
        action="store_true",
        help="自标定同时优化内参（会发现你那份标定偏了多少）",
    )
    p.add_argument(
        "--check-layout",
        action="store_true",
        help="自检模式：拿一帧反推「哪个 id 在哪个角 + 每个 tag 的贴装旋转」，",
    )
    p.add_argument(
        "--inspect",
        action="store_true",
        help="诊断模式：列出检测到的 tag（id/家族/质量/像素尺寸），不按 ID 过滤",
    )
    p.add_argument(
        "--config",
        default=None,
        help="config 目录(内含 target.yaml)，或单个合并的 YAML。"
        "默认依次找 ./config/、share/a_tag/config、$A_TAG_CONFIG",
    )
    p.add_argument(
        "--camera-info",
        default=None,
        help="相机内参文件(camera_info.yaml)，或含它的目录。离线求解必须给："
        "内参归相机驱动包所有，例如 src/rm_vision_ros2_hik_camera/config",
    )
    p.add_argument(
        "--backend",
        choices=("pupil", "aruco"),
        default=None,
        help="检测后端。pupil=libapriltag（tag36h11 最稳）；aruco=OpenCV"
        "（tag16h5 和真正的 ArUco 字典用它，见 README）。默认取 target.yaml 里的 detector",
    )
    p.add_argument(
        "--family",
        default=None,
        help=f"覆盖 tag 家族；--inspect 默认 {DEFAULT_FAMILY}，求解模式默认用 target.yaml 里的。"
        " --family all 会把所有常见家族各起一个进程试一遍"
        " （libapriltag 在一个进程里轮多个家族会把堆弄坏，所以必须分进程）",
    )
    p.add_argument(
        "--tag-size",
        type=float,
        default=None,
        help="tag 黑框边长(米)，--inspect 推算布局时用；默认取 target.yaml 里的值",
    )
    p.add_argument("--min-tags", type=int, default=None, help="覆盖最少可见 tag 数")
    p.add_argument(
        "--quad-decimate",
        type=float,
        default=None,
        help="覆盖检测降采样，精度优先请用 1.0",
    )
    p.add_argument("--no-filter", action="store_true", help="关闭时间滤波")
    p.add_argument(
        "--frames", type=int, default=0, help="最多处理多少帧，0 = 不限（话题源 Ctrl-C 退出）"
    )
    p.add_argument(
        "--print-every", type=int, default=30, help="每 N 帧打印一次位姿，0 = 只打汇总"
    )
    p.add_argument(
        "--save-annotated", default=None, help="把带标注的最后一帧写成图片（排查用）"
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="话题源等一帧的超时秒数（--topic 时生效）",
    )
    return p


def _load_config(args) -> AppConfig | None:
    """读配置；--inspect 允许没有配置（那就没有期望 ID 可比）。"""
    try:
        return AppConfig.load(args.config, camera_info_path=args.camera_info)
    except FileNotFoundError as exc:
        if args.inspect:
            print(f"（没有找到 target.yaml，跳过 ID 对比：{exc}）")
            return None
        raise


def _make_source(args):
    """按 --topic / --source 造帧源。"""
    if args.topic:
        return RosFrameSource(args.topic, timeout=args.timeout)
    if not args.source:
        raise SystemExit(
            "错误: 需要 --source <图片或视频> 或 --topic <ROS 图像话题>\n"
            "      ROS 流程整条链路请用 ros2 launch a_tag a_tag.launch.py"
        )
    return FrameSource(args.source)


def _sweep_families(bgr, args) -> tuple:
    """逐个家族各起一个进程试一遍，返回 (命中的家族, 汇总文本)。

    必须分进程：见 inspect.py 里 FAMILIES 的注释，同一个进程里轮几个家族会把
    libapriltag 的堆弄坏。
    """
    import os
    import subprocess
    import sys
    import tempfile

    import cv2

    with tempfile.TemporaryDirectory() as tmp:
        frame = os.path.join(tmp, "frame.png")
        cv2.imwrite(frame, bgr)
        rows, best, best_backend = [], None, "pupil"
        for backend, name in [("pupil", f) for f in SUPPORTED_FAMILIES] + [
            ("aruco", ALL_FAMILIES_FLAG)
        ]:
            cmd = [
                sys.executable,
                "-m",
                "a_tag.main",
                "--inspect",
                "--source",
                frame,
                "--backend",
                backend,
                "--family",
                name,
            ]
            if args.config:
                cmd += ["--config", args.config]
            if args.tag_size:
                cmd += ["--tag-size", str(args.tag_size)]
            label = name if backend == "pupil" else "aruco:全部字典"
            print(f"  正在试 {label} ...", flush=True)
            try:
                out = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=120
                ).stdout
            except Exception as exc:
                rows.append(f"  {label:<20} 跑挂了（{type(exc).__name__}）")
                continue
            n, hit_name = 0, None
            for line in out.splitlines():
                m = re.match(r"\s+(\S+)\s+(\d+) 个", line)
                if m and int(m.group(2)):
                    n, hit_name = int(m.group(2)), m.group(1)
                    break
            m = re.search(r"检测到 (\d+) 个 tag", out)
            if m and int(m.group(1)):
                n, hit_name = int(m.group(1)), name
            rows.append(f"  {label:<20} {n} 个" + ("   <- 命中" if n else ""))
            if n and best is None:
                best = hit_name or name
                best_backend = backend

    lines = ["逐家族扫描（每个家族单独进程，避开 libapriltag 的多家族 bug）:"] + rows
    if best:
        lines.append(f"\n命中家族: {best}（下面按它出详细报告）")
    else:
        lines.append("\n所有家族都没有检测到 —— 多半不是家族的问题，看下面的排查项")
    return best, best_backend, "\n".join(lines)


def run_self_calibrate(cfg: AppConfig, args: argparse.Namespace) -> int:
    """多视角自标定：解出板面几何 + 体检报告，可选写出 target.yaml。"""
    from pathlib import Path as _Path

    from .selfcal import calibrate, collect_views, format_report, render_target_yaml

    if not args.tag_size:
        print("错误: 自标定必须给 --tag-size（卡尺量**打印件**黑框外沿，单位米）")
        print("      单目图像的绝对尺度不可观测，这是唯一的尺度锚。")
        return 2
    try:
        K, D = cfg.require_intrinsics()
    except MissingIntrinsicsError as exc:
        print(f"错误: {exc}")
        return 2

    if not args.topic and not args.source:
        print("错误: 需要 --topic <图像话题>（边拍边采）或 --source <图片目录>")
        return 2

    print("自标定：举着板在相机前换 10~30 个姿势（不同距离 / 倾角 / 面内旋转），每个姿势停半秒。")
    print("  要求：4 个 tag 都在画面里、tag 边长 ≥80px、曝光固定别走自动、别糊。")
    if args.source and not str(args.source).startswith("/"):
        print(f"  从目录读图: {args.source}")
    src = args.topic or args.source
    views = collect_views(
        src,
        families=cfg.families,
        backend=cfg.detector,
        max_views=args.max_views,
    )
    print(f"\n采到 {len(views)} 个可用视角")
    if len(views) < 6:
        print("视角太少（<6），姿势要换得更开一些；至少 10 个更稳。")
        return 1

    print("联合优化中（板面几何 + 每个视角位姿）...", flush=True)
    res = calibrate(
        views, K, D, float(args.tag_size), refine_intrinsics=args.refine_intrinsics
    )
    print()
    print(format_report(res))
    yaml_text = render_target_yaml(res)
    print()
    print("=== 可直接覆盖进 target.yaml 的内容 ===")
    print(yaml_text)
    if args.out:
        import yaml as _yaml

        from .selfcal import board_dict

        out = _Path(args.out).expanduser()
        off = float(np.clip(3.0 * res.rms_px, 0.5, 20.0))
        # **合并**进已有文件：只替换 board 段、更新 max_reproj_px，
        # 保留 detector / families / min_tags 这些设置（早期版本是整文件覆盖，会把这些弄丢）
        old = {}
        if out.is_file():
            old = _yaml.safe_load(out.read_text(encoding="utf-8")) or {}
        old["board"] = board_dict(res)
        tracking = old.setdefault("tracking", {})
        tracking["max_reproj_px"] = round(off, 1)
        out.write_text(
            "# 由 a-tag --self-calibrate 更新（board 段来自自标定，tracking 保留原设置）\n"
            + _yaml.safe_dump(old, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        print(f"\n已把 board 段合并写进 {out}（原来的 tracking 设置保留）")
    return 0


def run_check_layout(cfg: AppConfig, args: argparse.Namespace) -> int:
    """拿一帧，搜「哪个 id 在哪个角 + 每个 tag 的贴装旋转」，找出 reproj 最小的布局。

    症状：tags 都检测到了（tags=4），但重投影几十上百 px —— 基本都是**布局对应关系**错了
    （id 贴错角、上下镜像、某个 tag 贴反）。手改配置要试很久，这里直接把正确答案搜出来。
    """
    import itertools

    try:
        K, D = cfg.require_intrinsics()
    except MissingIntrinsicsError as exc:
        print(f"错误: {exc}")
        return 2

    with _make_source(args) as src:
        ok, bgr = src.read()
    if not ok:
        print("没拿到图像。")
        return 1
    gray = FrameSource.to_gray(bgr)
    corners = AprilTagDetector(
        families=args.family or cfg.families,
        backend=args.backend or cfg.detector,
        nthreads=cfg.nthreads,
        quad_decimate=cfg.quad_decimate if args.quad_decimate is None else args.quad_decimate,
    ).detect(gray)
    ids = sorted(corners)
    print(f"检测到 {len(ids)} 个 tag: {ids}")
    if len(ids) < 3:
        print("少于 3 个 tag 没法自检布局；先解决检测（--inspect --family all）。")
        return 1

    xs = [c[0] for c in cfg.board.tag_centers.values()]
    ys = [c[1] for c in cfg.board.tag_centers.values()]
    dx = (max(xs) - min(xs)) or 0.09
    dy = (max(ys) - min(ys)) or 0.09
    POS = [(-dx / 2, dy / 2), (dx / 2, dy / 2), (-dx / 2, -dy / 2), (dx / 2, -dy / 2)]
    NAMES = ["左上", "右上", "左下", "右下"]
    print(f"网格间距按当前 target.yaml: {dx * 1000:.1f} x {dy * 1000:.1f} mm，tag_size={cfg.board.tag_size * 1000:.1f} mm")

    results = []
    for perm in itertools.permutations(ids, len(POS)):
        for rots in itertools.product((0, 90, 180, 270), repeat=len(POS)):
            centers = {tid: POS[p] for p, tid in enumerate(perm)}
            layout = BoardLayout(
                tag_size=cfg.board.tag_size,
                tag_centers=centers,
                tag_rotations=dict(zip(perm, rots)),
            )
            r = solve_board_pose(corners, layout, K, D, min_tags=len(ids),
                                 max_reproj_px=1e9, refine=False)
            if r.T_cam_board is None:
                continue
            results.append((r.reproj_rmse_px, perm, rots))
    if not results:
        print("所有假设都解不出位姿 —— 先确认 tag_size 和内参。")
        return 1
    results.sort(key=lambda t: t[0])

    print()
    print("=== 前 5 名（reproj 越小越对）===")
    for rms, perm, rots in results[:5]:
        where = "  ".join(f"{NAMES[p]}={t}" for p, t in enumerate(perm))
        rot = ", ".join(f"{t}:{a}°" for t, a in zip(perm, rots) if a) or "无"
        print(f"  reproj {rms:8.2f}px   {where}   贴装旋转 {rot}")

    rms, perm, rots = results[0]
    tied = [r for r in results[1:4] if r[0] - rms < 1.0]
    if tied:
        print()
        print(f"⚠️ 第 2~{len(tied) + 1} 名与第 1 名只差 {tied[0][0] - rms:.2f}px：方形布局 + 每个 tag 的 90° 对称性",
              "让它们几何上等价，单帧分不出来。**必须用可视化确认**：品红反投影点要正好压在",
              "橙黄检测角点上 → ros2 launch a_tag a_tag.launch.py publish_debug_image:=true")
    layout_best = BoardLayout(
        tag_size=cfg.board.tag_size,
        tag_centers={tid: POS[p] for p, tid in enumerate(perm)},
        tag_rotations=dict(zip(perm, rots)),
    )
    print()

    # ---- 分辨率先对一下：内参只在标定分辨率下有效 ----
    h, w = gray.shape[:2]
    if (w, h) != (cfg.image_width, cfg.image_height):
        print(f"⚠️ 画面 {w}x{h}，而标定文件是 {cfg.image_width}x{cfg.image_height}"
              f"（{cfg.camera_name or '未命名'}）—— 内参对不上，先把这个统一。")

    # ---- 用图像量"中心距/tag_size"的比值：能抓出"打印了另一块板"这种静默错误 ----
    ref = max(corners, key=lambda t: float(np.linalg.norm(corners[t][0] - corners[t][1])))
    half = cfg.board.tag_size / 2.0
    Href, _ = cv2.findHomography(corners[ref].astype(np.float32),
                                 (CORNER_SIGNS * half).astype(np.float32))
    plane = {}
    for t in ids:
        c = corners[t].mean(axis=0)
        p = Href @ np.array([c[0], c[1], 1.0])
        plane[t] = p[:2] / p[2]
    xs = [v[0] for v in plane.values()]
    ys = [v[1] for v in plane.values()]
    measured = ((max(xs) - min(xs)) / cfg.board.tag_size,
                (max(ys) - min(ys)) / cfg.board.tag_size)
    configured = (dx / cfg.board.tag_size, dy / cfg.board.tag_size)
    print(f"图像实测「中心距/tag_size」= {measured[0]:.2f} x {measured[1]:.2f}"
          f"   配置里是 {configured[0]:.2f} x {configured[1]:.2f}")
    for i, axis in enumerate(("X", "Y")):
        if configured[i] > 0 and abs(measured[i] - configured[i]) / configured[i] > 0.05:
            print(
                f"  ⚠️ {axis} 方向对不上（差 {abs(measured[i] - configured[i]) / configured[i] * 100:.0f}%）："
                "打印的可能是另一块板，或 tag_size 量错了 —— 这会让位姿**静默地**整体缩放"
                f" {measured[i] / configured[i]:.2f} 倍（reproj 看不出来）"
            )

    # ---- 几何 vs 内参：平面单应残差不需要内参，可以区分两者 ----
    obj = np.vstack([layout_best.tag_corners(t)[:, :2] for t in ids]).astype(np.float32)
    img = np.vstack([corners[t] for t in ids]).astype(np.float32)
    Hmat, _ = cv2.findHomography(obj, img)
    proj_h = cv2.perspectiveTransform(obj.reshape(-1, 1, 2), Hmat).reshape(-1, 2)
    hom_rms = float(np.sqrt(np.mean(np.sum((proj_h - img) ** 2, axis=1))))
    print(f"平面拟合残差（不用内参）{hom_rms:6.2f}px   ／   用文件内参的重投影 {rms:6.2f}px")

    if rms > 3.0 and hom_rms < rms * 0.5:
        print("→ 几何是自洽的（平面单应拟合得很好），说明**内参不准**。粗搜一下主点/焦距：")
        best_int, best_r = None, rms
        for sc in (0.85, 0.92, 1.0, 1.08, 1.15):
            for dcx in (-80, -40, 0, 40, 80):
                for dcy in (-80, -40, 0, 40, 80):
                    K2 = K.copy()
                    K2[0, 0] *= sc
                    K2[1, 1] *= sc
                    K2[0, 2] += dcx
                    K2[1, 2] += dcy
                    rr = solve_board_pose(corners, layout_best, K2, D,
                                          min_tags=len(ids), max_reproj_px=1e9, refine=False)
                    if rr.T_cam_board is not None and rr.reproj_rmse_px < best_r:
                        best_r, best_int = rr.reproj_rmse_px, (sc, dcx, dcy)
        if best_int is not None and best_r < rms * 0.5:
            sc, dcx, dcy = best_int
            print(f"   把 fx,fy 乘 {sc:.2f}、主点挪 ({dcx:+d}, {dcy:+d}) px 后 reproj = {best_r:.2f}px")
            print("   → 你这份 camera_info.yaml 跟这台相机/这个分辨率对不上，建议重新标定")
    elif rms > 3.0:
        print("→ 平面单应本身也拟合不好：板子可能不平（tag 是分别贴的、有翘曲），或者角点噪声大。")
        print("   先确认曝光正常、tag 在画面里 ≥50px；实在不行把 max_reproj_px 放宽到 5~10px 先用。")

    if rms > 5.0:
        print(f"⚠️ 最好的一组 reproj 也有 {rms:.1f}px，说明不只是布局问题，先查：")
        print("   tag_size（卡尺量黑框外沿）/ 内参是不是这台相机的 / 输入分辨率与标定是否一致")
    print("把下面这段覆盖进 config/target.yaml 的 board 段：")
    print("board:")
    print(f"  tag_size: {cfg.board.tag_size}")
    print("  tag_centers:")
    for p, tid in enumerate(perm):
        print(f"    {tid}: [{POS[p][0]:+.4f}, {POS[p][1]:+.4f}]".ljust(30) + f"# {NAMES[p]}")
    nonzero = {t: a for t, a in zip(perm, rots) if a}
    if nonzero:
        print("  tag_rotations:")
        for t, a in nonzero.items():
            print(f"    {t}: {a}")
    return 0 if rms <= 5.0 else 1


def run_inspect(cfg: AppConfig | None, args: argparse.Namespace) -> int:
    """诊断模式：列 tag、和 target.yaml 的 ID 对比、推一份 tag_centers。"""
    expected_ids = cfg.board.tag_ids if cfg is not None else None
    tag_size = args.tag_size if args.tag_size else (cfg.board.tag_size if cfg else None)

    with _make_source(args) as src:
        ok, bgr = src.read()
        if not ok:
            print("没拿到图像。文件源检查路径；话题源检查话题名和发布端是否在发。")
            return 1

    import cv2

    gray = FrameSource.to_gray(bgr)
    backend = args.backend or (cfg.detector if cfg is not None else "pupil")
    # 家族默认跟着 target.yaml 走（换板时只改配置，不用记命令行参数）
    family = args.family or (cfg.families if cfg is not None else DEFAULT_FAMILY)

    if family == ALL_FAMILIES_FLAG and backend == "aruco":
        # OpenCV 这边没有 libapriltag 的堆问题，所有字典可以在本进程里一次扫完
        print("逐字典扫描（OpenCV，一个进程）:")
        for name, n in scan_aruco_dicts(gray):
            print(f"  {name:<20} {n} 个   <- 命中")
        found = scan_aruco_dicts(gray)
        if not found:
            print("  一个字典都没命中")
            return 1
        family = found[0][0]
        print(f"\n命中字典: {family}（下面按它出详细报告）")

    elif family == ALL_FAMILIES_FLAG:
        family, backend, sweep = _sweep_families(bgr, args)
        print(sweep)
        if family is None:
            print(
                format_report(
                    [], gray.shape, expected_ids=expected_ids, family="all", backend=backend
                )
            )
            if args.save_annotated:
                cv2.imwrite(args.save_annotated, bgr)
                print(f"已写出原图（没检测到任何 tag）: {args.save_annotated}")
            return 1

    hits = detect_all(
        gray,
        family=family,
        backend=backend,
        quad_decimate=args.quad_decimate if args.quad_decimate is not None else 1.0,
    )
    print(
        format_report(
            hits,
            gray.shape,
            expected_ids=expected_ids,
            tag_size=tag_size,
            family=family,
            backend=backend,
        )
    )
    if args.save_annotated:
        cv2.imwrite(args.save_annotated, annotate(bgr, hits, expected_ids))
        print(f"已写出标注图: {args.save_annotated}")
    return 0 if hits else 1


def run(cfg: AppConfig, args: argparse.Namespace) -> int:
    """求解模式：跑完整段输入，返回进程退出码（0 = 至少有一帧有效）。"""
    try:
        K, D = cfg.require_intrinsics()
    except MissingIntrinsicsError as exc:
        print(f"错误: {exc}")
        return 2

    detector = AprilTagDetector(
        families=args.family or cfg.families,
        backend=args.backend or cfg.detector,
        nthreads=cfg.nthreads,
        quad_decimate=(
            cfg.quad_decimate if args.quad_decimate is None else args.quad_decimate
        ),
        expected_ids=cfg.board.tag_ids,
    )
    pose_filter = None if args.no_filter else PoseFilter()
    min_tags = cfg.min_tags if args.min_tags is None else args.min_tags

    n_frames = 0
    n_ok = 0
    warned_size = False
    last_bgr = None
    last_corners: dict = {}
    last_result = None
    last_T = None
    t_prev = time.perf_counter()

    with _make_source(args) as src:
        while True:
            if args.frames and n_frames >= args.frames:
                break
            ok, bgr = src.read()
            if not ok:
                break

            gray = FrameSource.to_gray(bgr)
            h, w = gray.shape[:2]
            if not warned_size and (w, h) != (cfg.image_width, cfg.image_height):
                warned_size = True
                print(
                    f"警告: 输入 {w}x{h}，而标定文件是 "
                    f"{cfg.image_width}x{cfg.image_height}"
                    f"（{cfg.camera_name or '未命名'}）；内参只在标定分辨率下有效"
                )

            now = time.perf_counter()
            dt = now - t_prev
            t_prev = now

            corners = detector.detect(gray)
            result = solve_board_pose(
                corners,
                cfg.board,
                K,
                D,
                min_tags=min_tags,
                max_reproj_px=cfg.max_reproj_px,
            )
            n_frames += 1
            n_ok += int(result.ok)
            last_bgr, last_corners = bgr, corners
            last_result = result

            T = result.T_board_cam if result.ok else None
            if pose_filter is not None:
                T = pose_filter(T, dt)
            last_T = T

            if args.print_every and n_frames % args.print_every == 0:
                _print_status(n_frames, result, T)

    if args.save_annotated and last_bgr is not None:
        # 和节点发的 debug 图同一套画法：检测框 + 板框投影 + 反投影角点 + 文字
        import cv2

        lines = [f"reproj {last_result.reproj_rmse_px:.2f} px  ({n_ok}/{n_frames} 帧有效)"]
        if last_T is not None:
            p = pose_from_T(last_T, degrees=True)
            lines.append(
                f"x {p['x'] * 1000:.1f}  y {p['y'] * 1000:.1f}  z {p['z'] * 1000:.1f} mm"
            )
            lines.append(f"r {p['roll']:.1f}  p {p['pitch']:.1f}  y {p['yaw']:.1f} deg")
        elif last_result is not None and last_result.reason:
            lines.append(last_result.reason)
        out = draw_overlay(
            last_bgr,
            last_corners,
            result=last_result,
            layout=cfg.board,
            camera_matrix=K,
            dist_coeffs=D,
            lines=lines,
            expected_ids=cfg.board.tag_ids,
        )
        cv2.imwrite(args.save_annotated, out)
        print(f"已写出标注图（检测框/板框投影/文字）: {args.save_annotated}")

    if n_frames:
        print(f"\n处理 {n_frames} 帧，有效 {n_ok} 帧 ({100.0 * n_ok / n_frames:.1f}%)")
    return 0 if n_ok else 1


def _print_status(n_frames: int, result, T_board_cam) -> None:
    if T_board_cam is None:
        print(f"[{n_frames:6d}] 无位姿  tags={result.n_tags}  {result.reason}")
        return
    p = pose_from_T(T_board_cam, degrees=True)
    print(
        f"[{n_frames:6d}] tags={result.n_tags} reproj={result.reproj_rmse_px:5.3f}px "
        f"x={p['x'] * 1000:8.1f} y={p['y'] * 1000:8.1f} z={p['z'] * 1000:8.1f} mm  "
        f"rpy=({p['roll']:6.2f},{p['pitch']:6.2f},{p['yaw']:6.2f}) deg"
    )


def main(argv: list[str] | None = None) -> int:
    """命令行入口（console_scripts: a-tag）。"""
    args = build_parser().parse_args(argv)
    cfg = _load_config(args)
    if args.self_calibrate:
        return run_self_calibrate(cfg, args)
    if args.check_layout:
        return run_check_layout(cfg, args)
    if args.inspect:
        return run_inspect(cfg, args)

    print(f"目标配置: {cfg.target_path}")
    if cfg.has_intrinsics:
        print(
            f"相机内参: {cfg.camera_info_path} "
            f"({cfg.camera_name or '未命名'} {cfg.image_width}x{cfg.image_height})"
        )
    else:
        print("相机内参: 无")
    print(
        f"标定板: tag_size={cfg.board.tag_size * 1000:.1f}mm  "
        f"ids={list(cfg.board.tag_ids)}  "
        f"外廓={cfg.board.board_size[0] * 1000:.0f}x{cfg.board.board_size[1] * 1000:.0f}mm"
    )
    return run(cfg, args)


if __name__ == "__main__":  # 也支持 python -m a_tag.main
    raise SystemExit(main())
