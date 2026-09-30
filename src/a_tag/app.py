"""命令行入口：采集 -> 检测 -> 求解 -> 显示。"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from .camera import Camera
from .config import AppConfig
from .detector import AprilTagDetector
from .filter import PoseFilter
from .pose import solve_board_pose
from .se3 import pose_from_T
from .viz import draw_board, draw_detections, draw_hud, draw_pose_axes

__all__ = ["build_parser", "run", "main"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="a-tag",
        description="用四个 AprilTag 标定板估计相机相对板的 6DoF 位姿",
    )
    p.add_argument("--config", default=None,
                   help="YAML 配置路径。默认依次找 ./config.yaml、$A_TAG_CONFIG、随包默认配置")
    p.add_argument("--source", default=None,
                   help="视频源: 摄像头索引(如 0) 或 视频/图片 路径。覆盖配置里的 source")
    p.add_argument("--max-frames", type=int, default=0, help="最多处理多少帧，0 表示不限")
    p.add_argument("--no-preview", action="store_true", help="不开窗口（无显示器时用）")
    p.add_argument("--no-filter", action="store_true", help="关闭时间滤波")
    p.add_argument("--print-every", type=int, default=30, help="每 N 帧打印一次位姿，0 表示不打印")
    p.add_argument("--quad-decimate", type=float, default=None,
                   help="覆盖检测降采样，精度优先请用 1.0")
    p.add_argument("--min-tags", type=int, default=None, help="覆盖最少可见 tag 数")
    p.add_argument("--save-video", default=None, help="把带标注的结果写成视频")
    return p


def run(cfg: AppConfig, args: argparse.Namespace) -> int:
    """跑主循环，返回进程退出码。"""
    detector = AprilTagDetector(
        families=cfg.families,
        nthreads=cfg.nthreads,
        quad_decimate=cfg.quad_decimate if args.quad_decimate is None else args.quad_decimate,
        expected_ids=cfg.board.tag_ids,
    )
    pose_filter = None if args.no_filter else PoseFilter()
    min_tags = cfg.min_tags if args.min_tags is None else args.min_tags

    source = cfg.source if args.source is None else _coerce_source(args.source)
    writer = None
    n_frames = 0
    n_ok = 0
    t_prev = time.perf_counter()
    fps = 0.0

    with Camera(source, cfg.image_width, cfg.image_height, cfg.fps) as cam:
        while True:
            ok, bgr = cam.read()
            if not ok:
                break

            now = time.perf_counter()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt

            gray = Camera.to_gray(bgr)
            corners = detector.detect(gray)
            result = solve_board_pose(
                corners, cfg.board, cfg.camera_matrix, cfg.dist_coeffs,
                min_tags=min_tags, max_reproj_px=cfg.max_reproj_px,
            )
            n_frames += 1
            n_ok += int(result.ok)

            T_filtered = result.T_board_cam if result.ok else None
            if pose_filter is not None:
                T_filtered = pose_filter(T_filtered, dt)
            stale = pose_filter.stale_frames if pose_filter is not None else 0

            if args.print_every and n_frames % args.print_every == 0:
                _print_status(n_frames, result, T_filtered, fps)

            if not args.no_preview or args.save_video:
                draw_detections(bgr, corners, cfg.board)
                if result.T_cam_board is not None:
                    draw_board(bgr, result, cfg.board, cfg.camera_matrix, cfg.dist_coeffs)
                    draw_pose_axes(bgr, result.T_cam_board,
                                   cfg.camera_matrix, cfg.dist_coeffs)
                draw_hud(bgr, result, T_filtered, fps, stale)
                if args.save_video:
                    if writer is None:
                        h, w = bgr.shape[:2]
                        writer = cv2.VideoWriter(args.save_video,
                                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                                 cfg.fps if cfg.fps > 0 else 30, (w, h))
                    writer.write(bgr)
                if not args.no_preview:
                    cv2.imshow("a-tag", bgr)
                    if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                        break

            if args.max_frames and n_frames >= args.max_frames:
                break

    if writer is not None:
        writer.release()
    if not args.no_preview:
        cv2.destroyAllWindows()

    if n_frames:
        print(f"\n处理 {n_frames} 帧，有效 {n_ok} 帧 ({100.0 * n_ok / n_frames:.1f}%)")
    return 0 if n_ok else 1


def _coerce_source(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _print_status(n_frames: int, result, T_filtered, fps: float) -> None:
    if T_filtered is None:
        print(f"[{n_frames:6d}] 无位姿  tags={result.n_tags}  {result.reason}")
        return
    p = pose_from_T(T_filtered, degrees=True)
    print(f"[{n_frames:6d}] tags={result.n_tags} reproj={result.reproj_rmse_px:5.3f}px "
          f"x={p['x'] * 1000:8.1f} y={p['y'] * 1000:8.1f} z={p['z'] * 1000:8.1f} mm  "
          f"rpy=({p['roll']:6.2f},{p['pitch']:6.2f},{p['yaw']:6.2f}) deg "
          f"fps={fps:4.1f}")


def main(argv: list[str] | None = None) -> int:
    """命令行入口（pyproject 里的 a-tag 脚本）。"""
    args = build_parser().parse_args(argv)
    cfg = AppConfig.load(args.config)
    print(f"配置: {cfg.source_path}")
    print(f"标定板: tag_size={cfg.board.tag_size * 1000:.1f}mm  "
          f"ids={list(cfg.board.tag_ids)}  "
          f"外廓={cfg.board.board_size[0] * 1000:.0f}x{cfg.board.board_size[1] * 1000:.0f}mm")
    return run(cfg, args)
