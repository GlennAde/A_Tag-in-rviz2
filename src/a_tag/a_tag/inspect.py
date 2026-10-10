"""看一眼画面里到底有哪些 tag —— 换相机 / 换板之后的第一步诊断。

不做 ID 过滤、也不依赖 target.yaml：libapriltag 报什么就列什么。回答两个问题：

* 我的 tag 家族对不对（tag36h11 最常见；也支持 tag25h9 / tag16h45 / tag36h10）；
* 我的 tag 编号是不是和 target.yaml 里写的一样。

顺带按参考 tag 的张成平面推一份 tag_centers，可以直接抄进 target.yaml。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import cv2
import numpy as np
from pupil_apriltags import Detector

from .board import CORNER_SIGNS
from .detector import ARUCO_DICTS, AprilTagDetector

__all__ = [
    "DEFAULT_FAMILY",
    "scan_aruco_dicts",
    "SUPPORTED_FAMILIES",
    "UNSAFE_FAMILIES",
    "TagHit",
    "annotate",
    "detect_all",
    "estimate_tag_centers",
    "format_report",
]

#: 默认家族，现实里绝大多数板都是它。
#:
#: **一次只用一个家族**，两个原因都是实测出来的：
#: 1) pupil_apriltags 的 bindings 里家族是 if/elif 链，传空格分隔的多个家族只会创建链上
#:    第一个匹配的，"tag36h11 tag25h9" 在 tag36h11 的图上会检出 0 个；
#: 2) 更麻烦的是同一个进程里反复建/销毁 Detector 会把堆弄坏（glibc: malloc():
#:    mismatching next->prev_size），一个 CLI 调用里轮几个家族会随机崩。
#: 所以换家族请单独再跑一次（--family tag25h9），别在一个进程里轮。
DEFAULT_FAMILY = "tag36h11"

#: libapriltag 认得的家族名。
SUPPORTED_FAMILIES = (
    "tag36h11",
    "tag25h9",
    "tagStandard41h12",
    "tagStandard52h13",
    "tagCircle21h7",
    "tagCircle49h12",
    "tagCustom48h12",
)

#: **别放进默认扫描**：实测 tag16h5 在同一个进程里解完图之后会把堆弄坏
#: （下一次建 Detector 就 malloc(): mismatching next->prev_size 直接崩），
#: 而且在 tag36h11 的板上会报 2 个误检。要试它请单独起一个进程。
UNSAFE_FAMILIES = ("tag16h5",)

_COLOR_OK = (0, 220, 0)
_COLOR_BAD = (0, 0, 255)


@dataclass
class TagHit:
    """一次检测结果（比 AprilTagDetector.detect 多带家族/质量信息）。"""

    tag_id: int
    family: str
    hamming: int
    margin: float
    center: np.ndarray      # (2,) 像素
    corners: np.ndarray     # (4, 2) 像素，顺序同 CORNER_SIGNS

    @property
    def size_px(self) -> float:
        """tag 黑框在图像里的平均边长（像素）。低于 ~50px 精度就开始崩。"""
        c = self.corners
        return float(np.mean([np.linalg.norm(c[(i + 1) % 4] - c[i]) for i in range(4)]))


def detect_all(
    gray: np.ndarray,
    family: str = DEFAULT_FAMILY,
    backend: str = "pupil",
    nthreads: int = 2,
    quad_decimate: float = 1.0,
) -> list:
    """检测画面里的所有 tag，**不按 ID 过滤**。

    backend="pupil" 时一次只用一个家族（原因见 DEFAULT_FAMILY）；backend="aruco" 用
    OpenCV，家族名是 ARUCO_DICTS 的键（tag16h5 / DICT_4X4_50 / ...）。
    """
    if backend == "aruco":
        det = AprilTagDetector(families=family, backend="aruco")
        return [
            TagHit(
                tag_id=tid,
                family=family,
                hamming=-1,                 # OpenCV 不给这些质量指标
                margin=float("nan"),
                center=pts.mean(axis=0),
                corners=pts,
            )
            for tid, pts in sorted(det.detect(gray).items())
        ]

    if family in UNSAFE_FAMILIES:
        print(
            f"（警告: {family} 在这个版本的 libapriltag 上解完会把进程堆弄坏，结果未必可信；"
            '建议改用 --backend aruco）',
            file=sys.stderr,
        )
    try:
        det = Detector(families=family, nthreads=nthreads, quad_decimate=quad_decimate)
    except Exception:
        raise ValueError(
            f"libapriltag 不认这个家族名: {family!r}。可用: "
            + ", ".join(SUPPORTED_FAMILIES + UNSAFE_FAMILIES)
        ) from None

    hits = [
        TagHit(
            tag_id=int(d.tag_id),
            family=(
                d.tag_family.decode()
                if isinstance(d.tag_family, bytes)
                else str(d.tag_family)
            ),
            hamming=int(d.hamming),
            margin=float(d.decision_margin),
            center=np.asarray(d.center, dtype=np.float64),
            corners=np.asarray(d.corners, dtype=np.float64).reshape(4, 2),
        )
        for d in det.detect(gray)
    ]
    hits.sort(key=lambda h: h.tag_id)
    return hits


def scan_aruco_dicts(gray: np.ndarray) -> list:
    """把所有 OpenCV 字典在同一个进程里试一遍（cv2 没有 libapriltag 那个堆问题）。

    Returns: [(字典名, 命中数), ...]，只列命中的。
    """
    found = []
    for name in ARUCO_DICTS:
        try:
            hits = detect_all(gray, family=name, backend="aruco")
        except Exception:
            continue
        if hits:
            found.append((name, len(hits)))
    return found


def annotate(bgr: np.ndarray, hits: list, expected_ids=None) -> np.ndarray:
    """把检测结果画到图上：四角（带序号）、id、像素边长。ID 不在期望里就画红。"""
    img = bgr.copy()
    known = None if expected_ids is None else {int(i) for i in expected_ids}
    for h in hits:
        color = _COLOR_OK if (known is None or h.tag_id in known) else _COLOR_BAD
        cv2.polylines(
            img, [h.corners.astype(np.int32).reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA
        )
        for i, (x, y) in enumerate(h.corners):
            cv2.circle(img, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)
            cv2.putText(img, str(i), (int(x) + 4, int(y) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cx, cy = h.center
        cv2.putText(img, f"id={h.tag_id}", (int(cx) - 26, int(cy) + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        cv2.putText(img, f"{h.size_px:.0f}px", (int(cx) - 26, int(cy) + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def estimate_tag_centers(hits: list, tag_size: float) -> dict:
    """按参考 tag 的张成平面，推各 tag 中心在板坐标系下的 [x, y]，单位米。

    把像素最大的那个 tag 的四个角点当作已知边长的正方形，求「图像 -> 该 tag 平面」
    的单应，再把所有 tag 的中心映射过去；最后平移到 tag 中心包围盒的中心，
    与本库「原点在板面中心」的约定一致。
    """
    if len(hits) < 2 or tag_size <= 0:
        return {}
    ref = max(hits, key=lambda h: h.size_px)
    half = float(tag_size) / 2.0
    H = cv2.getPerspectiveTransform(
        ref.corners.astype(np.float32), (CORNER_SIGNS * half).astype(np.float32)
    )
    centers = {}
    for h in hits:
        p = H @ np.array([h.center[0], h.center[1], 1.0], dtype=np.float64)
        centers[h.tag_id] = p[:2] / p[2]
    xs = np.array([c[0] for c in centers.values()])
    ys = np.array([c[1] for c in centers.values()])
    cx, cy = (xs.max() + xs.min()) / 2.0, (ys.max() + ys.min()) / 2.0
    return {i: (float(x - cx), float(y - cy)) for i, (x, y) in centers.items()}


def format_report(
    hits: list,
    shape,
    expected_ids=None,
    tag_size=None,
    family: str = DEFAULT_FAMILY,
    backend: str = "pupil",
) -> str:
    """把检测结果整理成给人看的报告。"""
    lines = [f"图像: {shape[1]}x{shape[0]}   检测到 {len(hits)} 个 tag"]
    if not hits:
        lines += [
            f"一个都没检测到（后端 {backend}，{family}）。按这个顺序排查：",
            "  1) 家族/字典不对？直接全扫一遍，它会告诉你命中的是哪个：",
            "       --family all",
        ]
        lines += [
            f"     也可以单独指定（{'后端 pupil 一次只能用一个家族' if backend == 'pupil' else 'OpenCV 可以一次扫完'}）："
            f"       --family {name}"
            for name in (SUPPORTED_FAMILIES if backend == "pupil" else tuple(ARUCO_DICTS))
        ]
        lines += [
            "  2) 板上的图案是另一种编码？pupil_apriltags 只认 AprilTag 家族；真正的 ArUco",
            "     （4x4/5x5/6x6）要用 --backend aruco --family DICT_*；",
            "  3) 图像质量：tag 在画面里要 ≥50px，锁死曝光和增益（自动曝光会让角点抖）。",
        ]
        return "\n".join(lines)

    lines.append(
        f"{'id':>6}  {'family':<10} {'hamming':>7} {'margin':>7} {'中心(px)':>18} {'边长(px)':>9}"
    )
    for h in hits:
        hamming = "-" if h.hamming < 0 else str(h.hamming)
        margin = "    -" if h.margin != h.margin else f"{h.margin:7.1f}"  # nan 检查
        lines.append(
            f"{h.tag_id:>6}  {h.family:<10} {hamming:>7} {margin} "
            f"{f'({h.center[0]:.1f}, {h.center[1]:.1f})':>18} {h.size_px:>9.1f}"
        )

    found = sorted(h.tag_id for h in hits)
    if expected_ids is not None:
        expected = sorted(int(i) for i in expected_ids)
        missing = [i for i in expected if i not in found]
        extra = [i for i in found if i not in expected]
        if not missing and not extra:
            lines.append(f"\nID 对得上：target.yaml 期望 {expected}，实测 {found} ✓")
        elif not extra and missing:
            lines += [
                f"\n只有 {len(found)} 个 tag 在画面里，少了 {missing}（**不是 ID 错**）",
                "  多半是出画面/被挡住了：把板挪到画面中央、四个 tag 全进来再看。",
                "  少于 4 个时先别动 target.yaml —— 单帧 3 个 tag 推出来的布局有 2~3% 误差，",
                "  会把你已经自标定好的（σ≈0.1mm）那份覆盖掉。",
            ]
        elif extra:
            lines += [
                f"\n**画面里有 target.yaml 没列的 id**：多了 {extra}",
                "  a_tag 只保留 target.yaml 里列出的 ID（滤干扰 tag 用），所以这些会被丢掉。",
                "  要么是板上真有别的 id（改 tag_centers 的键），要么是误检（看 hamming/margin）。",
            ]
        if missing and extra:
            lines.append("  期望与实测对不上：期望 %s，实测 %s" % (expected, found))

    if tag_size and len(hits) >= 4:
        centers = estimate_tag_centers(hits, tag_size)
        if centers:
            lines += [
                f"\n按检测结果推的布局（tag_size={tag_size} m，可直接抄进 target.yaml）：",
                "board:",
                f"  tag_size: {tag_size}",
                "  tag_centers:",
            ]
            lines += [f"    {i}: [{x:+.3f}, {y:+.3f}]" for i, (x, y) in sorted(centers.items())]
            lines.append(
                "（这是**单帧外推**的估计，误差 2~3%；要精确覆盖 target.yaml 请用 "
                "--self-calibrate 多视角自标定，别拿这个直接换掉）"
            )
    return "\n".join(lines)
