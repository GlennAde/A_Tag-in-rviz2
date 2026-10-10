"""生成一块可打印的 AprilTag 标定板（矢量 SVG + 位图 PNG）。

为什么需要它：手工贴的板很容易把某个 tag 贴转，导致 a_tag 的角点对应关系错位
（症状：tags 检测到了但 reproj 几十上百 px）。这里直接把整块板印在一张纸上，
四个 tag 的朝向由程序保证一致 —— 打印出来就是"标准贴法"，配置里不用写 tag_rotations。

用法::

    python3 tools/make_board.py --out /tmp/board            # 默认 4/14/6/22, 4cm tag
    python3 tools/make_board.py --out board --tag-size 30 --dx 70 --dy 70 --ids 0,1,2,3

输出 <out>.svg（打印用，单位 mm，浏览器/Inkscape 打开按 100% 打印）
和 <out>.png（300dpi，预览/检查用）。打印后**务必用卡尺量一下图上的 100mm 校验条**：
不是 100.0±0.5mm 就重打（或用 --scale 校正），否则位姿会整体缩放。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

#: 家族名 -> (OpenCV 字典, 模块数)
FAMILIES = {
    "tag16h5": (cv2.aruco.DICT_APRILTAG_16h5, 6),
    "tag25h9": (cv2.aruco.DICT_APRILTAG_25h9, 7),
    "tag36h11": (cv2.aruco.DICT_APRILTAG_36h11, 8),
}

MM_PER_INCH = 25.4


def tag_modules(family: str, tag_id: int):
    """返回该 tag 的模块矩阵（True=黑），形状 (n, n)，行从上到下。"""
    dict_id, modules = FAMILIES[family]
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    img = cv2.aruco.generateImageMarker(dictionary, tag_id, modules * 10, borderBits=1)
    grid = img.reshape(modules, 10, modules, 10)[:, 0, :, 0] < 128
    return grid


def build(args):
    """算出所有几何（单位 mm，原点=板中心，X 右、Y 上）。返回 dict。"""
    ids = [int(v) for v in args.ids.split(",")]
    if len(ids) != 4:
        raise SystemExit("--ids 需要 4 个（左上,右上,左下,右下）")
    # 左上 右上 左下 右下 的中心坐标（Y 向上，所以"上"是正 y）
    centers = {
        ids[0]: (-args.dx / 2, +args.dy / 2),
        ids[1]: (+args.dx / 2, +args.dy / 2),
        ids[2]: (-args.dx / 2, -args.dy / 2),
        ids[3]: (+args.dx / 2, -args.dy / 2),
    }
    half = args.tag_size / 2
    width = args.dx + args.tag_size + 2 * args.margin
    height = args.dy + args.tag_size + 2 * args.margin
    return {
        "ids": ids,
        "centers": centers,
        "dx": args.dx,
        "dy": args.dy,
        "half": half,
        "tag_size": args.tag_size,
        "width": width,
        "height": height,
        "family": args.family,
    }


# ------------------------------------------------------------------ SVG（打印用）
def write_svg(geo, path: Path) -> None:
    W, H, half = geo["width"], geo["height"], geo["half"]
    mod = geo["tag_size"] / FAMILIES[geo["family"]][1]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.3f}mm" height="{H:.3f}mm" '
        f'viewBox="0 0 {W:.3f} {H:.3f}">',
        f'<rect x="0" y="0" width="{W:.3f}" height="{H:.3f}" fill="#ffffff"/>',
        '<g fill="#000000" font-family="DejaVu Sans, Arial, sans-serif">',
    ]

    def X(x):  # 板坐标(mm, Y 上) -> SVG 坐标(Y 下)
        return W / 2 + x

    def Y(y):
        return H / 2 - y

    # tag 模块（矢量方块，缩放打印也清楚）
    for tid, (cx, cy) in geo["centers"].items():
        grid = tag_modules(geo["family"], tid)
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                if not grid[row, col]:
                    continue
                x = cx - half + col * mod
                y = cy + half - (row + 1) * mod      # 第一行在最上面
                parts.append(
                    f'<rect x="{X(x):.3f}" y="{Y(y + mod):.3f}" '
                    f'width="{mod:.3f}" height="{mod:.3f}"/>'
                )

    # 坐标轴、原点、标注、校验条
    parts += [
        f'<line x1="{X(0):.3f}" y1="{Y(0):.3f}" x2="{X(60):.3f}" y2="{Y(0):.3f}" '
        'stroke="#d00000" stroke-width="0.6"/>',
        f'<line x1="{X(0):.3f}" y1="{Y(0):.3f}" x2="{X(0):.3f}" y2="{Y(60):.3f}" '
        'stroke="#008000" stroke-width="0.6"/>',
        f'<text x="{X(62):.3f}" y="{Y(-3):.3f}" font-size="4" fill="#d00000">+X (右)</text>',
        f'<text x="{X(2):.3f}" y="{Y(62):.3f}" font-size="4" fill="#008000">+Y (上)</text>',
        f'<text x="{X(2):.3f}" y="{Y(-6):.3f}" font-size="3.2" fill="#404040">原点 = 板中心，'
        'Z 垂直纸面向外（朝相机）</text>',
    ]
    # 每个 tag 的 id 与朝向提示（放在 tag 下方，不进 tag 内部）
    for tid, (cx, cy) in geo["centers"].items():
        parts.append(
            f'<text x="{X(cx):.3f}" y="{Y(cy - half - 3):.3f}" font-size="4.5" '
            f'text-anchor="middle">{tid}</text>'
        )
        parts.append(
            f'<text x="{X(cx):.3f}" y="{Y(cy + half + 6):.3f}" font-size="4" '
            f'text-anchor="middle" fill="#404040">↑ 朝向</text>'
        )
    # 100mm 校验条
    bar_y = -H / 2 + 10
    bar_x0 = -50
    parts += [
        f'<line x1="{X(bar_x0):.3f}" y1="{Y(bar_y):.3f}" x2="{X(bar_x0 + 100):.3f}" '
        f'y2="{Y(bar_y):.3f}" stroke="#000" stroke-width="0.6"/>',
        f'<line x1="{X(bar_x0):.3f}" y1="{Y(bar_y - 3):.3f}" x2="{X(bar_x0):.3f}" '
        f'y2="{Y(bar_y + 3):.3f}" stroke="#000" stroke-width="0.6"/>',
        f'<line x1="{X(bar_x0 + 100):.3f}" y1="{Y(bar_y - 3):.3f}" '
        f'x2="{X(bar_x0 + 100):.3f}" y2="{Y(bar_y + 3):.3f}" stroke="#000" stroke-width="0.6"/>',
        f'<text x="{X(bar_x0 + 50):.3f}" y="{Y(bar_y - 4):.3f}" font-size="3.6" '
        'text-anchor="middle">打印后量这里 = 100.0 mm</text>',
        f'<text x="{X(0):.3f}" y="{Y(bar_y + 9):.3f}" font-size="3.6" text-anchor="middle">'
        f'{geo["family"]}  ids={geo["ids"]}  tag={geo["tag_size"]:g}mm  '
        f'中心距 {geo["dx"]:g}x{geo["dy"]:g}mm  打印 100% 不缩放</text>',
    ]
    parts.append("</g></svg>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


# ------------------------------------------------------------------ 渲染成位图
def render_bgr(geo, dpi: int) -> np.ndarray:
    scale = dpi / MM_PER_INCH                      # px per mm
    W = int(round(geo["width"] * scale))
    H = int(round(geo["height"] * scale))
    img = np.full((H, W, 3), 255, np.uint8)

    def to_px(x, y):
        return int(round(W / 2 + x * scale)), int(round(H / 2 - y * scale))

    for tid, (cx, cy) in geo["centers"].items():
        grid = tag_modules(geo["family"], tid)
        mod_px = geo["tag_size"] * scale / grid.shape[0]
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                if not grid[row, col]:
                    continue
                x0 = W / 2 + (cx - geo["half"] + col * geo["tag_size"] / grid.shape[0]) * scale
                y0 = H / 2 - (cy + geo["half"] - row * geo["tag_size"] / grid.shape[0]) * scale
                cv2.rectangle(img, (int(round(x0)), int(round(y0))),
                              (int(round(x0 + mod_px)), int(round(y0 + mod_px))),
                              (0, 0, 0), -1)
        p = to_px(cx, cy - geo["half"] - 3)
        cv2.putText(img, str(tid), (p[0] - 10, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        p2 = to_px(cx, cy + geo["half"] + 5)
        cv2.putText(img, "up", (p2[0] - 10, p2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)

    cv2.arrowedLine(img, to_px(0, 0), to_px(60, 0), (0, 0, 200), 2, cv2.LINE_AA)
    cv2.arrowedLine(img, to_px(0, 0), to_px(0, 60), (0, 128, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "+X", to_px(62, -3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1, cv2.LINE_AA)
    cv2.putText(img, "+Y", to_px(2, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 0), 1, cv2.LINE_AA)
    by = -geo["height"] / 2 + 10
    cv2.line(img, to_px(-50, by), to_px(50, by), (0, 0, 0), 2)
    cv2.putText(img, "100mm check", to_px(-50, by + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def write_png(img: np.ndarray, path: Path) -> None:
    cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, 6])


# ------------------------------------------------------------------ PDF（A4，打印用）
def write_pdf(img: np.ndarray, path: Path, width_mm: float, height_mm: float) -> None:
    """把板面无损嵌进一张 A4 PDF 并居中，按 100% 打印就是实际尺寸。"""
    import zlib

    page_w, page_h = 210.0, 297.0                       # A4，单位 mm
    if width_mm > page_w or height_mm > page_h:
        page_w, page_h = width_mm + 20, height_mm + 20
    pt = 72.0 / MM_PER_INCH
    pw, ph = page_w * pt, page_h * pt
    bw, bh = width_mm * pt, height_mm * pt
    x0, y0 = (pw - bw) / 2.0, (ph - bh) / 2.0

    h, w = img.shape[:2]
    raw = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).tobytes()
    data = zlib.compress(raw, 9)

    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {pw:.2f} {ph:.2f}] "
         "/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>").encode(),
        (f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} "
         "/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode "
         f"/Length {len(data)} >>\nstream\n").encode() + data + b"\nendstream",
        (f"<< /Length 64 >>\nstream\nq\n{bw:.2f} 0 0 {bh:.2f} {x0:.2f} {y0:.2f} cm\n"
         "/Im0 Do\nQ\nendstream").encode(),
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n"
            "%%EOF\n").encode()
    Path(path).write_bytes(bytes(out))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="生成可打印的 AprilTag 标定板（SVG + PNG）")
    p.add_argument("--out", required=True, help="输出前缀，会生成 <out>.svg 和 <out>.png")
    p.add_argument("--family", default="tag16h5", choices=sorted(FAMILIES))
    p.add_argument("--ids", default="4,14,6,22", help="左上,右上,左下,右下")
    p.add_argument("--tag-size", type=float, default=40.0, help="tag 黑框边长，mm")
    p.add_argument("--dx", type=float, default=98.2, help="左右 tag 中心距，mm")
    p.add_argument("--dy", type=float, default=100.0, help="上下 tag 中心距，mm")
    p.add_argument("--margin", type=float, default=16.0, help="板边到 tag 外边界的空白，mm")
    p.add_argument("--dpi", type=int, default=300, help="PNG/PDF 的分辨率")
    p.add_argument("--no-pdf", action="store_true", help="不生成 A4 PDF")
    args = p.parse_args(argv)

    geo = build(args)
    out = Path(args.out)
    write_svg(geo, out.with_suffix(".svg"))
    img = render_bgr(geo, args.dpi)
    write_png(img, out.with_suffix(".png"))
    if not args.no_pdf:
        write_pdf(img, out.with_suffix(".pdf"), geo["width"], geo["height"])
    print(f"板尺寸 {geo['width']:.1f} x {geo['height']:.1f} mm   "
          f"{geo['family']}  ids={geo['ids']}")
    made = ".svg（矢量，浏览器/Inkscape 打印） + .png（预览）"
    if not args.no_pdf:
        made += " + .pdf（A4，直接打印，100% 不缩放）"
    print(f"已写出 {out}{made}")
    print(f"对应的 target.yaml: tag_size={geo['tag_size'] / 1000:g}, "
          f"x=±{geo['dx'] / 2000:g}, y=±{geo['dy'] / 2000:g}, tag_rotations 留空")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
