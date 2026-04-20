#!/usr/bin/env python3
"""
华为水印批量替换工具 v3
原理：从新水印图中自动检测文字区域，缩放后精准贴到原图 HUAWEI 文字的位置。
输出命名：原文件名_gai.jpg

依赖：pip install pillow numpy

用法：
  批量处理目录   python3 replace_watermark.py <输入目录> <输出目录> <水印图路径>
  处理单张图片   python3 replace_watermark.py <输入图片> <输出目录>  <水印图路径>

可选参数：
  --jobs  并行进程数，默认 CPU 核心数
  --ext   图片格式，默认 jpg,jpeg

示例：
  python3 replace_watermark.py ./原图 ./新图 ./P90水印.jpg
  python3 replace_watermark.py ./原图 ./新图 ./P90水印.jpg --jobs 8
  python3 replace_watermark.py photo.jpg ./新图 ./P90水印.jpg
"""

import sys, os, argparse, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image

# ── 检测参数 ──────────────────────────────────────────────────────────────────
DARK_THRESH  = 80    # 判断原图文字像素的阈值
WM_THRESH    = 100   # 判断水印图文字像素的阈值
WHITE_MEAN   = 240   # 判断「纯白行」的均值下限
WHITE_STD    = 15    # 判断「纯白行」的标准差上限
TEXT_PAD     = 8     # 贴图时文字框上下左右的额外留白（px，用于覆盖旧文字边缘）
# ─────────────────────────────────────────────────────────────────────────────


def find_white_area_top(arr: np.ndarray) -> int:
    """
    从底部向上扫描，找到白色水印区域的最顶行。
    策略：从底向上，遇到第一个「纯白行」之后继续向上，
          直到遇到非白行（照片内容），返回上一白行的位置。
    """
    h = arr.shape[0]
    in_white = False
    white_start = h - 1

    for r in range(h - 1, max(h - 1000, 0), -1):
        row   = arr[r, :, :]
        mean  = float(row.mean())
        std   = float(row.std())
        is_white = (mean > WHITE_MEAN and std < WHITE_STD) or \
                   (mean > WHITE_MEAN - 10 and std < WHITE_STD + 30)  # 允许文字行

        if is_white and not in_white:
            in_white   = True
            white_start = r
        elif not is_white and in_white:
            # 连续白区域结束，判断是否够宽（避免噪声）
            if white_start - r > 20:
                return r + 1
            in_white = False   # 只是噪声，继续向上找

    return white_start


def find_huawei_text_bbox(arr: np.ndarray, white_top: int) -> tuple:
    """
    在白色水印区域内的左半部分，找 HUAWEI 文字的紧边界框。
    返回 (x0, y0, x1, y1) 绝对坐标，失败时返回 fallback。
    """
    h, w = arr.shape[:2]
    region = arr[white_top:, :w // 2, :]
    dark   = region.min(axis=2) < DARK_THRESH

    rows = np.where(dark.any(axis=1))[0]
    cols = np.where(dark.any(axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        # fallback：3072×4440 实测坐标
        return 137, 4232, 1057, 4304

    return (int(cols[0]),
            int(white_top + rows[0]),
            int(cols[-1]),
            int(white_top + rows[-1]))


def prepare_patch(wm_path: str, target_w: int, target_h: int) -> Image.Image:
    """
    从水印图中裁出文字区域（含等比留白），缩放到 (target_w, target_h)。
    """
    wm  = Image.open(wm_path).convert("RGB")
    arr = np.array(wm)
    wh, ww = arr.shape[:2]

    dark = arr.min(axis=2) < WM_THRESH
    rows = np.where(dark.any(axis=1))[0]
    cols = np.where(dark.any(axis=0))[0]

    if len(rows) == 0:
        raise ValueError(f"水印图中未检测到文字（阈值={WM_THRESH}），请检查图片。")

    # 文字紧边界
    r0, r1 = int(rows[0]),  int(rows[-1])
    c0, c1 = int(cols[0]),  int(cols[-1])

    # 计算与目标尺寸匹配的留白，使文字在贴片内的比例与原水印图一致
    text_h_orig = r1 - r0
    text_w_orig = c1 - c0
    # 按高度比例计算留白
    v_ratio  = text_h_orig / wh          # 文字占原图高度的比例
    pad_top  = int(r0 / wh * target_h)   # 顶部留白（缩放后）
    pad_bot  = int((wh - r1) / wh * target_h)

    # 裁剪区域：保留完整列范围，行按留白裁
    crop_t = max(0, r0 - max(2, pad_top))
    crop_b = min(wh, r1 + max(2, pad_bot))
    crop_l = max(0, c0 - 20)
    crop_r = min(ww, c1 + 20)

    patch = wm.crop((crop_l, crop_t, crop_r, crop_b))
    patch = patch.resize((target_w, target_h), Image.LANCZOS)
    return patch


def process_one(input_path: str, output_path: str, wm_path: str) -> tuple:
    """处理单张图片，返回 (success: bool, message: str)。"""
    try:
        img = Image.open(input_path).convert("RGB")
        arr = np.array(img)
        h, w = arr.shape[:2]

        # 1. 找白色水印区域起点
        white_top = find_white_area_top(arr)

        # 2. 找 HUAWEI 文字边界框
        x0, y0, x1, y1 = find_huawei_text_bbox(arr, white_top)
        text_w = x1 - x0
        text_h = y1 - y0

        # 3. 准备贴片（加上 PAD 以彻底覆盖旧文字边缘锯齿）
        patch_w = text_w + TEXT_PAD * 2
        patch_h = text_h + TEXT_PAD * 2
        patch   = prepare_patch(wm_path, patch_w, patch_h)

        # 4. 贴图（从文字框往外扩 PAD，确保旧文字被完整覆盖）
        paste_x = max(0, x0 - TEXT_PAD)
        paste_y = max(0, y0 - TEXT_PAD)
        img.paste(patch, (paste_x, paste_y))

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        img.save(output_path, quality=97, subsampling=0)

        return True, (f"{w}x{h}  white_top={white_top}  "
                      f"文字框=[{x0},{y0},{x1},{y1}]  "
                      f"贴片={patch.size}  贴位=({paste_x},{paste_y})")

    except Exception as e:
        import traceback
        return False, traceback.format_exc()


# ── 批量 ──────────────────────────────────────────────────────────────────────

def make_output_path(f: Path, out_dir: Path) -> Path:
    return out_dir / f"{f.stem}_gai{f.suffix}"


def collect_images(d: Path, exts: set) -> list:
    return sorted([p for p in d.rglob("*") if p.suffix.lower() in exts])


def batch_worker(args):
    in_path, out_path, wm_path = args
    return str(in_path), process_one(str(in_path), str(out_path), wm_path)


def run_batch(input_dir: Path, output_dir: Path, wm_path: str, exts: set, jobs: int):
    files = collect_images(input_dir, exts)
    if not files:
        print(f"  在 {input_dir} 中未找到 {'/'.join(exts)} 图片。")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(files)

    print(f"  水印图：{wm_path}")
    print(f"  输入目录：{input_dir}")
    print(f"  输出目录：{output_dir}")
    print(f"  共 {total} 张，并行进程数：{jobs}")
    print(f"  命名规则：原文件名_gai.jpg")
    print("-" * 60)

    tasks = [
        (p,
         make_output_path(p, output_dir / p.relative_to(input_dir).parent),
         wm_path)
        for p in files
    ]

    ok, fail = 0, 0
    t0 = time.time()
    width = len(str(total))

    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(batch_worker, t): t for t in tasks}
        done = 0
        for future in as_completed(futures):
            done += 1
            in_str, (success, msg) = future.result()
            name    = Path(in_str).name
            elapsed = time.time() - t0
            eta     = (elapsed / done) * (total - done)

            if success:
                ok += 1
                out_name = f"{Path(in_str).stem}_gai{Path(in_str).suffix}"
                print(f"[{done:{width}}/{total}] OK  {name}  ->  {out_name}")
                print(f"         {msg}")
            else:
                fail += 1
                print(f"[{done:{width}}/{total}] ERR {name}")
                print(f"         {msg.splitlines()[-1]}")

            if eta > 1:
                print(f"         已用 {elapsed:.1f}s，预计剩余 {eta:.1f}s")

    elapsed = time.time() - t0
    print("-" * 60)
    print(f"完成！成功 {ok}/{total} 张，总耗时 {elapsed:.1f}s（平均 {elapsed/total:.1f}s/张）")
    if fail:
        print(f"  {fail} 张失败，请查看上方 ERR 日志。")


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="华为水印直接替换工具（贴图方式，像素级精准）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  批量：  python3 replace_watermark.py ./原图 ./新图 ./P90水印.jpg
  并行：  python3 replace_watermark.py ./原图 ./新图 ./P90水印.jpg --jobs 8
  单张：  python3 replace_watermark.py photo.jpg ./新图 ./P90水印.jpg
        """,
    )
    parser.add_argument("input",     help="输入目录或单张图片")
    parser.add_argument("output",    help="输出目录")
    parser.add_argument("watermark", help="新水印图片路径（如 P90水印.jpg）")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--ext",  default="jpg,jpeg")
    args = parser.parse_args()

    if not Path(args.watermark).exists():
        print(f"  水印图不存在：{args.watermark}")
        sys.exit(1)

    exts        = {"." + e.strip().lower() for e in args.ext.split(",")}
    input_path  = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_file():
        output_path.mkdir(parents=True, exist_ok=True)
        out_file = make_output_path(input_path, output_path)
        print(f"单张模式：{input_path.name}  ->  {out_file.name}")
        ok, msg = process_one(str(input_path), str(out_file), args.watermark)
        if ok:
            print(f"OK -> {out_file}\n   {msg}")
        else:
            print(f"ERR:\n{msg}")
            sys.exit(1)

    elif input_path.is_dir():
        run_batch(input_path, output_path, args.watermark, exts, args.jobs)

    else:
        print(f"  路径不存在：{input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
