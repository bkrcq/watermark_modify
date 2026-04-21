#!/usr/bin/env python3
"""
华为水印批量替换工具 v5（支持横竖屏双水印）
原理：自动检测图片方向（竖屏/横屏），使用对应的水印图精准贴到左下角位置。
     竖屏照片 → 用竖屏水印
     横屏照片 → 用横屏水印
输出命名：原文件名_gai.jpg

依赖：pip install pillow numpy

用法：
  批量处理目录   python3 replace_watermark.py <输入目录> <输出目录> <竖屏水印> <横屏水印>
  处理单张图片   python3 replace_watermark.py <输入图片> <输出目录>  <竖屏水印> <横屏水印>

可选参数：
  --jobs  并行进程数，默认 CPU 核心数
  --ext   图片格式，默认 jpg,jpeg

示例：
  python3 replace_watermark.py ./原图 ./新图 ./P90竖屏水印.jpg ./P90横屏水印.jpg
  python3 replace_watermark.py ./原图 ./新图 ./P90竖屏水印.jpg ./P90横屏水印.jpg --jobs 8

提示：如果你只有一张水印图，两个位置传同一个路径即可。
"""

import sys, os, argparse, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image

# ══════════════════════════════════════════════════════════════════════════════
#  固定参考坐标
#  华为水印按绝对像素布局：文字距左侧 137px，距底部 136px，文字高 72px。
# ══════════════════════════════════════════════════════════════════════════════

# 竖屏参考（3072 × 4440，HUAWEI Pura80 Pro 实测）
REF_PORTRAIT = {
    "ref_w":  3072,
    "ref_h":  4440,
    "text_left":   137,
    "text_top":    4232,
    "text_right":  1057,
    "text_bottom": 4304,
}

# 横屏参考（假设分辨率为 4440 × 3072，文字与底部/左侧保持相同像素距离）
# 若与实际横屏照片不匹配，调整这里的值即可
REF_LANDSCAPE = {
    "ref_w":  4440,
    "ref_h":  3072,
    "text_left":   137,       # 距左侧仍为 137px
    "text_top":    2864,      # 3072 - 208 = 2864（距底部 208px）
    "text_right":  1057,
    "text_bottom": 2936,      # 3072 - 136 = 2936（距底部 136px）
}

# 水印图中文字的检测阈值
WM_DARK_THRESH = 100
# ─────────────────────────────────────────────────────────────────────────────


def prepare_patch(wm_path: str, target_text_height: int):
    """
    读取水印图，自动检测文字边界，裁出带少量留白的贴片，缩放到目标文字高度。
    返回 (patch_image, text_offset_x_in_patch, text_offset_y_in_patch)。
    """
    wm  = Image.open(wm_path).convert("RGB")
    arr = np.array(wm)

    dark = arr.min(axis=2) < WM_DARK_THRESH
    rows = np.where(dark.any(axis=1))[0]
    cols = np.where(dark.any(axis=0))[0]
    if len(rows) == 0:
        raise ValueError(f"水印图中未检测到深色文字：{wm_path}")

    r0, r1 = int(rows[0]), int(rows[-1])
    c0, c1 = int(cols[0]), int(cols[-1])
    text_h_wm = r1 - r0

    pad_v = max(10, int(text_h_wm * 0.18))
    pad_h = max(12, int(text_h_wm * 0.12))
    crop_l = max(0, c0 - pad_h)
    crop_t = max(0, r0 - pad_v)
    crop_r = min(arr.shape[1], c1 + pad_h)
    crop_b = min(arr.shape[0], r1 + pad_v)
    patch  = wm.crop((crop_l, crop_t, crop_r, crop_b))

    offset_x = c0 - crop_l
    offset_y = r0 - crop_t

    scale = target_text_height / text_h_wm
    new_w = max(1, int(patch.size[0] * scale))
    new_h = max(1, int(patch.size[1] * scale))
    patch = patch.resize((new_w, new_h), Image.LANCZOS)

    offset_x = int(offset_x * scale)
    offset_y = int(offset_y * scale)
    return patch, offset_x, offset_y


def process_one(input_path: str, output_path: str,
                wm_portrait: str, wm_landscape: str) -> tuple:
    """处理单张图片，自动识别方向选择水印。返回 (success, message)。"""
    try:
        img  = Image.open(input_path).convert("RGB")
        w, h = img.size

        # ── 根据宽高判断方向 ──
        is_landscape = w > h
        if is_landscape:
            wm_path  = wm_landscape
            ref      = REF_LANDSCAPE
            orient   = "横屏"
        else:
            wm_path  = wm_portrait
            ref      = REF_PORTRAIT
            orient   = "竖屏"

        # ── 按比例换算目标文字位置 ──
        scale_w = w / ref["ref_w"]
        scale_h = h / ref["ref_h"]
        tgt_left   = int(ref["text_left"]   * scale_w)
        tgt_top    = int(ref["text_top"]    * scale_h)
        tgt_right  = int(ref["text_right"]  * scale_w)
        tgt_bottom = int(ref["text_bottom"] * scale_h)
        tgt_text_h = tgt_bottom - tgt_top

        # ── 准备水印贴片 ──
        patch, off_x, off_y = prepare_patch(wm_path, tgt_text_h)

        paste_x = max(0, tgt_left - off_x)
        paste_y = max(0, tgt_top  - off_y)

        img.paste(patch, (paste_x, paste_y))

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        img.save(output_path, quality=97, subsampling=0)

        return True, (f"{orient}  {w}x{h}  "
                      f"目标文字=[{tgt_left},{tgt_top},{tgt_right},{tgt_bottom}]  "
                      f"贴片={patch.size}  贴位=({paste_x},{paste_y})")

    except Exception as e:
        import traceback
        return False, traceback.format_exc()


# ══════════════════════════════════════════════════════════════════════════════
#  批量处理
# ══════════════════════════════════════════════════════════════════════════════

def make_output_path(f: Path, out_dir: Path) -> Path:
    return out_dir / f"{f.stem}_gai{f.suffix}"


def collect_images(d: Path, exts: set) -> list:
    return sorted([p for p in d.rglob("*") if p.suffix.lower() in exts])


def batch_worker(args):
    in_path, out_path, wm_p, wm_l = args
    return str(in_path), process_one(str(in_path), str(out_path), wm_p, wm_l)


def run_batch(input_dir: Path, output_dir: Path,
              wm_portrait: str, wm_landscape: str,
              exts: set, jobs: int):
    files = collect_images(input_dir, exts)
    if not files:
        print(f"  在 {input_dir} 中未找到 {'/'.join(exts)} 图片。")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(files)

    print(f"  竖屏水印：{wm_portrait}")
    print(f"  横屏水印：{wm_landscape}")
    print(f"  输入目录：{input_dir}")
    print(f"  输出目录：{output_dir}")
    print(f"  共 {total} 张，并行进程数：{jobs}")
    print(f"  命名规则：原文件名_gai.jpg")
    print("-" * 60)

    tasks = [
        (p,
         make_output_path(p, output_dir / p.relative_to(input_dir).parent),
         wm_portrait, wm_landscape)
        for p in files
    ]

    ok, fail = 0, 0
    portrait_count = 0
    landscape_count = 0
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
                if msg.startswith("横屏"):
                    landscape_count += 1
                else:
                    portrait_count += 1
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
    print(f"完成！成功 {ok}/{total} 张"
          f"（竖屏 {portrait_count} 张，横屏 {landscape_count} 张）")
    print(f"总耗时 {elapsed:.1f}s（平均 {elapsed/total:.1f}s/张）")
    if fail:
        print(f"  {fail} 张失败，请查看上方 ERR 日志。")


# ══════════════════════════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="华为水印直接替换工具（支持横竖屏双水印）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  批量：  python3 replace_watermark.py ./原图 ./新图 ./P90竖屏水印.jpg ./P90横屏水印.jpg
  并行：  python3 replace_watermark.py ./原图 ./新图 ./P90竖屏水印.jpg ./P90横屏水印.jpg --jobs 8
  单张：  python3 replace_watermark.py photo.jpg ./新图 ./P90竖屏水印.jpg ./P90横屏水印.jpg

提示：
  如果你只有一张水印图（例如只有竖屏水印），两个位置传同一个路径即可，
  但处理横屏照片时位置会不准。建议横竖屏都准备一张匹配的水印图。
        """,
    )
    parser.add_argument("input",        help="输入目录或单张图片")
    parser.add_argument("output",       help="输出目录")
    parser.add_argument("wm_portrait",  help="竖屏水印图路径")
    parser.add_argument("wm_landscape", help="横屏水印图路径")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--ext",  default="jpg,jpeg")
    args = parser.parse_args()

    for label, path in [("竖屏水印", args.wm_portrait), ("横屏水印", args.wm_landscape)]:
        if not Path(path).exists():
            print(f"  {label}不存在：{path}")
            sys.exit(1)

    exts        = {"." + e.strip().lower() for e in args.ext.split(",")}
    input_path  = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_file():
        output_path.mkdir(parents=True, exist_ok=True)
        out_file = make_output_path(input_path, output_path)
        print(f"单张模式：{input_path.name}  ->  {out_file.name}")
        ok, msg = process_one(str(input_path), str(out_file),
                              args.wm_portrait, args.wm_landscape)
        if ok:
            print(f"OK -> {out_file}\n   {msg}")
        else:
            print(f"ERR:\n{msg}")
            sys.exit(1)

    elif input_path.is_dir():
        run_batch(input_path, output_path,
                  args.wm_portrait, args.wm_landscape, exts, args.jobs)

    else:
        print(f"  路径不存在：{input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
