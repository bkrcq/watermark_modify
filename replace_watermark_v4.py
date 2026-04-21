#!/usr/bin/env python3
"""
华为水印批量替换工具 v4（固定坐标版）
原理：按「图片比例」直接把 P90 水印贴到左下角对应位置，
      不做任何自动检测，所以不会出现「没修改」或「贴错位置」的问题。
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
"""

import sys, os, argparse, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image

# ══════════════════════════════════════════════════════════════════════════════
#  固定参考坐标（华为 Pura80 Pro 标准图片 3072×4440）
#  HUAWEI Pura80 Pro 文字占据：
#      左=137  上=4232  右=1057  下=4304
#  所有坐标会按输入图片尺寸等比缩放，适配不同分辨率。
# ══════════════════════════════════════════════════════════════════════════════
REF_W = 3072
REF_H = 4440
REF_TEXT_LEFT   = 137
REF_TEXT_TOP    = 4232
REF_TEXT_RIGHT  = 1057
REF_TEXT_BOTTOM = 4304

# 水印图中文字的检测阈值
WM_DARK_THRESH  = 100
# ─────────────────────────────────────────────────────────────────────────────


def prepare_patch(wm_path: str, target_text_height: int):
    """
    读取水印图，自动检测文字边界，裁出带少量留白的贴片，并缩放到目标文字高度。
    返回 (patch_image, text_offset_x_in_patch, text_offset_y_in_patch)，
    贴图时按 text_offset 对齐到目标文字位置即可。
    """
    wm  = Image.open(wm_path).convert("RGB")
    arr = np.array(wm)

    # 找水印图中的文字边界
    dark = arr.min(axis=2) < WM_DARK_THRESH
    rows = np.where(dark.any(axis=1))[0]
    cols = np.where(dark.any(axis=0))[0]
    if len(rows) == 0:
        raise ValueError(f"水印图中未检测到深色文字：{wm_path}")

    r0, r1 = int(rows[0]), int(rows[-1])
    c0, c1 = int(cols[0]), int(cols[-1])
    text_h_wm = r1 - r0

    # 裁剪区域：文字框 + 少量留白（以白色覆盖原图旧文字的边缘锯齿）
    pad_v = max(10, int(text_h_wm * 0.18))
    pad_h = max(12, int(text_h_wm * 0.12))
    crop_l = max(0, c0 - pad_h)
    crop_t = max(0, r0 - pad_v)
    crop_r = min(arr.shape[1], c1 + pad_h)
    crop_b = min(arr.shape[0], r1 + pad_v)
    patch  = wm.crop((crop_l, crop_t, crop_r, crop_b))

    # 文字在裁切出的贴片中的相对偏移
    offset_x = c0 - crop_l
    offset_y = r0 - crop_t

    # 按目标文字高度等比缩放
    scale = target_text_height / text_h_wm
    new_w = max(1, int(patch.size[0] * scale))
    new_h = max(1, int(patch.size[1] * scale))
    patch = patch.resize((new_w, new_h), Image.LANCZOS)

    offset_x = int(offset_x * scale)
    offset_y = int(offset_y * scale)
    return patch, offset_x, offset_y


def process_one(input_path: str, output_path: str, wm_path: str) -> tuple:
    """处理单张图片，返回 (success: bool, message: str)。"""
    try:
        img  = Image.open(input_path).convert("RGB")
        w, h = img.size

        # 按比例换算目标文字位置
        scale_w = w / REF_W
        scale_h = h / REF_H
        tgt_left   = int(REF_TEXT_LEFT   * scale_w)
        tgt_top    = int(REF_TEXT_TOP    * scale_h)
        tgt_right  = int(REF_TEXT_RIGHT  * scale_w)
        tgt_bottom = int(REF_TEXT_BOTTOM * scale_h)
        tgt_text_h = tgt_bottom - tgt_top

        # 准备水印贴片
        patch, off_x, off_y = prepare_patch(wm_path, tgt_text_h)

        # 贴片内文字对齐到目标文字位置
        paste_x = tgt_left - off_x
        paste_y = tgt_top  - off_y

        # 防越界（极少数情况下）
        paste_x = max(0, paste_x)
        paste_y = max(0, paste_y)

        img.paste(patch, (paste_x, paste_y))

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        img.save(output_path, quality=97, subsampling=0)

        return True, (f"{w}x{h}  目标文字=[{tgt_left},{tgt_top},{tgt_right},{tgt_bottom}]  "
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
    print(f"  定位方式：按 3072×4440 参考图比例换算（固定坐标，不做检测）")
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


# ══════════════════════════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="华为水印直接替换工具（固定坐标版，按比例贴到左下角）",
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
