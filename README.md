# 批量处理（最常用）
python3 replace_watermark_v3.py ./原图 ./新图 ./90_watermark.jpeg

# 单张
python3 replace_watermark_v3.py photo.jpg ./新图 ./90_watermark.jpeg

# 多核加速
python3 replace_watermark_v3.py ./原图 ./新图 ./90_watermark.jpeg --jobs 8
