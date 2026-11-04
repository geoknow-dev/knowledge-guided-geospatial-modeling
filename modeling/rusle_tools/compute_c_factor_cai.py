# file: tools/compute_c_factor_cai.py
import os
import json
import numpy as np
import rasterio

def compute_c_factor_cai(input_str: str) -> str:
    """
    C因子计算工具（蔡崇法模型）：
    根据 resample_to_1km 输出的 JSON，选择其中命名包含 'ndvi' 的栅格数据，
    计算植被覆盖度 c = (NDVI - NDVI_min) / (NDVI_max - NDVI_min)，
    再根据蔡崇法经验模型计算 C 因子：
      if c == 0: C = 1
      elif 0 < c <= 0.783: C = 0.6508 - 0.3436 * np.log(c)
      else: C = 0

    输入: JSON 字符串 { name: raster_path, ... }
    返回: C因子 GeoTIFF 路径，或错误信息。
    """
    try:
        data = json.loads(input_str)
        if isinstance(data, dict) and 'error' in data:
            return input_str
        # 查找 NDVI 栅格路径
        ndvi_paths = [p for k, p in data.items() if 'ndvi' in k.lower()]
        if len(ndvi_paths) != 1:
            return json.dumps({'error': '未找到唯一 NDVI 栅格'}, ensure_ascii=False)
        ndvi_path = ndvi_paths[0]
        # 读取 NDVI 数据
        with rasterio.open(ndvi_path) as src:
            ndvi = src.read(1).astype(np.float32)
            profile = src.profile.copy()
        # 计算 NDVI_min 和 NDVI_max（排除无效值）
        valid = ndvi[np.isfinite(ndvi)]
        if valid.size == 0:
            return json.dumps({'error': 'NDVI 栅格无有效数据'}, ensure_ascii=False)
        ndvi_min, ndvi_max = np.min(valid), np.max(valid)
        if ndvi_max - ndvi_min == 0:
            return json.dumps({'error': 'NDVI 栅格最大最小值相同，无法计算'}, ensure_ascii=False)
        c = (ndvi - ndvi_min) / (ndvi_max - ndvi_min)
        # 计算 C 因子
        C = np.zeros_like(c, dtype=np.float32)
        # c == 0
        mask0 = (c == 0)
        C[mask0] = 1.0
        # 0 < c <= 0.783
        mask1 = (c > 0) & (c <= 0.783)
        C[mask1] = 0.6508 - 0.3436 * np.log(c[mask1])
        # c > 0.783 -> C default 0
        # 输出 C 因子栅格
        out_dir = os.path.join(os.getenv('DATA_DIR', '/data/gis'), 'output')
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(ndvi_path))[0]
        out_path = os.path.join(out_dir, f"C_factor_{base}.tif")
        profile.update(dtype='float32', count=1)
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(C, 1)
        return out_path
    except Exception as e:
        return json.dumps({'error': f'compute_c_factor_cai 出错: {e}'}, ensure_ascii=False)
