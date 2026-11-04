# file: tools/compute_p_factor_landuse.py
import os
import json
import numpy as np
import rasterio

def compute_p_factor_landuse(input_str: str) -> str:
    """
    P因子计算工具：
    根据 resample_to_1km 输出的 JSON，选择其中命名包含 'landuse' 的栅格，
    使用重分类映射生成 P 因子：
      原值 0,13 -> 0
      原值 1,2,3,4,5,6,7,15 -> 1
      原值 8,9,10 -> 0.8
      原值 11 -> 0.01
      原值 12,14 -> 0.4
    其他值保持原值不变。

    输入: JSON 字符串 { name: raster_path, ... }
    返回: P因子 GeoTIFF 路径，或错误信息。
    """
    try:
        data = json.loads(input_str)
        if isinstance(data, dict) and 'error' in data:
            return input_str
        # 查找 landuse 栅格路径
        landuse_paths = [p for k, p in data.items() if 'landuse' in k.lower()]
        if len(landuse_paths) != 1:
            return json.dumps({'error': '未找到唯一 landuse 栅格'}, ensure_ascii=False)
        landuse_path = landuse_paths[0]
        with rasterio.open(landuse_path) as src:
            arr = src.read(1)
            profile = src.profile.copy()
        # 定义映射
        mapping = {
            0: 0.0, 13: 0.0,
            1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 15: 1.0,
            8: 0.8, 9: 0.8, 10: 0.8,
            11: 0.01,
            12: 0.4, 14: 0.4
        }
        # 构建输出数组
        P = arr.astype(np.float32)
        for oldv, newv in mapping.items():
            P[arr == oldv] = newv
        # 输出 P 因子栅格
        out_dir = os.path.join(os.getenv('DATA_DIR', '/data/gis'), 'output')
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(landuse_path))[0]
        out_path = os.path.join(out_dir, f"P_factor_{base}.tif")
        profile.update(dtype='float32', count=1)
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(P, 1)
        return out_path
    except Exception as e:
        return json.dumps({'error': f'compute_p_factor_landuse 出错: {e}'}, ensure_ascii=False)
