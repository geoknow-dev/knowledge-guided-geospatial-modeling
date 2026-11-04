# file: tools/compute_ls_factor.py
import os
import json
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject


def compute_ls_factor(input_str: str) -> str:
    """
    LS因子计算工具：
    输入为 reproject_selected_data 返回的 JSON 字符串，包含 'dem' 栅格路径。
    步骤：
      1. 读取 DEM，计算坡度角 θ（以度为单位）。
      2. 根据 θ 使用 McCool 等公式计算坡度因子 S：
         S = 10.8*sin(θ)+0.03, if θ<5°
         S = 16.8*sin(θ)-0.05, if 5°≤θ≤10°
         S = 21.92*sin(θ)-0.96, if θ>10°
      3. 取 λ=30，计算坡长因子 L = (λ/22.13)^m，
         m = 0.5 if θ≥3°, m = 0.4 if 1.5°≤θ<3°, m = 0.3 if 0.5°≤θ<1.5°, m = 0.2 if θ<0.5°
      4. LS = L * S
    返回 LS 因子栅格路径或错误信息。
    """
    try:
        data = json.loads(input_str)
        if isinstance(data, dict) and 'error' in data:
            return input_str
        # 获取DEM路径
        dem_paths = [p for k,p in data.items() if 'dem' in k.lower()]
        if len(dem_paths) != 1:
            return json.dumps({'error': '未找到唯一 DEM 栅格'}, ensure_ascii=False)
        dem_path = dem_paths[0]
        # 读取DEM
        with rasterio.open(dem_path) as src:
            dem = src.read(1).astype(np.float32)
            profile = src.profile.copy()
            transform = src.transform
        # 计算坡度
        # 计算分辨率
        res_x = transform.a
        res_y = abs(transform.e)
        # 使用numpy梯度
        grad_y, grad_x = np.gradient(dem, res_y, res_x)
        slope_rad = np.arctan(np.hypot(grad_x, grad_y))  # 弧度
        slope_deg = np.degrees(slope_rad)
        # 计算S
        S = np.zeros_like(slope_deg, dtype=np.float32)
        mask1 = slope_deg < 5
        mask2 = (slope_deg >= 5) & (slope_deg <= 10)
        mask3 = slope_deg > 10
        S[mask1] = 10.8 * np.sin(np.radians(slope_deg[mask1])) + 0.03
        S[mask2] = 16.8 * np.sin(np.radians(slope_deg[mask2])) - 0.05
        S[mask3] = 21.92 * np.sin(np.radians(slope_deg[mask3])) - 0.96
        # 计算L
        lam = 30.0
        L = np.zeros_like(slope_deg, dtype=np.float32)
        # 确定m
        m = np.where(slope_deg >= 3, 0.5,
            np.where(slope_deg >= 1.5, 0.4,
                np.where(slope_deg >= 0.5, 0.3, 0.2)))
        L = np.power(lam/22.13, m)
        # 计算LS
        LS = (L * S).astype(np.float32)
        # 输出LS
        out_dir = os.path.join(os.getenv('DATA_DIR', '/data/gis'), 'output')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'LS_factor.tif')
        profile.update(dtype='float32', count=1)
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(LS, 1)
        return out_path
    except Exception as e:
        return json.dumps({'error': f'compute_ls_factor 出错: {e}'}, ensure_ascii=False)
