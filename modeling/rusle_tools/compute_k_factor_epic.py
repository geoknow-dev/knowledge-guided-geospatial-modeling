# file: tools/compute_k_factor_epic.py

import os
import json
import numpy as np
import rasterio

def compute_k_factor_epic(input_str: str) -> str:
    """
    K 因子计算工具（EPIC 模型）：
    - 输入：resample_to_1km 输出的 JSON 字符串
    - 在 data 中获取 'ms','msilt','mc','orgc' 四个键对应的栅格路径（已经是剪裁、投影、1km 分辨率）
    - 按 EPIC 经验公式计算 K，并乘以 0.1317
    - 输出：写出与输入栅格同大小、同投影的 GeoTIFF（output/K_factor_epic.tif）
    """
    try:
        data = json.loads(input_str)
        # 如果上一步已经有 error，直接透传
        if isinstance(data, dict) and data.get("error"):
            return input_str

        # 辅助：提取单路径
        def get_path(key):
            val = data.get(key)
            if isinstance(val, list):
                if len(val) != 1:
                    raise ValueError(f"{key} 返回了 {len(val)} 个文件，应该是单个栅格")
                return val[0]
            return val

        ms_path    = get_path('ms')
        msilt_path = get_path('msilt')
        mc_path    = get_path('mc')
        orgc_path  = get_path('orgc')
        for p in (ms_path, msilt_path, mc_path, orgc_path):
            if not p or not os.path.isfile(p):
                return json.dumps({"error": f"{p} 文件不存在或路径错误"}, ensure_ascii=False)

        # 打开四个栅格
        with rasterio.open(ms_path) as src:
            ms = src.read(1).astype(np.float32)
            profile = src.profile.copy()
        with rasterio.open(msilt_path) as src:
            msilt = src.read(1).astype(np.float32)
        with rasterio.open(mc_path) as src:
            mc = src.read(1).astype(np.float32)
        with rasterio.open(orgc_path) as src:
            orgc = src.read(1).astype(np.float32)

        # EPIC 模型公式（Wischmeier & Smith 1978），并乘以0.1317
        eps = 1e-6
        base = (
            0.2
            + 0.3 * np.exp(-0.0256 * ms * (1 - msilt / 100.0))
            * (1 - (0.25 * orgc) / (orgc + np.exp(3.72 - 2.95 * orgc) + eps))
        )
        K = base * 0.1317

        # 写出结果
        out_dir = os.path.join(os.getenv("DATA_DIR", "/data/gis"), "output")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "K_factor_epic.tif")

        profile.update(dtype="float32", count=1, driver="GTiff")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(K, 1)

        return out_path

    except Exception as e:
        return json.dumps({"error": f"compute_k_factor_epic 出错: {e}"}, ensure_ascii=False)
