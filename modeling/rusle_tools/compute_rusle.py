# file: tools/compute_rusle.py

import os
import json
import numpy as np
import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling

def compute_rusle(input_str: str) -> str:
    """
    RUSLE 最终结果计算：
    - 输入为 JSON 字符串，包含 R、K、C、LS、P 五个键，对应各因子栅格路径
    - 以 R 因子栅格为参考，逐一打开并重投影/重采样 K、C、LS、P 到同一网格
    - A = R * K * C * LS * P
    - 保存结果到 output/soil_erosion_RUSLE.tif
    """
    try:
        data = json.loads(input_str)
        if isinstance(data, dict) and data.get("error"):
            return input_str

        # 五个因子路径
        R_path  = data["R"]
        K_path  = data["K"]
        C_path  = data["C"]
        LS_path = data["LS"]
        P_path  = data["P"]

        # 打开 R 因子，获取参考网格信息
        with rasterio.open(R_path) as srcR:
            R = srcR.read(1).astype(np.float32)
            profile = srcR.profile.copy()
            dst_transform = srcR.transform
            dst_crs = srcR.crs
            dst_shape = R.shape  # (height, width)

        # 工具：打开 path，将其重投影/重采样到 R 的网格
        def load_align(path):
            with rasterio.open(path) as src:
                arr = src.read(1).astype(np.float32)
                dst = np.zeros(dst_shape, dtype=np.float32)
                reproject(
                    source=arr,
                    destination=dst,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest
                )
                return dst

        # 加载并对齐
        K  = load_align(K_path)
        C  = load_align(C_path)
        LS = load_align(LS_path)
        P  = load_align(P_path)

        # 计算 A
        A = R * K * C * LS * P

        # 写出结果
        out_dir = os.path.join(os.getenv("DATA_DIR", "/data/gis"), "output")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "soil_erosion_RUSLE.tif")

        profile.update(dtype="float32", count=1, driver="GTiff")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(A, 1)

        return out_path

    except Exception as e:
        return json.dumps({"error": f"compute_rusle 出错: {e}"}, ensure_ascii=False)
