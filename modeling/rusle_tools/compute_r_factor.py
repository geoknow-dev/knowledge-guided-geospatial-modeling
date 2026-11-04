# file: tools/compute_r_factor.py

import os
import json
import numpy as np
import rasterio

def compute_r_factor(input_str: str) -> str:
    """
    R 因子计算工具：根据 resample_to_1km 输出的 JSON，
    寻找其中包含 'annual' 的单个栅格作为年降雨量 P，
    寻找包含 'monthly' 的键，如果值是列表则当作 12 个月份栅格列表，
    否则按老逻辑（多个键）收集月量栅格，
    使用 Wischmeier 经验公式计算 R 因子：
      R = sum_{i=1..12}[1.735 * 10^(1.5 * log10(Pi/P) - 0.8188)]
    输入：JSON 字符串 { name: raster_path or [raster_path,...], ... }
    输出：保存的 R 因子 GeoTIFF 路径 或 错误 JSON 字符串。
    """
    try:
        data = json.loads(input_str)
        # 如果已经是上一步的 error 透传
        if isinstance(data, dict) and data.get("error"):
            return input_str

        # 1) 找年降雨 P
        annual_paths = [p for n, p in data.items() if 'annual' in n.lower()]
        if len(annual_paths) != 1:
            return json.dumps(
                {"error": "未找到唯一的 annual_precipitation 栅格"},
                ensure_ascii=False
            )
        P_path = annual_paths[0]

        # 2) 找月降雨 Pi
        monthly_paths = []
        # 如果有键 'monthly_precipitation'，且其值为列表，就直接用它
        if 'monthly_precipitation' in data and isinstance(data['monthly_precipitation'], list):
            monthly_paths = data['monthly_precipitation']
        else:
            # 否则按老逻辑：所有键名里带 'monthly' 的值
            for name, val in data.items():
                if 'monthly' in name.lower():
                    if isinstance(val, list):
                        monthly_paths.extend(val)
                    else:
                        monthly_paths.append(val)

        # 校验数量
        if len(monthly_paths) != 12:
            return json.dumps(
                {"error": f"未找到 12 个 月降雨量 栅格，当前共 {len(monthly_paths)} 个"},
                ensure_ascii=False
            )

        # 3) 读取 P 栅格
        with rasterio.open(P_path) as srcP:
            P = srcP.read(1).astype(np.float32)
            profile = srcP.profile.copy()

        # 4) 初始化 R 数组
        R = np.zeros_like(P, dtype=np.float32)

        # 5) 逐月叠加
        for mpath in sorted(monthly_paths):
            with rasterio.open(mpath) as srcM:
                Pi = srcM.read(1).astype(np.float32)
            # 只在 P>0 且 Pi>0 时计算比值，否则视作比值=1
            ratio = np.where((P > 0) & (Pi > 0), Pi / P, 1.0)
            R += 1.735 * np.power(10, (1.5 * np.log10(ratio) - 0.8188))

        # 6) 写出 R 因子
        base = os.path.splitext(os.path.basename(P_path))[0]
        out_dir = os.path.join(os.getenv('DATA_DIR', '/data/gis'), 'output')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"R_factor_{base}.tif")

        profile.update(dtype='float32', count=1, driver='GTiff')
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(R, 1)

        return out_path

    except Exception as e:
        return json.dumps(
            {"error": f"compute_r_factor 出错: {e}"},
            ensure_ascii=False
        )
