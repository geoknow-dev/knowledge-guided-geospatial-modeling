# Knowledge-Guided Geospatial Modeling Reproducibility Package

本仓库提供一套面向精细化地理建模的知识注入、工作流规划、代码生成和结果评价工具。当前案例以 RUSLE 土壤侵蚀建模为主。

> 原始地理栅格数据未随仓库发布；运行完整实验前，需要使用者自行准备具有合法使用权限的数据，并填写本地数据路径。

## 1. 项目解决什么问题

大语言模型能够生成地理建模代码，但在缺少领域知识时，可能出现工作流步骤缺失、公式或参数选择错误、数据依赖关系不完整等问题。本项目将地理建模知识划分为六个累计注入层级，并在相同任务条件下比较模型的工作流、代码和栅格结果，从而回答以下问题：

- 不同层级的领域知识对大语言模型地理建模能力有何影响？
- 哪一类知识对工作流规划、代码生成和结果精度贡献最大？
- 如何使用统一指标量化模型对领域知识的依赖程度？

项目实现的是“生成与评价框架”，不是无需配置即可完成所有地区土壤侵蚀计算的一键式软件。`scripts/run_experiment.py` 负责调用 OpenAI 兼容接口生成 DAG 工作流和 Python 代码，但不会自动执行生成代码；生成代码需要在具有相应栅格数据和地理计算依赖的环境中单独检查与运行。

## 2. 主要功能

### 2.1 六级累计知识注入

`knowledge/` 包含以下知识层级：

| 层级 | 知识类型 |
| --- | --- |
| L1-1 | 地理概念知识 |
| L1-2 | 地理过程知识 |
| L2-1 | 模型结构知识 |
| L2-2 | 数据与工具知识 |
| L3-1 | 公式与算法知识 |
| L3-2 | 参数配置知识 |

选择某一层级时，会累计加载该层级及之前的知识文件。

### 2.2 工作流规划与代码生成

- 读取自然语言任务、知识层级和数据路径配置。
- 通过 OpenAI 兼容接口调用指定模型。
- 生成具有 `step_id`、输入、输出和依赖关系的 DAG 工作流 JSON。
- 根据工作流生成 Python 地理建模代码。
- 支持关闭知识注入或开启有限对话历史，用于对照实验。

### 2.3 评价指标

`evaluation/metrics.py` 实现：

- 工作流评价：WVR、WGD、WKHR、WQI。
- 代码评价：BLEU、加权 n-gram、AST Match、Data-flow Match 和轻量级 CodeBLEU。
- 栅格数值评价：R²、nRMSE。
- 知识依赖评价：KCPI、KLG。

其中代码评价实现为仓库内可复查的轻量版本。如果研究要求严格复现官方 CodeBLEU，应另外固定官方实现、解析器和版本。

### 2.4 RUSLE 参考工具

`modeling/rusle_tools/` 提供 R、K、LS、C、P 因子及最终 RUSLE 结果的参考计算模块。这些模块需要真实且空间一致的栅格输入，不会由实验入口脚本自动执行。

### 2.5 结果与图表复现

`results/` 保存图表使用的结构化 CSV 数值；`scripts/plot_results.py` 可重新生成主要结果图，输出 330 dpi PNG 和矢量 SVG。

## 3. 安装方法

### 3.1 环境要求

- Python 3.10 或 3.11。
- 能访问所选模型的 OpenAI 兼容接口；可使用本地推理服务或经授权的远程服务。
- 完整栅格计算需要 Rasterio、Fiona、GeoPandas、Shapely 和 NumPy。

### 3.2 下载代码

可以在 GitHub 页面选择 `Code → Download ZIP`，解压后进入项目目录；也可以使用 Git：

```bash
git clone <repository-url>
cd geospatial-modeling-reproducibility
```

### 3.3 创建虚拟环境并安装依赖

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux 或 macOS：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

安装完成后运行基础检查：

```bash
python -m unittest evaluation.test_metrics
python scripts/validate_release.py
```

## 4. 使用方法

### 4.1 准备数据路径配置

Windows PowerShell：

```powershell
Copy-Item configs\data_paths.example.json configs\data_paths.local.json
```

然后编辑 `configs/data_paths.local.json`，将示例路径替换为本机真实文件。需要的主要数据包括 DEM、填洼 DEM、汇流累积量、年/月降水、土地覆被、土壤砂粉黏粒和有机碳、NDVI。所有输入栅格应在运行生成代码之前完成研究所需的坐标系、范围、分辨率和 NoData 一致性处理。

`configs/data_paths.local.json` 已被 `.gitignore` 忽略，避免将私人路径上传到 GitHub。

### 4.2 设置模型接口

程序不会在代码中保存密钥。Windows PowerShell 示例：

```powershell
$env:OPENAI_API_KEY="replace-with-your-key-or-local-placeholder"
$env:OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
$env:MODEL_NAME="replace-with-exact-model-or-checkpoint-id"
```

如果使用远程接口，请填写服务商提供的地址和密钥；如果使用本地 OpenAI 兼容服务，请填写其本地地址和实际模型标识。不要把真实密钥写入 `.env.example`、代码、README 或提交记录。

### 4.3 运行一次知识注入实验

```powershell
python scripts\run_experiment.py `
  --level L3-2 `
  --task "Calculate the soil erosion modulus of the Loess Plateau in 2022." `
  --data-config configs\data_paths.local.json
```

常用参数：

| 参数 | 含义 |
| --- | --- |
| `--level` | 必填，知识层级：L1-1 至 L3-2 |
| `--task` | 必填，自然语言建模任务 |
| `--data-config` | 数据路径 JSON；默认使用示例配置 |
| `--output-dir` | 输出目录；默认是 `outputs/` |
| `--temperature` | 模型采样温度；默认 0.1 |
| `--no-kb` | 不注入外部知识，用于基线实验 |
| `--history` | 启用有限对话历史；默认关闭 |

### 4.4 重新生成结果图

```bash
python scripts/plot_results.py
```

图件写入 `figures/generated/`。PNG 使用 330 dpi，SVG 为矢量格式。

### 4.5 调用评价指标

```python
import json
from evaluation.metrics import evaluate_all

with open("outputs/workflow_L3-2_kb1_mem0.json", encoding="utf-8") as file:
    generated_workflow = json.load(file)
with open("reference_workflows/expert_rusle_workflow.json", encoding="utf-8") as file:
    reference_workflow = json.load(file)
with open("outputs/script_L3-2_kb1_mem0.py", encoding="utf-8") as file:
    generated_code = file.read()
with open("path/to/expert_reference.py", encoding="utf-8") as file:
    reference_code = file.read()

metrics = evaluate_all(
    generated_workflow=generated_workflow,
    expected_workflow=reference_workflow,
    generated_code=generated_code,
    expected_code=reference_code,
    y_true=[1.0, 2.0, 3.0],
    y_pred=[0.9, 2.1, 2.8],
)
print(metrics)
```

真实栅格评价时，应将专家参考栅格与模型输出栅格的共同有效像元展开后传入 `y_true` 和 `y_pred`。仓库没有提供虚构的专家参考代码，示例中的 `path/to/expert_reference.py` 必须替换为经过确认的参考实现。

## 5. 输入输出示例

### 5.1 输入示例

数据配置文件的简化示例：

```json
{
  "dem_filled": "data/processed/dem_filled_2022.tif",
  "FlowAcc": "data/processed/flow_accumulation_2022.tif",
  "rainfall_year": "data/raw/rainfall_annual_2022.tif",
  "soil_sand": "data/raw/soil_sand.tif",
  "soil_silt": "data/raw/soil_silt.tif",
  "soil_clay": "data/raw/soil_clay.tif",
  "soil_toc": "data/raw/soil_toc.tif",
  "landcover": "data/raw/landcover_2022.tif",
  "ndvi": "data/raw/ndvi_2022.tif"
}
```

实验输入由三部分组成：

1. `--level L3-2`：累计加载 L1-1 至 L3-2 的全部知识。
2. `--task "..."`：用户的自然语言建模任务。
3. `--data-config ...`：可供模型规划时引用的逻辑数据键和本地路径。

### 5.2 工作流与代码输出示例

程序会生成类似以下结构的 JSON，而具体步骤和参数由所选模型、任务和知识层级决定：

```json
[
  {
    "step_id": 1,
    "operation": "Calculate soil erodibility factor",
    "description": "Calculate the K factor from sand, silt, clay and organic carbon rasters.",
    "factor_name": "K",
    "input_files": ["soil_sand", "soil_silt", "soil_clay", "soil_toc"],
    "output_file": "K_factor_2022.tif",
    "dependencies": [],
    "tool_library": "rasterio, numpy",
    "algorithm_formula": "EPIC soil erodibility equation",
    "parameters": {
      "constants": {},
      "thresholds": {},
      "lookup_table": {}
    }
  }
]
```

使用上述 L3-2、知识库开启、历史关闭的命令时，入口脚本会保存：

```text
outputs/
├── workflow_L3-2_kb1_mem0.json
└── script_L3-2_kb1_mem0.py
```

- `workflow_*.json`：模型生成的 DAG 工作流。
- `script_*.py`：模型根据该工作流生成的 Python 代码。

这两个文件是“模型生成产物”，不代表已经执行成功。运行 `script_*.py` 前应检查数据路径、公式、参数、单位、空间参考和输出目录。

### 5.3 图表输出示例

运行 `python scripts/plot_results.py` 后将生成：

```text
figures/generated/
├── figure7a_task_kcpi.png
├── figure7a_task_kcpi.svg
├── figure7b_knowledge_level_gain.png
├── figure8_wqi.svg
└── figure9_spatial_accuracy.svg
```

## 项目结构

```text
├── configs/                 # 实验设置和数据路径示例
├── data/                    # 数据说明；不包含公开原始栅格
├── evaluation/              # 评价指标和单元测试
├── figures/generated/       # 可复现图件
├── knowledge/               # 六级结构化知识库
├── modeling/rusle_tools/    # RUSLE 参考计算模块
├── reference_workflows/     # 专家和低知识工作流参考
├── results/                 # 结构化结果 CSV
├── scripts/                 # 实验、作图和发布检查脚本
└── src/                     # GeoAgent 核心实现
```

## 已知限制

- 仓库不提供原始栅格数据，也不保证第三方数据允许再分发。
- 模型生成具有随机性；严格复现应记录模型精确版本、推理框架、硬件、随机设置、运行日期和原始响应。
- `run_experiment.py` 只生成工作流和代码，不自动执行生成代码或验证栅格结果。
- 正式的软件许可证和引用信息应由全体作者确认后补充。

更多信息见 `docs/REPRODUCIBILITY.md`。
