import os
import json
import re
from typing import Any, Dict, List, Tuple, Union

from openai import OpenAI


class GeoAgent:
    """
    GeoAgent with two independent switches:
      1) conversation_history_enabled: whether to include dialogue history in each API call
      2) kb_injection_enabled: whether to inject external KB json files into system prompt

    Also supports:
      - temperature: sampling temperature
      - max_history_messages: cap for stored messages (user+assistant each count as 1 message)
    """

    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_key: str,
        data_paths: Dict[str, Any],
        knowledge_base_dir: str,
        conversation_history_enabled: bool = False,
        kb_injection_enabled: bool = True,
        temperature: float = 0.1,
        max_history_messages: int = 12,
    ):
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model = model_name
        self.data_paths = data_paths
        self.kb_dir = knowledge_base_dir

        # ✅ switches
        self.conversation_history_enabled = bool(conversation_history_enabled)
        self.kb_injection_enabled = bool(kb_injection_enabled)

        # ✅ generation controls
        self.temperature = float(temperature)

        # ✅ history buffer
        self._history: List[Dict[str, str]] = []
        self.max_history_messages = max(0, int(max_history_messages))

        # Knowledge level mapping
        self.level_mapping = {
            "L1-1": ["L1_1_concepts.json"],
            "L1-2": ["L1_1_concepts.json", "L1_2_processes.json"],
            "L2-1": ["L1_1_concepts.json", "L1_2_processes.json", "L2_1_structure.json"],
            "L2-2": ["L1_1_concepts.json", "L1_2_processes.json", "L2_1_structure.json", "L2_2_tools.json"],
            "L3-1": [
                "L1_1_concepts.json",
                "L1_2_processes.json",
                "L2_1_structure.json",
                "L2_2_tools.json",
                "L3_1_formulas.json",
            ],
            "L3-2": [
                "L1_1_concepts.json",
                "L1_2_processes.json",
                "L2_1_structure.json",
                "L2_2_tools.json",
                "L3_1_formulas.json",
                "L3_2_parameters.json",
            ],
        }

    # -------------------------
    # Switch controls
    # -------------------------
    def set_conversation_history(self, enabled: bool, clear_when_disable: bool = True) -> None:
        """Enable/disable dialogue history; optionally clear when disabling."""
        self.conversation_history_enabled = bool(enabled)
        if (not self.conversation_history_enabled) and clear_when_disable:
            self.clear_conversation_history()

    def clear_conversation_history(self) -> None:
        """Clear stored dialogue history."""
        self._history = []

    def set_kb_injection(self, enabled: bool) -> None:
        """Enable/disable KB injection."""
        self.kb_injection_enabled = bool(enabled)

    def set_temperature(self, temperature: float) -> None:
        """Set sampling temperature."""
        self.temperature = float(temperature)

    # -------------------------
    # Knowledge loading
    # -------------------------
    def _load_knowledge(self, level: str) -> str:
        """Load and merge KB json files for given level (if kb_injection_enabled)."""
        if not self.kb_injection_enabled:
            return "{}"

        if level not in self.level_mapping:
            print(f"⚠️ Warning: 等级 {level} 未定义，将仅依赖模型内部知识。")
            return "{}"

        files = self.level_mapping[level]
        combined: Dict[str, Any] = {}

        for fname in files:
            path = os.path.join(self.kb_dir, fname)
            if not os.path.exists(path):
                print(f"⚠️ Warning: 找不到知识文件 {path}")
                continue

            try:
                with open(path, "r", encoding="utf-8") as f:
                    key_name = fname.replace(".json", "")
                    combined[key_name] = json.load(f)
            except json.JSONDecodeError:
                print(f"❌ Error: 文件 {fname} JSON 格式错误，已跳过。")

        return json.dumps(combined, indent=2, ensure_ascii=False)

    # -------------------------
    # Content extraction
    # -------------------------
    def _extract_content(self, text: str, type: str = "json") -> str:
        """Extract content from fenced code blocks if present."""
        if type == "json":
            pattern = r"```json(.*?)```"
        else:
            pattern = r"```python(.*?)```"

        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # fallback: strip any fences
        return (
            text.replace(f"```{type}", "")
            .replace("```json", "")
            .replace("```python", "")
            .replace("```", "")
            .strip()
        )

    # -------------------------
    # LLM call (streaming)
    # -------------------------
    def _trim_history(self) -> None:
        """Keep only the last max_history_messages."""
        if self.max_history_messages <= 0:
            self._history = []
            return
        if len(self._history) > self.max_history_messages:
            self._history = self._history[-self.max_history_messages :]

    def _call_llm(self, sys_msg: str, user_msg: str) -> str:
        """Stream LLM output; optionally include conversation history."""
        print("\n🤖 [AI 生成中...] ", end="", flush=True)

        messages: List[Dict[str, str]] = [{"role": "system", "content": sys_msg}]
        if self.conversation_history_enabled and self._history:
            messages.extend(self._history)
        messages.append({"role": "user", "content": user_msg})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                stream=True,
            )

            full_content: List[str] = []
            for chunk in response:
                delta = getattr(chunk.choices[0], "delta", None)
                content = getattr(delta, "content", None) if delta is not None else None
                if content:
                    print(content, end="", flush=True)
                    full_content.append(content)

            assistant_text = "".join(full_content)
            print("\n")

            if self.conversation_history_enabled:
                self._history.append({"role": "user", "content": user_msg})
                self._history.append({"role": "assistant", "content": assistant_text})
                self._trim_history()

            return assistant_text

        except Exception as e:
            print(f"\n❌ API Error: {str(e)}")
            return ""

    # -------------------------
    # Planning
    # -------------------------
    def plan(self, task: str, level: str) -> Tuple[str, Union[List[Dict[str, Any]], str]]:
        """Stage 2: 认知与规划"""

        knowledge_text = self._load_knowledge(level)
        data_env_str = json.dumps(self.data_paths, indent=2, ensure_ascii=False)

        landcover_metadata = """
【数据语义补充 - 土地覆被分类体系 (UMD)】
注意：以下内容仅为 'landcover' 数据的像素值含义定义。请根据这些定义，结合你的地理知识或外部规则库，自行决定如何处理 P 因子赋值。

0: 水体
1: 常绿针叶林
2: 常绿阔叶林
3: 落叶针叶林
4: 落叶阔叶林
5: 混交林
6: 郁闭灌丛
7: 开放灌丛
8: 萨瓦纳
9: 稀树草原
10: 草地
11: 永久性湿地
12: 农田
13: 城市和建筑用地
14: 农田/自然植被镶嵌
15: 非植被土地
"""

        system_prompt = f"""
你是一名专业的地理空间智能体 (GeoAI Agent)。

【认知模式】
1) 内部知识：利用你预训练的通用逻辑推理能力和编程常识。
2) 外部知识（已注入，可能为空）：
{knowledge_text}

决策规则：当外部知识（特定公式、参数、工具推荐）与你的内部常识冲突时，必须优先遵循外部知识。

【数据环境 (只读)】
本地数据路径如下，规划时请参考这些可用的数据资源：
{data_env_str}

{landcover_metadata}

【重要约束】
- 所有输入数据均已经过预处理（对齐坐标系、填洼、分辨率一致）。
- 严禁生成任何数据预处理步骤（Reproject/Resample/Fill Sinks 等）。
- 直接规划核心计算与建模步骤。
- 每个计算步骤必须显式给出 parameters（常数/阈值/分段/查表等）。
"""

        user_prompt = f"""
【用户任务】
{task}

【输出要求】
请规划一个详细的科学计算工作流 (DAG)。
必须以 JSON 列表格式返回（只返回 JSON，不要额外解释文字）。

JSON 模板:
[
  {{
    "step_id": <整数, 从1开始>,
    "operation": "<操作名称>",
    "description": "<步骤详细描述>",
    "factor_name": "<R/K/LS/C/P/A等>",
    "input_files": ["<文件名或数据键>"],
    "output_file": "<文件名>",
    "dependencies": [<依赖step_id列表>],
    "tool_library": "<推荐库>",
    "algorithm_formula": "<算法或公式名称(具体到模型名)>",
    "parameters": {{
      "constants": {{ "<参数名>": <数值> }},
      "thresholds": {{ "<阈值名>": <数值> }},
      "lookup_table": {{
        "<表名>": [
          {{ "condition": "<条件>", "value": <数值或表达式> }}
        ]
      }},
      "units": "<单位可选>",
      "notes": "<口径说明可选>"
    }}
  }}
]

【关键提示（必须遵守）】
1) 依赖闭合：dependencies 引用的步骤，其 output_file 必须出现在本步骤 input_files（或在 description 明确说明派生关系）。
2) 必须显式写 dependencies 构建 DAG。
3) 禁止用 ALGO_XXX/默认参数/查表 等占位符替代具体参数；参数要落到 constants/thresholds/lookup_table。
"""

        raw = self._call_llm(system_prompt, user_prompt)

        clean = self._extract_content(raw, type="json")
        try:
            workflow = json.loads(clean)
            return system_prompt, workflow
        except json.JSONDecodeError:
            return system_prompt, clean

    # -------------------------
    # Code generation
    # -------------------------
    def generate_code(self, sys_context: str, workflow: Union[List[Dict[str, Any]], str]) -> str:
        """Stage 3: 代码生成"""

        workflow_str = workflow if isinstance(workflow, str) else json.dumps(workflow, ensure_ascii=False, indent=2)

        user_prompt = f"""
基于以下详细工作流计划编写 Python 代码：
{workflow_str}

【编程约束】
1) 严格实现：依据 tool_library / algorithm_formula / parameters 尽可能实现指定逻辑与参数。
2) 数据流控制：变量传递与 input/output/dependencies 一致。
3) 环境一致性：直接使用【数据环境】中提供的路径或键映射。
4) 结果保存：保存到 output/ 文件夹。
5) 鲁棒性：处理 NoData（-9999 或 NaN）。
6) 仅输出 Python 代码块。
"""

        raw = self._call_llm(sys_context, user_prompt)
        return self._extract_content(raw, type="python")
