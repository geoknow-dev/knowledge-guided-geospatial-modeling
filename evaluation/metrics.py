"""Metrics for evaluating generated geospatial workflows and Python code.

The formulas follow the supplied evaluation tables:

* WVR = N_valid / N_total
* WGD = abs(N_gen - N_exp) / N_exp
* WKHR = sum(Hit_i) / sum(Total_i)
* WQI = w1*WVR + w2*(1-WGD) + w3*WKHR
* CodeBLEU = alpha*BLEU + beta*WNM + gamma*AST + delta*DF
* KCPI = (WQI + CodeBLEU + (1-nRMSE)) / 3
* KLG = KCPI(L_k) - KCPI(L_{k-1})

The module intentionally uses only the Python standard library.  Workflow
records can be the JSON list format already used by ``知识LLM实验/output``.
"""

from __future__ import annotations

import ast
import io
import math
import re
import tokenize
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


DEFAULT_CODEBLEU_WEIGHTS = (0.25, 0.25, 0.25, 0.25)
KNOWLEDGE_CATEGORIES = ("Model", "Factor", "Tool", "Data", "Formula", "Param")


def _as_list(values: Iterable[Any] | None) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        return [values]
    return list(values)


def _require_same_length(y_true: Sequence[float], y_pred: Sequence[float]) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("at least one observation is required")


def _validate_unit_interval(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def _normalised_weights(weights: Sequence[float], expected_length: int) -> tuple[float, ...]:
    if len(weights) != expected_length:
        raise ValueError(f"expected {expected_length} weights")
    values = tuple(float(x) for x in weights)
    if any(x < 0 for x in values) or not math.isclose(sum(values), 1.0, abs_tol=1e-9):
        raise ValueError("weights must be non-negative and sum to 1")
    return values


def _tokens(code: str) -> list[str]:
    """Tokenise Python code while ignoring formatting and comments."""
    if not isinstance(code, str):
        raise TypeError("code must be a string")
    result: list[str] = []
    try:
        stream = io.StringIO(code).readline
        ignored = {tokenize.ENCODING, tokenize.ENDMARKER, tokenize.NL,
                   tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
                   tokenize.COMMENT}
        for item in tokenize.generate_tokens(stream):
            if item.type not in ignored:
                result.append(item.string)
    except (tokenize.TokenError, IndentationError):
        # Text/code fragments that are not complete Python are still usable
        # by the lexical metrics.
        result = re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|[^\s]", code)
    return result


def _ngrams(tokens: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _modified_precision(candidate: Sequence[str], reference: Sequence[str], n: int) -> float:
    candidate_counts = _ngrams(candidate, n)
    if not candidate_counts:
        return 0.0
    reference_counts = _ngrams(reference, n)
    clipped = sum(min(count, reference_counts[gram]) for gram, count in candidate_counts.items())
    return clipped / sum(candidate_counts.values())


def ngram_bleu(candidate: str, reference: str, max_n: int = 4,
               weights: Sequence[float] | None = None, smoothing: float = 1e-9) -> float:
    """Compute token-level BLEU with modified n-gram precision."""
    if max_n < 1:
        raise ValueError("max_n must be positive")
    candidate_tokens, reference_tokens = _tokens(candidate), _tokens(reference)
    if not candidate_tokens:
        return 1.0 if not reference_tokens else 0.0
    if not reference_tokens:
        return 0.0
    weights = _normalised_weights(weights or (1.0 / max_n,) * max_n, max_n)
    precisions = [_modified_precision(candidate_tokens, reference_tokens, n) for n in range(1, max_n + 1)]
    if any(p == 0 for p in precisions) and smoothing <= 0:
        return 0.0
    log_precision = sum(w * math.log(max(p, smoothing)) for w, p in zip(weights, precisions))
    brevity_penalty = 1.0 if len(candidate_tokens) > len(reference_tokens) else math.exp(
        1.0 - len(reference_tokens) / len(candidate_tokens)
    )
    return brevity_penalty * math.exp(log_precision)


def weighted_ngram_match(candidate: str, reference: str, max_n: int = 4,
                         weights: Sequence[float] | None = None) -> float:
    """Compute WNM = sum(w_n * p'_n), where p'_n is modified precision."""
    if max_n < 1:
        raise ValueError("max_n must be positive")
    weights = _normalised_weights(weights or (1.0 / max_n,) * max_n, max_n)
    candidate_tokens, reference_tokens = _tokens(candidate), _tokens(reference)
    if not candidate_tokens:
        return 1.0 if not reference_tokens else 0.0
    return sum(w * _modified_precision(candidate_tokens, reference_tokens, n)
               for w, n in zip(weights, range(1, max_n + 1)))


def _ast_signatures(code: str) -> Counter[str] | None:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, TypeError):
        return None
    return Counter(ast.dump(node, annotate_fields=False, include_attributes=False)
                   for node in ast.walk(tree))


def ast_match(candidate: str, reference: str) -> float:
    """AST match = |T_gen intersection T_ref| / |T_ref| (multiset form)."""
    candidate_nodes, reference_nodes = _ast_signatures(candidate), _ast_signatures(reference)
    if candidate_nodes is None or reference_nodes is None:
        return 0.0
    reference_total = sum(reference_nodes.values())
    if reference_total == 0:
        return 1.0 if not candidate_nodes else 0.0
    return sum((candidate_nodes & reference_nodes).values()) / reference_total


class _DataflowExtractor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.edges: set[tuple[str, str]] = set()

    @staticmethod
    def _names(node: ast.AST, context: type[ast.expr_context]) -> set[str]:
        return {item.id for item in ast.walk(node)
                if isinstance(item, ast.Name) and isinstance(item.ctx, context)}

    @staticmethod
    def _defined_names(node: ast.AST) -> set[str]:
        names: set[str] = set()
        for item in ast.walk(node):
            if isinstance(item, ast.Name) and isinstance(item.ctx, (ast.Store, ast.Del)):
                names.add(item.id)
            elif isinstance(item, ast.arg):
                names.add(item.arg)
        return names

    def _add_assignment_edges(self, target: ast.AST, value: ast.AST) -> None:
        sources = self._names(value, ast.Load)
        for target_name in self._defined_names(target):
            for source in sources:
                if source != target_name:
                    self.edges.add((source, target_name))

    def visit_Assign(self, node: ast.Assign) -> Any:
        for target in node.targets:
            self._add_assignment_edges(target, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        if node.value is not None:
            self._add_assignment_edges(node.target, node.value)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> Any:
        sources = self._names(node.value, ast.Load) | self._names(node.target, ast.Load)
        for target_name in self._defined_names(node.target):
            for source in sources:
                if source != target_name:
                    self.edges.add((source, target_name))
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> Any:
        self._add_assignment_edges(node.target, node.iter)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> Any:
        if node.value is not None:
            for source in self._names(node.value, ast.Load):
                self.edges.add((source, "RETURN"))
        self.generic_visit(node)


def _dataflow_edges(code: str) -> set[tuple[str, str]] | None:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, TypeError):
        return None
    extractor = _DataflowExtractor()
    extractor.visit(tree)
    return extractor.edges


def dataflow_match(candidate: str, reference: str) -> float:
    """Dataflow match = |D_gen intersection D_ref| / |D_ref|."""
    candidate_edges, reference_edges = _dataflow_edges(candidate), _dataflow_edges(reference)
    if candidate_edges is None or reference_edges is None:
        return 0.0
    if not reference_edges:
        return 1.0 if not candidate_edges else 0.0
    return len(candidate_edges & reference_edges) / len(reference_edges)


def codebleu(candidate: str, reference: str, alpha: float = 0.25, beta: float = 0.25,
             gamma: float = 0.25, delta: float = 0.25) -> float:
    """Compute CodeBLEU = alpha*BLEU + beta*WNM + gamma*AST + delta*DF."""
    weights = _normalised_weights((alpha, beta, gamma, delta), 4)
    components = (ngram_bleu(candidate, reference), weighted_ngram_match(candidate, reference),
                  ast_match(candidate, reference), dataflow_match(candidate, reference))
    return sum(weight * value for weight, value in zip(weights, components))


_CATEGORY_ALIASES = {
    "Model": ("model", "models", "model_name", "model_type"),
    "Factor": ("factor", "factors", "factor_name", "factor_names"),
    "Tool": ("tool", "tools", "tool_library", "tool_libraries", "library"),
    "Data": ("data", "dataset", "datasets", "input_files", "output_file", "output_files"),
    "Formula": ("formula", "formulas", "algorithm_formula", "algorithm", "method"),
    "Param": ("param", "params", "parameter", "parameters"),
}


def _workflow_steps(workflow: Any) -> list[Mapping[str, Any]]:
    if isinstance(workflow, Mapping):
        workflow = workflow.get("steps", workflow.get("workflow", []))
    steps = _as_list(workflow)
    if not all(isinstance(step, Mapping) for step in steps):
        raise TypeError("workflow must contain mapping-like step records")
    return list(steps)


def _nonempty(value: Any) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def _step_is_valid(step: Mapping[str, Any], all_ids: set[Any]) -> bool:
    if not _nonempty(step.get("operation")) or not _nonempty(step.get("description")):
        return False
    step_id = step.get("step_id")
    if step_id is not None and (not isinstance(step_id, int) or isinstance(step_id, bool)):
        return False
    dependencies = step.get("dependencies", [])
    if not isinstance(dependencies, (list, tuple)) or any(dep not in all_ids for dep in dependencies):
        return False
    return True


def workflow_validity_rate(workflow: Any) -> float:
    """WVR = number of complete, internally referenced steps / total steps."""
    steps = _workflow_steps(workflow)
    if not steps:
        return 0.0
    ids = {step.get("step_id") for step in steps}
    return sum(_step_is_valid(step, ids) for step in steps) / len(steps)


def workflow_granularity_deviation(generated: Any, expected: Any) -> float:
    """WGD = abs(number of generated steps - expected steps) / expected steps."""
    generated_count, expected_count = len(_workflow_steps(generated)), len(_workflow_steps(expected))
    if expected_count == 0:
        return 0.0 if generated_count == 0 else 1.0
    return abs(generated_count - expected_count) / expected_count


def _flatten(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        result: list[str] = []
        for item in value.values():
            result.extend(_flatten(item))
        return result
    if isinstance(value, (list, tuple, set)):
        result = []
        for item in value:
            result.extend(_flatten(item))
        return result
    return [str(value).strip().lower()]


def _knowledge_counts(workflow: Any, categories: Sequence[str]) -> tuple[int, int]:
    steps = _workflow_steps(workflow)
    total = hit = 0
    for category in categories:
        aliases = _CATEGORY_ALIASES.get(category, (category.lower(),))
        values: list[str] = []
        for step in steps:
            for key in aliases:
                if key in step:
                    values.extend(_flatten(step[key]))
        counts = Counter(values)
        total += sum(counts.values())
        hit += sum(counts.values())  # replaced by intersection in caller
    return hit, total


def knowledge_accuracy(generated: Any, expected: Any,
                       categories: Sequence[str] = KNOWLEDGE_CATEGORIES) -> float:
    """Backward-compatible name for the paper's WKHR metric."""
    generated_steps, expected_steps = _workflow_steps(generated), _workflow_steps(expected)
    hit = total = 0
    for category in categories:
        aliases = _CATEGORY_ALIASES.get(category, (category.lower(),))
        generated_values: list[str] = []
        expected_values: list[str] = []
        for step in generated_steps:
            for key in aliases:
                if key in step:
                    generated_values.extend(_flatten(step[key]))
        for step in expected_steps:
            for key in aliases:
                if key in step:
                    expected_values.extend(_flatten(step[key]))
        generated_counts, expected_counts = Counter(generated_values), Counter(expected_values)
        total += sum(expected_counts.values())
        hit += sum((generated_counts & expected_counts).values())
    return 1.0 if total == 0 and hit == 0 else hit / total


def workflow_knowledge_hit_rate(generated: Any, expected: Any,
                                categories: Sequence[str] = KNOWLEDGE_CATEGORIES) -> float:
    """WKHR = matched reference knowledge items / all reference knowledge items."""
    return knowledge_accuracy(generated, expected, categories)


def workflow_quality_index(wvr: float, wgd: float, wka: float,
                           weights: Sequence[float] = (1 / 3, 1 / 3, 1 / 3)) -> float:
    """WQI = w1*WVR + w2*(1-WGD) + w3*WKA."""
    wvr, wka = _validate_unit_interval(wvr, "wvr"), _validate_unit_interval(wka, "wka")
    wgd = float(wgd)
    if wgd < 0:
        raise ValueError("wgd must be non-negative")
    w1, w2, w3 = _normalised_weights(weights, 3)
    return w1 * wvr + w2 * (1.0 - wgd) + w3 * wka


def r2_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Coefficient of determination R²."""
    _require_same_length(y_true, y_pred)
    mean = sum(float(value) for value in y_true) / len(y_true)
    residual_sum = sum((float(actual) - float(predicted)) ** 2 for actual, predicted in zip(y_true, y_pred))
    total_sum = sum((float(actual) - mean) ** 2 for actual in y_true)
    if total_sum == 0:
        return 1.0 if residual_sum == 0 else 0.0
    return 1.0 - residual_sum / total_sum


def normalized_rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """nRMSE = RMSE / (y_max - y_min)."""
    _require_same_length(y_true, y_pred)
    values = [float(value) for value in y_true]
    rmse = math.sqrt(sum((actual - float(predicted)) ** 2
                         for actual, predicted in zip(values, y_pred)) / len(values))
    data_range = max(values) - min(values)
    if data_range == 0:
        if rmse == 0:
            return 0.0
        raise ValueError("nRMSE is undefined when y_max equals y_min")
    return rmse / data_range


def kcpi(wqi: float, codebleu_value: float, nrmse: float) -> float:
    """KCPI = 1/3 * (WQI + CodeBLEU + (1-nRMSE))."""
    wqi = _validate_unit_interval(wqi, "wqi")
    codebleu_value = _validate_unit_interval(codebleu_value, "codebleu")
    nrmse = float(nrmse)
    if nrmse < 0:
        raise ValueError("nrmse must be non-negative")
    return (wqi + codebleu_value + (1.0 - nrmse)) / 3.0


def knowledge_layer_gain(current_kcpi: float, previous_kcpi: float) -> float:
    """KLG(L_k) = KCPI(L_k) - KCPI(L_{k-1})."""
    return float(current_kcpi) - float(previous_kcpi)


def evaluate_all(generated_workflow: Any, expected_workflow: Any,
                 generated_code: str | None = None, expected_code: str | None = None,
                 y_true: Sequence[float] | None = None,
                 y_pred: Sequence[float] | None = None) -> dict[str, float]:
    """Evaluate all applicable metrics and return a flat, JSON-friendly mapping."""
    wvr = workflow_validity_rate(generated_workflow)
    wgd = workflow_granularity_deviation(generated_workflow, expected_workflow)
    wkhr = workflow_knowledge_hit_rate(generated_workflow, expected_workflow)
    result = {"WVR": wvr, "WGD": wgd, "WKHR": wkhr,
              "WQI": workflow_quality_index(wvr, wgd, wkhr)}
    if generated_code is not None and expected_code is not None:
        result.update({"BLEU": ngram_bleu(generated_code, expected_code),
                       "WNM": weighted_ngram_match(generated_code, expected_code),
                       "AST": ast_match(generated_code, expected_code),
                       "Dataflow": dataflow_match(generated_code, expected_code)})
        result["CodeBLEU"] = codebleu(generated_code, expected_code)
    if y_true is not None and y_pred is not None:
        result["R2"] = r2_score(y_true, y_pred)
        result["nRMSE"] = normalized_rmse(y_true, y_pred)
    if "CodeBLEU" in result and "nRMSE" in result:
        result["KCPI"] = kcpi(result["WQI"], result["CodeBLEU"], result["nRMSE"])
    return result


__all__ = [
    "ast_match", "codebleu", "dataflow_match", "evaluate_all", "kcpi",
    "knowledge_accuracy", "knowledge_layer_gain", "ngram_bleu", "normalized_rmse",
    "r2_score", "weighted_ngram_match", "workflow_granularity_deviation",
    "workflow_knowledge_hit_rate", "workflow_quality_index", "workflow_validity_rate",
]
