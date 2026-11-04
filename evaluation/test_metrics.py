import unittest


from evaluation.metrics import (
    ast_match,
    codebleu,
    dataflow_match,
    knowledge_accuracy,
    knowledge_layer_gain,
    kcpi,
    normalized_rmse,
    r2_score,
    weighted_ngram_match,
    workflow_granularity_deviation,
    workflow_quality_index,
    workflow_validity_rate,
    ngram_bleu,
    evaluate_all,
)


class EvaluationMetricsTest(unittest.TestCase):
    def setUp(self):
        self.reference_workflow = [
            {
                "step_id": 1,
                "operation": "读取DEM",
                "description": "加载数字高程模型",
                "factor_name": "DEM",
                "tool_library": "rasterio",
                "algorithm_formula": "read",
                "input_files": ["dem.tif"],
                "output_file": "dem_filled.tif",
            },
            {
                "step_id": 2,
                "operation": "计算坡度",
                "description": "从DEM计算坡度",
                "factor_name": "S",
                "tool_library": "numpy",
                "algorithm_formula": "gradient",
                "input_files": ["dem_filled.tif"],
                "output_file": "slope.tif",
                "dependencies": [1],
            },
        ]

    def test_workflow_metrics_follow_the_formulas(self):
        self.assertEqual(workflow_validity_rate(self.reference_workflow), 1.0)
        generated = self.reference_workflow[:1]
        self.assertAlmostEqual(workflow_granularity_deviation(generated, self.reference_workflow), 0.5)
        self.assertAlmostEqual(knowledge_accuracy(self.reference_workflow, self.reference_workflow), 1.0)
        self.assertAlmostEqual(workflow_quality_index(1.0, 0.5, 1.0), 0.8333333333333334)

    def test_workflow_invalid_step_is_not_counted(self):
        workflow = [{"step_id": 1, "operation": "", "description": ""}]
        self.assertEqual(workflow_validity_rate(workflow), 0.0)

    def test_code_metrics_detect_exact_and_partial_matches(self):
        reference = "def add(a, b):\n    return a + b\n"
        candidate = "def add(a, b):\n    return a + b\n"
        self.assertEqual(ngram_bleu(candidate, reference), 1.0)
        self.assertEqual(weighted_ngram_match(candidate, reference), 1.0)
        self.assertEqual(ast_match(candidate, reference), 1.0)
        self.assertEqual(dataflow_match(candidate, reference), 1.0)
        self.assertEqual(codebleu(candidate, reference), 1.0)

        partial = "def add(a, b):\n    return a - b\n"
        self.assertLess(codebleu(partial, reference), 1.0)
        self.assertLess(ast_match(partial, reference), 1.0)

    def test_regression_metrics_and_knowledge_metrics(self):
        self.assertAlmostEqual(r2_score([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertAlmostEqual(normalized_rmse([0, 1, 2], [0, 1, 1]), (1 / 3) ** 0.5 / 2)
        self.assertAlmostEqual(kcpi(0.8, 0.7, 0.2), (0.8 + 0.7 + 0.8) / 3)
        self.assertAlmostEqual(knowledge_layer_gain(0.8, 0.6), 0.2)

    def test_evaluate_all_returns_the_complete_flat_result(self):
        result = evaluate_all(
            self.reference_workflow,
            self.reference_workflow,
            "x = 1\nreturn_value = x\n",
            "x = 1\nreturn_value = x\n",
            [0, 1, 2],
            [0, 1, 1],
        )
        self.assertEqual(
            set(result),
            {"WVR", "WGD", "WKHR", "WQI", "BLEU", "WNM", "AST", "Dataflow", "CodeBLEU", "R2", "nRMSE", "KCPI"},
        )


if __name__ == "__main__":
    unittest.main()
