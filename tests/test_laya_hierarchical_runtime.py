import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from laya_hierarchical_runtime import (
    CalibrationConfigError,
    HierarchicalRuntime,
    ModelIntegrityError,
    RuntimeInferenceError,
    resolve_temperature_bucket,
)
FROZEN_PHASE3_TAXONOMY_SHA256 = "dc4f29b07ff75fea5fc235cc83c83326d1370088e64200385351a5f0dc095159"
from laya_hierarchical_runtime import STAGE_DEFINITIONS


class FakeModel(torch.nn.Module):
    def forward(self, logits):
        return logits, torch.zeros((1, 1))


class FakeAgent:
    temperature = [1.0, 1.0, 1.0]
    temperature_by_options = {}
    device = torch.device("cpu")

    def __init__(self, predictions=None, fail=False):
        self.model = FakeModel()
        self.predictions = predictions or {}
        self.fail = fail
        self.seen_states = []

    def predict(self, state, questions):
        if self.fail:
            raise RuntimeError("simulated forward failure")
        self.seen_states.append(state)
        qid, qdef = next(iter(questions.items()))
        keys = list(qdef["criteria"])
        logits = self.predictions.get(qid, [2.0] + [0.0] * (len(keys) - 1))
        tensor = torch.tensor([logits], dtype=torch.float32)
        self.model(tensor)
        p = np.exp(np.asarray(logits, dtype=float) - np.max(logits))
        p /= p.sum()
        probs = {key: round(float(value), 4) for key, value in zip(keys, p)}
        return {"answers": {qid: {"choice": keys[int(np.argmax(p))], "probabilities": probs}}}


class RuntimeTestCase(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.model = root / "model"
        self.model.mkdir()
        self.weights = self.model / "model.safetensors"
        self.weights.write_bytes(b"fixture model weights")
        digest = hashlib.sha256(self.weights.read_bytes()).hexdigest()
        self.config = root / "runtime_config.json"
        self.config.write_text(json.dumps({
            "model_sha256": digest,
            "temperature_by_options": {"choice:2": 5.0, "choice:3-5": 4.202408887068698},
            "status": "frozen_after_preregistered_acceptance_rule",
            "method": "deterministic geometric grid search minimizing NLL",
        }), encoding="utf-8")
        self.predictions = {
            "information_sufficiency": [3.0, 0.0],
            "current_readiness": [3.0, 0.0],
            "remedy_type": [3.0, 0.0, -1.0, -2.0],
        }

    def tearDown(self):
        self.tempdir.cleanup()

    def runtime(self, predictions=None, fail=False, config=None):
        digest = hashlib.sha256(self.weights.read_bytes()).hexdigest()
        return HierarchicalRuntime(
            model_path=self.model,
            calibration_config_path=config or self.config,
            agent=FakeAgent(predictions if predictions is not None else self.predictions, fail=fail),
            expected_model_sha256=digest,
        )

    def test_stage1_insufficient_stops_hierarchy(self):
        predictions = dict(self.predictions)
        predictions["information_sufficiency"] = [0.0, 4.0]
        result = self.runtime(predictions).predict("A complete state description")
        self.assertEqual(result["final_label"], "insufficient_information")
        self.assertEqual([x["stage"] for x in result["stages"]], ["stage1"])

    def test_stage2_ready_stops_hierarchy(self):
        result = self.runtime().predict("A complete state description")
        self.assertEqual(result["final_label"], "ready_now")
        self.assertEqual([x["stage"] for x in result["stages"]], ["stage1", "stage2"])

    def test_stage3_final_label_routes_are_mapped(self):
        expected = {
            "local_setup": "needs_local_setup",
            "external_access": "needs_external_access",
            "other_environment": "needs_other_environment",
            "no_allowed_remedy": "blocked",
        }
        for index, (stage3_label, final_label) in enumerate(expected.items()):
            with self.subTest(stage3_label=stage3_label):
                predictions = dict(self.predictions)
                predictions["current_readiness"] = [0.0, 4.0]
                logits = [-2.0] * 4
                logits[index] = 4.0
                predictions["remedy_type"] = logits
                result = self.runtime(predictions).predict("A complete state description")
                self.assertEqual(result["final_label"], final_label)
                self.assertEqual([x["stage"] for x in result["stages"]], ["stage1", "stage2", "stage3"])

    def test_output_schema_bucket_calibration_and_uncertainty(self):
        result = self.runtime().predict_stage("A nonempty state", "information_sufficiency")
        self.assertEqual(set(result), {"final_label", "path", "stages", "calibration", "runtime"})
        stage = result["stages"][0]
        self.assertEqual(stage["temperature_bucket"], "choice:2")
        self.assertEqual(stage["temperature"], 5.0)
        self.assertEqual(len(stage["raw_probabilities"]), 2)
        self.assertAlmostEqual(sum(stage["calibrated_probabilities"]), 1.0)
        self.assertEqual(stage["prediction"], stage["raw_prediction"])
        self.assertNotIn("confidence", stage)
        self.assertNotIn("correctness_probability", stage)
        self.assertAlmostEqual(stage["normalized_entropy"], stage["entropy"] / math.log(2), places=10)
        self.assertAlmostEqual(stage["concentration"], 1 - stage["normalized_entropy"], places=10)
        self.assertEqual(result["calibration"]["choice:2_temperature"], 5.0)
        full = self.runtime().predict("A complete state description")
        self.assertEqual(full["stages"][1]["temperature_bucket"], "choice:2")

    def test_string_state_uses_frozen_evaluator_normalization(self):
        runtime = self.runtime()
        runtime.predict_stage("short state", "information_sufficiency")
        self.assertEqual(runtime.agent.seen_states[-1], {"situation": "short state"})

    def test_stage3_uses_choice_3_5_bucket_and_argmax_is_unchanged(self):
        predictions = dict(self.predictions)
        predictions["current_readiness"] = [0.0, 4.0]
        result = self.runtime(predictions).predict("A complete state description")
        stage = result["stages"][2]
        self.assertEqual(stage["temperature_bucket"], "choice:3-5")
        self.assertAlmostEqual(stage["temperature"], 4.202408887068698)
        self.assertEqual(stage["raw_prediction"], stage["calibrated_prediction"])
        self.assertEqual(stage["prediction"], stage["raw_prediction"])

    def test_temperature_buckets_and_frozen_taxonomy_match_phase3(self):
        self.assertEqual(resolve_temperature_bucket(2), "choice:2")
        self.assertEqual(resolve_temperature_bucket(4), "choice:3-5")
        with self.assertRaises(ValueError):
            resolve_temperature_bucket(6)
        taxonomy = {
            stage_id: {"instructions": value["instructions"], "criteria": value["criteria"]}
            for stage_id, value in STAGE_DEFINITIONS.items()
        }
        payload = json.dumps(taxonomy, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.assertEqual(hashlib.sha256(payload).hexdigest(), FROZEN_PHASE3_TAXONOMY_SHA256)
    def test_invalid_state_stage_and_options_are_rejected(self):
        runtime = self.runtime()
        with self.assertRaises(ValueError):
            runtime.predict("")
        with self.assertRaises(ValueError):
            runtime.predict_stage("state", "not_a_stage")
        with self.assertRaises(ValueError):
            runtime.predict_stage("state", "information_sufficiency", options=["only_one"])
        with self.assertRaises(ValueError):
            runtime.predict_stage("state", "information_sufficiency", options=["sufficient_information", "wrong"])

    def test_missing_calibration_and_invalid_temperature_fail_loudly(self):
        with self.assertRaises(FileNotFoundError):
            self.runtime(config=self.config.parent / "missing.json")
        bad = self.config.parent / "bad.json"
        bad.write_text(json.dumps({"model_sha256": hashlib.sha256(self.weights.read_bytes()).hexdigest(),
                                   "temperature_by_options": {"choice:2": float("nan"), "choice:3-5": 1.0}}), encoding="utf-8")
        with self.assertRaises(CalibrationConfigError):
            self.runtime(config=bad)

    def test_model_sha_is_verified(self):
        config = self.config.parent / "wrong_hash.json"
        config.write_text(json.dumps({"model_sha256": "0" * 64,
                                      "temperature_by_options": {"choice:2": 1.0, "choice:3-5": 1.0}}), encoding="utf-8")
        with self.assertRaises(ModelIntegrityError):
            HierarchicalRuntime(model_path=self.model, calibration_config_path=self.config, agent=FakeAgent())
        with self.assertRaises(CalibrationConfigError):
            self.runtime(config=config)

    def test_missing_model_fails_before_inference(self):
        with self.assertRaises(FileNotFoundError):
            HierarchicalRuntime(model_path=self.model / "missing", calibration_config_path=self.config, agent=FakeAgent())

    def test_cuda_unavailable_warns_and_uses_laya_cpu_path(self):
        import laya
        digest = hashlib.sha256(self.weights.read_bytes()).hexdigest()
        with patch("torch.cuda.is_available", return_value=False), patch("laya.load", return_value=FakeAgent()) as loader:
            with self.assertWarnsRegex(RuntimeWarning, "CUDA is unavailable"):
                runtime = HierarchicalRuntime(model_path=self.model, calibration_config_path=self.config,
                                              expected_model_sha256=digest)
        loader.assert_called_once_with(str(self.model.resolve()), device="cpu")
        self.assertEqual(runtime.device, "cpu")

    def test_inference_failure_is_not_silently_fallbacked(self):
        with self.assertRaises(RuntimeInferenceError):
            self.runtime(fail=True).predict("A complete state description")


if __name__ == "__main__":
    unittest.main()
