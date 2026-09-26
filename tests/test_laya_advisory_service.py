import json
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from laya_advisory_service import (
    AdvisoryService,
    FROZEN_CALIBRATION_SHA256,
    FROZEN_MODEL_SHA256,
    FROZEN_POLICY_SHA256,
    RequestValidationError,
    create_http_server,
)


POLICY = {
    "status": "frozen_from_tune",
    "policy_family": "B",
    "policy_name": "concentration",
    "thresholds": {"choice:2": 0.9, "choice:3-5": 0.9},
}


def stage(name, prediction, bucket, concentration=0.95):
    return {
        "stage": name,
        "prediction": prediction,
        "temperature_bucket": bucket,
        "entropy": 0.1,
        "normalized_entropy": 0.05,
        "concentration": concentration,
        "top1_probability": 0.95,
        "top2_probability": 0.05,
        "top1_top2_margin": 0.9,
        "raw_probabilities": [0.97, 0.03],
        "calibrated_probabilities": [0.95, 0.05],
    }


class FakeRuntime:
    model_sha256 = FROZEN_MODEL_SHA256
    config_sha256 = FROZEN_CALIBRATION_SHA256
    device = "cpu-test"

    def __init__(self, results=None):
        self.results = list(results or [])
        self.calls = []

    def predict(self, state):
        self.calls.append(state)
        if self.results:
            return self.results.pop(0)
        return {
            "final_label": "ready_now",
            "path": [{"stage": "stage1", "prediction": "sufficient_information"},
                     {"stage": "stage2", "prediction": "ready_now"}],
            "stages": [stage("stage1", "sufficient_information", "choice:2"),
                       stage("stage2", "ready_now", "choice:2")],
        }


def result(final_label, paths, stages):
    return {"final_label": final_label, "path": paths, "stages": stages}


class AdvisoryServiceTests(unittest.TestCase):
    def make_service(self, runtime_factory=None, runtime=None):
        factory = runtime_factory or (lambda: runtime or FakeRuntime())
        return AdvisoryService(
            runtime_factory=factory,
            policy_config=POLICY,
            policy_sha256=FROZEN_POLICY_SHA256,
        )

    def test_service_loads_one_runtime_and_reuses_it_for_repeated_decisions(self):
        runtime = FakeRuntime()
        loads = []

        def factory():
            loads.append(1)
            return runtime

        service = self.make_service(runtime_factory=factory)
        first = service.handle_decide({"state": "state A", "task": "task A", "request_id": "r1"})
        second = service.handle_decide({"state": "state B", "task": "task B", "request_id": "r2"})
        self.assertEqual(len(loads), 1)
        self.assertEqual(runtime.calls, [{"situation": "state A", "task": "task A"},
                                         {"situation": "state B", "task": "task B"}])
        self.assertEqual((first["request_id"], second["request_id"]), ("r1", "r2"))

    def test_health_reports_verified_loaded_provenance(self):
        health = self.make_service(runtime=FakeRuntime()).health()
        self.assertEqual(health["status"], "ok")
        self.assertTrue(health["model_loaded"])
        self.assertEqual(health["model_sha256"], FROZEN_MODEL_SHA256)
        self.assertEqual(health["calibration_sha256"], FROZEN_CALIBRATION_SHA256)
        self.assertEqual(health["policy_sha256"], FROZEN_POLICY_SHA256)

    def test_response_contains_advisory_schema_and_never_authorizes_autonomy(self):
        response = self.make_service(runtime=FakeRuntime()).handle_decide({
            "state": "evidence", "request_id": "req-1",
        })
        self.assertEqual(response["advisory_status"], "advisory_only")
        self.assertEqual(response["decision"]["final_label"], "ready_now")
        self.assertEqual(len(response["decision"]["hierarchical_path"]), 2)
        self.assertEqual(len(response["stages"]), 2)
        self.assertEqual(response["autonomy_policy"], {
            "status": "experimental_failed_gate",
            "authorizes_autonomous_execution": False,
        })
        self.assertNotIn("safe_to_execute", response)
        self.assertEqual(response["provenance"]["model_sha256"], FROZEN_MODEL_SHA256)

    def test_uncertainty_is_reported_per_stage_without_correctness_probability(self):
        response = self.make_service(runtime=FakeRuntime()).handle_decide({"state": "evidence"})
        self.assertEqual(len(response["uncertainty"]["by_stage"]), 2)
        self.assertIn("entropy", response["uncertainty"]["by_stage"][0])
        self.assertIn("concentration", response["uncertainty"]["by_stage"][0])
        self.assertNotIn("correctness_probability", response["uncertainty"])
        self.assertTrue(all("raw_probabilities" in stage_row for stage_row in response["stages"]))
        self.assertTrue(all("calibrated_probabilities" in stage_row for stage_row in response["stages"]))

    def test_experimental_clear_is_not_autonomous_authorization(self):
        response = self.make_service(runtime=FakeRuntime()).handle_decide({"state": "evidence"})
        self.assertEqual(response["experimental_policy_result"], "CLEAR")
        self.assertFalse(response["autonomy_policy"]["authorizes_autonomous_execution"])

    def test_low_concentration_requests_review_without_changing_decision(self):
        runtime = FakeRuntime([result(
            "needs_local_setup",
            [{"stage": "stage1", "prediction": "sufficient_information"},
             {"stage": "stage2", "prediction": "not_ready_now"},
             {"stage": "stage3", "prediction": "local_setup"}],
            [stage("stage1", "sufficient_information", "choice:2", 0.95),
             stage("stage2", "not_ready_now", "choice:2", 0.95),
             stage("stage3", "local_setup", "choice:3-5", 0.2)],
        )])
        response = self.make_service(runtime=runtime).handle_decide({"state": "evidence"})
        self.assertEqual(response["decision"]["final_label"], "needs_local_setup")
        self.assertEqual(response["experimental_policy_result"], "REVIEW_REQUIRED")
        self.assertFalse(response["autonomy_policy"]["authorizes_autonomous_execution"])

    def test_stage1_and_stage2_stops_preserve_runtime_paths(self):
        cases = [
            result("insufficient_information",
                   [{"stage": "stage1", "prediction": "insufficient_information"}],
                   [stage("stage1", "insufficient_information", "choice:2")]),
            result("ready_now",
                   [{"stage": "stage1", "prediction": "sufficient_information"},
                    {"stage": "stage2", "prediction": "ready_now"}],
                   [stage("stage1", "sufficient_information", "choice:2"),
                    stage("stage2", "ready_now", "choice:2")]),
        ]
        for result_obj, expected_count in zip(cases, (1, 2)):
            with self.subTest(expected_count=expected_count):
                service = self.make_service(runtime=FakeRuntime([result_obj]))
                response = service.handle_decide({"state": "evidence"})
                self.assertEqual(len(response["stages"]), expected_count)
                self.assertEqual(len(response["decision"]["hierarchical_path"]), expected_count)

    def test_malformed_and_empty_requests_fail_closed(self):
        service = self.make_service(runtime=FakeRuntime())
        for payload in ({}, {"state": "  "}, {"state": []}, {"state": "x", "task": 4}):
            with self.subTest(payload=payload), self.assertRaises(RequestValidationError):
                service.handle_decide(payload)

    def test_runtime_and_frozen_provenance_mismatch_fail_closed(self):
        class BadRuntime(FakeRuntime):
            model_sha256 = "0" * 64

        with self.assertRaises(ValueError):
            self.make_service(runtime=BadRuntime())

    def test_http_health_decide_errors_and_loopback_binding(self):
        service = self.make_service(runtime=FakeRuntime())
        server = create_http_server("127.0.0.1", 0, service)
        self.assertEqual(server.server_address[0], "127.0.0.1")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            with urlopen(base + "/health", timeout=2) as response:
                self.assertEqual(json.load(response)["status"], "ok")
            request = Request(base + "/decide", data=json.dumps({"state": "evidence"}).encode(),
                              headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(request, timeout=2) as response:
                body = json.load(response)
                self.assertFalse(body["autonomy_policy"]["authorizes_autonomous_execution"])
            bad = Request(base + "/decide", data=b"{bad", headers={"Content-Type": "application/json"}, method="POST")
            with self.assertRaises(HTTPError) as caught:
                urlopen(bad, timeout=2)
            self.assertEqual(caught.exception.code, 400)
            empty = Request(base + "/decide", data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
            with self.assertRaises(HTTPError) as caught:
                urlopen(empty, timeout=2)
            self.assertEqual(caught.exception.code, 422)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
