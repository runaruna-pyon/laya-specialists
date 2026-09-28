from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

from phase7D_experience import VaultExperienceStore
from phase7D_specialist import FLAG_LABELS, PRIMARY_LABELS
from phase7D_specialist_service import SpecialistService, create_server
from phase7D_specialist_client import SpecialistUnavailable
from phase7D_specialist_service import read_checkpoint_path
from laya_specialist_client import (
    AdapterUnavailable,
    ModelAgnosticSpecialistClient,
    normalize_common_request,
    read_specialist_endpoint,
    to_c1_input,
)


class RecordingRuntime:
    device = "cpu"

    def __init__(self):
        self.inputs = []

    def predict(self, text):
        self.inputs.append(text)
        primary = "verification_required"
        return {
            "primary_logits": {label: 2.0 if label == primary else 0.0 for label in PRIMARY_LABELS},
            "primary_probabilities": {label: 0.7 if label == primary else 0.1 for label in PRIMARY_LABELS},
            "flag_logits": {flag: -1.0 for flag in FLAG_LABELS},
            "flag_probabilities": {flag: 0.2 for flag in FLAG_LABELS},
        }


class Phase7EModelAgnosticTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        vault = Path(self.tmp.name) / "vault"
        vault.mkdir()
        self.runtime = RecordingRuntime()
        service = SpecialistService(self.runtime, VaultExperienceStore(vault), "f" * 64)
        self.server = create_server("127.0.0.1", 0, service)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.client = ModelAgnosticSpecialistClient(self.url, expected_checkpoint_sha256="f" * 64)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.tmp.cleanup()

    @staticmethod
    def request(agent):
        return {
            "request_id": "same-semantic-request",
            "source_agent": {"provider": "openai", "model": agent, "interface": "common-json"},
            "project": "HPCV",
            "problem_context": "Audio decoder latency experiment",
            "proposal": "Adopt decoder X for the next bounded test",
            "claim_under_review": "Decoder X caused the measured latency improvement",
            "known_evidence": {
                "observed_facts": "Latency fell in one run, while batch size also changed.",
                "available_evidence": "One before/after run; no isolated comparison.",
                "missing_evidence": "A controlled comparison separating decoder and batch size.",
            },
            "requested_decision": "Assess whether the causal claim is supported.",
        }

    def test_luna_astra_sol_codex_same_content_produces_identical_c1_decision(self):
        results = [self.client.decide(self.request(agent)) for agent in ("luna", "astra", "sol", "codex")]
        self.assertEqual(len(set(self.runtime.inputs)), 1)
        stable = [{key: result[key] for key in ("primary", "flags", "uncertainty", "advisory", "authorizes_autonomous_execution")}
                  for result in results]
        self.assertTrue(all(value == stable[0] for value in stable[1:]))
        self.assertTrue(all(result["authorizes_autonomous_execution"] is False for result in results))
        self.assertEqual([result["source_agent"]["model"] for result in results], ["luna", "astra", "sol", "codex"])

    def test_common_request_normalizes_to_only_the_frozen_c1_input_fields(self):
        request = normalize_common_request(self.request("future-model"))
        self.assertEqual(set(to_c1_input(request)), {
            "context", "observed_facts", "proposed_claim_or_action", "available_evidence", "decision_scope", "missing_evidence"
        })
        self.assertNotIn("source_agent", json.dumps(to_c1_input(request)))

    def test_experience_roundtrip_preserves_source_agent_and_request_ids(self):
        response = self.client.decide(self.request("astra"))
        record = {
            "experience_id": "phase7e-source-agent-roundtrip",
            "timestamp": "2026-09-28T00:00:00Z",
            "project": "HPCV",
            "problem_context": {"summary": "decoder latency experiment"},
            "llm_proposal": {"summary": "adopt decoder X", "claim_under_review": "causal latency improvement"},
            "source_agent": self.request("astra")["source_agent"],
            "proposal_id": "proposal-17",
            "laya_request_id": response["request_id"],
            "laya_prediction": {
                "primary": response["primary"]["label"],
                "probabilities": response["primary"]["probabilities"],
                "flags": response["flags"],
                "uncertainty": response["uncertainty"],
                "model_version": response["specialist"]["model_version"],
                "checkpoint_sha256": response["specialist"]["checkpoint_sha"],
            },
            "human_decision": {"decision": "pending"},
            "experiment": {"action": "none", "measurements": [], "observations": []},
            "resolution": {"status": "unresolved", "supported_claims": [], "rejected_claims": [],
                           "resolved_primary": None, "resolved_flags": []},
            "source_provenance": [],
            "derived_inference_contract": None,
            "training_candidate": {"eligible": False, "reason": "unresolved"},
        }
        saved = self.client.save_experience(record)
        loaded = self.client.load_experience(record["experience_id"])
        self.assertEqual(saved["status"], "saved")
        self.assertEqual(loaded["source_agent"], record["source_agent"])
        self.assertEqual(loaded["proposal_id"], "proposal-17")
        self.assertEqual(loaded["laya_request_id"], response["request_id"])

    def test_unknown_direct_adapter_fails_gracefully(self):
        with self.assertRaises(AdapterUnavailable):
            self.client.direct_adapter("unknown-provider-interface")

    def test_unreachable_service_fails_closed_without_specialist_result(self):
        client = ModelAgnosticSpecialistClient("http://127.0.0.1:1", timeout=0.2,
                                               expected_checkpoint_sha256="f" * 64)
        with self.assertRaises(SpecialistUnavailable) as ctx:
            client.decide(self.request("sol"))
        self.assertNotIn("ready_to_implement", str(ctx.exception))

    def test_local_endpoint_is_configurable_without_provider_specific_enum(self):
        config = Path(self.tmp.name) / "local.toml"
        config.write_text('[specialist]\nendpoint = "http://127.0.0.1:9876"\n', encoding="utf-8")
        self.assertEqual(read_specialist_endpoint(config), "http://127.0.0.1:9876")
        self.assertEqual(normalize_common_request(self.request("future-agent-model"))["source_agent"]["model"],
                         "future-agent-model")

    def test_checkpoint_path_is_configurable_as_a_local_path(self):
        checkpoint = Path(self.tmp.name) / "pinned-model.safetensors"
        checkpoint.write_bytes(b"test checkpoint placeholder")
        config = Path(self.tmp.name) / "checkpoint.toml"
        config.write_text(f'[specialist]\ncheckpoint_path = "{checkpoint.as_posix()}"\n', encoding="utf-8")
        self.assertEqual(read_checkpoint_path(config), checkpoint.resolve())

    def test_all_agent_templates_validate_against_common_request_contract(self):
        adapters = Path(__file__).resolve().parents[1] / "adapters"
        for agent in ("luna", "astra", "sol", "codex"):
            with self.subTest(agent=agent):
                template = json.loads((adapters / agent / "request.template.json").read_text(encoding="utf-8"))
                self.assertEqual(normalize_common_request(template)["source_agent"]["model"], agent)


if __name__ == "__main__":
    unittest.main()
