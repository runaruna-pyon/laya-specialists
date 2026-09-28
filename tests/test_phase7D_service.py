from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from phase7D_experience import VaultExperienceStore
from phase7D_specialist import FLAG_LABELS, PRIMARY_LABELS
from phase7D_specialist_service import SpecialistService, create_server, read_vault_root


class FakeRuntime:
    device = "cpu"

    def predict(self, text: str) -> dict:
        assert "Observed facts: one controlled comparison" in text
        primary = "ready_to_implement"
        return {
            "primary_logits": {label: 1.0 if label == primary else 0.0 for label in PRIMARY_LABELS},
            "primary_probabilities": {label: 1.0 if label == primary else 0.0 for label in PRIMARY_LABELS},
            "flag_logits": {flag: -5.0 for flag in FLAG_LABELS},
            "flag_probabilities": {flag: 0.01 for flag in FLAG_LABELS},
        }


class Phase7DSpecialistServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.vault = Path(self.tmp.name) / "vault"
        self.vault.mkdir()
        service = SpecialistService(
            FakeRuntime(), VaultExperienceStore(self.vault),
            checkpoint_sha256="d" * 64,
            model_version="phase7D-c1-pilot",
        )
        self.server = create_server("127.0.0.1", 0, service)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.tmp.cleanup()

    def get(self, path):
        with urlopen(self.base + path, timeout=3) as response:
            return response.status, json.loads(response.read())

    def post(self, path, body):
        req = Request(self.base + path, data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=3) as response:
            return response.status, json.loads(response.read())

    @staticmethod
    def input_payload():
        return {"input": {
            "context": "local audio experiment",
            "observed_facts": "one controlled comparison",
            "proposed_claim_or_action": "continue a bounded trial",
            "available_evidence": "matched settings and repeated measurement",
            "decision_scope": "advisory only",
            "missing_evidence": "none identified",
        }, "request_id": "request-1"}

    def test_local_config_selects_vault_root_without_embedding_it_in_records(self):
        (self.vault / "60_Knowledge").mkdir()
        configured = Path(self.tmp.name) / "local.toml"
        configured.write_text(f'[vault]\nroot = "{self.vault.as_posix()}"\n', encoding="utf-8")
        self.assertEqual(read_vault_root(configured), self.vault.resolve())

    def test_service_refuses_non_loopback_bind(self):
        with self.assertRaises(ValueError):
            create_server("0.0.0.0", 0, object())

    def test_health_and_decide_preserve_c1_provenance_and_non_authorizing_contract(self):
        status, health = self.get("/health")
        self.assertEqual(status, 200)
        self.assertEqual(health["checkpoint_sha256"], "d" * 64)
        status, result = self.post("/decide", self.input_payload())
        self.assertEqual(status, 200)
        self.assertEqual(result["primary"]["label"], "ready_to_implement")
        self.assertFalse(result["autonomy_policy"]["authorizes_autonomous_execution"])

    def test_decide_rejects_non_allowlisted_input_schema(self):
        payload = self.input_payload()
        payload["input"]["gold"] = "ready_to_implement"
        with self.assertRaises(HTTPError) as ctx:
            self.post("/decide", payload)
        self.assertEqual(ctx.exception.code, 422)

    def test_experience_record_can_be_saved_and_read_back_through_service(self):
        from test_phase7D_specialist import ExperienceRecordTests
        record = ExperienceRecordTests().record()
        status, saved = self.post("/experience", record)
        self.assertEqual(status, 201)
        self.assertEqual(saved["record_path"], "60_Knowledge/Laya_Experience/Active/exp-test-001.json")
        status, loaded = self.get("/experience/exp-test-001")
        self.assertEqual(status, 200)
        self.assertEqual(loaded, record)


if __name__ == "__main__":
    unittest.main()
