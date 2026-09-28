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
from phase7D_specialist_client import SpecialistClient, SpecialistProtocolError
from phase7D_specialist_service import SpecialistService, create_server


class FakeRuntime:
    device = "cpu"

    def predict(self, text):
        primary = "insufficient_evidence"
        return {
            "primary_logits": {label: 10.0 if label == primary else -1.0 for label in PRIMARY_LABELS},
            "primary_probabilities": {label: 0.999 if label == primary else 0.001 / 3 for label in PRIMARY_LABELS},
            "flag_logits": {flag: -4.0 for flag in FLAG_LABELS},
            "flag_probabilities": {flag: 0.01 for flag in FLAG_LABELS},
        }


class Phase7DClientTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        vault = Path(self.tmp.name) / "vault"
        vault.mkdir()
        self.server = create_server("127.0.0.1", 0, SpecialistService(
            FakeRuntime(), VaultExperienceStore(vault), "e" * 64,
        ))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.tmp.cleanup()

    def test_client_returns_raw_typed_output_without_confidence_routing(self):
        client = SpecialistClient(self.url, expected_checkpoint_sha256="e" * 64)
        response = client.decide({
            "context": "engineering review",
            "observed_facts": "measurement missing",
            "proposed_claim_or_action": "claim improvement",
            "available_evidence": "one run",
            "decision_scope": "research decision",
            "missing_evidence": "replication",
        })
        self.assertEqual(response["primary"]["label"], "insufficient_evidence")
        self.assertEqual(response["routing"]["route"], "MORE_INFORMATION_OR_HUMAN")
        self.assertFalse(response["routing"]["confidence_used_for_routing"])
        self.assertFalse(response["autonomy_policy"]["authorizes_autonomous_execution"])

    def test_client_fails_closed_on_checkpoint_provenance_mismatch(self):
        client = SpecialistClient(self.url, expected_checkpoint_sha256="f" * 64)
        with self.assertRaises(SpecialistProtocolError):
            client.health()

    def test_client_rejects_non_loopback_urls(self):
        with self.assertRaises(ValueError):
            SpecialistClient("http://example.com:8766", expected_checkpoint_sha256="e" * 64)

    def test_client_saves_and_reads_an_unresolved_experience_record(self):
        from test_phase7D_specialist import ExperienceRecordTests
        record = ExperienceRecordTests().record()
        client = SpecialistClient(self.url, expected_checkpoint_sha256="e" * 64)
        saved = client.save_experience(record)
        self.assertEqual(saved["status"], "saved")
        self.assertEqual(client.load_experience(record["experience_id"]), record)


if __name__ == "__main__":
    unittest.main()
