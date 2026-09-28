from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from phase7D_experience import VaultExperienceStore, validate_experience
from phase7D_specialist import (
    FLAG_LABELS,
    PRIMARY_LABELS,
    build_advisory_response,
    serialize_specialist_input,
)


def fake_prediction(primary: str, flags: dict[str, float] | None = None) -> dict:
    return {
        "primary_logits": {label: 0.4 if label == primary else 0.1 for label in PRIMARY_LABELS},
        "primary_probabilities": {label: (0.7 if label == primary else 0.1) for label in PRIMARY_LABELS},
        "flag_logits": {label: 0.0 for label in FLAG_LABELS},
        "flag_probabilities": {label: (flags or {}).get(label, 0.1) for label in FLAG_LABELS},
    }


class SpecialistOutputTests(unittest.TestCase):
    def test_input_serialization_uses_only_frozen_six_fields_and_order(self):
        payload = {
            "context": "audio decoder change",
            "observed_facts": "RTF fell in one run",
            "proposed_claim_or_action": "adopt decoder",
            "available_evidence": "before/after timing",
            "decision_scope": "bounded local experiment",
            "missing_evidence": "second seed",
        }
        self.assertEqual(
            serialize_specialist_input(payload),
            "Context: audio decoder change\n"
            "Observed facts: RTF fell in one run\n"
            "Proposed claim or action: adopt decoder\n"
            "Available evidence: before/after timing\n"
            "Decision scope: bounded local experiment\n"
            "Missing evidence: second seed",
        )

    def test_input_serialization_rejects_gold_or_metadata_fields(self):
        payload = {"context": "x", "primary_disposition": "ready_to_implement"}
        with self.assertRaises(ValueError):
            serialize_specialist_input(payload)

    def test_input_serialization_preserves_user_text_exactly_after_label(self):
        payload = {
            "context": "  leading and trailing  ",
            "observed_facts": "x",
            "proposed_claim_or_action": "x",
            "available_evidence": "x",
            "decision_scope": "x",
            "missing_evidence": "x",
        }
        self.assertTrue(serialize_specialist_input(payload).startswith("Context:   leading and trailing  \n"))

    def test_response_is_advisory_and_semantic_route_ignores_confidence(self):
        response = build_advisory_response(
            fake_prediction("high_regression_risk"),
            model_version="phase7D-c1-pilot",
            checkpoint_sha256="a" * 64,
        )
        self.assertEqual(response["routing"]["route"], "HUMAN_REVIEW")
        self.assertFalse(response["autonomy_policy"]["authorizes_autonomous_execution"])
        self.assertFalse(response["primary"]["calibrated"])
        self.assertIn("not correctness probabilities", response["uncertainty"]["semantics"])

    def test_ready_label_is_advisory_even_at_extreme_probability(self):
        result = fake_prediction("ready_to_implement")
        result["primary_probabilities"] = {label: float(label == "ready_to_implement") for label in PRIMARY_LABELS}
        response = build_advisory_response(result, "phase7D-c1-pilot", "b" * 64)
        self.assertEqual(response["routing"]["route"], "PROCEED_ADVISORY")
        self.assertFalse(response["autonomy_policy"]["authorizes_autonomous_execution"])


class ExperienceRecordTests(unittest.TestCase):
    def record(self) -> dict:
        return {
            "experience_id": "exp-test-001",
            "timestamp": "2026-09-28T00:00:00Z",
            "project": "HPCV",
            "source_agent": {"provider": "openai", "model": "codex", "interface": "test"},
            "problem_context": {"summary": "test-only unresolved case"},
            "llm_proposal": {"summary": "test proposal"},
            "laya_prediction": {
                "primary": "verification_required",
                "probabilities": {label: 0.25 for label in PRIMARY_LABELS},
                "flags": {label: {"probability": 0.0, "active": False} for label in FLAG_LABELS},
                "uncertainty": {"margin": 0.0, "entropy": 1.0, "concentration": 0.0},
                "model_version": "phase7D-c1-pilot",
                "checkpoint_sha256": "c" * 64,
            },
            "human_decision": {"decision": "pending", "notes": "not reviewed"},
            "experiment": {"action": "none", "measurements": [], "observations": []},
            "resolution": {
                "status": "unresolved",
                "supported_claims": [],
                "rejected_claims": [],
                "resolved_primary": None,
                "resolved_flags": [],
            },
            "source_provenance": [],
            "derived_inference_contract": None,
            "training_candidate": {"eligible": False, "reason": "unresolved; no verified outcome"},
        }

    def test_unresolved_record_cannot_be_training_eligible(self):
        record = self.record()
        record["training_candidate"] = {"eligible": True, "reason": "model predicted it"}
        with self.assertRaises(ValueError):
            validate_experience(record)

    def test_store_round_trips_without_machine_absolute_vault_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = VaultExperienceStore(Path(tmp))
            record = self.record()
            path = store.save(record)
            loaded = store.load("exp-test-001")
            self.assertEqual(loaded, record)
            self.assertNotIn(str(Path(tmp)), path.read_text(encoding="utf-8"))
            self.assertEqual(path.parent.name, "Active")
            self.assertNotIn(b"\r\n", path.read_bytes())

    def test_resolution_moves_record_from_active_to_resolved_without_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = VaultExperienceStore(Path(tmp))
            record = self.record()
            active = store.save(record)
            record["human_decision"] = {"decision": "resolved", "resolved_by_human": True}
            record["resolution"].update({"status": "resolved", "resolved_primary": "ready_to_implement"})
            record["training_candidate"] = {"eligible": False, "reason": "reviewed; candidate selection pending"}
            resolved = store.save(record)
            self.assertFalse(active.exists())
            self.assertEqual(resolved.parent.name, "Resolved")
            self.assertEqual(store.load(record["experience_id"]), record)

    def test_system_test_record_is_kept_out_of_active_experience_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = VaultExperienceStore(Path(tmp))
            record = self.record()
            record["record_kind"] = "system_test"
            path = store.save(record)
            self.assertEqual(path.parent.name, "_System_Tests")
            self.assertEqual(store.load(record["experience_id"]), record)
            self.assertEqual(list((Path(tmp) / "60_Knowledge/Laya_Experience/Active").glob("*.json")), [])

    def test_system_test_id_cannot_collide_with_real_experience_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = VaultExperienceStore(Path(tmp))
            test_record = self.record()
            test_record["record_kind"] = "system_test"
            store.save(test_record)
            real_record = self.record()
            with self.assertRaises(ValueError):
                store.save(real_record)

    def test_resolved_record_requires_independent_provenance_for_training(self):
        record = self.record()
        record["resolution"].update({
            "status": "resolved",
            "resolved_primary": "ready_to_implement",
            "resolved_flags": [],
        })
        record["human_decision"] = {"decision": "ready_to_implement", "notes": "explicit human resolution"}
        record["training_candidate"] = {"eligible": True, "reason": "explicit human resolution and verified outcome"}
        record["experiment"] = {"action": "run bounded test", "measurements": [{"verified": True, "name": "latency"}], "observations": ["measured"]}
        self.assertTrue(validate_experience(record)["training_candidate"]["eligible"])

    def test_explicit_human_resolution_is_independent_training_provenance(self):
        record = self.record()
        record["resolution"].update({
            "status": "resolved",
            "resolved_primary": "ready_to_implement",
            "resolved_flags": [],
        })
        record["human_decision"] = {"decision": "ready_to_implement", "resolved_by_human": True}
        record["training_candidate"] = {"eligible": True, "reason": "explicit human resolution"}
        self.assertTrue(validate_experience(record)["training_candidate"]["eligible"])


if __name__ == "__main__":
    unittest.main()
