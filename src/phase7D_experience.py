"""Validated Vault storage for reviewed Laya experience records."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from phase7D_specialist import FLAG_LABELS, PRIMARY_LABELS


_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _is_independently_supported(record: dict[str, Any]) -> bool:
    if (record["human_decision"].get("independently_confirmed") is True
            or record["human_decision"].get("resolved_by_human") is True):
        return True
    if any(item.get("verified") is True for item in record["experiment"].get("measurements", []) if isinstance(item, dict)):
        return True
    if any(item.get("verified") is True and item.get("type") in {"authoritative_source", "reference", "independent_measurement"}
           for item in record.get("source_provenance", []) if isinstance(item, dict)):
        return True
    return False


def validate_experience(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("experience record must be an object")
    required = {
        "experience_id", "timestamp", "project", "problem_context", "llm_proposal",
        "laya_prediction", "human_decision", "experiment", "resolution",
        "source_provenance", "derived_inference_contract", "training_candidate",
    }
    optional = {"record_kind", "source_agent", "proposal_id", "laya_request_id"}
    if not required.issubset(record) or set(record) - required - optional:
        raise ValueError(f"experience record fields must contain {sorted(required)} and only documented optional fields")
    source_agent = record.get("source_agent")
    if source_agent is not None:
        if not isinstance(source_agent, dict) or set(source_agent) != {"provider", "model", "interface"}:
            raise ValueError("source_agent must contain provider, model, and interface")
        for key, value in source_agent.items():
            if not isinstance(value, str) or not value.strip() or len(value) > 256:
                raise ValueError(f"source_agent.{key} must be a nonempty string of at most 256 characters")
    for key in ("proposal_id", "laya_request_id"):
        if key in record and (not isinstance(record[key], str) or not record[key].strip() or len(record[key]) > 128):
            raise ValueError(f"{key} must be a nonempty string of at most 128 characters")
    record_kind = record.get("record_kind", "experience")
    if record_kind not in {"experience", "system_test"}:
        raise ValueError("record_kind must be experience or system_test")
    experience_id = record["experience_id"]
    if not isinstance(experience_id, str) or not _ID_PATTERN.fullmatch(experience_id):
        raise ValueError("experience_id must be a safe relative identifier")
    if not isinstance(record["timestamp"], str) or not record["timestamp"].endswith("Z"):
        raise ValueError("timestamp must be an ISO-8601 UTC string ending in Z")
    try:
        from datetime import datetime
        datetime.fromisoformat(record["timestamp"][:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("timestamp must be a valid ISO-8601 UTC timestamp") from exc
    if not isinstance(record["project"], str) or not record["project"].strip():
        raise ValueError("project is required")
    for key in ("problem_context", "llm_proposal", "laya_prediction", "human_decision", "experiment", "resolution", "training_candidate"):
        if not isinstance(record[key], dict):
            raise ValueError(f"{key} must be an object")
    prediction = record["laya_prediction"]
    if prediction.get("primary") not in PRIMARY_LABELS:
        raise ValueError("laya_prediction.primary is not in the frozen taxonomy")
    probabilities = prediction.get("probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) != set(PRIMARY_LABELS):
        raise ValueError("laya_prediction.probabilities must contain all primary labels")
    if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 <= p <= 1 for p in probabilities.values()):
        raise ValueError("primary probabilities must be finite values in [0, 1]")
    if abs(sum(probabilities.values()) - 1.0) > 1e-4:
        raise ValueError("primary probabilities must sum to one")
    if not isinstance(prediction.get("flags"), dict) or set(prediction["flags"]) != set(FLAG_LABELS):
        raise ValueError("laya_prediction.flags must contain every frozen required-check flag")
    for flag, item in prediction["flags"].items():
        if not isinstance(item, dict) or not isinstance(item.get("active"), bool):
            raise ValueError(f"flag {flag} requires probability and active")
        prob = item.get("probability")
        if isinstance(prob, bool) or not isinstance(prob, (int, float)) or not 0 <= prob <= 1:
            raise ValueError(f"flag {flag} probability must be in [0, 1]")
    if not _SHA_PATTERN.fullmatch(str(prediction.get("checkpoint_sha256", ""))):
        raise ValueError("laya_prediction.checkpoint_sha256 must be a lowercase SHA-256")
    if not isinstance(record["source_provenance"], list):
        raise ValueError("source_provenance must be a list")
    resolution = record["resolution"]
    if resolution.get("status") not in {"unresolved", "resolved", "rejected"}:
        raise ValueError("resolution.status must be unresolved, resolved, or rejected")
    candidate = record["training_candidate"]
    if not isinstance(candidate.get("eligible"), bool) or not isinstance(candidate.get("reason"), str) or not candidate["reason"].strip():
        raise ValueError("training_candidate requires boolean eligible and a reason")
    if candidate["eligible"]:
        if record_kind == "system_test":
            raise ValueError("system-test records cannot be training candidates")
        if resolution["status"] != "resolved":
            raise ValueError("unresolved or rejected experiences cannot be training candidates")
        if not _is_independently_supported(record):
            raise ValueError("model prediction alone cannot establish training eligibility")
    return record


class VaultExperienceStore:
    """Stores records by Vault-relative path; absolute vault roots stay local."""

    def __init__(self, vault_root: str | Path) -> None:
        self.vault_root = Path(vault_root).expanduser().resolve()
        if not self.vault_root.is_dir():
            raise FileNotFoundError(f"configured Obsidian Vault does not exist: {self.vault_root}")
        self.base = self.vault_root / "60_Knowledge" / "Laya_Experience"

    def _destination(self, record: dict[str, Any]) -> Path:
        if record.get("record_kind") == "system_test":
            return self.base / "_System_Tests" / f"{record['experience_id']}.json"
        status = record["resolution"]["status"]
        area = "Active" if status == "unresolved" else "Resolved"
        return self.base / area / f"{record['experience_id']}.json"

    def save(self, record: Any) -> Path:
        value = validate_experience(record)
        target = self._destination(value)
        existing_other_locations = [
            self.base / area / f"{value['experience_id']}.json"
            for area in ("Active", "Resolved", "_System_Tests")
            if area != target.parent.name and (self.base / area / f"{value['experience_id']}.json").is_file()
        ]
        if existing_other_locations:
            is_status_move = (value.get("record_kind", "experience") == "experience"
                              and target.parent.name in {"Active", "Resolved"}
                              and all(path.parent.name in {"Active", "Resolved"} for path in existing_other_locations))
            if not is_status_move:
                raise ValueError("experience_id already belongs to a different Vault record")
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(".json.tmp")
        with temp.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        temp.replace(target)
        loaded = json.loads(target.read_text(encoding="utf-8"))
        validate_experience(loaded)
        if loaded != value:
            raise IOError("saved experience record failed round-trip equality")
        if target.parent.name in {"Active", "Resolved"}:
            other_area = "Resolved" if target.parent.name == "Active" else "Active"
            previous_location = self.base / other_area / target.name
            if previous_location.exists():
                previous_location.unlink()
        queue = self.base / "Training_Candidates" / f"{value['experience_id']}.ref.json"
        if value["training_candidate"]["eligible"]:
            queue.parent.mkdir(parents=True, exist_ok=True)
            relative_record = target.relative_to(self.vault_root).as_posix()
            with queue.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps({"experience_id": value["experience_id"], "record_path": relative_record,
                                         "reason": value["training_candidate"]["reason"]}, ensure_ascii=False, indent=2) + "\n")
        elif queue.exists():
            queue.unlink()
        return target

    def load(self, experience_id: str) -> dict[str, Any]:
        if not isinstance(experience_id, str) or not _ID_PATTERN.fullmatch(experience_id):
            raise ValueError("experience_id must be a safe relative identifier")
        candidates = [self.base / area / f"{experience_id}.json" for area in ("Active", "Resolved", "_System_Tests")]
        matches = [path for path in candidates if path.is_file()]
        if len(matches) != 1:
            raise FileNotFoundError(f"expected exactly one Vault record for {experience_id}, found {len(matches)}")
        record = json.loads(matches[0].read_text(encoding="utf-8"))
        validate_experience(record)
        return record
