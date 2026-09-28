"""Model-agnostic JSON client for the local Laya Deep Learning Specialist."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tomllib
from pathlib import Path
from typing import Any

from phase7D_specialist_client import SpecialistClient, SpecialistProtocolError
from phase7D_specialist import FLAG_LABELS, PRIMARY_LABELS


COMMON_REQUEST_FIELDS = {
    "request_id", "source_agent", "project", "problem_context", "proposal",
    "claim_under_review", "known_evidence", "requested_decision",
}
EVIDENCE_FIELDS = {"observed_facts", "available_evidence", "missing_evidence"}
SOURCE_AGENT_FIELDS = {"provider", "model", "interface"}
ROOT = Path(__file__).resolve().parents[1]
LOCAL_CONFIG = ROOT / "config" / "laya_local.toml"
DEFAULT_ENDPOINT = "http://127.0.0.1:8766"


class CommonRequestError(ValueError):
    pass


class AdapterUnavailable(RuntimeError):
    """Raised when a caller asks for an unverified provider-specific transport."""


def read_specialist_endpoint(config_path: str | Path = LOCAL_CONFIG) -> str:
    path = Path(config_path)
    if not path.is_file():
        return DEFAULT_ENDPOINT
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    endpoint = config.get("specialist", {}).get("endpoint", DEFAULT_ENDPOINT)
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("[specialist].endpoint must be a nonempty URL")
    return endpoint


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CommonRequestError(f"{name} must be a nonempty string")
    return value


def normalize_common_request(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != COMMON_REQUEST_FIELDS:
        raise CommonRequestError(f"request must contain exactly: {', '.join(sorted(COMMON_REQUEST_FIELDS))}")
    source = value["source_agent"]
    if not isinstance(source, dict) or set(source) != SOURCE_AGENT_FIELDS:
        raise CommonRequestError("source_agent must contain provider, model, and interface strings")
    normalized_source = {key: _required_text(source[key], f"source_agent.{key}") for key in ("provider", "model", "interface")}
    evidence = value["known_evidence"]
    if not isinstance(evidence, dict) or set(evidence) != EVIDENCE_FIELDS:
        raise CommonRequestError("known_evidence must contain observed_facts, available_evidence, and missing_evidence")
    normalized_evidence = {key: _required_text(evidence[key], f"known_evidence.{key}") for key in sorted(EVIDENCE_FIELDS)}
    normalized = {
        "request_id": _required_text(value["request_id"], "request_id"),
        "source_agent": normalized_source,
        "project": _required_text(value["project"], "project"),
        "problem_context": _required_text(value["problem_context"], "problem_context"),
        "proposal": _required_text(value["proposal"], "proposal"),
        "claim_under_review": _required_text(value["claim_under_review"], "claim_under_review"),
        "known_evidence": normalized_evidence,
        "requested_decision": _required_text(value["requested_decision"], "requested_decision"),
    }
    if len(normalized["request_id"]) > 128:
        raise CommonRequestError("request_id must be at most 128 characters")
    for key, text in normalized_source.items():
        if len(text) > 256:
            raise CommonRequestError(f"source_agent.{key} must be at most 256 characters")
    return normalized


def to_c1_input(request: Any) -> dict[str, str]:
    value = normalize_common_request(request)
    evidence = value["known_evidence"]
    return {
        "context": f"Project: {value['project']}\n{value['problem_context']}",
        "observed_facts": evidence["observed_facts"],
        "proposed_claim_or_action": f"Claim under review: {value['claim_under_review']}\nProposal: {value['proposal']}",
        "available_evidence": evidence["available_evidence"],
        "decision_scope": value["requested_decision"],
        "missing_evidence": evidence["missing_evidence"],
    }


def normalize_common_response(response: Any, request: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise SpecialistProtocolError("legacy service response must be an object")
    primary = response.get("primary")
    flags = response.get("flags")
    uncertainty = response.get("uncertainty")
    routing = response.get("routing")
    autonomy = response.get("autonomy_policy")
    provenance = response.get("provenance")
    if not all(isinstance(value, dict) for value in (primary, flags, uncertainty, routing, autonomy, provenance)):
        raise SpecialistProtocolError("legacy service response is missing typed fields")
    if autonomy.get("authorizes_autonomous_execution") is not False:
        raise SpecialistProtocolError("service response violates the advisory-only contract")
    if response.get("request_id") != request["request_id"]:
        raise SpecialistProtocolError("service response request_id does not match request")
    if primary.get("calibrated") is not False or routing.get("confidence_used_for_routing") is not False:
        raise SpecialistProtocolError("service response violates frozen C1 uncertainty/routing semantics")
    probabilities = primary.get("probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) != set(PRIMARY_LABELS):
        raise SpecialistProtocolError("primary probabilities do not match the frozen taxonomy")
    if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1
           for p in probabilities.values()) or abs(sum(probabilities.values()) - 1.0) > 1e-4:
        raise SpecialistProtocolError("primary probabilities are invalid")
    if primary.get("label") not in PRIMARY_LABELS:
        raise SpecialistProtocolError("primary label is outside the frozen taxonomy")
    if not isinstance(flags, dict) or set(flags) != set(FLAG_LABELS):
        raise SpecialistProtocolError("flags do not match the frozen taxonomy")
    for flag, item in flags.items():
        if not isinstance(item, dict) or not isinstance(item.get("active"), bool):
            raise SpecialistProtocolError(f"invalid typed flag response for {flag}")
        probability = item.get("probability")
        if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise SpecialistProtocolError(f"invalid flag probability for {flag}")
    if routing.get("route") not in {"PROCEED_ADVISORY", "CHECK_REQUIRED", "MORE_INFORMATION_OR_HUMAN", "HUMAN_REVIEW"}:
        raise SpecialistProtocolError("service returned an unknown semantic action class")
    if not isinstance(provenance.get("model_version"), str) or not isinstance(provenance.get("checkpoint_sha256"), str):
        raise SpecialistProtocolError("service provenance is incomplete")
    if len(provenance["checkpoint_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in provenance["checkpoint_sha256"]):
        raise SpecialistProtocolError("service checkpoint provenance is malformed")
    scalar_ranges = {
        "top1_probability": (0.0, 1.0), "top2_probability": (0.0, 1.0),
        "margin": (0.0, 1.0), "entropy": (0.0, math.log(len(PRIMARY_LABELS))),
        "normalized_entropy": (0.0, 1.0), "concentration": (0.0, 1.0),
    }
    for name, (lower, upper) in scalar_ranges.items():
        scalar = primary.get(name) if name in {"top1_probability", "top2_probability", "margin"} else uncertainty.get(name)
        if isinstance(scalar, bool) or not isinstance(scalar, (int, float)) or not math.isfinite(scalar) or not lower <= scalar <= upper:
            raise SpecialistProtocolError(f"invalid uncertainty field {name}")
    if not isinstance(routing.get("human_summary"), str):
        raise SpecialistProtocolError("service response has no human summary")
    return {
        "schema_version": "laya-specialist-common-response-v1",
        "request_id": request["request_id"],
        "source_agent": dict(request["source_agent"]),
        "project": request["project"],
        "specialist": {
            "name": "Laya Deep Learning Specialist",
            "model_version": provenance.get("model_version"),
            "checkpoint_sha": provenance.get("checkpoint_sha256"),
        },
        "primary": {
            "label": primary.get("label"),
            "logits": primary.get("logits"),
            "probabilities": dict(probabilities),
        },
        "flags": flags,
        "uncertainty": {
            "top1_probability": primary["top1_probability"],
            "top2_probability": primary["top2_probability"],
            "margin": primary["margin"],
            "entropy": uncertainty.get("entropy"),
            "normalized_entropy": uncertainty.get("normalized_entropy"),
            "concentration": uncertainty.get("concentration"),
            "calibrated": False,
            "semantics": "distribution properties; not correctness probabilities",
        },
        "advisory": {
            "action_class": routing.get("route"),
            "human_summary": routing.get("human_summary"),
            "suggested_next_check": " ".join(routing.get("suggested_checks", [])) or None,
        },
        "authorizes_autonomous_execution": False,
    }


class ModelAgnosticSpecialistClient:
    """Common request/response contract over the existing fail-closed C1 client."""

    # No provider environment was available to validate a direct agent-to-loopback
    # call. Callers use the shared JSON bridge/CLI until that is proven per runtime.
    DIRECT_ADAPTERS = frozenset()
    TEMPLATE_ADAPTERS = frozenset({"luna", "astra", "sol", "codex", "generic"})

    def __init__(self, base_url: str | None = None, timeout: float = 30.0,
                 expected_checkpoint_sha256: str | None = None, config_path: str | Path = LOCAL_CONFIG):
        endpoint = base_url or read_specialist_endpoint(config_path)
        self._client = SpecialistClient(endpoint, timeout, expected_checkpoint_sha256)

    def direct_adapter(self, interface: str) -> Any:
        if interface not in self.DIRECT_ADAPTERS:
            raise AdapterUnavailable(f"UNAVAILABLE_DIRECT_LOCAL_CALL: {interface}; use the common JSON bridge")
        return self

    def decide(self, value: Any) -> dict[str, Any]:
        request = normalize_common_request(value)
        raw = self._client.decide(to_c1_input(request), request["request_id"])
        return normalize_common_response(raw, request)

    def save_experience(self, record: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(record, dict) or not isinstance(record.get("source_agent"), dict):
            raise CommonRequestError("new model-agnostic Experience Records require source_agent")
        return self._client.save_experience(record)

    def load_experience(self, experience_id: str) -> dict[str, Any]:
        return self._client.load_experience(experience_id)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--request", type=Path, help="Common request JSON file")
    actions.add_argument("--save-experience", type=Path, help="Experience Record JSON file; source_agent is required")
    actions.add_argument("--get-experience", help="Read an Experience Record by ID")
    parser.add_argument("--url", help="Override configured local specialist endpoint")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    try:
        client = ModelAgnosticSpecialistClient(args.url, args.timeout)
        if args.request:
            request = json.loads(args.request.read_text(encoding="utf-8-sig"))
            output = client.decide(request)
        elif args.save_experience:
            record = json.loads(args.save_experience.read_text(encoding="utf-8-sig"))
            output = client.save_experience(record)
        else:
            output = client.load_experience(args.get_experience)
        print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__, "detail": str(exc),
                          "advisory_status": "unavailable"}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
