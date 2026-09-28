"""Typed, non-authorizing output contract for the C1 DL specialist."""

from __future__ import annotations

import math
import re
from typing import Any


PRIMARY_LABELS = (
    "insufficient_evidence",
    "ready_to_implement",
    "verification_required",
    "high_regression_risk",
)
FLAG_LABELS = (
    "needs_measurement",
    "needs_ablation",
    "needs_reference_check",
    "needs_baseline_or_replication",
    "needs_train_inference_parity_check",
    "needs_leakage_or_distribution_check",
    "needs_latency_accounting",
    "needs_implementation_spec",
    "needs_regression_plan",
    "needs_sample_breadth",
)
INPUT_FIELDS = (
    "context",
    "observed_facts",
    "proposed_claim_or_action",
    "available_evidence",
    "decision_scope",
    "missing_evidence",
)
INPUT_LABELS = (
    "Context",
    "Observed facts",
    "Proposed claim or action",
    "Available evidence",
    "Decision scope",
    "Missing evidence",
)
FLAG_THRESHOLD = 0.50  # Reporting threshold only; not a calibrated execution gate.
SERVICE_VERSION = "phase7d-c1-specialist-1.0"

FLAG_NEXT_CHECK = {
    "needs_measurement": "対象指標を同じ条件で測定してください。",
    "needs_ablation": "変更要因を切り分ける比較またはablationを行ってください。",
    "needs_reference_check": "該当versionと実行環境に一致する公式仕様・一次資料を確認してください。",
    "needs_baseline_or_replication": "baselineとの比較または独立runで再現性を確認してください。",
    "needs_train_inference_parity_check": "training条件とruntime条件の差を確認してください。",
    "needs_leakage_or_distribution_check": "split漏洩と評価分布の差を点検してください。",
    "needs_latency_accounting": "未計上の処理を含めたend-to-end latencyを測定してください。",
    "needs_implementation_spec": "変更対象・設定・実装条件を特定してください。",
    "needs_regression_plan": "変更後にcritical pathの副作用を確認するテストを定めてください。",
    "needs_sample_breadth": "結論を支えるrun・seed・sample数を追加してください。",
}


def serialize_specialist_input(value: Any) -> str:
    """Serialize only the frozen six natural-language fields used in C1 training."""
    if not isinstance(value, dict) or set(value) != set(INPUT_FIELDS):
        raise ValueError(f"input must contain exactly these fields: {', '.join(INPUT_FIELDS)}")
    parts: list[str] = []
    for key, label in zip(INPUT_FIELDS, INPUT_LABELS, strict=True):
        item = value[key]
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"input field {key!r} must be a nonempty string")
        parts.append(f"{label}: {item}")
    return "\n".join(parts)


def _finite_probability_map(values: Any, labels: tuple[str, ...], *, normalized: bool) -> dict[str, float]:
    if not isinstance(values, dict) or set(values) != set(labels):
        raise ValueError("prediction probability keys do not match the frozen taxonomy")
    result: dict[str, float] = {}
    for label in labels:
        value = values[label]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"invalid probability for {label}")
        result[label] = float(value)
    if normalized and abs(sum(result.values()) - 1.0) > 1e-4:
        raise ValueError("primary probabilities must sum to one")
    return result


def _finite_logit_map(values: Any, labels: tuple[str, ...]) -> dict[str, float]:
    if not isinstance(values, dict) or set(values) != set(labels):
        raise ValueError("prediction logit keys do not match the frozen taxonomy")
    result = {}
    for label in labels:
        value = values[label]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"invalid logit for {label}")
        result[label] = float(value)
    return result


def _route(primary: str) -> tuple[str, str]:
    # Routing deliberately depends on semantic class only. Probabilities,
    # margins, entropy, and concentration are recorded but never authorize action.
    return {
        "ready_to_implement": ("PROCEED_ADVISORY", "限定した提案をadvisoryとして返します。実行許可ではありません。"),
        "verification_required": ("CHECK_REQUIRED", "判断前に必要な確認を行ってください。"),
        "insufficient_evidence": ("MORE_INFORMATION_OR_HUMAN", "判断材料を追加するか、人間に確認してください。"),
        "high_regression_risk": ("HUMAN_REVIEW", "回帰リスクが高いため、人間レビューへ戻してください。"),
    }[primary]


def build_advisory_response(
    prediction: dict[str, Any],
    model_version: str,
    checkpoint_sha256: str,
    request_id: str | None = None,
) -> dict[str, Any]:
    primary_logits = _finite_logit_map(prediction.get("primary_logits"), PRIMARY_LABELS)
    primary_probabilities = _finite_probability_map(prediction.get("primary_probabilities"), PRIMARY_LABELS, normalized=True)
    flag_logits = _finite_logit_map(prediction.get("flag_logits"), FLAG_LABELS)
    flag_probabilities = _finite_probability_map(prediction.get("flag_probabilities"), FLAG_LABELS, normalized=False)
    if not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256):
        raise ValueError("checkpoint_sha256 must be lowercase SHA-256 hex")
    if not isinstance(model_version, str) or not model_version.strip():
        raise ValueError("model_version is required")

    primary_label = max(PRIMARY_LABELS, key=primary_probabilities.__getitem__)
    ranked = sorted(primary_probabilities.items(), key=lambda pair: pair[1], reverse=True)
    top1_label, top1 = ranked[0]
    _top2_label, top2 = ranked[1]
    entropy = -sum(p * math.log(p) for p in primary_probabilities.values() if p > 0)
    normalized_entropy = entropy / math.log(len(PRIMARY_LABELS))
    concentration = 1.0 - normalized_entropy
    route, route_summary = _route(primary_label)

    flags = {
        flag: {
            "logit": flag_logits[flag],
            "probability": flag_probabilities[flag],
            "active": flag_probabilities[flag] >= FLAG_THRESHOLD,
            "reporting_threshold": FLAG_THRESHOLD,
            "suggested_check": FLAG_NEXT_CHECK[flag] if flag_probabilities[flag] >= FLAG_THRESHOLD else None,
        }
        for flag in FLAG_LABELS
    }
    checks = [flags[name]["suggested_check"] for name in FLAG_LABELS if flags[name]["active"]]
    if primary_label == "ready_to_implement":
        summary = "提案を限定的に進める側の出力です。自動実行を許可するものではありません。"
    elif primary_label == "verification_required":
        summary = "追加確認が必要です。" + (" " + " ".join(checks) if checks else "具体的な確認項目は人間が特定してください。")
    elif primary_label == "insufficient_evidence":
        summary = "判断に必要な情報が不足しています。証拠を追加するか、人間に確認してください。"
    else:
        summary = "回帰リスクが高い出力です。変更前に人間レビューを行ってください。"

    return {
        "schema_version": "phase7d-specialist-response-v1",
        "request_id": request_id,
        "advisory_status": "advisory_only",
        "primary": {
            "label": primary_label,
            "logits": primary_logits,
            "probabilities": primary_probabilities,
            "top1_label": top1_label,
            "top1_probability": top1,
            "top2_probability": top2,
            "margin": top1 - top2,
            "calibrated": False,
        },
        "flags": flags,
        "uncertainty": {
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "concentration": concentration,
            "semantics": "distribution properties; not correctness probabilities",
        },
        "routing": {
            "route": route,
            "semantic_route_summary": route_summary,
            "human_summary": summary,
            "suggested_checks": checks,
            "confidence_used_for_routing": False,
        },
        "autonomy_policy": {
            "authorizes_autonomous_execution": False,
            "status": "advisory_pilot",
        },
        "provenance": {
            "model_version": model_version,
            "checkpoint_sha256": checkpoint_sha256,
        },
    }
