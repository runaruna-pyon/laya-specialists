"""Fail-closed loopback client for the C1 Deep Learning Specialist pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "http://127.0.0.1:8766"
FREEZE_MANIFEST = ROOT / "results" / "phase7D_c1_freeze.json"


class SpecialistUnavailable(RuntimeError):
    pass


class SpecialistProtocolError(RuntimeError):
    pass


class SpecialistClient:
    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 30.0,
                 expected_checkpoint_sha256: str | None = None) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or not parsed.port:
            raise ValueError("base_url must use http://127.0.0.1:<port>")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.expected_checkpoint_sha256 = expected_checkpoint_sha256 or self._load_expected_sha()

    @staticmethod
    def _load_expected_sha() -> str:
        if not FREEZE_MANIFEST.is_file():
            raise FileNotFoundError("C1 freeze manifest is unavailable; refusing unpinned service")
        manifest = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8-sig"))
        if manifest.get("status") != "PILOT_DEPLOYABLE":
            raise SpecialistProtocolError("C1 is not marked PILOT_DEPLOYABLE")
        expected = manifest.get("checkpoint_sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise SpecialistProtocolError("freeze manifest has no valid checkpoint SHA")
        return expected

    def _request(self, path: str, payload: Any | None = None, method: str | None = None) -> Any:
        data = None if payload is None else json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        request = Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"} if data is not None else {},
            method=method or ("POST" if data is not None else "GET"),
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise SpecialistUnavailable(f"C1 Specialist service unavailable: {exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SpecialistProtocolError("service returned malformed JSON") from exc
        return result

    def health(self) -> dict[str, Any]:
        health = self._request("/health")
        if not isinstance(health, dict) or health.get("status") != "ok" or health.get("model_loaded") is not True:
            raise SpecialistProtocolError("C1 Specialist service is not ready")
        if health.get("checkpoint_sha256") != self.expected_checkpoint_sha256:
            raise SpecialistProtocolError("service checkpoint does not match the frozen C1 checkpoint")
        if health.get("calibrated") is not False or health.get("authorizes_autonomous_execution") is not False:
            raise SpecialistProtocolError("service metadata violates the advisory-only C1 contract")
        return health

    def decide(self, input_fields: dict[str, str], request_id: str | None = None) -> dict[str, Any]:
        self.health()
        payload: dict[str, Any] = {"input": input_fields}
        if request_id is not None:
            payload["request_id"] = request_id
        result = self._request("/decide", payload)
        if not isinstance(result, dict) or result.get("advisory_status") != "advisory_only":
            raise SpecialistProtocolError("response is not explicitly advisory")
        if result.get("autonomy_policy", {}).get("authorizes_autonomous_execution") is not False:
            raise SpecialistProtocolError("response lacks the mandatory false-autonomy guard")
        if result.get("primary", {}).get("calibrated") is not False:
            raise SpecialistProtocolError("C1 output must be marked uncalibrated")
        if result.get("routing", {}).get("confidence_used_for_routing") is not False:
            raise SpecialistProtocolError("C1 routing must not use confidence")
        if result.get("provenance", {}).get("checkpoint_sha256") != self.expected_checkpoint_sha256:
            raise SpecialistProtocolError("response checkpoint provenance mismatch")
        if not isinstance(result.get("flags"), dict) or not isinstance(result.get("primary", {}).get("probabilities"), dict):
            raise SpecialistProtocolError("response lacks typed primary/flag outputs")
        return result

    def save_experience(self, record: dict[str, Any]) -> dict[str, Any]:
        self.health()
        result = self._request("/experience", record)
        if not isinstance(result, dict) or result.get("status") != "saved":
            raise SpecialistProtocolError("experience save did not return a saved status")
        return result

    def load_experience(self, experience_id: str) -> dict[str, Any]:
        self.health()
        result = self._request("/experience/" + experience_id)
        if not isinstance(result, dict) or result.get("experience_id") != experience_id:
            raise SpecialistProtocolError("experience load returned an unexpected record")
        return result


def human_summary(result: dict[str, Any]) -> str:
    primary = result["primary"]
    lines = [
        f"Decision: {result['routing']['route']} — {result['routing']['human_summary']}",
        f"Primary: {primary['label']} (uncalibrated model probability {primary['top1_probability']:.3f})",
        f"Margin: {primary['margin']:.3f}; entropy: {result['uncertainty']['entropy']:.3f}; concentration: {result['uncertainty']['concentration']:.3f}",
        "Flags:",
    ]
    active = [name for name, value in result["flags"].items() if value["active"]]
    if active:
        lines.extend(f"- {name}: {result['flags'][name]['probability']:.3f}; {result['flags'][name]['suggested_check']}" for name in active)
    else:
        lines.append("- none active at the fixed 0.50 reporting threshold")
    lines.append("確率・margin・concentrationは正答確率ではありません。自動実行は許可されません。")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--input", type=Path, help="UTF-8 JSON file containing the six-field input object")
    actions.add_argument("--save-experience", type=Path, help="UTF-8 JSON Experience Record to validate and save to the Vault")
    actions.add_argument("--get-experience", help="Read an Experience Record by experience_id")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--format", choices=("json", "human"), default="json")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    try:
        client = SpecialistClient(args.url, args.timeout)
        if args.input:
            payload = json.loads(args.input.read_text(encoding="utf-8-sig"))
            if not isinstance(payload, dict) or "input" not in payload:
                raise ValueError("request file must contain an input object")
            result = client.decide(payload["input"], payload.get("request_id"))
            output = json.dumps(result, ensure_ascii=args.format == "json", indent=2) if args.format == "json" else human_summary(result)
        elif args.save_experience:
            record = json.loads(args.save_experience.read_text(encoding="utf-8-sig"))
            output = json.dumps(client.save_experience(record), ensure_ascii=False, indent=2)
        else:
            output = json.dumps(client.load_experience(args.get_experience), ensure_ascii=False, indent=2)
        print(output)
        return 0
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__, "detail": str(exc), "advisory_status": "unavailable"}, ensure_ascii=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
