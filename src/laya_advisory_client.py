"""Small fail-closed client and JSON CLI for the persistent Laya advisory service."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from typing import Any

from laya_advisory_service import (
    FROZEN_CALIBRATION_SHA256, FROZEN_MODEL_SHA256, FROZEN_POLICY_SHA256,
)


class AdvisoryServiceUnavailable(RuntimeError):
    pass


class AdvisoryProtocolError(RuntimeError):
    pass


def cli_json_text(value: dict[str, Any]) -> str:
    """Serialize CLI output as ASCII-only JSON so Windows pipe codepages cannot corrupt it."""
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


class AdvisoryClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8765", timeout: float = 30.0) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or not parsed.port:
            raise ValueError("base_url must use http://127.0.0.1:<port>")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(self.base_url + path, data=data,
                          headers={"Content-Type": "application/json"} if data is not None else {},
                          method="POST" if data is not None else "GET")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (URLError, TimeoutError, HTTPError, OSError) as exc:
            raise AdvisoryServiceUnavailable(f"Laya advisory service unavailable: {exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AdvisoryProtocolError("service returned malformed JSON") from exc
        if not isinstance(result, dict):
            raise AdvisoryProtocolError("service response root must be an object")
        return result

    def health(self) -> dict[str, Any]:
        result = self._request("/health")
        if result.get("status") != "ok" or result.get("model_loaded") is not True:
            raise AdvisoryProtocolError("service is not ready")
        self._validate_provenance(result, health=True)
        return result

    @staticmethod
    def _validate_provenance(result: dict[str, Any], health: bool = False) -> None:
        actual = result if health else result.get("provenance", {})
        expected = {"model_sha256": FROZEN_MODEL_SHA256,
                    "calibration_sha256": FROZEN_CALIBRATION_SHA256}
        expected["policy_sha256"] = FROZEN_POLICY_SHA256
        for key, value in expected.items():
            if actual.get(key) != value:
                raise AdvisoryProtocolError(f"service {key} does not match the frozen value")

    def decide(self, state: Any, task: str | None = None, request_id: str | None = None) -> dict[str, Any]:
        if isinstance(state, str) and not state.strip() or not isinstance(state, (str, dict, list)) or isinstance(state, (dict, list)) and not state:
            raise ValueError("state must be a nonempty string, object, or list")
        payload: dict[str, Any] = {"state": state, "request_id": request_id or str(uuid.uuid4())}
        if task is not None:
            payload["task"] = task
        result = self._request("/decide", payload)
        if result.get("advisory_status") != "advisory_only":
            raise AdvisoryProtocolError("service response is not explicitly advisory")
        autonomy = result.get("autonomy_policy")
        if not isinstance(autonomy, dict) or autonomy.get("authorizes_autonomous_execution") is not False:
            raise AdvisoryProtocolError("response lacks the mandatory false-autonomy guard")
        self._validate_provenance(result)
        decision = result.get("decision")
        if not isinstance(decision, dict) or not isinstance(decision.get("final_label"), str) or not isinstance(decision.get("hierarchical_path"), list):
            raise AdvisoryProtocolError("response has malformed decision schema")
        if not isinstance(result.get("stages"), list) or not isinstance(result.get("uncertainty", {}).get("by_stage"), list):
            raise AdvisoryProtocolError("response has malformed stage/uncertainty schema")
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="JSON request file; if omitted, read JSON from stdin")
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    try:
        payload = json.load(open(args.input, encoding="utf-8")) if args.input else json.load(sys.stdin)
        if not isinstance(payload, dict) or "state" not in payload:
            raise ValueError("input must be an object containing state")
        result = AdvisoryClient(args.url, args.timeout).decide(
            payload["state"], payload.get("task"), payload.get("request_id"))
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0
    except Exception as exc:
        print(cli_json_text({"error": type(exc).__name__, "detail": str(exc),
                             "advisory_status": "unavailable"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
