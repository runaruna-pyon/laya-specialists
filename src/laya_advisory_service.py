"""Loopback-only persistent advisory service for the frozen Laya v0e runtime."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "hierarchical_pilot_v0e"
CALIBRATION_PATH = ROOT / "results" / "phase4_calibration_runtime_config.json"
POLICY_PATH = ROOT / "results" / "phase5_policy_candidate_config.json"
FROZEN_MODEL_SHA256 = "761ad958e6879c73cb6bf5ba9529b68aa8eb6a6eaf23eaa18f602a4e4ee38ff6"
FROZEN_CALIBRATION_SHA256 = "cca0607acf2249caea4ad520830930be824099a8d8d79e507c35d4978d38481e"
FROZEN_POLICY_SHA256 = "16fc7a19c2df48a48a6dd3fe367be8443a567f7561a84d9e90730f5cef7f0b06"
SERVICE_VERSION = "phase5b-advisory-1.0"
MAX_REQUEST_BYTES = 1_048_576


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RequestValidationError(ValueError):
    pass


def _read_frozen_policy() -> tuple[dict[str, Any], str]:
    digest = sha256_file(POLICY_PATH)
    if digest != FROZEN_POLICY_SHA256:
        raise RuntimeError(f"frozen policy SHA-256 mismatch: {digest}")
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    expected = {"choice:2": 0.9, "choice:3-5": 0.9}
    if (policy.get("status") != "frozen_from_tune"
            or policy.get("policy_name") != "concentration"
            or policy.get("thresholds") != expected):
        raise RuntimeError("frozen experimental policy contents do not match the Phase 5A record")
    return policy, digest


def load_frozen_runtime() -> Any:
    from laya_hierarchical_runtime import HierarchicalRuntime

    if sha256_file(MODEL_PATH / "model.safetensors") != FROZEN_MODEL_SHA256:
        raise RuntimeError("frozen model SHA-256 mismatch")
    if sha256_file(CALIBRATION_PATH) != FROZEN_CALIBRATION_SHA256:
        raise RuntimeError("frozen calibration config SHA-256 mismatch")
    runtime = HierarchicalRuntime(MODEL_PATH, CALIBRATION_PATH)
    if runtime.model_sha256 != FROZEN_MODEL_SHA256 or runtime.config_sha256 != FROZEN_CALIBRATION_SHA256:
        raise RuntimeError("runtime loaded unexpected frozen provenance")
    return runtime


class AdvisoryService:
    """Owns exactly one initialized runtime and never grants execution authority."""

    def __init__(self, runtime_factory: Callable[[], Any] | None = None,
                 policy_config: dict[str, Any] | None = None,
                 policy_sha256: str | None = None) -> None:
        if runtime_factory is None:
            runtime_factory = load_frozen_runtime
        if policy_config is None:
            policy_config, measured_policy_sha = _read_frozen_policy()
            policy_sha256 = measured_policy_sha
        if policy_sha256 != FROZEN_POLICY_SHA256:
            raise ValueError("experimental policy provenance does not match the frozen Phase 5A config")
        self.policy_config = policy_config
        self.policy_sha256 = policy_sha256
        self.runtime = runtime_factory()
        if getattr(self.runtime, "model_sha256", None) != FROZEN_MODEL_SHA256:
            raise ValueError("runtime model SHA does not match the frozen model")
        if getattr(self.runtime, "config_sha256", None) != FROZEN_CALIBRATION_SHA256:
            raise ValueError("runtime calibration SHA does not match the frozen config")
        self.started_at = time.time()
        self.device = str(getattr(self.runtime, "device", "unknown"))
        self._inference_lock = threading.Lock()

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok", "model_loaded": True, "device": self.device,
            "model_sha256": FROZEN_MODEL_SHA256,
            "calibration_sha256": FROZEN_CALIBRATION_SHA256,
            "policy_sha256": self.policy_sha256,
            "service_version": SERVICE_VERSION, "pid": os.getpid(),
        }

    @staticmethod
    def _validate(payload: Any) -> tuple[Any, str, str | None]:
        if not isinstance(payload, dict):
            raise RequestValidationError("request root must be a JSON object")
        if "state" not in payload:
            raise RequestValidationError("state is required")
        state = payload["state"]
        if isinstance(state, str):
            if not state.strip():
                raise RequestValidationError("state must not be empty")
        elif isinstance(state, (dict, list)):
            if not state:
                raise RequestValidationError("state object/list must not be empty")
        else:
            raise RequestValidationError("state must be a nonempty string, object, or list")
        task = payload.get("task")
        if task is not None and (not isinstance(task, str) or not task.strip()):
            raise RequestValidationError("task must be a nonempty string when provided")
        request_id = payload.get("request_id")
        if request_id is not None and (not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 128):
            raise RequestValidationError("request_id must be a nonempty string of at most 128 characters")
        return state, request_id or str(uuid.uuid4()), task

    def handle_decide(self, payload: Any) -> dict[str, Any]:
        state, request_id, _task = self._validate(payload)
        # Laya's loaded model object is shared by all requests; serialize inference.
        runtime_state = state if _task is None else {"situation": state, "task": _task}
        with self._inference_lock:
            result = self.runtime.predict(runtime_state)
        stages = result.get("stages")
        path = result.get("path")
        if not isinstance(stages, list) or not stages or not isinstance(path, list) or len(path) != len(stages):
            raise RuntimeError("runtime returned malformed hierarchical output")
        uncertainty = []
        clear = True
        thresholds = self.policy_config["thresholds"]
        for row in stages:
            bucket = row.get("temperature_bucket")
            if bucket not in thresholds:
                raise RuntimeError(f"runtime returned unsupported temperature bucket: {bucket}")
            concentration = row.get("concentration")
            if isinstance(concentration, bool) or not isinstance(concentration, (float, int)):
                raise RuntimeError("runtime returned invalid concentration")
            threshold = float(thresholds[bucket])
            clear = clear and float(concentration) >= threshold
            uncertainty.append({key: row.get(key) for key in (
                "stage", "entropy", "normalized_entropy", "concentration",
                "top1_probability", "top2_probability", "top1_top2_margin")})
        return {
            "request_id": request_id,
            "advisory_status": "advisory_only",
            "decision": {"final_label": result["final_label"], "hierarchical_path": path},
            "stages": stages,
            "uncertainty": {"by_stage": uncertainty,
                "semantics": {"probabilities": "model output distributions; not correctness probabilities",
                              "entropy": "distribution entropy",
                              "concentration": "1 - entropy / ln(K)"}},
            "experimental_policy_result": "CLEAR" if clear else "REVIEW_REQUIRED",
            "autonomy_policy": {"status": "experimental_failed_gate",
                                "authorizes_autonomous_execution": False},
            "provenance": {"model_sha256": FROZEN_MODEL_SHA256,
                           "calibration_sha256": FROZEN_CALIBRATION_SHA256,
                           "policy_sha256": self.policy_sha256,
                           "runtime_version": SERVICE_VERSION},
        }


def create_http_server(host: str, port: int, service: AdvisoryService,
                       shutdown_token: str | None = None) -> ThreadingHTTPServer:
    if host != "127.0.0.1":
        raise ValueError("service must bind only to 127.0.0.1")

    class Handler(BaseHTTPRequestHandler):
        server_version = SERVICE_VERSION

        def log_message(self, fmt: str, *args: Any) -> None:
            print("[http] " + (fmt % args), flush=True)

        def _send(self, status: int, obj: dict[str, Any]) -> None:
            data = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, service.health())
            elif self.path == "/metadata":
                self._send(200, {"service_version": SERVICE_VERSION,
                                 "advisory_status": "advisory_only",
                                 "autonomy_authorized": False,
                                 "uncertainty_semantics": "distribution properties, not correctness probabilities"})
            else:
                self._send(404, {"error": "not_found"})

        def do_POST(self) -> None:
            if self.path == "/shutdown":
                supplied = self.headers.get("X-Laya-Shutdown-Token", "")
                if not shutdown_token or not hmac.compare_digest(supplied, shutdown_token):
                    self._send(403, {"error": "shutdown_forbidden"})
                    return
                self._send(200, {"status": "shutting_down"})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            if self.path != "/decide":
                self._send(404, {"error": "not_found"})
                return
            if self.headers.get_content_type() != "application/json":
                self._send(415, {"error": "content_type_must_be_application_json"})
                return
            try:
                size = int(self.headers.get("Content-Length", "-1"))
                if size < 0:
                    raise RequestValidationError("Content-Length is required")
                if size > MAX_REQUEST_BYTES:
                    self._send(413, {"error": "request_too_large"})
                    return
                raw = self.rfile.read(size)
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    self._send(400, {"error": "malformed_json", "detail": str(exc)})
                    return
                try:
                    response = service.handle_decide(payload)
                except RequestValidationError as exc:
                    self._send(422, {"error": "invalid_request", "detail": str(exc)})
                    return
                self._send(200, response)
            except RequestValidationError as exc:
                self._send(400, {"error": "invalid_http_request", "detail": str(exc)})
            except Exception as exc:
                self._send(503, {"error": "inference_failed", "detail": str(exc)})

        def do_PUT(self) -> None: self._send(405, {"error": "method_not_allowed"})
        def do_DELETE(self) -> None: self._send(405, {"error": "method_not_allowed"})

    return ThreadingHTTPServer((host, port), Handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    service = AdvisoryService()
    token = os.environ.get("LAYA_ADVISORY_SHUTDOWN_TOKEN")
    server = create_http_server(args.host, args.port, service, token)
    print(json.dumps({"event": "ready", **service.health(), "host": args.host,
                      "port": server.server_address[1]}, ensure_ascii=False), flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
