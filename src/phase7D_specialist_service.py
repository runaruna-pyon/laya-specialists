"""Loopback-only persistent HTTP service for the frozen Phase 7C C1 model."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from phase7D_experience import VaultExperienceStore, validate_experience
from phase7D_specialist import (
    INPUT_FIELDS,
    SERVICE_VERSION,
    build_advisory_response,
    serialize_specialist_input,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "laya_local.toml"
MODEL_VERSION = "phase7C-c1-pilot"
CHECKPOINT_PATH = ROOT / "models" / "phase6_specialist" / "c1" / "best" / "model.safetensors"
MAX_REQUEST_BYTES = 1_048_576
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_vault_root(config_path: str | Path = DEFAULT_CONFIG) -> Path:
    import tomllib

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"local Laya config is required: {path}")
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    root = config.get("vault", {}).get("root")
    if not isinstance(root, str) or not root.strip():
        raise ValueError("config must set [vault].root")
    vault = Path(root).expanduser().resolve()
    if not vault.is_dir() or not (vault / "60_Knowledge").is_dir():
        raise FileNotFoundError(f"configured Obsidian Vault root is invalid: {vault}")
    return vault


def read_checkpoint_path(config_path: str | Path = DEFAULT_CONFIG) -> Path:
    import tomllib

    path = Path(config_path)
    config = tomllib.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    configured = config.get("specialist", {}).get("checkpoint_path")
    checkpoint = Path(configured).expanduser() if isinstance(configured, str) and configured.strip() else CHECKPOINT_PATH
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"configured C1 checkpoint does not exist: {checkpoint}")
    return checkpoint


def load_c1_runtime(checkpoint_path: str | Path | None = None) -> tuple[Any, str, str]:
    """Verify frozen provenance, then load the C1 specialist and pinned tokenizer."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    import train_phase6_specialist as c0

    selected_checkpoint = Path(checkpoint_path).resolve() if checkpoint_path is not None else CHECKPOINT_PATH

    c0_config = c0.read_config(ROOT / "configs" / "phase6C_c0.json")
    c1_config = json.loads((ROOT / "configs" / "phase7C_c1.json").read_text(encoding="utf-8-sig"))
    training_manifest_path = ROOT / "results" / "phase7C_training_manifest.json"
    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8-sig"))
    dataset_audit = json.loads((ROOT / "results" / "phase7C_contract_dataset_audit.json").read_text(encoding="utf-8-sig"))
    freeze_path = ROOT / "results" / "phase7D_c1_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8-sig"))
    if freeze.get("status") != "PILOT_DEPLOYABLE":
        raise RuntimeError("C1 freeze record is not marked PILOT_DEPLOYABLE")
    if sha256_file(training_manifest_path) != freeze.get("training_manifest_sha256"):
        raise RuntimeError("C1 training manifest changed after pilot freeze")
    if sha256_file(ROOT / "results" / "phase7C_contract_dataset_audit.json") != freeze.get("contract_dataset_audit_sha256"):
        raise RuntimeError("Phase 7C contract dataset audit changed after pilot freeze")
    if sha256_file(ROOT / "results" / "phase7B_source_manifest.json") != freeze.get("phase7b_concept_contract_source_manifest_sha256"):
        raise RuntimeError("Phase 7B source manifest changed after pilot freeze")
    for name, path in (("phase6C_c0", ROOT / "configs" / "phase6C_c0.json"),
                       ("phase7C_c1", ROOT / "configs" / "phase7C_c1.json")):
        if sha256_file(path) != freeze.get("architecture_config_sha256", {}).get(name):
            raise RuntimeError(f"frozen {name} configuration changed")
    if sha256_file(ROOT / "src" / "train_phase6_specialist.py") != freeze.get("architecture_source_sha256"):
        raise RuntimeError("C1 architecture source changed after pilot freeze")
    if training_manifest.get("status") != "C1_TRAINING_COMPLETE":
        raise RuntimeError("C1 training manifest is not complete")
    if dataset_audit.get("status") != "PASS" or dataset_audit.get("semantic_failure_count") != 0:
        raise RuntimeError("Phase 7C contract dataset audit is not semantically clean")
    expected_sha = training_manifest.get("selected_checkpoint_sha256")
    if (not isinstance(expected_sha, str) or expected_sha != freeze.get("checkpoint_sha256")
            or sha256_file(selected_checkpoint) != expected_sha):
        raise RuntimeError("C1 best checkpoint SHA-256 mismatch")
    for key, path in (("original_train", ROOT / "data/specialist/phase6_specialist_train_final.jsonl"),
                      ("contract_train", ROOT / "data/specialist/phase7C_contract_train.jsonl"),
                      ("original_development", ROOT / "data/specialist/phase6_specialist_development_final.jsonl"),
                      ("contract_development", ROOT / "data/specialist/phase7C_contract_dev.jsonl")):
        expected = freeze.get("training_datasets_sha256", {}).get(key)
        if key.endswith("development"):
            expected = freeze.get("development_datasets_sha256", {}).get(key)
        if sha256_file(path) != expected:
            raise RuntimeError(f"C1 source dataset changed after pilot freeze: {key}")
    if c1_config.get("dataset", {}).get("contract_train_sha256") != dataset_audit.get("dataset_sha256", {}).get("train"):
        raise RuntimeError("C1 contract TRAIN hash differs from the audited dataset")
    if c1_config.get("dataset", {}).get("contract_development_sha256") != dataset_audit.get("dataset_sha256", {}).get("development"):
        raise RuntimeError("C1 contract DEVELOPMENT hash differs from the audited dataset")
    if training_manifest.get("source_sha256", {}).get("phase7b_source_manifest") != sha256_file(ROOT / "results" / "phase7B_source_manifest.json"):
        raise RuntimeError("Phase 7B concept/contract source manifest changed after C1 training")

    base_checkpoint = c0.resolve_path(c0_config["initial_checkpoint"]["path"])
    encoder_dir = base_checkpoint.parent / "encoder"
    model_config = AutoConfig.from_pretrained(encoder_dir, local_files_only=True)
    encoder = AutoModel.from_config(model_config, attn_implementation="sdpa")
    model = c0.SpecialistModel(encoder, len(c0.PRIMARY_LABELS), len(c0.FLAG_LABELS), dropout=0.1)
    state = load_file(str(selected_checkpoint), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"C1 checkpoint tensors mismatch; missing={missing}, unexpected={unexpected}")
    tokenizer = AutoTokenizer.from_pretrained(c0.resolve_path(c0_config["tokenizer"]["path"]), local_files_only=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    precision = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"

    class C1Runtime:
        def predict(self, text: str) -> dict[str, dict[str, float]]:
            encoded = tokenizer(text, add_special_tokens=True, truncation=True, max_length=c0_config["input"]["max_length"],
                                padding=False, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                if precision == "bf16":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        primary_logits, flag_logits = model(encoded["input_ids"], encoded["attention_mask"])
                else:
                    primary_logits, flag_logits = model(encoded["input_ids"], encoded["attention_mask"])
            primary_logits = primary_logits[0].float().cpu()
            flag_logits = flag_logits[0].float().cpu()
            primary_probabilities = torch.softmax(primary_logits, dim=-1)
            flag_probabilities = torch.sigmoid(flag_logits)
            if not torch.isfinite(primary_logits).all() or not torch.isfinite(flag_logits).all():
                raise RuntimeError("C1 returned non-finite logits")
            return {
                "primary_logits": {label: float(primary_logits[i]) for i, label in enumerate(c0.PRIMARY_LABELS)},
                "primary_probabilities": {label: float(primary_probabilities[i]) for i, label in enumerate(c0.PRIMARY_LABELS)},
                "flag_logits": {label: float(flag_logits[i]) for i, label in enumerate(c0.FLAG_LABELS)},
                "flag_probabilities": {label: float(flag_probabilities[i]) for i, label in enumerate(c0.FLAG_LABELS)},
            }

    runtime = C1Runtime()
    runtime.device = str(device)
    runtime.precision = precision
    return runtime, expected_sha, precision


class RequestValidationError(ValueError):
    pass


class SpecialistService:
    def __init__(self, runtime: Any, store: VaultExperienceStore, checkpoint_sha256: str,
                 model_version: str = MODEL_VERSION, precision: str = "unknown") -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256):
            raise ValueError("checkpoint SHA-256 must be lowercase hexadecimal")
        self.runtime = runtime
        self.store = store
        self.checkpoint_sha256 = checkpoint_sha256
        self.model_version = model_version
        self.precision = precision
        self.device = str(getattr(runtime, "device", "unknown"))
        self.started_at = time.time()
        self._inference_lock = threading.Lock()

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok", "model_loaded": True, "service_version": SERVICE_VERSION,
            "model_version": self.model_version, "checkpoint_sha256": self.checkpoint_sha256,
            "device": self.device, "precision": self.precision, "calibrated": False,
            "authorizes_autonomous_execution": False, "started_at_unix": self.started_at,
        }

    @staticmethod
    def validate_decision_payload(payload: Any) -> tuple[dict[str, str], str]:
        if not isinstance(payload, dict) or set(payload) - {"input", "request_id"} or "input" not in payload:
            raise RequestValidationError("request must contain input and optional request_id only")
        try:
            text = serialize_specialist_input(payload["input"])
        except (ValueError, TypeError) as exc:
            raise RequestValidationError(str(exc)) from exc
        request_id = payload.get("request_id")
        if request_id is None:
            request_id = str(uuid.uuid4())
        if not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 128:
            raise RequestValidationError("request_id must be a nonempty string of at most 128 characters")
        return payload["input"], request_id

    def decide(self, payload: Any) -> dict[str, Any]:
        input_fields, request_id = self.validate_decision_payload(payload)
        serialized = serialize_specialist_input(input_fields)
        with self._inference_lock:
            prediction = self.runtime.predict(serialized)
        return build_advisory_response(prediction, self.model_version, self.checkpoint_sha256, request_id)

    def save_experience(self, record: Any) -> dict[str, str]:
        validate_experience(record)
        path = self.store.save(record)
        relative = path.relative_to(self.store.vault_root).as_posix()
        return {"status": "saved", "experience_id": record["experience_id"], "record_path": relative}

    def load_experience(self, experience_id: str) -> dict[str, Any]:
        if not _ID_PATTERN.fullmatch(experience_id):
            raise RequestValidationError("invalid experience_id")
        return self.store.load(experience_id)


def create_server(host: str, port: int, service: SpecialistService,
                  shutdown_token: str | None = None) -> ThreadingHTTPServer:
    if host != "127.0.0.1":
        raise ValueError("Phase 7D specialist must bind only to 127.0.0.1")

    class Handler(BaseHTTPRequestHandler):
        server_version = SERVICE_VERSION

        def log_message(self, fmt: str, *args: Any) -> None:
            print("[phase7d-http] " + (fmt % args), flush=True)

        def _send(self, status: int, obj: Any) -> None:
            data = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _read_json(self) -> Any:
            if self.headers.get_content_type() != "application/json":
                raise RequestValidationError("content_type_must_be_application_json")
            try:
                size = int(self.headers.get("Content-Length", "-1"))
            except ValueError as exc:
                raise RequestValidationError("invalid_content_length") from exc
            if size < 0 or size > MAX_REQUEST_BYTES:
                raise RequestValidationError("invalid_or_oversized_request")
            try:
                return json.loads(self.rfile.read(size).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RequestValidationError("malformed_json") from exc

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, service.health())
                return
            if self.path == "/metadata":
                self._send(200, {"service_version": SERVICE_VERSION, "advisory_status": "advisory_only",
                                 "calibrated": False, "autonomy_authorized": False,
                                 "routing_basis": "frozen semantic primary; confidence is not used"})
                return
            if self.path.startswith("/experience/"):
                experience_id = unquote(self.path.removeprefix("/experience/"))
                try:
                    self._send(200, service.load_experience(experience_id))
                except (RequestValidationError, ValueError) as exc:
                    self._send(422, {"error": "invalid_experience_id", "detail": str(exc)})
                except FileNotFoundError:
                    self._send(404, {"error": "experience_not_found"})
                return
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
            try:
                payload = self._read_json()
                if self.path == "/decide":
                    self._send(200, service.decide(payload))
                elif self.path == "/experience":
                    result = service.save_experience(payload)
                    self._send(201, result)
                else:
                    self._send(404, {"error": "not_found"})
            except RequestValidationError as exc:
                self._send(422, {"error": "invalid_request", "detail": str(exc)})
            except (ValueError, TypeError) as exc:
                self._send(422, {"error": "invalid_experience", "detail": str(exc)})
            except Exception as exc:
                self._send(503, {"error": "service_error", "detail": str(exc)})

        def do_PUT(self) -> None:
            self._send(405, {"error": "method_not_allowed"})

        def do_DELETE(self) -> None:
            self._send(405, {"error": "method_not_allowed"})

    return ThreadingHTTPServer((host, port), Handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Optional C1 checkpoint path; the frozen checkpoint SHA is still mandatory")
    args = parser.parse_args()
    vault_root = read_vault_root(args.config)
    runtime, checkpoint_sha, precision = load_c1_runtime(args.checkpoint or read_checkpoint_path(args.config))
    service = SpecialistService(runtime, VaultExperienceStore(vault_root), checkpoint_sha, precision=precision)
    token = os.environ.get("LAYA_SPECIALIST_SHUTDOWN_TOKEN")
    server = create_server(args.host, args.port, service, token)
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
