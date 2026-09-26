"""Frozen v0e hierarchical inference API with Phase 4A temperature scaling."""

from __future__ import annotations

import hashlib
import json
import math
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT / "models" / "hierarchical_pilot_v0e"
DEFAULT_CALIBRATION_PATH = ROOT / "results" / "phase4_calibration_runtime_config.json"
FROZEN_MODEL_SHA256 = "761ad958e6879c73cb6bf5ba9529b68aa8eb6a6eaf23eaa18f602a4e4ee38ff6"

# Copied without changes from the Phase 3 frozen evaluator.
STAGE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "information_sufficiency": {
        "stage": "stage1",
        "instructions": (
            "現在記述されている事実だけを使って、"
            "実行可能性を判断するための情報が十分か判定してください。"
            "対象物やサービスの種類だけから推測しないでください。"
            "必要な事実が未確認なら insufficient_information を選んでください。"
        ),
        "criteria": {
            "sufficient_information": (
                "実行可能性について次の判断へ進むために必要な事実が、"
                "現在の記述だけで十分に確認されている。"
            ),
            "insufficient_information": (
                "実行可能性を判断するために必要な事実の一部が未確認であり、"
                "現在の情報だけでは判断できない。"
            ),
        },
    },
    "current_readiness": {
        "stage": "stage2",
        "instructions": (
            "必要な事実は十分に確認されているものとして、"
            "現在すでに追加準備なしで実行できるか判定してください。"
            "対象物の種類ではなく、現在の状態を見て判断してください。"
        ),
        "criteria": {
            "ready_now": (
                "必要な資源、設定、権限、環境がすでに利用可能であり、"
                "追加準備なしで現在すぐ実行できる。"
            ),
            "not_ready_now": (
                "必要条件の少なくとも一つが現在満たされていないことが"
                "確認されており、追加の対応なしでは実行できない。"
            ),
        },
    },
    "remedy_type": {
        "stage": "stage3",
        "instructions": (
            "現在は追加対応なしでは実行できないことが確認されています。"
            "不足している条件を満たすために必要な解決経路を一つ選んでください。"
        ),
        "criteria": {
            "local_setup": (
                "現在の環境内でのインストール、設定変更、"
                "ファイル配置などのローカル作業だけで解決できる。"
            ),
            "external_access": (
                "外部サービス、API、アカウント、認証、ライセンス、"
                "契約、権限など、新しい外部アクセスの取得が必要である。"
            ),
            "other_environment": (
                "現在の環境では要件を満たせないが、"
                "利用可能であることが確認済みの別環境なら実行できる。"
            ),
            "no_allowed_remedy": (
                "現在示されている条件と許可された手段の範囲では、"
                "要件を満たす解決方法が存在しない。"
            ),
        },
    },
}
STAGE3_TO_FINAL = {
    "local_setup": "needs_local_setup",
    "external_access": "needs_external_access",
    "other_environment": "needs_other_environment",
    "no_allowed_remedy": "blocked",
}


class CalibrationConfigError(ValueError):
    """The frozen temperature config is missing, malformed, or inconsistent."""


class ModelIntegrityError(RuntimeError):
    """The loaded model files do not match the frozen checkpoint digest."""


class RuntimeInferenceError(RuntimeError):
    """The underlying Laya inference failed or returned an invalid result."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_temperature_bucket(option_count: int) -> str:
    if option_count == 2:
        return "choice:2"
    if 3 <= option_count <= 5:
        return "choice:3-5"
    raise ValueError(f"unsupported choice option count: {option_count}; supported counts are 2 through 5")


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    exps = np.exp(scaled - np.max(scaled))
    return exps / exps.sum()


def _validate_state(state: Any) -> None:
    if isinstance(state, str):
        if not state.strip():
            raise ValueError("state must not be empty")
        return
    if isinstance(state, (dict, list)) and state:
        return
    raise ValueError("state must be a nonempty string, object, or list")


class HierarchicalRuntime:
    """Inference-only wrapper for the frozen hierarchical v0e model."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        calibration_config_path: str | Path = DEFAULT_CALIBRATION_PATH,
        device: str | None = None,
        *,
        agent: Any | None = None,
        expected_model_sha256: str = FROZEN_MODEL_SHA256,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.calibration_config_path = Path(calibration_config_path).resolve()
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"model directory not found: {self.model_path}")
        self.weights_path = self.model_path / "model.safetensors"
        if not self.weights_path.is_file():
            raise FileNotFoundError(f"model weights not found: {self.weights_path}")
        self.model_sha256 = sha256_file(self.weights_path)
        if self.model_sha256 != expected_model_sha256:
            raise ModelIntegrityError(
                f"frozen model SHA-256 mismatch: expected {expected_model_sha256}, got {self.model_sha256}"
            )
        self.config = self._read_calibration_config()
        self.temperatures = self.config["temperature_by_options"]
        self.config_sha256 = sha256_file(self.calibration_config_path)
        self.fallback_warnings: list[str] = []
        self.cold_model_load_seconds = 0.0
        if agent is None:
            self.agent, self.device = self._load_laya_agent(device)
        else:
            self.agent = agent
            self.device = str(getattr(agent, "device", "injected"))
            self._assert_laya_agent_t1(agent)

    def _read_calibration_config(self) -> dict[str, Any]:
        if not self.calibration_config_path.is_file():
            raise FileNotFoundError(f"frozen calibration config not found: {self.calibration_config_path}")
        try:
            config = json.loads(self.calibration_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CalibrationConfigError(f"cannot read calibration config: {exc}") from exc
        if not isinstance(config, dict):
            raise CalibrationConfigError("calibration config root must be an object")
        if config.get("model_sha256") != self.model_sha256:
            raise CalibrationConfigError("calibration config model SHA does not match the loaded model")
        temps = config.get("temperature_by_options")
        if not isinstance(temps, dict) or set(temps) != {"choice:2", "choice:3-5"}:
            raise CalibrationConfigError("calibration config must define exactly choice:2 and choice:3-5")
        for bucket, value in temps.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise CalibrationConfigError(f"invalid temperature for {bucket}: expected a number")
            if not math.isfinite(float(value)) or not 0.5 <= float(value) <= 5.0:
                raise CalibrationConfigError(f"temperature for {bucket} must be finite and within [0.5, 5.0]")
        if config.get("status") != "frozen_after_preregistered_acceptance_rule":
            raise CalibrationConfigError("calibration config is not marked as frozen")
        return config

    @staticmethod
    def _assert_laya_agent_t1(agent: Any) -> None:
        if [float(x) for x in agent.temperature] != [1.0, 1.0, 1.0] or dict(agent.temperature_by_options):
            raise CalibrationConfigError(
                "Laya agent must expose raw T=1 logits/probabilities; shipped model temperatures are not neutral"
            )

    def _load_laya_agent(self, requested_device: str | None) -> tuple[Any, str]:
        import torch
        import laya

        if requested_device not in (None, "cuda", "cpu"):
            raise ValueError("device must be 'cuda', 'cpu', or omitted")
        if requested_device == "cpu":
            selected_device = "cpu"
        elif torch.cuda.is_available():
            selected_device = "cuda"
        else:
            selected_device = "cpu"
            message = "CUDA is unavailable; Laya's native CPU device path is being used. Inference will be slower."
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            self.fallback_warnings.append("cuda_unavailable_cpu_fallback")
        started = time.perf_counter()
        try:
            agent = laya.load(str(self.model_path), device=selected_device)
        except Exception as exc:
            raise RuntimeInferenceError(f"failed to load frozen Laya model on {selected_device}: {exc}") from exc
        self.cold_model_load_seconds = time.perf_counter() - started
        self._assert_laya_agent_t1(agent)
        return agent, str(agent.device)

    @staticmethod
    def _stage_def(stage: str) -> dict[str, Any]:
        try:
            return STAGE_DEFINITIONS[stage]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"unknown stage: {stage!r}") from exc

    def _runtime_metadata(self) -> dict[str, Any]:
        try:
            import laya
            version = getattr(laya, "__version__", None)
        except Exception:
            version = None
        return {
            "model_path": str(self.model_path),
            "model_sha256": self.model_sha256,
            "device": self.device,
            "cold_model_load_seconds": self.cold_model_load_seconds,
            "laya_version": version,
            "warnings": list(self.fallback_warnings),
        }

    def _calibration_metadata(self) -> dict[str, Any]:
        return {
            "status": "calibrated",
            "config_path": str(self.calibration_config_path),
            "config_sha256": self.config_sha256,
            "choice:2_temperature": self.temperatures["choice:2"],
            "choice:3-5_temperature": self.temperatures["choice:3-5"],
            "method": self.config.get("method", "temperature scaling; details in frozen runtime config"),
            "model_sha256": self.model_sha256,
        }

    def _infer_stage(self, state: Any, stage: str, options: list[str] | None) -> dict[str, Any]:
        _validate_state(state)
        definition = self._stage_def(stage)
        canonical_options = list(definition["criteria"])
        if options is not None:
            if not isinstance(options, list) or not all(isinstance(x, str) and x for x in options):
                raise ValueError("options must be a nonempty list of nonempty strings")
            if len(options) != len(canonical_options):
                raise ValueError(f"unsupported option count for {stage}: expected {len(canonical_options)}")
            if options != canonical_options:
                raise ValueError("options must exactly match the frozen ordered taxonomy for this stage")
        bucket = resolve_temperature_bucket(len(canonical_options))
        if bucket not in self.temperatures:
            raise CalibrationConfigError(f"frozen calibration config has no temperature for {bucket}")
        temperature = float(self.temperatures[bucket])
        question_id = stage
        question = {
            "type": "choice",
            "instructions": definition["instructions"],
            "criteria": definition["criteria"],
        }
        # Match the frozen Phase 3 evaluator's state normalization exactly.
        model_state = {"situation": state} if isinstance(state, str) else state
        try:
            captured: list[np.ndarray] = []

            def capture_logits(module: Any, inputs: Any, output: Any) -> None:
                if not isinstance(output, (tuple, list)) or not output:
                    raise RuntimeInferenceError("Laya model returned an invalid output structure")
                logits = output[0]
                captured.append(logits.detach().float().cpu().numpy())

            hook = self.agent.model.register_forward_hook(capture_logits)
            try:
                response = self.agent.predict(model_state, {question_id: question})
            finally:
                hook.remove()
            if len(captured) != 1:
                raise RuntimeInferenceError(f"expected one model forward pass, observed {len(captured)}")
            matrix = np.asarray(captured[0])
            if matrix.ndim != 2 or matrix.shape[0] != 1 or matrix.shape[1] < len(canonical_options):
                raise RuntimeInferenceError(f"invalid logits shape: {matrix.shape}")
            logits = matrix[0, :len(canonical_options)].astype(np.float64)
            if not np.isfinite(logits).all():
                raise RuntimeInferenceError("Laya returned non-finite logits")
            public_answer = response["answers"][question_id]
            public_probs = public_answer["probabilities"]
            raw = _softmax(logits, 1.0)
            public = np.asarray([float(public_probs[key]) for key in canonical_options])
            if not np.allclose(public, raw, atol=0.00011, rtol=0.0):
                raise RuntimeInferenceError("Laya T=1 distribution differs from captured raw logits")
        except RuntimeInferenceError:
            raise
        except Exception as exc:
            raise RuntimeInferenceError(f"Laya inference failed at {stage}: {exc}") from exc

        calibrated = _softmax(logits, temperature)
        raw_index = int(np.argmax(raw))
        calibrated_index = int(np.argmax(calibrated))
        if raw_index != calibrated_index:
            raise RuntimeInferenceError("positive temperature changed argmax; refusing to return inconsistent result")
        entropy = float(-np.sum(calibrated[calibrated > 0] * np.log(calibrated[calibrated > 0])))
        normalized = entropy / math.log(len(canonical_options))
        order = np.argsort(calibrated)[::-1]
        top1, top2 = float(calibrated[order[0]]), float(calibrated[order[1]])
        return {
            "stage": definition["stage"],
            "question_id": question_id,
            "question_type": "choice",
            "option_count": len(canonical_options),
            "temperature_bucket": bucket,
            "temperature": temperature,
            "options": canonical_options,
            "raw_logits": [float(x) for x in logits],
            "raw_probabilities": [float(x) for x in raw],
            "calibrated_probabilities": [float(x) for x in calibrated],
            "raw_prediction": canonical_options[raw_index],
            "calibrated_prediction": canonical_options[calibrated_index],
            "prediction": canonical_options[calibrated_index],
            "entropy": entropy,
            "normalized_entropy": float(normalized),
            "concentration": float(1.0 - normalized),
            "top1_probability": top1,
            "top2_probability": top2,
            "top1_top2_margin": top1 - top2,
        }

    @staticmethod
    def _terminal_label(stage: str, prediction: str) -> str | None:
        if stage == "information_sufficiency" and prediction == "insufficient_information":
            return "insufficient_information"
        if stage == "current_readiness" and prediction == "ready_now":
            return "ready_now"
        if stage == "remedy_type":
            return STAGE3_TO_FINAL[prediction]
        return None

    def predict_stage(self, state: Any, stage: str, options: list[str] | None = None) -> dict[str, Any]:
        """Evaluate one frozen taxonomy stage without performing hierarchy routing."""
        result = self._infer_stage(state, stage, options)
        final_label = self._terminal_label(stage, result["prediction"])
        return {
            "final_label": final_label,
            "path": [{"stage": result["stage"], "prediction": result["prediction"]}],
            "stages": [result],
            "calibration": self._calibration_metadata(),
            "runtime": self._runtime_metadata(),
        }

    def predict(self, state: Any) -> dict[str, Any]:
        """Run the frozen Stage1 -> Stage2 -> Stage3 hierarchy and map to a final label."""
        _validate_state(state)
        stages = []
        path = []
        first = self._infer_stage(state, "information_sufficiency", None)
        stages.append(first)
        path.append({"stage": first["stage"], "prediction": first["prediction"]})
        final_label = self._terminal_label("information_sufficiency", first["prediction"])
        if final_label is None:
            second = self._infer_stage(state, "current_readiness", None)
            stages.append(second)
            path.append({"stage": second["stage"], "prediction": second["prediction"]})
            final_label = self._terminal_label("current_readiness", second["prediction"])
            if final_label is None:
                third = self._infer_stage(state, "remedy_type", None)
                stages.append(third)
                path.append({"stage": third["stage"], "prediction": third["prediction"]})
                final_label = self._terminal_label("remedy_type", third["prediction"])
        if final_label is None:
            raise RuntimeInferenceError("hierarchy ended without a terminal final label")
        return {
            "final_label": final_label,
            "path": path,
            "stages": stages,
            "calibration": self._calibration_metadata(),
            "runtime": self._runtime_metadata(),
        }
