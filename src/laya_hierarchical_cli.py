"""Minimal JSON CLI for laya_hierarchical_runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from laya_hierarchical_runtime import HierarchicalRuntime


def _read_request(input_path: str | None) -> dict[str, Any]:
    if input_path:
        request = json.loads(Path(input_path).read_text(encoding="utf-8"))
    else:
        request = json.load(sys.stdin)
    if not isinstance(request, dict):
        raise ValueError("request JSON must be an object")
    return request


def _dispatch(runtime: HierarchicalRuntime, request: dict[str, Any]) -> dict[str, Any]:
    if "state" not in request:
        raise ValueError("request requires 'state'")
    mode = request.get("mode", "hierarchical")
    if mode == "hierarchical":
        return runtime.predict(request["state"])
    if mode == "stage":
        if "stage" not in request:
            raise ValueError("stage mode requires 'stage'")
        return runtime.predict_stage(request["state"], request["stage"], request.get("options"))
    raise ValueError("mode must be 'hierarchical' or 'stage'")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Frozen v0e hierarchical decision runtime")
    parser.add_argument("--input", help="UTF-8 JSON request file; defaults to stdin")
    parser.add_argument("--model-path", help="defaults to the frozen v0e model directory")
    parser.add_argument("--calibration-config", help="defaults to the Phase 4A frozen runtime config")
    parser.add_argument("--device", choices=("cuda", "cpu"), help="defaults to CUDA or explicit CPU fallback")
    args = parser.parse_args(argv)
    try:
        request = _read_request(args.input)
        kwargs = {}
        if args.model_path:
            kwargs["model_path"] = args.model_path
        if args.calibration_config:
            kwargs["calibration_config_path"] = args.calibration_config
        if args.device:
            kwargs["device"] = args.device
        response = _dispatch(HierarchicalRuntime(**kwargs), request)
        json.dump(response, sys.stdout, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        sys.stdout.write("\n")
        return 0
    except Exception as exc:
        error = {"error": {"type": type(exc).__name__, "message": str(exc)}}
        json.dump(error, sys.stderr, ensure_ascii=False)
        sys.stderr.write("\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
