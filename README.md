# Laya Specialist — C1 Research Preview

**Research Preview / Pilot.** Laya C1 is a local, model-agnostic specialist interface for structured advisory decisions. It accepts a common JSON request and returns a typed primary decision, required-check flags, uncertainty metadata, and a human-readable advisory. It is not a truth verifier or an autonomous agent.

## What it is

This checkpoint publishes the C1 Deep Learning Specialist integration layer: a loopback-only persistent service and client, common request/response schemas, an Experience record schema, Skills, tests, and selected research summaries. The checkpoint weights, datasets, raw predictions, and private Experience records are not included.

## Why it exists

Laya explores Jev-style functional inspiration: express a bounded decision as typed outputs in one specialist inference, then let a human or calling agent decide what to do. This is an inspiration only; this project does not claim to be an exact Jev clone or reproduce unpublished internals.

## Architecture

```text
Luna / Astra / Sol / Codex / other caller
                    |
                    | common JSON request
                    v
        model-agnostic local client
                    |
                    | loopback HTTP (127.0.0.1)
                    v
          persistent C1 service
                    |
                    +--> typed primary + required-check flags
                    +--> uncertainty and advisory metadata
                    +--> optional reviewed Experience record
                    |
                    v
          human review / next check
```

`source_agent` is retained as request/response provenance metadata. It is excluded from model input serialization.

## Current C1 status and limitations

- C1 is frozen as a research pilot and remains uncalibrated.
- A synthetic stress probe contains high-confidence wrong decisions. Confidence, concentration, margin, and entropy describe model outputs; none is a probability that the answer is correct.
- Real-world HPCV competence has not been established.
- Codex was connected to the local C1 service through the common Python client in the tested Windows environment. Luna, Astra, and Sol have JSON templates and a prompt bridge; direct localhost access from those runtimes has not been verified.
- The repository does not include model weights, training/evaluation datasets, private freeze/provenance records, or the local Vault. The service source is published, but running the frozen C1 model requires separately available, matching runtime assets. No download or license for those assets is implied here.

## Quick start

1. Use a compatible Python environment with the dependencies required by the service and the separately supplied C1 runtime assets.
2. Copy `config/laya_local.example.toml` to `config/laya_local.toml`. Set the Vault root and checkpoint path for your machine. Keep the local file private.
3. Start the service from the repository root:

   ```console
   python src/phase7D_specialist_service.py --host 127.0.0.1 --port 8766
   ```

4. Submit a UTF-8 JSON request through the common client:

   ```console
   python src/laya_specialist_client.py --request request.json
   ```

`request.json` must conform to `config/laya_specialist_request.schema.json`. The service listens on loopback by default; do not expose it to a network interface.

## Common request/response protocol

A request includes `request_id`, `source_agent`, `project`, `problem_context`, `proposal`, `claim_under_review`, `known_evidence`, and `requested_decision`. `known_evidence` separates observed facts, available evidence, and missing evidence.

The response includes the typed `primary` distribution, `flags` for required checks, `uncertainty` metadata, an `advisory`, specialist/checkpoint provenance, and `authorizes_autonomous_execution: false`. See the JSON Schemas in `config/` for field constraints. Probability values are uncalibrated model outputs, not correctness probabilities.

## Experience Vault and growth loop

The versioned Experience schema supports recording a proposal, Laya's advisory, the human decision, later resolution, and provenance in a user's private Obsidian Vault. A record can inform a future reviewed inference contract only after resolution and evidence checks. System-test records are isolated from training candidates. The Vault itself and all real Experience records remain private and are not in this repository.

```text
advisory -> human decision -> later evidence/resolution
       -> reviewed Experience -> candidate contract for future research
```

## Safety and advisory behavior

The service and client fail closed on unavailable or malformed responses, checkpoint provenance mismatch, or violation of the advisory-only contract. `authorizes_autonomous_execution` is always `false`. A high-confidence result does not reduce the need for independent verification or human review.

## Research results

A single fresh **synthetic** stress comparison recorded the following descriptive results:

| Metric | C0 | C1 |
|---|---:|---:|
| Primary accuracy | 0.366 | 0.872 |
| Required-flag exact-set accuracy | 0.653 | 0.925 |
| Matched contrast pairs with both cases correct | 0/160 | 119/160 |

These are results on a constructed synthetic probe, not real-world performance estimates, calibration results, or evidence of HPCV competence. C1 still made high-confidence errors. The selected summary is in [`docs/research/c1_synthetic_stress_results.md`](docs/research/c1_synthetic_stress_results.md).

For Phase 7E, the same semantic request sent with Luna, Astra, Sol, and Codex `source_agent` metadata produced identical specialist outputs. Since `source_agent` is excluded from model input, this confirms protocol-level metadata independence, not comparative model performance. Codex's local connection was verified; the other three use JSON/prompt bridges without verified direct localhost access. See [`docs/research/model_agnostic_integration.md`](docs/research/model_agnostic_integration.md).

## Roadmap

1. Run a bounded real-world HPCV pilot with human review.
2. Resolve Experience records using independent evidence and identify only well-supported gaps.
3. Consider C2 research after real pilot evidence accumulates. C2 training, calibration, and routing-threshold changes are outside this checkpoint.

## Licensing and upstream attribution

This repository and its C1 research-preview additions are distributed under Apache-2.0, as stated in the root [`LICENSE`](LICENSE). The fork retains the upstream Apache-2.0 license and attribution for upstream-derived content. No separate upstream `NOTICE` file was present at this checkpoint.

## Author and contact

とくめーの民 — [@Tokumeies on X](https://x.com/Tokumeies)
