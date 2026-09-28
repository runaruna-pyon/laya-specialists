---
name: laya-specialist
description: Consult the local Laya C1 specialist for typed, advisory-only review of bounded technical decisions.
---

# Laya C1 Specialist

Use the model-agnostic local client when a structured second opinion would help evaluate a proposal. The response is advisory evidence, not a truth verifier and not an instruction that overrides project rules, evidence, or the user.

Submit a UTF-8 JSON request matching `config/laya_specialist_request.schema.json`:

```console
python src/laya_specialist_client.py --request request.json
```

The service must already be running on the configured loopback endpoint. Do not expose it on a network interface. The client fails closed if the service is unavailable, malformed, or violates checkpoint provenance or advisory-only requirements.

Interpret primary and flag probabilities, margin, entropy, and concentration as uncalibrated model outputs. They are not probabilities of correctness. High-confidence errors are known. Keep independent checks and human review in the loop. `authorizes_autonomous_execution` is always `false`.

`source_agent` is provenance metadata and is excluded from model input. Luna/Astra/Sol direct localhost support has not been verified; use their JSON/prompt bridge until direct access is independently confirmed.
