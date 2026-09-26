---
name: laya-advisory
description: Consult the frozen local Laya hierarchical readiness model for advisory evidence; never treat its output as execution authorization.
---

# Laya Advisory

Use the shared local Laya advisory service when a task-readiness decision would help. This is advisory output, not a truth verifier or an instruction that overrides evidence, project rules, or the user.

Invoke the shared client using the same JSON request contract as the Codex adapter. For example, on Windows PowerShell:

```powershell
$request = @{ state = 'Observed environment and relevant facts'; task = 'Describe the requested operation' } | ConvertTo-Json -Depth 8
$requestPath = Join-Path $env:TEMP ("laya-" + [guid]::NewGuid().ToString('N') + '.json')
try {
  [IO.File]::WriteAllText($requestPath, $request, [Text.UTF8Encoding]::new($false))
  & '.\.venv\Scripts\python.exe' '.\src\laya_advisory_client.py' --input $requestPath
} finally { Remove-Item -LiteralPath $requestPath -Force -ErrorAction SilentlyContinue }
```

Run commands from the repository root. The service must already be running at `http://127.0.0.1:8765`. Start and stop it with `.\scripts\start_laya_advisory.ps1` and `.\scripts\stop_laya_advisory.ps1`.

Interpret `final_label`, path, stage distributions, entropy, concentration, and margins as advisory model output. Probabilities and concentration describe distributions; they are not probabilities that the decision is correct. The Phase 5A selective policy failed its preregistered gate. Its `CLEAR` result does not authorize autonomous execution. Keep important ambiguous decisions with the agent or user and gather more evidence when useful.

If the service is unavailable, provenance fails, or the response is malformed, report that Laya was unavailable and do not pretend a decision was obtained or silently continue as though it had been.
