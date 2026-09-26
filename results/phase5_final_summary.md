# Laya Phase 5 最終記録

記録日: 2026-09-26
状態: **PHASE 5 COMPLETE — GENERIC LAYA ADVISORY SKILL V1 COMPLETE**

## Phase 1–4の基盤

- Phase 1–3で階層型task-readinessモデルv0eを構築・freezeし、preregisteredなsealed final holdout評価を完了した。結果記録は `diagnostic_v3_final_v0e.txt`、Phase 3の事前登録は `phase3_preregistration.md` を参照。
- Phase 4Aでfresh calibration fit/evaluationを完了し、temperature configをfreezeした。結果は `phase4_calibration_eval.txt` と `phase4_calibration_e2e.txt`。
- Phase 4Bで凍結modelを使うhierarchical runtime API、CLI、smoke testを完成した。結果は `phase4_runtime_report.txt` と `phase4_runtime_latency.json`。
- Phase 5C時点のmodel SHA-256は `761ad958e6879c73cb6bf5ba9529b68aa8eb6a6eaf23eaa18f602a4e4ee38ff6`。Phase 4 calibration config SHA-256は `cca0607acf2249caea4ad520830930be824099a8d8d79e507c35d4978d38481e`。

## Phase 5A — Selective policy: FAIL

凍結したgeneric concentration policyはheld-out gateを満たさなかった。thresholdを再探索せず、Generic Laya Skill v1のautonomous execution gateとしては採用しない。正式な判断と限定的な解釈は `phase5A_final_decision.md` に記録した。

## Phase 5B — Advisory Skill infrastructure: PASS

`src/laya_advisory_service.py`、`src/laya_advisory_client.py`、Codex/Claude Code adapterを実装した。serviceはloopback上のpersistent processでmodelを再利用し、凍結SHAを検証する。adapterは同じclientを利用し、独自taxonomy logicを持たない。詳細は `phase5B_advisory_skill_report.txt`。

## Phase 5C — Integration replay: PASS

24 fresh replay casesをCodex/Claude Code adapterから各1回ずつ、合計48実行した。6 final labelを含む通常advisory、CLEAR/REVIEW_REQUIRED両branch、安全なfail-closed経路を確認した。

- accounted adapter executions: 48/48
- Codex/Claude final label・path mismatch: 0
- provenance欠落: 0
- `authorizes_autonomous_execution = true`: 0
- CLEAR誤用: 0
- failure時fabricated decision / silent continuation: 0
- model load: 1回、runtime inference request: 40回
- Phase 5C acceptance gates: 全項目PASS

Replay set SHA-256: `14c63f9c864a25db991a1c87fbdefeb8eef1ae7dd6ad6094ac824ef5aa37fcfe`。詳細なcase結果・失敗fixture・provenanceは `phase5C_replay_results.json`、読みやすい結果は `phase5C_replay_report.txt`。

Codex/Claudeの各Skill instructionが指定する同一common-client CLIをadapter名ごとに起動した決定論的replayであり、interactive LLM agent sessionの比較ではない。CLEAR/REVIEW_REQUIREDの8例は両policy出力branchを通すためのintegration branch-coverage例であり、代表標本ではない。Phase 5Cは分類精度、policy性能、Codex/Claudeの能力差、未知分布への一般化を評価していない。初回のWindows pipe encoding failureとbranch coverage未達runはattempt1 artifactとして保持している。凍結policy・model・calibrationは変更していない。

## Generic Laya v1の位置づけ

名称: **Generic Laya Advisory Skill v1**

- advisory only。truth verifierではない。
- probabilities、entropy、concentrationはmodel distributionの情報で、correctness probabilityではない。
- Phase 5A selective policyはpreregistered autonomy gateに失敗した。
- `CLEAR`は自律実行を許可しない。runtimeは常に `authorizes_autonomous_execution: false` を返す。
- model/service/provenance failure時はdecisionをfabricateせず、明示的にfail closedする。

## 次段階

Phase 5を完了した。次工程候補はPhase 6 Deep Learning Specialistであるが、この記録ではPhase 6の設計・実装・学習を開始していない。Phase 6ではgeneric concentration thresholdの流用を前提にせず、別のsemantic decision taxonomyとして新規に事前登録する。
