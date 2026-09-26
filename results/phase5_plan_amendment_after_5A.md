# Phase 5 plan amendment after 5A

記録日: 2026-09-26
適用範囲: Phase 5B / Phase 5Cの計画。元のpreregistrationを上書きせず、Phase 5A final decisionを反映する。
状態: **計画変更のみ。Skill、service、adapterは未実装。**

## Phase 5Aから引き継ぐ制約

Phase 5AはOperational PASS / Policy Performance FAILで終了した。freeze済みpolicy（Concentration、`choice:2 = 0.90`、`choice:3-5 = 0.90`）は記録・表示用に保持できるが、Generic Laya Skill v1のautonomous execution gateには採用しない。

runtimeまたはadapterでこのpolicyを表現する場合、状態は明示的に `experimental_failed_gate` とする。CLEARを「安全に自動続行してよい」というauthorizationへ変換しない。threshold retuning、label-specific / stage-specific threshold、新signalやsignal combinationの追加はPhase 5では行わない。既存Phase 5A policy config、model、calibration、taxonomy、routingは変更しない。

## Phase 5B — Advisory Generic Skill Infrastructure

Phase 5Bの目的を自律controllerではなく、**ADVISORY GENERIC SKILL INFRASTRUCTURE**へ限定する。

実装対象は次のとおり。

- frozen v0eとcalibrationを利用するpersistent Laya runtime service
- 共通のSkill adapter interface
- Codex adapterとClaude Code adapter
- structured hierarchical decision output
- uncertainty metadataとmodel/calibration provenance
- explicit policy status
- inference失敗、service停止、schema不整合時の明示的error handling

初期のservice方式はPhase 5 preregistrationに沿ったloopback限定のpersistent local HTTP/JSONを候補とする。Codex/Claude adapterは同一coreを使い、model decision logicやpolicyをそれぞれへ複製しない。persistent processを再利用してcold model loadを毎回避ける。

Skill responseには最低限、decision、final_label、hierarchical_path、calibrated_probabilities、entropy、concentration、top1_top2_marginを含める。併せて次のようなpolicy metadataを返す。

```json
{
  "autonomy_policy": {
    "status": "experimental_failed_gate",
    "authorizes_autonomous_execution": false
  }
}
```

uncertainty distribution statisticsをcorrectness probabilityと呼ばない。Codex/Claude Codeはこのgeneric resultやCLEARを根拠に、重要な操作を自動承認してはならない。advisory出力は根拠情報として利用し、実行権限やhuman approvalを置き換えない。

Phase 5B acceptance確認では、model/calibration provenance、response schema、共通coreの利用、runtime再利用、起動・停止・error behavior、cold load回避を確認する。service error時にsilent continuationしない。

## Phase 5C — 縮小したSkill integration smoke/replay

大規模selective-policy benchmarkは実施しない。freshな小規模smoke/replayで次を確認する。

- agentがLayaのstructured resultを取得できる
- model/calibration provenanceが保持される
- `experimental_failed_gate`がauthorizationとして誤用されない
- Codex/Claude adapterが同じcoreを呼ぶ
- persistent runtimeを再利用しcold loadを避ける
- runtime、schema、transport失敗でsilent continuationしない

これはpolicyの再評価やthreshold selectionではなく、Skill integrationと安全なadvisory contractの検証とする。Phase 5A EVALUATION v5は再利用しない。

## Phase 5 completion naming

Phase 5Bと縮小したPhase 5Cの条件が完了した場合の名称は **GENERIC LAYA ADVISORY SKILL V1** とする。「safe autonomous agent controller」等とは呼ばない。Phase 5完了はPhase 6 Specialistの開始を自動承認するものではない。

## Phase 6への引き継ぎ

Deep Learning Specialistは別のsemantic decision taxonomyとして新規設計・事前登録する。候補概念はevidence sufficiency、measurement / ablationの必要性、reference verification、regression risk、implementation readinessなど。generic concentration thresholdをそのまま再利用する前提にしない。human escalationは分布の不確実性だけでなく、specialist semantic decisionとrisk/evidence stateを利用する方向で設計する。

今回の文書は計画修正であり、Phase 5B実装、policy変更、モデル変更、Phase 6設計開始は行っていない。
