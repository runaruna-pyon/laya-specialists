# Phase 5A final decision record

記録日: 2026-09-26
状態: **CLOSED — FAIL**

## 決定

- **PHASE 5A COMPLETE — FAIL**
- Operational: **PASS**
- Policy Performance: **FAIL**
- Generic Laya Skill v1では、freeze済みselective policyをautonomous execution gateとして採用しない。
- Phase 5Aのthreshold retuning、label-specific / stage-specific threshold、新signal、signal combination等の再最適化は行わない。
- TUNE、EVALUATION v5、model、calibration、policy configの結果・内容を変更しない。EVALUATION v5は正式評価を完了しており、再実行しない。

## TUNE記録

TUNEは1,200件。Laya単体は1,020/1,200 correct（85.00%）、wrong 180件だった。事前登録済みselection ruleによって、Gateを満たすcandidateがない状態からfallbackとして次が選ばれ、freezeされた。

- Policy: **B — Concentration**
- Thresholds: `choice:2 = 0.90`、`choice:3-5 = 0.90`
- CLEAR 926、REVIEW 274
- Coverage: 77.17%
- Selective accuracy: 90.39%
- False autonomous rate: 7.42%
- Error capture rate: 50.56%
- TUNE Gate: **FAIL**

TUNEでの値は選択記録であり、held-out performanceの代わりにはしない。

## Fresh held-out EVALUATION v5

1,200件を一度だけ実行し、全件inference・保存・集計が正常終了した。実行とfrozen inputのSHA確認は成功したためOperationalはPASS。performance metricsは次のとおり。

- Laya accuracy: 939/1,200 = **78.25%**（wrong 261）
- CLEAR: **829**（correct_CLEAR 733、wrong_CLEAR 96）
- REVIEW_REQUIRED: **371**（correct_REVIEW 206、wrong_REVIEW 165）
- Coverage: **69.08%**
- Selective accuracy: **88.42%**
- False autonomous rate: **8.00%**
- Error capture rate: **63.22%**
- Unnecessary escalation rate: **21.94%**

事前登録Gateの結果:

| 条件 | 結果 |
|---|---|
| Selective accuracy ≥97% | FAIL |
| False autonomous rate ≤2% | FAIL |
| Error capture rate ≥80% | FAIL |
| Overall coverage ≥60% | PASS |
| 各final label coverage ≥40% | PASS |
| 退化解の禁止（minimum coverageを満たす） | PASS |
| Execution errorなし | PASS |

したがって **Operational PASS / Policy Performance FAIL** と判定する。v5はconsumedとし、resume・再評価・threshold変更は行わない。

## Class-dependentな記述結果

結果はlabel間で一様ではなかった。

- `needs_other_environment`: selective accuracy 100%、error capture 100%
- `insufficient_information`: selective accuracy 97.30%、error capture 95.38%
- `needs_external_access`: selective accuracy 74.68%、error capture 34.43%
- 参考: `needs_local_setup`のselective accuracyは73.00%、error captureは57.81%。

これらはこのfrozen model・taxonomy・generic concentration policyと本EVALUATIONに限った記述結果である。これを使ってlabel-specific policyをfitしない。

## 科学的解釈と限界

calibrated concentrationは、この評価でrisk signalとして一定の情報を持った。しかし、現在のgeneric taxonomy/model/policyでは、事前登録したgeneric correctness-detector acceptance levelに達しなかった。CLEAR subsetには高いconcentrationでも誤ったdecisionが多数残った。uncertaintyとcorrectnessの関係もfinal label間で均一ではなかった。

probability、entropy、concentrationはmodel distributionの要約であり、correctness probabilityではない。今回の結果から「Layaのconfidenceは役に立たない」「uncertainty estimationは不可能」「specialist policyでも失敗する」とは結論しない。専門タスク向けpolicyや別taxonomyの成否も評価していない。

## 次工程

Phase 5Bへ進むが目的を**ADVISORY GENERIC SKILL INFRASTRUCTURE**に限定する。failed policyを自律実行の許可に使わず、Phase 5A再最適化も行わない。実装開始は別の明示的作業指示後とする。

Phase 5B/5Cの変更後計画は `results/phase5_plan_amendment_after_5A.md` に記録する。元のpreregistration、policy config、evaluation artifactsはそのまま保持する。

## 参照成果物

- TUNE metrics: `results/phase5_policy_tune_report.txt`。TUNE dataset SHA-256: `4fc261f53dbe2643501e1517837bbd3f71cb7586c13bc9e46c05e6fd6aeba6db`。
- 正式held-out metricsとGate: `results/phase5_policy_eval_v5_report.json` / `.txt`。EVALUATION v5 SHA-256: `b5bf02c79d7346869105bffe1d5160bd59923ce6908c7aa5b81853732d666616`。
- 保存済みprediction records: `results/phase5_policy_eval_v5_raw.jsonl`（1,200 case prediction records）。
- Freeze provenance: `results/phase5_policy_eval_v5_execution_freeze.md`。
- Model SHA-256: `761ad958e6879c73cb6bf5ba9529b68aa8eb6a6eaf23eaa18f602a4e4ee38ff6`。
- Calibration config SHA-256: `cca0607acf2249caea4ad520830930be824099a8d8d79e507c35d4978d38481e`。
- Frozen policy config SHA-256: `16fc7a19c2df48a48a6dd3fe367be8443a567f7561a84d9e90730f5cef7f0b06`。
