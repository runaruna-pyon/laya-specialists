# C1 synthetic stress comparison

C1 was compared with C0 on one fresh, contract-derived synthetic stress probe containing 320 cases and 160 matched evidence-flip pairs.

| Metric | C0 | C1 |
|---|---:|---:|
| Primary accuracy | 0.366 | 0.872 |
| Required-flag exact-set accuracy | 0.653 | 0.925 |
| Matched pairs with both cases correct | 0/160 | 119/160 |

This is a descriptive synthetic-probe result only. It does not establish real-world HPCV competence, probability calibration, or a safe confidence threshold. C1 retained high-confidence errors: 41 wrong primary predictions in the probe had top-1 output at or above 0.90. Therefore confidence is not a correctness probability and must not be used to bypass review. The frozen service does not use confidence to route and never authorizes autonomous execution.

Raw probe cases, consumed stress probes, datasets, predictions, and checkpoint files are not part of this report or repository.
