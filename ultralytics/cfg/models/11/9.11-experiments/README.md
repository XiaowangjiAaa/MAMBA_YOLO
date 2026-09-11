# 9.11 CPSB three-part ablation configurations

Only new configurations live here. Frozen 9.3 memory/family YAMLs and frozen
9.1 placement YAMLs are reused through aliases in `batch_train.py`, preventing
an identical experiment from being silently redefined.

- `K02`-`K06`: equal-maximum-budget scan-order comparison.
- `K20`, `K21`, `K23`, `K24`, `K26`: missing placement controls.
- `K28/K29`: native YOLOv9c-Seg pair.
- `K38/K39`: official-architecture YOLO12n-Seg pair backported to this fork.
- Other K identifiers are documented aliases to existing frozen YAMLs.
