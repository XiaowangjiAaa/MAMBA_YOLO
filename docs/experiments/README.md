# Crack-Aware Experiment Index

Read the documents chronologically:

1. [Initial Scan/Memory design](README_CRACK_STRUCTURE_EXPERIMENTS.md)
2. [8.9 results and next experiments](8.9_CRACK_STRUCTURE_RESULTS_AND_NEXT_EXPERIMENTS.md)
3. [8.12 unified redesign](8.12_CRACK_MAMBA_UNIFIED_REDESIGN.md)
4. [8.17 fixed method and tuning](8.17_FIXED_CRACK_METHOD_AND_TUNING.md)
5. [8.19 causal controls](8.19_CRACK_UNIFIED_CAUSAL_CONTROL_AND_OPTIMIZATION.md)
6. [8.23 Q00 and component ablations](8.23_P5_DIRECTIONAL_UNIFIED_AND_COMPONENT_ABLATION.md)
7. [8.24 corrected H/V mapping and stability experiments](8.24_Q00_CORRECTED_HV_AND_STABILITY_EXPERIMENTS.md)
8. [8.26 true YOLO11 efficient crack-aligned state propagation](8.26_EFFICIENT_YOLO11_CRACK_ALIGNED_STATE_PROPAGATION.md)
9. [8.27 Mask-mAP50-driven full-CASP optimization](8.27_MASK_MAP50_DRIVEN_FULL_CASP_OPTIMIZATION.md)
10. [8.28 Sparse adaptive crack-path Mamba](8.28_SPARSE_ADAPTIVE_CRACK_PATH_MAMBA.md)

Current status: 8.27 showed that probability guidance can help but fixed-family direction weighting does not realize the original crack-aligned scan objective. The 8.28 Y-series replaces it with sparse, image-adaptive curve tracing and path-level Mamba state propagation. All Y-series configs tune the complete method with Mask mAP50 as the primary metric.
