# Paper 1 Kaggle Benchmark Results

This report records the outputs already stored in `notebooks/paper1_kaggle_benchmark.ipynb`. The notebook was executed on a Kaggle Tesla T4 and contains more than one benchmark run.

## Dataset audit

The dataset audit output reports:

- 800 CXR images found
- 406 Normal images
- 394 TB-consistent images
- Shenzhen: 326 Normal and 336 TB-consistent
- Montgomery: 80 Normal and 58 TB-consistent
- Five stratified folds with `random_state=42`
- Fold class counts: fold 0 = 82/78; folds 1-4 = 81/79
- Metadata saved as `/kaggle/working/tb_metadata.csv`

## Final protocol-compliant benchmark output

The later benchmark run in notebook cell 5 uses the requested Paper 1 protocol:

- Architectures: ResNet50, DenseNet121, EfficientNet-B0
- Batch size: 32
- Maximum epochs: 30
- Early stopping patience: 7
- AdamW learning rate: `1e-4`
- AdamW weight decay: `1e-2`
- Training augmentation: rotation +/-10 degrees, horizontal flip `p=0.5`, affine shear +/-5 degrees
- Validation preprocessing: deterministic CLAHE and ImageNet normalization
- Device: Kaggle Tesla T4

Recorded summary:

| Architecture | Params (M) | GFLOPs | Disk (MB) | Latency (ms) | Global OOF ROC-AUC | Five-fold ROC-AUC mean +/- SD | Sensitivity mean +/- SD | Specificity mean +/- SD | F1 mean +/- SD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DenseNet121 | 6.96 | 5.80 | 27.1 | 14.21 | 0.9586 | 0.9615 +/- 0.0048 | 0.8731 +/- 0.0277 | 0.9262 +/- 0.0317 | 0.8958 +/- 0.0159 |
| EfficientNet-B0 | 4.01 | 0.82 | 15.6 | 7.86 | 0.9422 | 0.9477 +/- 0.0105 | 0.8757 +/- 0.0267 | 0.9040 +/- 0.0377 | 0.8870 +/- 0.0221 |
| ResNet50 | 23.51 | 8.26 | 90.0 | 5.39 | 0.9394 | 0.9472 +/- 0.0062 | 0.9037 +/- 0.0456 | 0.8281 +/- 0.1092 | 0.8698 +/- 0.0245 |

### Recorded Paper 1 selection

Under the notebook's stated primary selection rule, highest global OOF ROC-AUC, DenseNet121 is the empirical Paper 1 selection in the later protocol-compliant run:

- Global OOF ROC-AUC: `0.9586`
- Five-fold mean ROC-AUC: `0.9615 +/- 0.0048`

This is a research benchmark result, not a clinical validation result. It must not be described as a clinical diagnosis capability.

## Earlier run in the notebook

The earlier benchmark run in notebook cell 4 used different settings, including batch size 16 and a different augmentation definition. It recorded DenseNet121 global OOF ROC-AUC `0.9566` and five-fold mean `0.9586 +/- 0.0086`.

That earlier result is retained as notebook history but is not the final protocol-compliant Paper 1 result.

## Generated Kaggle artifacts

The notebook records these outputs under `/kaggle/working/`:

- `tb_metadata.csv`
- `paper1_rigorous_benchmark.csv`
- `pareto_frontier_paper1.png`
- `oof_predictions_resnet50.csv`
- `oof_predictions_densenet121.csv`
- `oof_predictions_efficientnet_b0.csv`
- `paper1_global_oof_audit.csv`
- `paper1_cohort_oof_audit.csv`
- `paper1_fold_auc_audit.csv`
- `paper1_fold_summary_audit.csv`
- Audited OOF CSV files for all three architectures
- Five fold checkpoint files for each architecture

The model checkpoints and generated CSV/PNG artifacts are intentionally not committed to GitHub by default. Download the selected DenseNet121 checkpoint from Kaggle and place it in `checkpoints/` for local application use.

## Reproducibility note

The result table above is transcribed from the stored notebook output. If the notebook is rerun, the newly generated Kaggle artifacts should be reviewed and this report updated from the actual output rather than copied forward automatically.
