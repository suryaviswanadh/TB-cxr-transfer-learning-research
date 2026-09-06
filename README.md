# Robust and Explainable Tuberculosis Detection from Chest X-Ray Images Using Transfer Learning

This repository contains the reusable research codebase for the tuberculosis chest X-ray transfer-learning study. It is organized to support a clean, reproducible workflow for model loading, preprocessing, evaluation, Grad-CAM attribution, and a lightweight research application prototype.

## 1. Project purpose

The project focuses on tuberculosis-related chest X-ray classification using image preprocessing and transfer learning. The codebase is intentionally modular and architecture-agnostic so it can support multiple backbone models and future paper phases without creating competing training pipelines.

## 2. Paper 1 dataset

Paper 1 uses the Shenzhen Hospital CXR set and Montgomery County CXR set, combined into a total dataset of 800 images:

- 406 Normal
- 394 TB-consistent

The dataset has been independently verified on Kaggle and is used as the benchmark dataset for the Paper 1 comparison.

## 3. Paper 1 methodology

Paper 1 compares exactly these three architectures:

1. ResNet50 using torchvision.models.ResNet50_Weights.IMAGENET1K_V2
2. DenseNet121 using torchvision.models.DenseNet121_Weights.IMAGENET1K_V1
3. EfficientNet-B0 using torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1

No architecture is assumed to be the winner in this repository. The benchmark is executed separately on Kaggle, and the final winner is determined by the actual experimental result.

## 4. Preprocessing

Inference and validation preprocessing is defined as follows:

1. Read the image as grayscale
2. Resize to 224x224
3. Apply CLAHE with clipLimit = 2.0 and tileGridSize = (8, 8)
4. Convert to 3-channel RGB
5. Convert to tensor
6. Apply ImageNet normalization with:
   - mean = [0.485, 0.456, 0.406]
   - std = [0.229, 0.224, 0.225]

No random augmentation is used during inference or validation.

## 5. Augmentation

The Kaggle Paper 1 training protocol uses the following augmentation during training only:

- RandomRotation ±10 degrees
- RandomHorizontalFlip p = 0.5
- RandomAffine shear ±5 degrees

These values are not altered in this repository’s reusable preprocessing utilities.

## 6. Three model architectures

This repository supports the following architectures through src.models.create_model():

- resnet50
- densenet121
- efficientnet_b0

The final classifier is replaced with a two-class output layer:

- class 0 = Normal
- class 1 = TB-consistent

## 7. 5-fold validation

The Paper 1 benchmark uses stratified 5-fold cross-validation with random_state = 42.

This repository does not create a second full training pipeline. The actual training and fold evaluation for Paper 1 are performed separately on Kaggle.

## 8. Evaluation metrics

The reusable evaluation utilities in src.metrics.py support:

- ROC-AUC
- sensitivity
- specificity
- precision
- F1-score
- accuracy
- confusion matrix

The implementation handles zero-division safely and raises a clear error for ROC-AUC calculation when only one class is present.

## 9. Efficiency metrics

The Paper 1 benchmark includes efficiency measurements such as:

- trainable parameter count
- GFLOPs/MACs
- GPU inference latency
- checkpoint size

These are treated as benchmark outputs produced by the Kaggle experiment, not fabricated results in this repository.

## 10. Paper 2 planned methodology

Paper 2 begins only after the actual best architecture is selected from the Paper 1 Kaggle benchmark. The planned sequence is:

1. Best architecture from Paper 1
2. Independent external TB dataset
3. External baseline evaluation
4. Model improvement
5. Improved external evaluation
6. Baseline compared with improved model
7. Grad-CAM explainability
8. Prediction analysis

This repository remains architecture-agnostic and does not assume the winning Paper 1 model in advance.

## 11. Kaggle benchmark notebook and recorded output

The complete Kaggle benchmark notebook is available at `notebooks/paper1_kaggle_benchmark.ipynb`. It performs dataset auditing, stratified fold generation, model benchmarking, OOF evaluation, efficiency profiling, artifact export, and a formal OOF audit.

The attached notebook contains multiple benchmark runs. The later run uses the requested Paper 1 settings, including batch size 32 and the specified training augmentation. Its stored output reports:

- DenseNet121 global OOF ROC-AUC: 0.9586
- DenseNet121 five-fold mean ROC-AUC: 0.9615 +/- 0.0048
- EfficientNet-B0 global OOF ROC-AUC: 0.9422
- ResNet50 global OOF ROC-AUC: 0.9394

Under the notebook's primary selection rule, highest global OOF ROC-AUC, DenseNet121 is the empirical Paper 1 selection from that recorded run. The complete table and the earlier run distinction are documented in `results/paper1_kaggle_results.md`.

These are benchmark outputs from the stored Kaggle notebook, not external validation results and not clinical diagnostic evidence.

## 12. Grad-CAM

The reusable Grad-CAM implementation in src.gradcam.py provides class activation visualization for the supported architectures. It resolves the target layer programmatically, verifies that the activation tensor is spatially valid, registers hooks safely, captures activations and gradients, computes a Grad-CAM heatmap, normalizes it, and resizes it to the original image dimensions.

Grad-CAM is an exploratory model-attribution visualization and is not validated clinical ground truth or a lesion-localization method.

## 13. Installation

Create and activate a virtual environment, then install the project requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 14. Application usage

Launch the Gradio research prototype:

```bash
python app/app.py
```

Then open the generated local URL in a browser, upload a chest X-ray image, choose an architecture, select a checkpoint, and run prediction.

## 15. Checkpoint placement

Place trained model checkpoints in the checkpoints directory. The application checks for:

- raw state_dict
- dictionary containing state_dict or model_state_dict

If loading fails, the UI shows a clear error rather than generating a fake output.

## 16. Limitations

- This repository is not a substitute for the Kaggle Paper 1 benchmark training experiment.
- Paper 1 training is performed separately on Kaggle.
- No external validation results are claimed here.
- Grad-CAM outputs are exploratory and not clinical ground truth.
- This application is a research prototype and should not be used for clinical diagnosis.

## 17. Research-only disclaimer

This project is a research prototype only and is not for clinical diagnosis or patient clinical decision-making.

## Reproducibility notes

- Random seed is configurable in the YAML config file.
- Device selection uses CUDA when available, otherwise CPU.
- Preprocessing is deterministic for inference and validation.
- No undocumented transformations are added to the reusable preprocessing code.

## Dataset and training status

Paper 1 training is performed separately on Kaggle. This repository does not fabricate training results or claim external validation.
