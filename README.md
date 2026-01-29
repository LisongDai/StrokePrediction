# Knowledge-Guided Deep Learning for Infarct Prediction

Implementation for the paper **"Predicting Infarct Outcomes Following Extended Time Window Thrombectomy in Large Vessel Occlusion Using Knowledge-Guided Deep Learning"** (JNIS, under review).

---

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Inference](#inference)
- [Model Variants & Inputs](#model-variants--inputs)
- [Citation](#citation)

---

## Overview

This project predicts **final infarct outcomes** after mechanical thrombectomy (MT) in acute ischemic stroke (AIS) patients treated **beyond the conventional 6-hour window**. It combines:

- **CT perfusion (CTP)** imaging (CBF, CBV, MTT, Tmax) as the primary input
- **Prior medical knowledge** integrated via a dedicated auxiliary pathway:
  - **Collateral circulation** scores (artery, perfusion, vein)
  - **Population-derived infarct probability** atlas
  - **Cerebral artery territory** mapping

Models are implemented as 3D U-Net–style architectures with a Swin Transformer backbone and learnable fusion of CTP-derived and prior-knowledge features at the bottleneck.

---

## Model Architecture

All variants share:

- **Primary pathway:** CTP maps → initial 3D conv → **Swin Transformer 3D** backbone → multi-scale features
- **Auxiliary pathway (when enabled):** Collateral scores + probability atlas + territory atlas → **ResNet3D** → bottleneck features
- **Fusion:** Learnable gated fusion (primary + α × auxiliary) at the bottleneck
- **Decoder:** U-Net-like decoder with skip connections from the primary encoder
- **Output:** Optional post-hoc modulation by the infarct probability atlas (Bayesian-style prior)

| Model | Collateral | Probability atlas | Territory atlas |
|-------|------------|-------------------|-----------------|
| **BaselineNet** | — | — | — |
| **CollateralFlowNet** | ✓ | — | — |
| **InfarctProbabilityNet** | — | ✓ | — |
| **ArterialTerritoryNet** | — | — | ✓ |
| **UnifiedNet** | ✓ | ✓ | ✓ |

---

## Project Structure

```
StrokePrediction-main/
├── README.md                 # This file
├── knowledge_guided_unet.py  # Model definitions (KnowledgeGuidedUNet3D, factory, building blocks)
├── train.py                  # Training script (dataset, augmentation, training loop)
├── inference.py              # Inference script (preprocess, load model, save NIfTI)
└── Predicting infarct outcomes ... .pdf   # Paper (if included)
```

- **`knowledge_guided_unet.py`**: Implements `KnowledgeGuidedUNet3D`, `create_knowledge_guided_unet_3d()`, and all building blocks (ConvBnRelu3d, DecoderBlock3d, ResNet3D, FusionModule). Expects an external **SwinTransformer3D** module (see [Requirements](#requirements)).
- **`train.py`**: Dataset (`StrokeDataset`), 3D augmentation, Dice/BCE loss, training/validation/test loops, checkpointing, early stopping.
- **`inference.py`**: Loads a checkpoint, preprocesses CTP (and optional atlases/collateral), runs inference, saves binary mask and probability map as NIfTI.

---

## Requirements

- **Python** 3.8+
- **PyTorch** ≥ 1.12.0 (with CUDA if training on GPU)
- **SwinTransformer3D**: The code imports `SwinTransformer3D` from a module named `swin_transformer_3d`. That module is **not** included in this repo. You must provide a compatible 3D Swin implementation (e.g. from [MONAI](https://github.com/Project-MONAI/MONAI) Swin UNETR or a custom 3D Swin) that:
  - Accepts input shape `[B, C, D, H, W]`
  - Returns a **list** of multi-scale feature tensors, each `[B, C, D, H, W]`
  - Exposes a `feature_info` attribute (list of dicts with `"num_chs"` per stage) for decoder skip connections

Other Python dependencies:

```
torch>=1.12.0
torchvision>=0.13.0
numpy>=1.20.0
scikit-learn>=1.0.0
nibabel>=3.2.0
tqdm>=4.62.0
matplotlib>=3.5.0
pandas>=1.3.0
h5py>=3.6.0
SimpleITK>=2.1.0
```

---

## Installation

1. Clone the repository and enter the project directory.
2. Create a virtual environment (recommended) and install dependencies:

   ```bash
   pip install torch torchvision numpy scikit-learn nibabel tqdm matplotlib pandas h5py SimpleITK
   ```

3. Add or implement a `swin_transformer_3d` module so that `from swin_transformer_3d import SwinTransformer3D` works (see [Requirements](#requirements)).

---

## Data Preparation

### Expected inputs

- **CTP parameter maps**: 4 channels — CBF, CBV, MTT, Tmax (NIfTI, shape `[4, D, H, W]` or `[D, H, W, 4]`).
- **DWI** (training only): Ground-truth infarct segmentation (binary mask).
- **Collateral scores** (CollateralFlowNet / UnifiedNet): Three values — artery, perfusion, vein (e.g. 0–10 scale). One row per subject (e.g. CSV with columns `artery, perfusion, vein`).
- **Infarct probability atlas** (InfarctProbabilityNet / UnifiedNet): Single 3D volume (NIfTI), population-derived infarct probability per voxel.
- **Arterial territory atlas** (ArterialTerritoryNet / UnifiedNet): Single 3D volume (NIfTI), e.g. integer labels per territory; will be resampled with nearest-neighbor.

### Directory layout

Data should be organized as follows (paths are examples; the training script builds path lists from `--data_dir`):

```
data_directory/
├── patient_0/
│   ├── ctp.nii.gz
│   ├── dwi.nii.gz
│   └── collateral_scores.csv
├── patient_1/
│   ├── ctp.nii.gz
│   ├── dwi.nii.gz
│   └── collateral_scores.csv
├── ...
└── atlases/
    ├── probability_atlas.nii.gz
    └── territory_atlas.nii.gz
```

- **CTP**: 4D NIfTI with channels (CBF, CBV, MTT, Tmax). The training script assumes a fixed spatial size (e.g. 128×128×128); preprocess (resample, skull-strip, normalize) as needed.
- **collateral_scores.csv**: At least three numeric columns (e.g. `artery`, `perfusion`, `vein`) or a single row of three comma-separated values per subject.
- **Atlases**: Align to the same space as CTP; the code will interpolate to the model’s spatial size.

---

## Training

**Usage:**

```bash
python train.py --data_dir /path/to/data_directory --output_dir ./output [OPTIONS]
```

**Main arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_type` | `UnifiedNet` | One of: `BaselineNet`, `CollateralFlowNet`, `InfarctProbabilityNet`, `ArterialTerritoryNet`, `UnifiedNet` |
| `--data_dir` | (required) | Root directory containing patient folders and `atlases/` |
| `--output_dir` | `./output` | Where to save checkpoints and results |
| `--batch_size` | 4 | Batch size |
| `--num_epochs` | 100 | Max training epochs |
| `--lr` | 1e-4 | Initial learning rate |
| `--weight_decay` | 1e-2 | AdamW weight decay |
| `--seed` | 42 | Random seed |

**Outputs:**

- Best model (by validation Dice): `{output_dir}/{model_type}_best.pth`
- Training/validation metrics printed per epoch; early stopping after 15 epochs without improvement (configurable in script).
- Final test metrics and a results file: `{output_dir}/{model_type}_results.txt`

**Example:**

```bash
python train.py --data_dir ./data --output_dir ./runs --model_type UnifiedNet --batch_size 4 --num_epochs 80
```

---

## Inference

**Usage:**

```bash
python inference.py --model_path /path/to/checkpoint.pth --ctp_path /path/to/ctp.nii.gz --output_path /path/to/pred.nii.gz [OPTIONS]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `--model_type` | No (default: `UnifiedNet`) | Must match the trained model |
| `--model_path` | Yes | Path to `.pth` checkpoint |
| `--ctp_path` | Yes | 4D CTP NIfTI (CBF, CBV, MTT, Tmax) |
| `--output_path` | Yes | Output NIfTI path (binary mask) |
| `--collateral_scores` | For CollateralFlowNet/UnifiedNet | Comma-separated: `artery,perfusion,vein` |
| `--probability_atlas` | For InfarctProbabilityNet/UnifiedNet | Path to probability atlas NIfTI |
| `--territory_atlas` | For ArterialTerritoryNet/UnifiedNet | Path to territory atlas NIfTI |
| `--threshold` | No (default: 0.5) | Binary threshold on sigmoid output |

**Outputs:**

- Binary segmentation: `--output_path` (e.g. `pred.nii.gz`)
- Probability map: same path with `_probability` inserted before `.nii.gz` (e.g. `pred_probability.nii.gz`)

**Examples:**

```bash
# BaselineNet (CTP only)
python inference.py --model_path ./output/BaselineNet_best.pth --ctp_path ./subject/ctp.nii.gz --output_path ./subject/pred.nii.gz --model_type BaselineNet

# UnifiedNet (all inputs)
python inference.py --model_path ./output/UnifiedNet_best.pth --ctp_path ./subject/ctp.nii.gz --output_path ./subject/pred.nii.gz \
  --collateral_scores 6,5,4 \
  --probability_atlas ./atlases/probability_atlas.nii.gz \
  --territory_atlas ./atlases/territory_atlas.nii.gz
```

---

## Model Variants & Inputs

| Model | CTP | Collateral (3 values) | Probability atlas | Territory atlas |
|-------|-----|------------------------|-------------------|-----------------|
| BaselineNet | ✓ | — | — | — |
| CollateralFlowNet | ✓ | ✓ | — | — |
| InfarctProbabilityNet | ✓ | — | ✓ | — |
| ArterialTerritoryNet | ✓ | — | — | ✓ |
| UnifiedNet | ✓ | ✓ | ✓ | ✓ |

Collateral dimensions are **artery**, **perfusion**, and **vein** (e.g. ASITN/SIR-style grading). The model normalizes and aggregates them with learnable weights.

---

## Citation

If you use this code or the method, please cite:

```bibtex
@article{dai2025predicting,
  title={Predicting infarct outcomes after extended time window thrombectomy in large vessel occlusion using knowledge guided deep learning},
  author={Dai, Lisong and Yuan, Lei and Zhang, Houwang and Sun, Zheng and Jiang, Jingxuan and Li, Zhaohui and Li, Yuehua and Zha, Yunfei},
  journal={Journal of NeuroInterventional Surgery},
  year={2025},
  publisher={British Medical Journal Publishing Group}
}
```

---

## License

See the repository for license information.
