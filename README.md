# Knowledge-Guided Deep Learning for Infarct Prediction

This repository contains the implementation of the paper "Predicting Infarct Outcomes Following Extended Time Window Thrombectomy in Large Vessel Occlusion Using Knowledge-Guided Deep Learning" (currently under review).

## Overview

This project proposes a deep learning approach to predict final infarct outcomes after mechanical thrombectomy (MT) in acute ischemic stroke (AIS) patients treated beyond the conventional 6-hour treatment window. By incorporating prior medical knowledge into deep learning models, we significantly improve prediction accuracy compared to conventional threshold-based methods.

## Model Architecture

Five different model architectures were developed to evaluate the impact of different sources of prior knowledge:

1. **BaselineNet**: A standard 3D Swin Transformer-based model without any prior knowledge guidance.
2. **CollateralFlowNet**: Incorporates collateral circulation evaluation scores.
3. **InfarctProbabilityNet**: Integrates population-derived cerebral infarct probability maps.
4. **ArterialTerritoryNet**: Guided by cerebral artery territory mapping.
5. **UnifiedNet**: Combines all three sources of prior knowledge.

All models use a hybrid architecture with:
- Initial 3D convolutional layers for local feature extraction
- Swin Transformer backbone for capturing long-range dependencies
- U-Net-like decoder with skip connections
- Knowledge integration through dedicated auxiliary encoders and fusion modules

## Requirements

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

## Data Preparation

The model expects the following registered inputs:
- CT perfusion (CTP) parameter maps: CBF, CBV, MTT, and Tmax
- Diffusion-weighted imaging (DWI) for ground truth infarct segmentation
- Collateral circulation scores (for CollateralFlowNet and UnifiedNet)
- Infarct probability atlas (for InfarctProbabilityNet and UnifiedNet)
- Arterial territory atlas (for ArterialTerritoryNet and UnifiedNet)

Data should be organized as follows:
```
data_directory/
├── patient_1/
│   ├── ctp.nii.gz
│   ├── dwi.nii.gz
│   └── collateral_scores.csv
├── patient_2/
│   ├── ctp.nii.gz
│   ├── dwi.nii.gz
│   └── collateral_scores.csv
...
└── atlases/
    ├── probability_atlas.nii.gz
    └── territory_atlas.nii.gz
```