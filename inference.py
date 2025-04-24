import os
import argparse
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm

from knowledge_guided_unet import create_knowledge_guided_unet_3d

def preprocess_ctp(ctp_data, target_shape=(128, 128, 128)):
    """
    Preprocess CTP parameter maps (CBF, CBV, MTT, Tmax).
    
    Args:
        ctp_data: CTP parameter maps with shape [4, D, H, W]
        target_shape: Target shape to resize to
        
    Returns:
        Preprocessed CTP data with shape [4, 128, 128, 128]
    """
    # Create a brain mask (for example, where any parameter map > 0)
    brain_mask = np.any(ctp_data > 0, axis=0)
    
    # Z-score normalization within the brain mask
    preprocessed_data = np.zeros_like(ctp_data, dtype=np.float32)
    for i in range(ctp_data.shape[0]):
        channel_data = ctp_data[i]
        mask_values = channel_data[brain_mask]
        if len(mask_values) > 0:
            mean = np.mean(mask_values)
            std = np.std(mask_values)
            if std > 0:
                channel_normalized = (channel_data - mean) / std
                preprocessed_data[i] = channel_normalized
    
    # Convert to tensor
    preprocessed_tensor = torch.from_numpy(preprocessed_data).float()
    
    # Resize to target shape if necessary
    if preprocessed_tensor.shape[1:] != target_shape:
        # Add batch dimension for resizing
        preprocessed_tensor = preprocessed_tensor.unsqueeze(0)
        
        # Resize using trilinear interpolation
        preprocessed_tensor = torch.nn.functional.interpolate(
            preprocessed_tensor,
            size=target_shape,
            mode='trilinear',
            align_corners=False
        )
        
        # Remove batch dimension
        preprocessed_tensor = preprocessed_tensor.squeeze(0)
    
    return preprocessed_tensor

def load_atlas(atlas_path, target_shape=(128, 128, 128), affine=None):
    """
    Load and preprocess an atlas (probability or territory).
    
    Args:
        atlas_path: Path to the atlas file
        target_shape: Target shape to resize to
        affine: Affine transformation matrix for the target space
        
    Returns:
        Preprocessed atlas with shape [1, 128, 128, 128]
    """
    # Load the atlas
    atlas_nii = nib.load(atlas_path)
    atlas_data = atlas_nii.get_fdata()
    
    # Add channel dimension if needed
    if atlas_data.ndim == 3:
        atlas_data = atlas_data[np.newaxis, ...]
    
    # Convert to tensor
    atlas_tensor = torch.from_numpy(atlas_data).float()
    
    # Resize to target shape if necessary
    if atlas_tensor.shape[1:] != target_shape:
        # Add batch dimension for resizing
        atlas_tensor = atlas_tensor.unsqueeze(0)
        
        # Resize using appropriate interpolation method
        # Use trilinear for probability atlas, nearest for territory atlas
        interpolation_mode = 'trilinear' if 'probability' in atlas_path else 'nearest'
        atlas_tensor = torch.nn.functional.interpolate(
            atlas_tensor,
            size=target_shape,
            mode=interpolation_mode,
            align_corners=False if interpolation_mode == 'trilinear' else None
        )
        
        # Remove batch dimension
        atlas_tensor = atlas_tensor.squeeze(0)
    
    return atlas_tensor

def save_prediction(prediction, output_path, reference_nii=None):
    """
    Save the prediction as a NIfTI file.
    
    Args:
        prediction: Tensor with shape [1, D, H, W]
        output_path: Path to save the prediction
        reference_nii: Reference NIfTI file to get header information
    """
    # Convert to binary mask if needed
    prediction_np = prediction.squeeze().cpu().numpy()
    
    # Create NIfTI image
    if reference_nii is not None:
        # Use the same header as the reference image
        prediction_nii = nib.Nifti1Image(prediction_np, reference_nii.affine, reference_nii.header)
    else:
        # Create a new header
        prediction_nii = nib.Nifti1Image(prediction_np, np.eye(4))
    
    # Save the prediction
    nib.save(prediction_nii, output_path)
    
    print(f"Prediction saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Infarct prediction inference")
    parser.add_argument("--model_type", type=str, default="UnifiedNet", 
                        choices=["BaselineNet", "CollateralFlowNet", 
                                "InfarctProbabilityNet", "ArterialTerritoryNet", "UnifiedNet"],
                        help="Type of model to use")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the trained model checkpoint")
    parser.add_argument("--ctp_path", type=str, required=True, 
                        help="Path to the CTP parameter maps (4D NIfTI with CBF, CBV, MTT, Tmax)")
    parser.add_argument("--collateral_scores", type=str, default=None,
                        help="Comma-separated collateral scores (artery,perfusion,vein) required for CollateralFlowNet and UnifiedNet")
    parser.add_argument("--probability_atlas", type=str, default=None,
                        help="Path to the infarct probability atlas (required for InfarctProbabilityNet and UnifiedNet)")
    parser.add_argument("--territory_atlas", type=str, default=None,
                        help="Path to the arterial territory atlas (required for ArterialTerritoryNet and UnifiedNet)")
    parser.add_argument("--output_path", type=str, required=True, 
                        help="Path to save the prediction")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for binary prediction (default: 0.5)")
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if required inputs are provided based on model type
    if args.model_type in ["CollateralFlowNet", "UnifiedNet"] and args.collateral_scores is None:
        raise ValueError(f"Collateral scores are required for {args.model_type}")
    if args.model_type in ["InfarctProbabilityNet", "UnifiedNet"] and args.probability_atlas is None:
        raise ValueError(f"Probability atlas is required for {args.model_type}")
    if args.model_type in ["ArterialTerritoryNet", "UnifiedNet"] and args.territory_atlas is None:
        raise ValueError(f"Territory atlas is required for {args.model_type}")
    
    # Load model
    model = create_knowledge_guided_unet_3d(
        model_type=args.model_type,
        n_classes=1  # Binary segmentation
    )
    
    # Load model weights
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    # Load and preprocess CTP data
    ctp_nii = nib.load(args.ctp_path)
    ctp_data = ctp_nii.get_fdata()
    
    # Ensure CTP data has the correct shape [4, D, H, W]
    if len(ctp_data.shape) == 4 and ctp_data.shape[0] == 4:
        pass  # Already in the correct format
    elif len(ctp_data.shape) == 4 and ctp_data.shape[-1] == 4:
        # Channels last, convert to channels first
        ctp_data = np.transpose(ctp_data, (3, 0, 1, 2))
    else:
        raise ValueError(f"Unexpected CTP data shape: {ctp_data.shape}. Expected [4, D, H, W] or [D, H, W, 4]")
    
    # Preprocess CTP data
    ctp_tensor = preprocess_ctp(ctp_data)
    
    # Prepare model inputs
    model_inputs = {"ctp_maps": ctp_tensor.unsqueeze(0).to(device)}
    
    # Load and preprocess auxiliary inputs if needed
    if args.model_type in ["CollateralFlowNet", "UnifiedNet"] and args.collateral_scores is not None:
        # Parse collateral scores
        scores = [float(s) for s in args.collateral_scores.split(',')]
        if len(scores) != 3:
            raise ValueError("Expected 3 collateral scores (artery,perfusion,vein)")
        model_inputs["collateral_scores"] = torch.tensor([scores], dtype=torch.float32).to(device)
    
    if args.model_type in ["InfarctProbabilityNet", "UnifiedNet"] and args.probability_atlas is not None:
        # Load and preprocess probability atlas
        probability_atlas = load_atlas(args.probability_atlas)
        model_inputs["probability_atlas"] = probability_atlas.unsqueeze(0).to(device)
    
    if args.model_type in ["ArterialTerritoryNet", "UnifiedNet"] and args.territory_atlas is not None:
        # Load and preprocess territory atlas
        territory_atlas = load_atlas(args.territory_atlas)
        model_inputs["territory_atlas"] = territory_atlas.unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(**model_inputs)
        prediction = torch.sigmoid(output) > args.threshold
    
    # Save prediction
    save_prediction(prediction, args.output_path, ctp_nii)
    
    # Additionally save probability map if desired
    probability_path = args.output_path.replace('.nii.gz', '_probability.nii.gz')
    save_prediction(torch.sigmoid(output), probability_path, ctp_nii)

if __name__ == "__main__":
    main() 