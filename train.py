import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import nibabel as nib
from tqdm import tqdm

from knowledge_guided_unet import create_knowledge_guided_unet_3d

# Constants
SPATIAL_SIZE = (128, 128, 128)
BATCH_SIZE = 4
INITIAL_LR = 1e-4
MIN_LR = 1e-6
WEIGHT_DECAY = 1e-2
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
SEED = 42

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class StrokeDataset(Dataset):
    """Dataset for stroke prediction from CT perfusion images."""
    def __init__(self, 
                 data_paths: List[Dict],
                 use_collateral_scores: bool = False,
                 use_probability_atlas: bool = False, 
                 use_territory_atlas: bool = False,
                 transform=None):
        """
        Args:
            data_paths: List of dictionaries containing paths to CTP data, DWI, etc.
            use_collateral_scores: Whether to use collateral scores
            use_probability_atlas: Whether to use infarct probability atlas
            use_territory_atlas: Whether to use arterial territory atlas
            transform: Optional transforms to apply to the data
        """
        self.data_paths = data_paths
        self.use_collateral_scores = use_collateral_scores
        self.use_probability_atlas = use_probability_atlas
        self.use_territory_atlas = use_territory_atlas
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.data_paths)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Load and preprocess a single data item.
        
        Returns a dict containing:
        - ctp_maps: Tensor of shape [4, D, H, W] containing CBF, CBV, MTT, Tmax
        - target: Binary segmentation mask from DWI
        - collateral_scores: Collateral scores (if use_collateral_scores is True)
        - probability_atlas: Infarct probability atlas (if use_probability_atlas is True)
        - territory_atlas: Arterial territory atlas (if use_territory_atlas is True)
        """
        # For demonstration purposes - in practice you would load actual data
        # This is a placeholder implementation
        item_path = self.data_paths[idx]
        
        # Load CTP maps (CBF, CBV, MTT, Tmax)
        ctp_data = self._load_and_preprocess_ctp(item_path["ctp_path"])
        
        # Load the ground truth infarct segmentation from DWI
        target = self._load_and_preprocess_target(item_path["dwi_path"])
        
        result = {
            "ctp_maps": ctp_data,
            "target": target
        }
        
        # Load collateral scores if needed
        if self.use_collateral_scores:
            collateral_scores = self._load_collateral_scores(item_path["collateral_path"])
            result["collateral_scores"] = collateral_scores
            
        # Load probability atlas if needed
        if self.use_probability_atlas:
            probability_atlas = self._load_probability_atlas(item_path["probability_atlas_path"])
            result["probability_atlas"] = probability_atlas
            
        # Load territory atlas if needed
        if self.use_territory_atlas:
            territory_atlas = self._load_territory_atlas(item_path["territory_atlas_path"])
            result["territory_atlas"] = territory_atlas
            
        # Apply any transformations
        if self.transform:
            result = self.transform(result)
            
        return result
    
    def _load_and_preprocess_ctp(self, path: str) -> torch.Tensor:
        """Load and preprocess CTP maps (CBF, CBV, MTT, Tmax)."""
        # In a real implementation, you would:
        # 1. Load the Nifti file with nibabel
        # 2. Normalize each parameter map using Z-scoring
        # 3. Resample to isotropic spacing
        # 4. Resize to the standard dimensions
        # For demonstration, we create a random tensor
        ctp_data = torch.randn(4, *SPATIAL_SIZE)
        return ctp_data
    
    def _load_and_preprocess_target(self, path: str) -> torch.Tensor:
        """Load and preprocess the target infarct segmentation from DWI."""
        # In a real implementation, you would load the segmentation mask
        # For demonstration, we create a random binary mask
        target = torch.zeros(1, *SPATIAL_SIZE)
        # Create some random "infarct" regions
        target[0, 40:60, 40:70, 40:70] = 1
        return target
    
    def _load_collateral_scores(self, path: str) -> torch.Tensor:
        """Load collateral scores for artery, perfusion, and vein dimensions."""
        # In a real implementation, you would load actual scores from a clinical database
        # For demonstration, we create random scores between 0 and 10
        scores = torch.tensor([[np.random.uniform(0, 10) for _ in range(3)]])
        return scores
    
    def _load_probability_atlas(self, path: str) -> torch.Tensor:
        """Load the probability atlas of infarct likelihood."""
        # In a real implementation, you would:
        # 1. Load the pre-computed atlas
        # 2. Align it to the patient's space
        # 3. Resample to match the input dimensions
        probability_atlas = torch.zeros(1, *SPATIAL_SIZE)
        # Create a gradient of probabilities for demonstration
        x, y, z = np.indices(SPATIAL_SIZE)
        center = np.array(SPATIAL_SIZE) // 2
        distances = np.sqrt(((x - center[0])/center[0])**2 + 
                           ((y - center[1])/center[1])**2 + 
                           ((z - center[2])/center[2])**2)
        # Higher probability near center, lower at edges
        probability_atlas[0] = torch.from_numpy(np.clip(1.0 - distances, 0, 1) * 0.8).float()
        return probability_atlas
    
    def _load_territory_atlas(self, path: str) -> torch.Tensor:
        """Load the arterial territory atlas."""
        # In a real implementation, you would:
        # 1. Load the pre-computed atlas
        # 2. Align it to the patient's space
        # 3. Resample to match the input dimensions
        territory_atlas = torch.zeros(1, *SPATIAL_SIZE)
        # Create some territory regions for demonstration
        # MCA territory (example)
        territory_atlas[0, 30:80, 20:100, 30:90] = 0.8
        # ACA territory (example)
        territory_atlas[0, 40:70, 20:60, 90:110] = 0.6
        # PCA territory (example)
        territory_atlas[0, 50:90, 20:60, 20:50] = 0.4
        return territory_atlas

class SpatialAugmentation:
    """Class for 3D spatial augmentations."""
    def __init__(self, 
                 rotation_range=10, 
                 scale_range=(0.9, 1.1),
                 flip_prob=0.5):
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.flip_prob = flip_prob
        
    def __call__(self, sample: Dict) -> Dict:
        # Random rotation
        angle = np.random.uniform(-self.rotation_range, self.rotation_range, 3)
        angle_rad = np.deg2rad(angle)
        
        # Random scaling
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        
        # Apply transformations to all 3D volumes
        for key in sample:
            if isinstance(sample[key], torch.Tensor) and len(sample[key].shape) >= 4:
                # Skip non-volume data like collateral scores
                sample[key] = self._rotate_3d(sample[key], angle_rad)
                sample[key] = self._scale_3d(sample[key], scale)
                
                # Random flip along sagittal axis (left-right)
                if np.random.random() < self.flip_prob:
                    sample[key] = torch.flip(sample[key], [3])  # Flip along W dimension
                    
        # Random intensity shifts for CTP maps only
        if "ctp_maps" in sample:
            # Apply random intensity shifts to each CTP parameter map
            shift = torch.randn(4, 1, 1, 1) * 0.1  # 10% standard deviation
            sample["ctp_maps"] = sample["ctp_maps"] + shift
            
        return sample
    
    def _rotate_3d(self, volume: torch.Tensor, angles: np.ndarray) -> torch.Tensor:
        """Apply 3D rotation to volume."""
        # In practice, you would use torch's grid_sample or similar
        # For demonstration, we just return the original volume
        return volume
    
    def _scale_3d(self, volume: torch.Tensor, scale: float) -> torch.Tensor:
        """Apply 3D scaling to volume."""
        # In practice, you would use torch's grid_sample or similar
        # For demonstration, we just return the original volume
        return volume

class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks."""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        
        # Flatten the predictions and targets
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Calculate Dice coefficient
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

class CombinedLoss(nn.Module):
    """Combined Dice and BCE loss."""
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        return self.dice_weight * dice + self.bce_weight * bce

def calculate_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Calculate Dice score and IoU for evaluation."""
    pred_bin = (torch.sigmoid(pred) > 0.5).float()
    pred_flat = pred_bin.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum().item()
    pred_sum = pred_flat.sum().item()
    target_sum = target_flat.sum().item()
    union = pred_sum + target_sum - intersection
    dice = (2.0 * intersection) / (pred_sum + target_sum + 1e-8)
    iou = intersection / (union + 1e-8)
    return {"dice": dice, "iou": iou}

def train_epoch(model: nn.Module, 
                dataloader: DataLoader, 
                optimizer: torch.optim.Optimizer, 
                criterion: nn.Module, 
                device: torch.device) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    metrics = {"dice": 0.0, "iou": 0.0}
    
    for batch in tqdm(dataloader, desc="Training"):
        # Extract data
        ctp_maps = batch["ctp_maps"].to(device)
        target = batch["target"].to(device)
        
        # Prepare inputs based on model type
        forward_args = {"ctp_maps": ctp_maps}
        
        if "collateral_scores" in batch:
            forward_args["collateral_scores"] = batch["collateral_scores"].to(device)
            
        if "probability_atlas" in batch:
            forward_args["probability_atlas"] = batch["probability_atlas"].to(device)
            
        if "territory_atlas" in batch:
            forward_args["territory_atlas"] = batch["territory_atlas"].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(**forward_args)
        
        # Calculate loss
        loss = criterion(output, target)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        batch_metrics = calculate_metrics(output.detach(), target)
        metrics["dice"] += batch_metrics["dice"]
        metrics["iou"] += batch_metrics["iou"]
    
    # Calculate average metrics
    num_batches = len(dataloader)
    return {
        "loss": running_loss / num_batches,
        "dice": metrics["dice"] / num_batches,
        "iou": metrics["iou"] / num_batches
    }

def validate(model: nn.Module, 
             dataloader: DataLoader, 
             criterion: nn.Module, 
             device: torch.device) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    metrics = {"dice": 0.0, "iou": 0.0}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # Extract data
            ctp_maps = batch["ctp_maps"].to(device)
            target = batch["target"].to(device)
            
            # Prepare inputs based on model type
            forward_args = {"ctp_maps": ctp_maps}
            
            if "collateral_scores" in batch:
                forward_args["collateral_scores"] = batch["collateral_scores"].to(device)
                
            if "probability_atlas" in batch:
                forward_args["probability_atlas"] = batch["probability_atlas"].to(device)
                
            if "territory_atlas" in batch:
                forward_args["territory_atlas"] = batch["territory_atlas"].to(device)
            
            # Forward pass
            output = model(**forward_args)
            
            # Calculate loss
            loss = criterion(output, target)
            
            # Track metrics
            running_loss += loss.item()
            batch_metrics = calculate_metrics(output, target)
            metrics["dice"] += batch_metrics["dice"]
            metrics["iou"] += batch_metrics["iou"]
    
    # Calculate average metrics
    num_batches = len(dataloader)
    return {
        "loss": running_loss / num_batches,
        "dice": metrics["dice"] / num_batches,
        "iou": metrics["iou"] / num_batches
    }

def test(model: nn.Module, 
         dataloader: DataLoader, 
         device: torch.device) -> Dict[str, float]:
    """Test the model on the test set."""
    model.eval()
    metrics = {"dice": 0.0, "iou": 0.0}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            # Extract data
            ctp_maps = batch["ctp_maps"].to(device)
            target = batch["target"].to(device)
            
            # Prepare inputs based on model type
            forward_args = {"ctp_maps": ctp_maps}
            
            if "collateral_scores" in batch:
                forward_args["collateral_scores"] = batch["collateral_scores"].to(device)
                
            if "probability_atlas" in batch:
                forward_args["probability_atlas"] = batch["probability_atlas"].to(device)
                
            if "territory_atlas" in batch:
                forward_args["territory_atlas"] = batch["territory_atlas"].to(device)
            
            # Forward pass
            output = model(**forward_args)
            
            # Track metrics
            batch_metrics = calculate_metrics(output, target)
            metrics["dice"] += batch_metrics["dice"]
            metrics["iou"] += batch_metrics["iou"]
    
    # Calculate average metrics
    num_batches = len(dataloader)
    return {
        "dice": metrics["dice"] / num_batches,
        "iou": metrics["iou"] / num_batches
    }

def main():
    parser = argparse.ArgumentParser(description="Train infarct prediction model")
    parser.add_argument("--model_type", type=str, default="UnifiedNet", 
                        choices=["BaselineNet", "CollateralFlowNet", 
                                "InfarctProbabilityNet", "ArterialTerritoryNet", "UnifiedNet"],
                        help="Type of model to train")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Directory containing the dataset")
    parser.add_argument("--output_dir", type=str, default="./output", 
                        help="Directory to save model checkpoints")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, 
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=MAX_EPOCHS, 
                        help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=INITIAL_LR, 
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY, 
                        help="Weight decay for optimizer")
    parser.add_argument("--seed", type=int, default=SEED, 
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which prior knowledge sources to use based on model type
    use_collateral_scores = args.model_type in ["CollateralFlowNet", "UnifiedNet"]
    use_probability_atlas = args.model_type in ["InfarctProbabilityNet", "UnifiedNet"]
    use_territory_atlas = args.model_type in ["ArterialTerritoryNet", "UnifiedNet"]
    
    # Create dummy data paths for demonstration
    # In practice, you would load actual file paths from disk
    data_paths = []
    for i in range(221): 
        data_paths.append({
            "ctp_path": f"{args.data_dir}/patient_{i}/ctp.nii.gz",
            "dwi_path": f"{args.data_dir}/patient_{i}/dwi.nii.gz",
            "collateral_path": f"{args.data_dir}/patient_{i}/collateral_scores.csv",
            "probability_atlas_path": f"{args.data_dir}/atlases/probability_atlas.nii.gz",
            "territory_atlas_path": f"{args.data_dir}/atlases/territory_atlas.nii.gz"
        })
    
    # Split data
    train_paths, test_paths = train_test_split(
        data_paths, test_size=0.3, random_state=args.seed
    )
    val_paths, test_paths = train_test_split(
        test_paths, test_size=0.5, random_state=args.seed
    )
    
    # Data augmentation for training
    train_transform = SpatialAugmentation()
    
    # Create datasets
    train_dataset = StrokeDataset(
        train_paths, 
        use_collateral_scores=use_collateral_scores,
        use_probability_atlas=use_probability_atlas,
        use_territory_atlas=use_territory_atlas,
        transform=train_transform
    )
    
    val_dataset = StrokeDataset(
        val_paths, 
        use_collateral_scores=use_collateral_scores,
        use_probability_atlas=use_probability_atlas,
        use_territory_atlas=use_territory_atlas
    )
    
    test_dataset = StrokeDataset(
        test_paths, 
        use_collateral_scores=use_collateral_scores,
        use_probability_atlas=use_probability_atlas,
        use_territory_atlas=use_territory_atlas
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = create_knowledge_guided_unet_3d(
        model_type=args.model_type,
        n_classes=1  # Binary segmentation
    )
    model = model.to(device)
    
    # Define loss function
    criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
    
    # Define optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Define learning rate scheduler - cosine annealing with warmup
    def lr_lambda(epoch):
        if epoch < 10:  # 10 epochs of warmup
            return (epoch + 1) * (args.lr - MIN_LR) / 10 / args.lr + MIN_LR / args.lr
        else:
            # Cosine annealing after warmup
            return MIN_LR / args.lr + 0.5 * (1 - MIN_LR / args.lr) * (
                1 + np.cos(np.pi * (epoch - 10) / (args.num_epochs - 10))
            )
            
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    best_val_dice = 0.0
    patience_counter = 0
    
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}, IoU: {train_metrics['iou']:.4f}")
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        print(f"Val - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}")
        
        # Update learning rate
        scheduler.step()
        
        # Check for improvement
        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': best_val_dice,
            }, f"{args.output_dir}/{args.model_type}_best.pth")
            
            print(f"Saved new best model with Dice score: {best_val_dice:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
            
        # Check early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    # Load best model for testing (map_location for CPU/GPU portability)
    checkpoint = torch.load(
        f"{args.output_dir}/{args.model_type}_best.pth",
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Test
    test_metrics = test(model, test_loader, device)
    print(f"Test - Dice: {test_metrics['dice']:.4f}, IoU: {test_metrics['iou']:.4f}")
    
    # Save test results
    with open(f"{args.output_dir}/{args.model_type}_results.txt", "w") as f:
        f.write(f"Model: {args.model_type}\n")
        f.write(f"Test Dice: {test_metrics['dice']:.4f}\n")
        f.write(f"Test IoU: {test_metrics['iou']:.4f}\n")

if __name__ == "__main__":
    main() 
