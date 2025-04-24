import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Dict, Optional, Tuple, Union
from swin_transformer_3d import SwinTransformer3D

DEFAULT_INPUT_SIZE_3D = (128, 128, 128)
DEFAULT_CTP_CHANNELS = 4
DEFAULT_COLLATERAL_SCORE_DIM = 3

class ConvBnRelu3d(nn.Module):
    """3D convolution followed by batch normalization and ReLU activation."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))

class DecoderBlock3d(nn.Module):
    """Decoder block for the U-Net architecture with skip connections."""
    def __init__(self,
                 in_channels: int,
                 skip_channels: int,
                 out_channels: int):
        super().__init__()

        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)

        total_in_channels = out_channels + skip_channels

        self.conv1 = ConvBnRelu3d(total_in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = ConvBnRelu3d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.upsample(x)

        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                 skip = F.interpolate(skip, size=x.shape[2:], mode='trilinear', align_corners=False)

            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)

        return x

class BasicBlock3d(nn.Module):
    """Basic ResNet block for 3D data."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = ConvBnRelu3d(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv3d(planes, planes * self.expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet3D(nn.Module):
    """ResNet backbone for 3D data processing."""
    def __init__(self, block, layers, in_channels=3, out_channels=128):
        super().__init__()

        self.inplanes = 64
        self.conv1 = ConvBnRelu3d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        self.out_conv = ConvBnRelu3d(128 * block.expansion, out_channels, kernel_size=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.out_conv(x)

        return x

class FusionModule(nn.Module):
    """Module for fusing primary and auxiliary features with learnable weights."""
    def __init__(self, feature_channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, feature_channels, 1, 1, 1))

    def forward(self, primary_features: torch.Tensor, auxiliary_features: torch.Tensor) -> torch.Tensor:
        if primary_features.shape[2:] != auxiliary_features.shape[2:]:
             raise ValueError(f"Spatial dimensions of primary features {primary_features.shape[2:]} and auxiliary features {auxiliary_features.shape[2:]} must match for fusion.")

        fused_features = primary_features + self.alpha * auxiliary_features
        return fused_features

class KnowledgeGuidedUNet3D(nn.Module):
    """Knowledge-guided 3D UNet for infarct prediction with support for prior knowledge integration."""
    def __init__(self,
                 n_classes: int,
                 ctp_channels: int = DEFAULT_CTP_CHANNELS,
                 use_collateral_scores: bool = False,
                 collateral_score_dim: int = DEFAULT_COLLATERAL_SCORE_DIM,
                 use_probability_atlas: bool = False,
                 use_territory_atlas: bool = False,
                 swin_base_channels: int = 96,
                 swin_depths: Tuple[int, ...] = (2, 2, 6, 2),
                 swin_num_heads: Tuple[int, ...] = (3, 6, 12, 24),
                 swin_window_size: Tuple[int, ...] = (4, 4, 4),
                 swin_patch_size: Tuple[int, ...] = (2, 2, 2),
                 swin_output_channels: Tuple[int, ...] = (96, 192, 384, 768),
                 resnet_layers: List[int] = [2, 2],
                 auxiliary_encoder_output_channels: int = 384,
                 decoder_channels: Tuple[int, ...] = (384, 192, 96, 64),
                 initial_conv_channels: int = 64
                 ):
        super().__init__()

        self.n_classes = n_classes
        self.ctp_channels = ctp_channels
        self.use_collateral_scores = use_collateral_scores
        self.collateral_score_dim = collateral_score_dim
        self.use_probability_atlas = use_probability_atlas
        self.use_territory_atlas = use_territory_atlas
        self.auxiliary_encoder_output_channels = auxiliary_encoder_output_channels

        # Weight parameters for collateral score dimensions
        self.collateral_weights = nn.Parameter(torch.tensor([0.4, 0.4, 0.2]), requires_grad=True)
        
        # Normalization range for each dimension
        self.collateral_min_vals = torch.tensor([0.0] * collateral_score_dim)
        self.collateral_max_vals = torch.tensor([10.0] * collateral_score_dim)

        # Initial convolutional layers for CTP parameter maps
        self.initial_conv = nn.Sequential(
            ConvBnRelu3d(ctp_channels, initial_conv_channels, kernel_size=3, padding=1),
            ConvBnRelu3d(initial_conv_channels, initial_conv_channels, kernel_size=3, padding=1)
        )
        swin_input_channels = initial_conv_channels

        # Primary encoder using Swin Transformer 3D
        self.primary_encoder = SwinTransformer3D(
            in_chans=swin_input_channels,
            embed_dim=swin_base_channels,
            depths=swin_depths,
            num_heads=swin_num_heads,
            window_size=swin_window_size,
            patch_size=swin_patch_size,
            output_channels=swin_output_channels
        )

        primary_encoder_feature_info = self.primary_encoder.feature_info
        primary_skip_channels = [info['num_chs'] for info in primary_encoder_feature_info[:-1]]
        primary_deepest_channels = primary_encoder_feature_info[-1]['num_chs']

        # Determine input channels for auxiliary encoder
        auxiliary_encoder_input_channels = 0
        if self.use_probability_atlas:
            auxiliary_encoder_input_channels += 1
        if self.use_territory_atlas:
            auxiliary_encoder_input_channels += 1
        if self.use_collateral_scores:
            auxiliary_encoder_input_channels += 1

        # Auxiliary encoder using ResNet3D to process prior knowledge
        self.auxiliary_encoder = None
        if auxiliary_encoder_input_channels > 0:
            self.auxiliary_encoder = ResNet3D(
                BasicBlock3d,
                resnet_layers,
                in_channels=auxiliary_encoder_input_channels,
                out_channels=self.auxiliary_encoder_output_channels
            )

        # Projection layer to ensure compatible dimensions for fusion
        self.primary_fusion_projection = None
        if primary_deepest_channels != self.auxiliary_encoder_output_channels:
            self.primary_fusion_projection = ConvBnRelu3d(
                primary_deepest_channels, 
                self.auxiliary_encoder_output_channels, 
                kernel_size=1, 
                padding=0
            )

        # Feature fusion module
        self.fusion_module = None
        if self.auxiliary_encoder is not None:
            self.fusion_module = FusionModule(self.auxiliary_encoder_output_channels)

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        decoder_in_channels = self.auxiliary_encoder_output_channels if self.fusion_module else primary_deepest_channels

        skip_connection_channels = primary_skip_channels[::-1]

        if len(decoder_channels) != len(skip_connection_channels) + 1:
            raise ValueError(f"Number of decoder channels ({len(decoder_channels)}) must match number of skip connections ({len(skip_connection_channels)}) + 1")

        # First decoder block
        self.decoder_blocks.append(
            DecoderBlock3d(
                in_channels=decoder_in_channels,
                skip_channels=skip_connection_channels[0],
                out_channels=decoder_channels[0]
            )
        )
        
        # Remaining decoder blocks
        for i in range(1, len(decoder_channels)):
            self.decoder_blocks.append(
                DecoderBlock3d(
                    in_channels=decoder_channels[i-1],
                    skip_channels=skip_connection_channels[i] if i < len(skip_connection_channels) else 0,
                    out_channels=decoder_channels[i]
                )
            )

        # Segmentation head
        self.segmentation_head = nn.Conv3d(decoder_channels[-1], n_classes, kernel_size=1)

    def _broadcast_collateral_score(self, collateral_scores: torch.Tensor, spatial_shape: Tuple[int, int, int]) -> torch.Tensor:
        """Broadcasts collateral scores to required spatial dimensions."""
        if collateral_scores is None:
            raise ValueError("Collateral scores are required for broadcasting")
        if collateral_scores.shape[1] != self.collateral_score_dim:
            raise ValueError(f"Expected collateral score dimension {self.collateral_score_dim}, but got {collateral_scores.shape[1]}")

        # Normalize scores
        normalized_scores = self._normalize_collateral_scores(collateral_scores)

        # Calculate comprehensive score using weighted sum
        comprehensive_score = self._calculate_comprehensive_score(normalized_scores)

        # Broadcast to spatial dimensions
        batch_size = comprehensive_score.shape[0]
        broadcasted_score_map = comprehensive_score.view(batch_size, 1, 1, 1, 1).repeat(
            1, 1, spatial_shape[0], spatial_shape[1], spatial_shape[2]
        )

        return broadcasted_score_map

    def _normalize_collateral_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """Normalizes collateral scores to [0,1] range."""
        min_vals = self.collateral_min_vals.to(scores.device)
        max_vals = self.collateral_max_vals.to(scores.device)
        normalized_scores = (scores - min_vals) / (max_vals - min_vals + 1e-6)
        normalized_scores = torch.clamp(normalized_scores, 0.0, 1.0)
        return normalized_scores

    def _calculate_comprehensive_score(self, normalized_scores: torch.Tensor) -> torch.Tensor:
        """Calculates comprehensive collateral score using weighted sum."""
        weights = F.softmax(self.collateral_weights, dim=0).to(normalized_scores.device)
        comprehensive_score = torch.sum(weights.view(1, -1) * normalized_scores, dim=1, keepdim=True)
        return comprehensive_score

    def _adjust_with_infarct_probability(self, logits: torch.Tensor, probability_atlas: torch.Tensor) -> torch.Tensor:
        """Adjusts predictions using infarct probability atlas."""
        # Calculate mean probability from atlas
        prob_atlas_mean = torch.mean(probability_atlas)
        
        # Convert logits to probability
        prob = torch.sigmoid(logits)
        
        # Adjust probability using formula
        adjusted_prob = prob * probability_atlas / (prob_atlas_mean + 1e-6)
        
        # Convert back to logits
        epsilon = 1e-7
        adjusted_prob = torch.clamp(adjusted_prob, epsilon, 1 - epsilon)
        adjusted_logits = torch.log(adjusted_prob / (1 - adjusted_prob))
        
        return adjusted_logits

    def forward(self,
                ctp_maps: torch.Tensor,
                collateral_scores: Optional[torch.Tensor] = None,
                probability_atlas: Optional[torch.Tensor] = None,
                territory_atlas: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass implementing the full model architecture with knowledge integration.
        
        Args:
            ctp_maps: CT perfusion parameter maps (CBF, CBV, MTT, Tmax)
            collateral_scores: Collateral circulation scores (artery, perfusion, vein)
            probability_atlas: Infarct probability atlas
            territory_atlas: Arterial territory atlas
            
        Returns:
            Segmentation logits predicting final infarct
        """
        # Validate inputs
        auxiliary_input_list = []
        input_spatial_shape = ctp_maps.shape[2:]

        # Process collateral scores
        if self.use_collateral_scores:
            if collateral_scores is None: 
                raise ValueError("Collateral scores input is required when use_collateral_scores is True")
            broadcasted_score_map = self._broadcast_collateral_score(collateral_scores, input_spatial_shape)
            auxiliary_input_list.append(broadcasted_score_map)

        # Process probability atlas
        if self.use_probability_atlas:
            if probability_atlas is None: 
                raise ValueError("Probability atlas input is required when use_probability_atlas is True")
            if probability_atlas.shape[2:] != input_spatial_shape:
                probability_atlas = F.interpolate(
                    probability_atlas, 
                    size=input_spatial_shape, 
                    mode='trilinear', 
                    align_corners=False
                )
            auxiliary_input_list.append(probability_atlas)

        # Process territory atlas
        if self.use_territory_atlas:
            if territory_atlas is None: 
                raise ValueError("Territory atlas input is required when use_territory_atlas is True")
            if territory_atlas.shape[2:] != input_spatial_shape:
                territory_atlas = F.interpolate(
                    territory_atlas, 
                    size=input_spatial_shape, 
                    mode='nearest'
                )
            auxiliary_input_list.append(territory_atlas)

        # Prepare auxiliary encoder input
        auxiliary_encoder_input = None
        if auxiliary_input_list:
            auxiliary_encoder_input = torch.cat(auxiliary_input_list, dim=1)
            if auxiliary_encoder_input.shape[1] != (self.use_collateral_scores + self.use_probability_atlas + self.use_territory_atlas):
                raise RuntimeError("Mismatch in auxiliary input channel count")

        # Primary encoder path
        x_primary = self.initial_conv(ctp_maps)
        
        # Adjust Swin Transformer input shape
        x_primary = x_primary.permute(0, 2, 3, 4, 1)  # [B, C, D, H, W] -> [B, D, H, W, C]
        
        # Get multi-scale features
        primary_features = self.primary_encoder(x_primary.permute(0, 4, 1, 2, 3))  # [B, D, H, W, C] -> [B, C, D, H, W]
        
        # Convert features back to [B, D, H, W, C] format
        primary_features = [feat.permute(0, 2, 3, 4, 1) for feat in primary_features]
        
        primary_skip_features = primary_features[:-1]
        primary_deepest_feature = primary_features[-1]

        # Auxiliary encoder path
        auxiliary_feature = None
        if self.auxiliary_encoder is not None and auxiliary_encoder_input is not None:
            auxiliary_encoder_input = auxiliary_encoder_input.permute(0, 1, 2, 3, 4)  # Ensure input shape is correct
            auxiliary_feature = self.auxiliary_encoder(auxiliary_encoder_input)
            
            # Convert auxiliary features to [B, D, H, W, C] format
            auxiliary_feature = auxiliary_feature.permute(0, 2, 3, 4, 1)
            
            if auxiliary_feature.shape[1:4] != primary_deepest_feature.shape[1:4]:
                auxiliary_feature = F.interpolate(
                    auxiliary_feature.permute(0, 4, 1, 2, 3),  # [B, D, H, W, C] -> [B, C, D, H, W]
                    size=primary_deepest_feature.shape[1:4],
                    mode='trilinear',
                    align_corners=False
                ).permute(0, 2, 3, 4, 1)  # [B, C, D, H, W] -> [B, D, H, W, C]

        # Feature fusion
        fused_features = primary_deepest_feature
        if self.fusion_module is not None and auxiliary_feature is not None:
            if self.primary_fusion_projection is not None:
                projected_primary_feature = self.primary_fusion_projection(
                    primary_deepest_feature.permute(0, 4, 1, 2, 3)  # [B, D, H, W, C] -> [B, C, D, H, W]
                ).permute(0, 2, 3, 4, 1)  # [B, C, D, H, W] -> [B, D, H, W, C]
            else:
                projected_primary_feature = primary_deepest_feature

            fused_features = self.fusion_module(
                projected_primary_feature.permute(0, 4, 1, 2, 3),  # [B, D, H, W, C] -> [B, C, D, H, W]
                auxiliary_feature.permute(0, 4, 1, 2, 3)  # [B, D, H, W, C] -> [B, C, D, H, W]
            ).permute(0, 2, 3, 4, 1)  # [B, C, D, H, W] -> [B, D, H, W, C]

        # Decoder path
        x = fused_features.permute(0, 4, 1, 2, 3)  # [B, D, H, W, C] -> [B, C, D, H, W]
        reversed_primary_skip_features = [feat.permute(0, 4, 1, 2, 3) for feat in primary_skip_features[::-1]]

        for i, decoder_block in enumerate(self.decoder_blocks):
            skip_feature = reversed_primary_skip_features[i] if i < len(reversed_primary_skip_features) else None
            x = decoder_block(x, skip=skip_feature)

        # Generate segmentation logits
        logits = self.segmentation_head(x)

        # Apply probability atlas adjustment if enabled
        if self.use_probability_atlas and probability_atlas is not None:
            # Ensure probability atlas matches logits resolution
            if probability_atlas.shape[2:] != logits.shape[2:]:
                probability_atlas = F.interpolate(
                    probability_atlas, 
                    size=logits.shape[2:], 
                    mode='trilinear', 
                    align_corners=False
                )
            
            # Apply the infarct probability adjustment
            logits = self._adjust_with_infarct_probability(logits, probability_atlas)

        return logits

def create_knowledge_guided_unet_3d(model_type="UnifiedNet", n_classes=1, in_channels=4, depths=[2, 2, 2, 2], 
                                   embed_dim=96, swin_depths=[2, 2, 6, 2], swin_num_heads=[3, 6, 12, 24], 
                                   swin_window_size=(2, 7, 7), swin_patch_size=(2, 4, 4)):
    """
    Factory function for creating knowledge-guided 3D models for infarct prediction.
    
    Args:
        model_type: Type of model to create:
                    "BaselineNet" - Base model using only CTP input
                    "CollateralFlowNet" - Model using collateral flow information
                    "InfarctProbabilityNet" - Model using infarct probability atlas
                    "ArterialTerritoryNet" - Model using arterial territory atlas
                    "UnifiedNet" - Model using all prior knowledge sources
        n_classes: Number of output classes (default=1 for binary segmentation)
        in_channels: Number of input channels (default=4 for CTP sequences)
        depths: ResNet depth configuration
        embed_dim: Embedding dimension
        swin_depths: Swin Transformer depth at each stage 
        swin_num_heads: Swin Transformer number of heads at each stage
        swin_window_size: Swin Transformer window size
        swin_patch_size: Swin Transformer patch size
    """
    if model_type == "BaselineNet":
        model = KnowledgeGuidedUNet3D(
            in_channels=in_channels,
            n_classes=n_classes,
            depths=depths,
            embed_dim=embed_dim,
            use_collateral_scores=False,
            use_probability_atlas=False,
            use_territory_atlas=False,
            swin_depths=swin_depths,
            swin_num_heads=swin_num_heads,
            swin_window_size=swin_window_size,
            swin_patch_size=swin_patch_size
        )
    elif model_type == "CollateralFlowNet":
        model = KnowledgeGuidedUNet3D(
            in_channels=in_channels,
            n_classes=n_classes,
            depths=depths,
            embed_dim=embed_dim,
            use_collateral_scores=True,
            use_probability_atlas=False,
            use_territory_atlas=False,
            swin_depths=swin_depths,
            swin_num_heads=swin_num_heads,
            swin_window_size=swin_window_size,
            swin_patch_size=swin_patch_size
        )
    elif model_type == "InfarctProbabilityNet":
        model = KnowledgeGuidedUNet3D(
            in_channels=in_channels,
            n_classes=n_classes,
            depths=depths,
            embed_dim=embed_dim,
            use_collateral_scores=False,
            use_probability_atlas=True,
            use_territory_atlas=False,
            swin_depths=swin_depths,
            swin_num_heads=swin_num_heads,
            swin_window_size=swin_window_size,
            swin_patch_size=swin_patch_size
        )
    elif model_type == "ArterialTerritoryNet":
        model = KnowledgeGuidedUNet3D(
            in_channels=in_channels,
            n_classes=n_classes,
            depths=depths,
            embed_dim=embed_dim,
            use_collateral_scores=False,
            use_probability_atlas=False,
            use_territory_atlas=True,
            swin_depths=swin_depths,
            swin_num_heads=swin_num_heads,
            swin_window_size=swin_window_size,
            swin_patch_size=swin_patch_size
        )
    elif model_type == "UnifiedNet":
        model = KnowledgeGuidedUNet3D(
            in_channels=in_channels,
            n_classes=n_classes,
            depths=depths,
            embed_dim=embed_dim,
            use_collateral_scores=True,
            use_probability_atlas=True,
            use_territory_atlas=True,
            swin_depths=swin_depths,
            swin_num_heads=swin_num_heads,
            swin_window_size=swin_window_size,
            swin_patch_size=swin_patch_size
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model
