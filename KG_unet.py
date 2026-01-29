"""
Knowledge-Guided 3D U-Net for final infarct prediction after extended time window thrombectomy.

Corresponds to: "Predicting Infarct Outcomes Following Extended Time Window Thrombectomy in
Large Vessel Occlusion Using Knowledge-Guided Deep Learning" (JNIS, under review).

Architecture variants (paper nomenclature):
- BaselineNet: CTP-only; no prior knowledge.
- CollateralFlowNet: + collateral circulation scores (artery, perfusion, vein).
- InfarctProbabilityNet: + population-derived cerebral infarct probability atlas.
- ArterialTerritoryNet: + cerebral artery territory mapping.
- UnifiedNet: all three knowledge sources combined.

All share: initial 3D conv → Swin Transformer backbone → U-Net decoder with skip connections;
knowledge is integrated via a dedicated auxiliary encoder and learnable fusion at the bottleneck.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Dict, Optional, Tuple, Union
from swin_transformer_3d import SwinTransformer3D

DEFAULT_INPUT_SIZE_3D = (128, 128, 128)
DEFAULT_CTP_CHANNELS = 4  # CBF, CBV, MTT, Tmax
DEFAULT_COLLATERAL_SCORE_DIM = 3  # artery, perfusion, vein (e.g. ASITN/SIR-style grading)

class ConvBnRelu3d(nn.Module):
    """3D conv + BN + ReLU (building block for primary/auxiliary pathways and decoder)."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))

class DecoderBlock3d(nn.Module):
    """U-Net decoder block: upsample → optional skip concat → two convs (paper: decoder with skip connections)."""
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x) if self.downsample is not None else x
        out = self.conv1(x)
        out = self.bn(self.conv2(out))
        out = self.relu(out + identity)
        return out

class ResNet3D(nn.Module):
    """Lightweight 3D ResNet used as auxiliary encoder for prior-knowledge features (paper: dedicated auxiliary encoder)."""
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
    """
    Learnable gated fusion of primary (CTP-derived) and auxiliary (prior-knowledge) bottleneck features.
    Implements: fused = primary + alpha * auxiliary, with alpha per channel (paper: knowledge integration at bottleneck).
    """
    def __init__(self, feature_channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, feature_channels, 1, 1, 1))

    def forward(self, primary_features: torch.Tensor, auxiliary_features: torch.Tensor) -> torch.Tensor:
        if primary_features.shape[2:] != auxiliary_features.shape[2:]:
            auxiliary_features = F.interpolate(
                auxiliary_features, size=primary_features.shape[2:], mode="trilinear", align_corners=False
            )
        return primary_features + self.alpha * auxiliary_features

class KnowledgeGuidedUNet3D(nn.Module):
    """
    Knowledge-guided 3D U-Net for final infarct segmentation (paper: infarct outcome prediction).

    Primary pathway: CTP maps (CBF, CBV, MTT, Tmax) → initial conv → Swin Transformer 3D → multi-scale features.
    Auxiliary pathway (when enabled): collateral scores + probability atlas + territory atlas → ResNet3D → bottleneck features.
    Fusion: learnable gated fusion at bottleneck; decoder with skip connections from primary encoder.
    Optional: infarct probability atlas used again at output for Bayesian-style prior modulation (paper: population-derived prior).
    """
    def __init__(self,
                 n_classes: int,
                 ctp_channels: int = DEFAULT_CTP_CHANNELS,
                 use_collateral_scores: bool = False,
                 collateral_score_dim: int = DEFAULT_COLLATERAL_SCORE_DIM,
                 use_probability_atlas: bool = False,
                 use_territory_atlas: bool = False,
                 use_probability_post_adjustment: Optional[bool] = None,
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
        # Post-hoc adjustment by probability atlas (paper: prior modulation); default True when atlas is used
        self.use_probability_post_adjustment = (
            use_probability_atlas if use_probability_post_adjustment is None else use_probability_post_adjustment
        )
        self.auxiliary_encoder_output_channels = auxiliary_encoder_output_channels

        # --- Collateral: learnable weights over artery / perfusion / vein dimensions (paper: collateral flow information) ---
        self.collateral_weights = nn.Parameter(torch.tensor([0.4, 0.4, 0.2], dtype=torch.float32), requires_grad=True)
        self.register_buffer("collateral_min_vals", torch.zeros(collateral_score_dim))
        self.register_buffer("collateral_max_vals", torch.full((collateral_score_dim,), 10.0))

        # --- Primary pathway: CTP → conv → Swin 3D ---
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
        primary_skip_channels = [info["num_chs"] for info in primary_encoder_feature_info[:-1]]
        primary_deepest_channels = primary_encoder_feature_info[-1]["num_chs"]

        # --- Auxiliary pathway: prior knowledge (collateral / probability atlas / territory atlas) → ResNet3D ---
        auxiliary_encoder_input_channels = 0
        if self.use_probability_atlas:
            auxiliary_encoder_input_channels += 1
        if self.use_territory_atlas:
            auxiliary_encoder_input_channels += 1
        if self.use_collateral_scores:
            auxiliary_encoder_input_channels += 1

        self.auxiliary_encoder = None
        if auxiliary_encoder_input_channels > 0:
            self.auxiliary_encoder = ResNet3D(
                BasicBlock3d,
                resnet_layers,
                in_channels=auxiliary_encoder_input_channels,
                out_channels=self.auxiliary_encoder_output_channels
            )

        # --- Fusion: align primary bottleneck to auxiliary channels; learnable gated fusion ---
        self.primary_fusion_projection = None
        if primary_deepest_channels != self.auxiliary_encoder_output_channels:
            self.primary_fusion_projection = ConvBnRelu3d(
                primary_deepest_channels,
                self.auxiliary_encoder_output_channels,
                kernel_size=1,
                padding=0,
            )
        self.fusion_module = None
        if self.auxiliary_encoder is not None:
            self.fusion_module = FusionModule(self.auxiliary_encoder_output_channels)

        # --- Decoder: U-Net-style upsampling with skip connections from primary encoder ---
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
        """Broadcast collateral scores (artery, perfusion, vein) to a 3D volume for auxiliary encoder (paper: collateral flow)."""
        if collateral_scores is None:
            raise ValueError("Collateral scores are required for broadcasting")
        if collateral_scores.shape[1] != self.collateral_score_dim:
            raise ValueError(f"Expected collateral score dimension {self.collateral_score_dim}, but got {collateral_scores.shape[1]}")

        # Normalize scores
        normalized_scores = self._normalize_collateral_scores(collateral_scores)

        # Calculate comprehensive score using weighted sum
        comprehensive_score = self._calculate_comprehensive_score(normalized_scores)

        # Broadcast to spatial dimensions (expand avoids extra memory vs repeat)
        batch_size = comprehensive_score.shape[0]
        broadcasted_score_map = comprehensive_score.view(batch_size, 1, 1, 1, 1).expand(
            batch_size, 1, spatial_shape[0], spatial_shape[1], spatial_shape[2]
        )

        return broadcasted_score_map

    def _normalize_collateral_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """Normalize collateral dimensions to [0, 1] using registered min/max (paper: collateral grading)."""
        normalized_scores = (scores - self.collateral_min_vals) / (self.collateral_max_vals - self.collateral_min_vals + 1e-6)
        normalized_scores = torch.clamp(normalized_scores, 0.0, 1.0)
        return normalized_scores

    def _calculate_comprehensive_score(self, normalized_scores: torch.Tensor) -> torch.Tensor:
        """Aggregate normalized collateral dimensions with learnable softmax weights (paper: collateral flow)."""
        weights = F.softmax(self.collateral_weights, dim=0)
        comprehensive_score = (weights.unsqueeze(0) * normalized_scores).sum(dim=1, keepdim=True)
        return comprehensive_score

    def _adjust_with_infarct_probability(self, logits: torch.Tensor, probability_atlas: torch.Tensor) -> torch.Tensor:
        """Prior modulation: p_adj = p * (atlas / mean(atlas)); paper: population-derived infarct probability prior."""
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
        Forward pass: primary (CTP) + optional auxiliary (prior knowledge) → fusion → decoder → (optional) prior modulation.

        Args:
            ctp_maps: [B, 4, D, H, W] — CTP maps (CBF, CBV, MTT, Tmax).
            collateral_scores: [B, 3] — collateral scores (artery, perfusion, vein) for CollateralFlowNet/UnifiedNet.
            probability_atlas: [B, 1, D, H, W] — population-derived infarct probability (InfarctProbabilityNet/UnifiedNet).
            territory_atlas: [B, 1, D, H, W] — arterial territory mapping (ArterialTerritoryNet/UnifiedNet).

        Returns:
            [B, n_classes, D', H', W] — segmentation logits (final infarct prediction).
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

        # Territory atlas (paper: cerebral artery territory mapping); nearest to preserve label identity
        if self.use_territory_atlas:
            if territory_atlas is None:
                raise ValueError("Territory atlas input is required when use_territory_atlas is True")
            if territory_atlas.shape[2:] != input_spatial_shape:
                territory_atlas = F.interpolate(
                    territory_atlas, size=input_spatial_shape, mode="nearest"
                )
            auxiliary_input_list.append(territory_atlas)

        # Prepare auxiliary encoder input
        auxiliary_encoder_input = None
        if auxiliary_input_list:
            auxiliary_encoder_input = torch.cat(auxiliary_input_list, dim=1)
            if auxiliary_encoder_input.shape[1] != (self.use_collateral_scores + self.use_probability_atlas + self.use_territory_atlas):
                raise RuntimeError("Mismatch in auxiliary input channel count")

        # Primary encoder path: keep [B, C, D, H, W] throughout (PyTorch conv convention)
        x_primary = self.initial_conv(ctp_maps)
        primary_features = self.primary_encoder(x_primary)
        primary_skip_features = primary_features[:-1]
        primary_deepest_feature = primary_features[-1]

        # Auxiliary encoder path
        auxiliary_feature = None
        if self.auxiliary_encoder is not None and auxiliary_encoder_input is not None:
            auxiliary_feature = self.auxiliary_encoder(auxiliary_encoder_input)
            if auxiliary_feature.shape[2:] != primary_deepest_feature.shape[2:]:
                auxiliary_feature = F.interpolate(
                    auxiliary_feature,
                    size=primary_deepest_feature.shape[2:],
                    mode='trilinear',
                    align_corners=False
                )

        # Feature fusion (all tensors [B, C, D, H, W])
        fused_features = primary_deepest_feature
        if self.fusion_module is not None and auxiliary_feature is not None:
            projected_primary = (
                self.primary_fusion_projection(primary_deepest_feature)
                if self.primary_fusion_projection is not None
                else primary_deepest_feature
            )
            fused_features = self.fusion_module(projected_primary, auxiliary_feature)

        # Decoder path
        x = fused_features
        reversed_skip_features = list(reversed(primary_skip_features))
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip_feature = reversed_skip_features[i] if i < len(reversed_skip_features) else None
            x = decoder_block(x, skip=skip_feature)

        logits = self.segmentation_head(x)

        # Optional post-hoc prior modulation by infarct probability atlas (paper: population-derived prior)
        if self.use_probability_post_adjustment and probability_atlas is not None:
            atlas_at_logits = probability_atlas
            if atlas_at_logits.shape[2:] != logits.shape[2:]:
                atlas_at_logits = F.interpolate(
                    atlas_at_logits,
                    size=logits.shape[2:],
                    mode="trilinear",
                    align_corners=False,
                )
            logits = self._adjust_with_infarct_probability(logits, atlas_at_logits)

        return logits

def create_knowledge_guided_unet_3d(
    model_type: str = "UnifiedNet",
    n_classes: int = 1,
    ctp_channels: int = DEFAULT_CTP_CHANNELS,
    resnet_layers: Optional[List[int]] = None,
    swin_base_channels: int = 96,
    swin_depths: Tuple[int, ...] = (2, 2, 6, 2),
    swin_num_heads: Tuple[int, ...] = (3, 6, 12, 24),
    swin_window_size: Tuple[int, ...] = (4, 4, 4),
    swin_patch_size: Tuple[int, ...] = (2, 2, 2),
    **kwargs,
) -> KnowledgeGuidedUNet3D:
    """
    Create a knowledge-guided 3D model (paper: five variants for infarct outcome prediction).

    Args:
        model_type: Paper variants — "BaselineNet" | "CollateralFlowNet" | "InfarctProbabilityNet"
                    | "ArterialTerritoryNet" | "UnifiedNet".
        n_classes: Output classes (default 1 for binary segmentation).
        ctp_channels: CTP input channels (default 4: CBF, CBV, MTT, Tmax).
        resnet_layers: Auxiliary encoder ResNet block counts (default [2, 2]).
        swin_base_channels: Swin embedding dimension.
        swin_depths: Swin depth per stage.
        swin_num_heads: Attention heads per stage.
        swin_window_size: 3D window size (e.g. (4,4,4)).
        swin_patch_size: 3D patch size.
        **kwargs: Forwarded to KnowledgeGuidedUNet3D (e.g. use_probability_post_adjustment, decoder_channels).
    """
    if resnet_layers is None:
        resnet_layers = [2, 2]
    # Ensure 3-tuples for Swin
    swin_window_size = tuple(swin_window_size) if not isinstance(swin_window_size, tuple) else swin_window_size
    swin_patch_size = tuple(swin_patch_size) if not isinstance(swin_patch_size, tuple) else swin_patch_size
    if len(swin_window_size) != 3:
        swin_window_size = (swin_window_size[0],) * 3 if len(swin_window_size) >= 1 else (4, 4, 4)
    if len(swin_patch_size) != 3:
        swin_patch_size = (swin_patch_size[0],) * 3 if len(swin_patch_size) >= 1 else (2, 2, 2)

    common = dict(
        n_classes=n_classes,
        ctp_channels=ctp_channels,
        resnet_layers=resnet_layers,
        swin_base_channels=swin_base_channels,
        swin_depths=swin_depths,
        swin_num_heads=swin_num_heads,
        swin_window_size=swin_window_size,
        swin_patch_size=swin_patch_size,
        **kwargs,
    )
    if model_type == "BaselineNet":
        return KnowledgeGuidedUNet3D(use_collateral_scores=False, use_probability_atlas=False, use_territory_atlas=False, **common)
    if model_type == "CollateralFlowNet":
        return KnowledgeGuidedUNet3D(use_collateral_scores=True, use_probability_atlas=False, use_territory_atlas=False, **common)
    if model_type == "InfarctProbabilityNet":
        return KnowledgeGuidedUNet3D(use_collateral_scores=False, use_probability_atlas=True, use_territory_atlas=False, **common)
    if model_type == "ArterialTerritoryNet":
        return KnowledgeGuidedUNet3D(use_collateral_scores=False, use_probability_atlas=False, use_territory_atlas=True, **common)
    if model_type == "UnifiedNet":
        return KnowledgeGuidedUNet3D(use_collateral_scores=True, use_probability_atlas=True, use_territory_atlas=True, **common)
    raise ValueError(f"Unknown model type: {model_type}")
