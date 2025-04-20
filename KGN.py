import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Dict, Optional, Tuple, Union

DEFAULT_INPUT_SIZE_3D = (128, 128, 128)
DEFAULT_CTP_CHANNELS = 4
DEFAULT_COLLATERAL_SCORE_DIM = 3

class ConvBnRelu3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))

class DecoderBlock3d(nn.Module):
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
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = ConvBnRelu3d(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.conv2 = ConvBnRelu3d(planes, planes * self.expansion, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet3D(nn.Module):
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

class SwinTransformer3dPlaceholder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 96, depths: Tuple[int, ...] = (2, 2, 6, 2),
                 num_heads: Tuple[int, ...] = (3, 6, 12, 24), window_size: Tuple[int, ...] = (7, 7, 7),
                 output_channels: Tuple[int, ...] = (96, 192, 384, 768)):
        super().__init__()
        self.in_channels = in_channels
        self.output_channels = output_channels

        self.downsample1 = ConvBnRelu3d(in_channels, base_channels, kernel_size=3, stride=2, padding=1)
        self.downsample2 = ConvBnRelu3d(base_channels, output_channels[0], kernel_size=3, stride=2, padding=1)
        self.downsample3 = ConvBnRelu3d(output_channels[0], output_channels[1], kernel_size=3, stride=2, padding=1)
        self.downsample4 = ConvBnRelu3d(output_channels[1], output_channels[2], kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        f1 = self.downsample1(x)
        f2 = self.downsample2(f1)
        f3 = self.downsample3(f2)
        f4 = self.downsample4(f3)

        return [f2, f3, f4]

    @property
    def feature_info(self) -> List[Dict]:
        return [
            {'num_chs': self.output_channels[0], 'reduction': 4},
            {'num_chs': self.output_channels[1], 'reduction': 8},
            {'num_chs': self.output_channels[2], 'reduction': 16},
        ]

class FusionModule(nn.Module):
    def __init__(self, feature_channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, feature_channels, 1, 1, 1))

    def forward(self, primary_features: torch.Tensor, auxiliary_features: torch.Tensor) -> torch.Tensor:
        if primary_features.shape[2:] != auxiliary_features.shape[2:]:
             raise ValueError(f"Spatial dimensions of primary features {primary_features.shape[2:]} and auxiliary features {auxiliary_features.shape[2:]} must match for fusion.")

        fused_features = primary_features + self.alpha * auxiliary_features
        return fused_features

class KnowledgeGuidedUNet3D(nn.Module):
    def __init__(self,
                 n_classes: int,
                 ctp_channels: int = DEFAULT_CTP_CHANNELS,
                 use_collateral_scores: bool = False,
                 collateral_score_dim: int = DEFAULT_COLLATERAL_SCORE_DIM,
                 use_probability_atlas: bool = False,
                 use_territory_atlas: bool = False,
                 swin_base_channels: int = 96,
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

        self.collateral_weights = nn.Parameter(torch.ones(collateral_score_dim) / collateral_score_dim)
        self.collateral_min_vals = torch.tensor([0.0] * collateral_score_dim)
        self.collateral_max_vals = torch.tensor([10.0] * collateral_score_dim)


        self.initial_conv = nn.Sequential(
            ConvBnRelu3d(ctp_channels, initial_conv_channels, kernel_size=3, padding=1),
            ConvBnRelu3d(initial_conv_channels, initial_conv_channels, kernel_size=3, padding=1)
        )
        swin_input_channels = initial_conv_channels

        self.primary_encoder = SwinTransformer3dPlaceholder(
            in_channels=swin_input_channels,
            output_channels=swin_output_channels
        )

        primary_encoder_feature_info = self.primary_encoder.feature_info
        primary_skip_channels = [info['num_chs'] for info in primary_encoder_feature_info[:-1]][::-1]
        primary_deepest_channels = primary_encoder_feature_info[-1]['num_chs']

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

        self.primary_fusion_projection = None
        if primary_deepest_channels != self.auxiliary_encoder_output_channels:
             self.primary_fusion_projection = ConvBnRelu3d(primary_deepest_channels, self.auxiliary_encoder_output_channels, kernel_size=1, padding=0)

        self.fusion_module = None
        if self.auxiliary_encoder is not None:
             self.fusion_module = FusionModule(self.auxiliary_encoder_output_channels)

        self.decoder_blocks = nn.ModuleList()
        decoder_in_channels = self.auxiliary_encoder_output_channels if self.fusion_module else primary_deepest_channels

        skip_connection_channels = primary_skip_channels[::-1]

        if len(decoder_channels) != len(skip_connection_channels) + 1:
             raise ValueError(f"Number of decoder channels ({len(decoder_channels)}) must match number of skip connections ({len(skip_connection_channels)}) + 1.")

        self.decoder_blocks.append(
             DecoderBlock3d(
                 in_channels=decoder_in_channels,
                 skip_channels=skip_connection_channels[0],
                 out_channels=decoder_channels[0]
             )
        )
        for i in range(1, len(decoder_channels)):
             self.decoder_blocks.append(
                 DecoderBlock3d(
                     in_channels=decoder_channels[i-1],
                     skip_channels=skip_connection_channels[i],
                     out_channels=decoder_channels[i]
                 )
             )

        self.segmentation_head = nn.Conv3d(decoder_channels[-1], n_classes, kernel_size=1)

    def _broadcast_collateral_score(self, collateral_scores: torch.Tensor, spatial_shape: Tuple[int, int, int]) -> torch.Tensor:
        if collateral_scores is None:
            raise ValueError("Collateral scores are required for broadcasting.")
        if collateral_scores.shape[1] != self.collateral_score_dim:
             raise ValueError(f"Expected collateral score dimension {self.collateral_score_dim} but got {collateral_scores.shape[1]}.")

        normalized_scores = self._normalize_collateral_scores(collateral_scores)

        comprehensive_score = self._calculate_comprehensive_score(normalized_scores)

        batch_size = comprehensive_score.shape[0]
        broadcasted_score_map = comprehensive_score.view(batch_size, 1, 1, 1, 1).repeat(
            1, 1, spatial_shape[0], spatial_shape[1], spatial_shape[2]
        )

        return broadcasted_score_map

    def _normalize_collateral_scores(self, scores: torch.Tensor) -> torch.Tensor:
        min_vals = self.collateral_min_vals.to(scores.device)
        max_vals = self.collateral_max_vals.to(scores.device)
        normalized_scores = (scores - min_vals) / (max_vals - min_vals + 1e-6)
        normalized_scores = torch.clamp(normalized_scores, 0.0, 1.0)
        return normalized_scores

    def _calculate_comprehensive_score(self, normalized_scores: torch.Tensor) -> torch.Tensor:
         weights = self.collateral_weights.to(normalized_scores.device)
         comprehensive_score = torch.sum(weights * normalized_scores, dim=1, keepdim=True)
         return comprehensive_score

    def forward(self,
                ctp_maps: torch.Tensor,
                collateral_scores: Optional[torch.Tensor] = None,
                probability_atlas: Optional[torch.Tensor] = None,
                territory_atlas: Optional[torch.Tensor] = None) -> torch.Tensor:

        auxiliary_input_list = []
        input_spatial_shape = ctp_maps.shape[2:]

        if self.use_collateral_scores:
             if collateral_scores is None: raise ValueError("`collateral_scores` input is required when use_collateral_scores is True.")
             broadcasted_score_map = self._broadcast_collateral_score(collateral_scores, input_spatial_shape)
             auxiliary_input_list.append(broadcasted_score_map)

        if self.use_probability_atlas:
             if probability_atlas is None: raise ValueError("`probability_atlas` input is required when use_probability_atlas is True.")
             if probability_atlas.shape[2:] != input_spatial_shape:
                  raise ValueError(f"Probability atlas spatial dimensions {probability_atlas.shape[2:]} do not match CTP map dimensions {input_spatial_shape}.")
             auxiliary_input_list.append(probability_atlas)

        if self.use_territory_atlas:
             if territory_atlas is None: raise ValueError("`territory_atlas` input is required when use_territory_atlas is True.")
             if territory_atlas.shape[2:] != input_spatial_shape:
                  raise ValueError(f"Territory atlas spatial dimensions {territory_atlas.shape[2:]} do not match CTP map dimensions {input_spatial_shape}.")
             auxiliary_input_list.append(territory_atlas)

        auxiliary_encoder_input = None
        if auxiliary_input_list:
             auxiliary_encoder_input = torch.cat(auxiliary_input_list, dim=1)
             if auxiliary_encoder_input.shape[1] != (self.use_collateral_scores + self.use_probability_atlas + self.use_territory_atlas):
                 raise RuntimeError("Mismatch in auxiliary input channel count.")

        x_primary = self.initial_conv(ctp_maps)

        primary_features = self.primary_encoder(x_primary)

        primary_skip_features = primary_features[:-1]
        primary_deepest_feature = primary_features[-1]

        auxiliary_feature = None
        if self.auxiliary_encoder is not None and auxiliary_encoder_input is not None:
            auxiliary_feature = self.auxiliary_encoder(auxiliary_encoder_input)
            if auxiliary_feature.shape[2:] != primary_deepest_feature.shape[2:]:
                 auxiliary_feature = F.interpolate(auxiliary_feature, size=primary_deepest_feature.shape[2:], mode='trilinear', align_corners=False)

        fused_features = primary_deepest_feature

        if self.fusion_module is not None and auxiliary_feature is not None:
             if self.primary_fusion_projection is not None:
                  projected_primary_feature = self.primary_fusion_projection(primary_deepest_feature)
             else:
                  projected_primary_feature = primary_deepest_feature

             fused_features = self.fusion_module(projected_primary_feature, auxiliary_feature)
        elif auxiliary_feature is not None:
             pass

        x = fused_features
        reversed_primary_skip_features = primary_skip_features[::-1]

        for i, decoder_block in enumerate(self.decoder_blocks):
            skip_feature = reversed_primary_skip_features[i]
            x = decoder_block(x, skip=skip_feature)

        logits = self.segmentation_head(x)

        return logits

def create_knowledge_guided_unet_3d(model_type: str, n_classes: int, **kwargs) -> KnowledgeGuidedUNet3D:
    use_collateral_scores = False
    use_probability_atlas = False
    use_territory_atlas = False

    if model_type == 'BaselineNet':
        pass

    elif model_type == 'CollateralFlowNet':
        use_collateral_scores = True

    elif model_type == 'InfarctProbabilityNet':
        use_probability_atlas = True

    elif model_type == 'ArterialTerritoryNet':
        use_territory_atlas = True

    elif model_type == 'UnifiedNet':
        use_collateral_scores = True
        use_probability_atlas = True
        use_territory_atlas = True

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be one of 'BaselineNet', 'CollateralFlowNet', 'InfarctProbabilityNet', 'ArterialTerritoryNet', 'UnifiedNet'.")

    model = KnowledgeGuidedUNet3D(
        n_classes=n_classes,
        use_collateral_scores=use_collateral_scores,
        use_probability_atlas=use_probability_atlas,
        use_territory_atlas=use_territory_atlas,
        **kwargs
    )

    return model
