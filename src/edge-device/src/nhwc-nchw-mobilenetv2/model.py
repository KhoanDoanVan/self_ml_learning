import numpy as np
from typing import List, Tuple

from tensor import EdgeTensor, TensorLayout
from layouts.layout_aware_batchnorm2d import LayoutAwareBatchNorm2D
from layouts.layout_aware_conv2d import LayoutAwareConv2D
from layouts.relu import relu
from blocks.bottleneck_stack import BottleneckStack
from blocks.inverted_residual import InvertedResidual


class EdgeMobileNet:
    """
    Lightweight MobileNetV2-style model for edge devices

    Architecture designed to demonstrate layout impact:
    - Initial conv stem
    - Multiple inverted residual stages
    - Final classification head

    Same mathematical operations, different memory layouts
    """

    def __init__(
            self,
            num_classes: int = 1000,
            width_multiplier: float = 1.0,
            layout: TensorLayout = TensorLayout.NHWC,
            input_size: int = 224
    ):
        self.num_classes = num_classes
        self.width_multiplier = width_multiplier
        self.layout = layout
        self.input_size = input_size

        
    def _make_divisible(self, v: float, divisor: int = 8) -> int:
        """
        Ensure channel count is divisible by divisor
        Helps with hardware vectorization
        """

        new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v
    

    def _build_model(self):
        """
        Build MobileNetV2 architecture

        Configuration: [expand_ratio, channels, num_blocks, stride]
        """

        # Initial convolution stem
        input_channels = self._make_divisible(32 * self.width_multiplier)
        self.conv_stem = LayoutAwareConv2D(
            3, input_channels, kernel_size=3, stride=2, padding=1, layout=self.layout
        )
        self.bn_stem = LayoutAwareBatchNorm2D(input_channels)

        # MobileNetV2 configuration
        # Format: [expand_ratio, out_channels, num_blocks, stride]
        inverted_residual_config = [
            [1, 16, 1, 1], # Stage 1
            [6, 24, 2, 2], # Stage 2
            [6, 32, 3, 2], # Stage 3
            [6, 64, 4, 2], # Stage 4
            [6, 96, 3, 1] # Stage 5
        ]

        # Build inverted residual stages
        self.stages = []
        in_channels = input_channels

        for expand_ratio, out_channels, num_blocks, stride in inverted_residual_config:

            out_channels = self._make_divisible(out_channels * self.width_multiplier)

            stage = BottleneckStack(
                in_channels=in_channels,
                out_channels=out_channels,
                num_blocks=num_blocks,
                stride=stride,
                expand_ratio=expand_ratio,
                layout=self.layout
            )

            self.stages.append(stage)
            in_channels = out_channels

        
        # Final convolution
        final_channels = self._make_divisible(1280 * self.width_multiplier)
        self.conv_final = LayoutAwareConv2D(
            in_channels, final_channels, kernel_size=1, layout=self.layout
        )
        self.bn_final = LayoutAwareBatchNorm2D(final_channels)

        # Classification head
        self.fc_channels = final_channels
        self.fc_weight = np.random.randn(
            self.num_classes, final_channels
        ).astype(np.float32) * 0.01
        self.fc_bias = np.zeros(self.num_classes, dtype=np.float32)

    
    def forward(self, x: EdgeTensor) -> np.ndarray:
        """
        Forward pass through entire model.

        This demonstrates the cumulative effect of layout choice:
        - Layout conversions (if any) happen once at input
        - All intermediate operations in native layout
        - Final output is class logits
        """

        if x.layout != self.layout:
            x = x.to_layout(self.layout)

        # Stem
        x = self.conv_stem.forward(x)
        x = self.bn_stem.forward(x)
        x = relu(x)

        # Inverted residual stages
        for stage in self.stages:
            x = stage.forward(x)

        # Final convolution
        x = self.conv_final.forward(x)
        x = self.bn_final.forward(x)
        x = relu(x)

        # Global average pooling
        if self.layout == TensorLayout.NHWC:
            # (N, H, W, C) -> (N, C)
            pooled = np.mean(x.data, axis=(1, 2))
        else:
            # (N, C, H, W) -> (N, C)
            pooled = np.mean(x.data, axis=(2, 3))
        
        # Fully connected layer
        logits = pooled @ self.fc_weight.T + self.fc_bias

        return logits
    

    def count_parameters(self) -> int:
        """Count total number of parameters (not layout-dependent)."""
        
        # Simplified parameter counting
        total = 0
        
        # Stem
        total += self.conv_stem.weight.size + self.conv_stem.bias.size
        total += self.bn_stem.gamma.size + self.bn_stem.beta.size
        
        # Stages (approximation)
        # Each inverted residual has expand, depthwise, project
        for stage in self.stages:
            for block in stage.blocks:
                # Expand conv (if exists)
                if block.expand_conv is not None:
                    total += block.expand_conv.weight.size + block.expand_conv.bias.size
                    total += block.expand_bn.gamma.size + block.expand_bn.beta.size
                
                # Depthwise (simplified - each channel has kernel_size^2 params)
                total += block.depthwise_channels * 9  # 3x3 kernel
                total += block.depthwise_bn.gamma.size + block.depthwise_bn.beta.size
                
                # Project conv
                total += block.project_conv.weight.size + block.project_conv.bias.size
                total += block.project_bn.gamma.size + block.project_bn.beta.size
        
        # Final conv
        total += self.conv_final.weight.size + self.conv_final.bias.size
        total += self.bn_final.gamma.size + self.bn_final.beta.size
        
        # FC layer
        total += self.fc_weight.size + self.fc_bias.size
        
        return total
    
    def __repr__(self) -> str:
        param_count = self.count_parameters()
        return (
            f"EdgeMobileNet(\n"
            f"  layout={self.layout.value},\n"
            f"  num_classes={self.num_classes},\n"
            f"  width_multiplier={self.width_multiplier},\n"
            f"  input_size={self.input_size},\n"
            f"  parameters={param_count:,}\n"
            f")"
        )