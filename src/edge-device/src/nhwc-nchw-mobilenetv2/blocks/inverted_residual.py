import numpy as np
from typing import Optional
from layouts.depthwise_conv2d import depthwise_conv2d
from layouts.layout_aware_batchnorm2d import LayoutAwareBatchNorm2D
from layouts.layout_aware_conv2d import LayoutAwareConv2D
from layouts.relu import relu
from tensor import TensorLayout, EdgeTensor



class InvertedResidual:
    """
    Architecture:
    1. Expand: 1x1 conv to expend channels
    2. Depthwise: 3x3 depthwise conv
    3. Project: 1x1 conv to project back
    4. Residual connection (if stride=1 and in_channels=out_channels)
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            expand_ratio: int = 6,
            layout: TensorLayout = TensorLayout.NHWC
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.layout = layout

        # Calculate hidden dimension
        hidden_dim = in_channels * expand_ratio

        # Use residual connection if stride = 1 and channels match
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        # Build layers
        layers = []

        # 1. Expansion phase (1x1 conv)
        if expand_ratio != 1:
            self.expand_conv = LayoutAwareConv2D(
                in_channels,
                hidden_dim,
                kernel_size=1,
                layout=layout
            )
            self.expand_bn = LayoutAwareBatchNorm2D(
                hidden_dim
            )
        else:
            self.expand_conv = None
            self.expand_bn = None
            hidden_dim = in_channels

        
        # 2. Depthwise convolution phase
        self.depthwise_channels = hidden_dim
        self.depthwise_bn= LayoutAwareBatchNorm2D(hidden_dim)

        # 3. Projection phase (1x1 conv)
        self.project_conv = LayoutAwareConv2D(
            hidden_dim, out_channels, kernel_size=1, layout=layout
        )
        self.project_bn = LayoutAwareBatchNorm2D(out_channels)


    def forward(self, x: EdgeTensor) -> EdgeTensor:
        """
        Forward pass through inverted residual block

        This is where layout matters most:
        - Multiple conv operations in sequence
        - Layout conversions accumulate overhead
        - Cache behavior differs between NHWC/NCHW
        """

        identity = x

        # Ensure input is in block's preferred layout
        if x.layout != self.layout:
            x = x.to_layout(self.layout)

        # 1. Expansion (if needed)
        if self.expand_conv is not None:
            x = self.expand_conv.forward(x)
            x = self.expand_bn.forward(x)
            x = relu(x)

        # 2. Depthwise convolution
        # This is layout-sensitive: spatial operations on per-channel basis
        x = depthwise_conv2d(x, kernel_size=3, stride=self.stride, padding=1)
        x = self.depthwise_bn.forward(x)
        x = relu(x)

        # 3. Projection
        x = self.project_conv.forward(x)
        x = self.project_bn.forward(x)

        # 4. Residual connection
        if self.use_residual:
            # Ensure identity is in same layout as output
            if identity.layout != x.layout
                identity = identity.to_layout(x.layout)

            # Element-wise addition
            x = EdgeTensor(x.data + identity.data, x.layout)

        return x
    
    def __repr__(self) -> str:
        return (
            f"InvertedResidual(in={self.in_channels}, out={self.out_channels}, "
            f"stride={self.stride}, expand_ratio={self.expand_ratio}, "
            f"layout={self.layout.value}, use_residual={self.use_residual})"
        )