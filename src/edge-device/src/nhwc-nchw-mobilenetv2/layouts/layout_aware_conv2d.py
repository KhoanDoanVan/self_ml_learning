import numpy as np
from typing import Tuple, Optional
from tensor import TensorLayout, EdgeTensor



class LayoutAwareConv2D:
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            layout: TensorLayout = TensorLayout.NHWC
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.layout = layout


        # Initialize weights (same mathematical operation regradless of layout)
        # Weights stored in standard format: (out_channels, in_channels, kH, kW)
        self.weight = np.random.randn(
            out_channels,
            in_channels,
            kernel_size,
            kernel_size
        ).astype(np.float32) * 0.01

        self.bias = np.zeros(out_channels, dtype=np.float32)


    def forward(self, x: EdgeTensor) -> EdgeTensor:
        input_layout = x.layout

        # Convert to preferred layout if needed 
        if x.layout != self.layout:
            x = x.to_layout(self.layout)

        # Execute in native layout
        if self.layout == TensorLayout.NHWC:
            output = self._forward_nhwc(x)
        else:
            output = self._forward_nchw(x)

        # Convert back to input layout if needed
        if output.layout != input_layout:
            output = output.to_layout(input_layout)

        return output


    def _forward_nhwc(self, x: EdgeTensor) -> EdgeTensor:
        N, H, W, C = x.shape

        out_h = (H + 2*self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2*self.padding - self.kernel_size) // self.stride + 1

        if self.padding > 0:
            padded = np.pad(
                x.data,
                (
                    (0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)
                ),
                mode='constant'
            )
        else:
            padded = x.data

        output = np.zeros(
            (N, out_h, out_w, self.out_channels),
            dtype=np.float32
        )

        for n in range(N):
            for i in range(out_h):
                for j in range(out_w):

                    h_start = i * self.stride
                    w_start = j * self.stride

                    # Extract receptive field - contiguous in NHWC
                    # Shape: (kernel_size, kernel_size, in_channels)
                    receptive_field = padded[
                        n,
                        h_start: h_start + self.kernel_size,
                        w_start: w_start + self.kernel_size,
                        :
                    ]

                    # Convolution as matrix multiplication
                    # Weight shape: (out_channels, in_channels, kH, kW)
                    # Receptive_field shape: (kH, kW, in_channels)
                    for oc in range(self.out_channels):
                        # Element-wise multiply and sum over all dimensions
                        conv_sum = 0.0

                        for ic in range(self.in_channels):
                            for kh in range(self.kernel_size):
                                for kw in range(self.kernel_size):

                                    conv_sum += (
                                        receptive_field[kh, kw, ic] * self.weight[oc, ic, kh, kw]
                                    )

                        output[n, i, j, oc] = conv_sum + self.bias[oc]

        return EdgeTensor(output, TensorLayout.NHWC)

    
    def _forward_nchw(self, x: EdgeTensor) -> EdgeTensor:
        N, C, H, W = x.shape

        out_h = (H + 2*self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2*self.padding - self.kernel_size) // self.stride + 1

        if self.padding > 0:
            padded = np.pad(
                x.data,
                (
                    (0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)
                ),
                mode='constant'
            )
        else:
            padded = x.data


        output = np.zeros(
            (N, self.out_channels, out_h, out_w),
            dtype=np.float32
        )

        for n in range(N):
            for oc in range(self.out_channels):
                for i in range(out_h):
                    for j in range(out_w):

                        h_start = i * self.stride
                        w_start = j * self.stride

                        # Convolution as matrix multiplication
                        # Weight Shape: (out_channels, in_channels, kH, kW)
                        # Receptive_field Shape: (in_channels, kH, kW)
                        conv_sum = 0.0

                        for ic in range(self.in_channels):
                            for kh in range(self.kernel_size):
                                for kw in range(self.kernel_size):

                                    conv_sum += (
                                        padded[n, ic, h_start + kh, w_start + kw] * self.weight[oc, ic, kh, kw]
                                    )

                        output[n, oc, i, j] = conv_sum + self.bias[oc]


        return EdgeTensor(output, TensorLayout.NCHW)