from tensor import EdgeTensor, TensorLayout
import numpy as np


def depthwise_conv2d (
        x: EdgeTensor,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
) -> EdgeTensor:
    
    C = x.channels

    # Create depthwise convolution (groups = channels)
    # Each output channel computed from exactly one input channel
    output_data = np.zeros_like(x.data)

    if x.layout == TensorLayout.NHWC:
        N, H, W, C = x.shape

        if padding > 0:
            # pad_width is a tuple specifying how much padding to add before and after each axis.
            padded = np.pad(
                x.data,
                ((0, 0), (padding, padding), (padding, padding), (0, 0)),
                mode='constant'
            )
        else:
            padded = x.data

        out_h = (H + 2 * padding - kernel_size) // stride + 1
        out_w = (W + 2 * padding - kernel_size) // stride + 1


        output = np.zeros((N, out_h, out_w, C), dtype=np.float32)

        # Simplified depthwise - each channel processed independently
        for c in range(C):
            kernel = np.random.randn(kernel_size, kernel_size).astype(np.float32) * 0.01

            for n in range(N):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * stride
                        w_start = j * stride
                        receptive_field = padded[
                            n,
                            h_start:h_start + kernel_size,
                            w_start:w_start + kernel_size,
                            c
                        ]
                        output[n, i, j, c] = np.sum(receptive_field * kernel)
    
    else:
        N, C, H, W = x.shape

        if padded > 0:
            # pad_width is a tuple specifying how much padding to add before and after each axis.
            padded = np.pad(
                x.data,
                ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                mode='constant'
            )
        else:
            padded = x.data

        
        out_h = (H + 2 * padding - kernel_size) // stride + 1
        out_w = (W + 2 * padding - kernel_size) // stride + 1

        output = np.zeros((N, C, out_h, out_w), dtype=np.float32)

        for c in range(C):
            kernel = np.random.randn(kernel_size, kernel_size).astype(np.float32) * 0.01
            for n in range(N):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * stride
                        w_start = j * stride
                        receptive_field = padded[
                            n,
                            c,
                            h_start:h_start + kernel_size,
                            w_start:w_start + kernel_size
                        ]
                        output[n, c, i, j] = np.sum(receptive_field * kernel)


    return EdgeTensor(output, x.layout)