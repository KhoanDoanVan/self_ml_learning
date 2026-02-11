import numpy as np
from tensor import EdgeTensor, TensorLayout



class LayoutAwareBatchNorm2D:

    def __init__(
            self,
            num_channels: int,
            eps: float = 1e-5
    ):
        self.num_channels = num_channels
        self.eps = eps

        # Learnable parameters
        self.gamma = np.ones(num_channels, dtype=np.float32)
        self.beta = np.zeros(num_channels, dtype=np.float32)

        # Running statistics (for inference)
        self.running_mean = np.zeros(num_channels, dtype=np.float32)
        self.running_var = np.ones(num_channels, dtype=np.float32)


    def forward(self, x: EdgeTensor) -> EdgeTensor:

        if x.layout == TensorLayout.NHWC:
            mean = self.running_mean.reshape(1, 1, 1, -1)
            var = self.running_var.reshape(1, 1, 1, -1)
            gamma = self.gamma.reshape(1, 1, 1, -1)
            beta = self.beta.reshape(1, 1, 1, -1)
        else:
            mean = self.running_mean.reshape(1, -1, 1, 1)
            var = self.running_var.reshape(1, -1, 1, 1)
            gamma = self.gamma.reshape(1, -1, 1, 1)
            beta = self.beta.reshape(1, -1, 1, 1)

        # Normalize
        normalized = (x.data - mean) / np.sqrt(var + self.eps)
        output = gamma * normalized + beta

        return EdgeTensor(output, x.layout)