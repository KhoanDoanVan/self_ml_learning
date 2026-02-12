from tensor import TensorLayout, EdgeTensor
from blocks.inverted_residual import InvertedResidual


class BottleneckStack:

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_blocks: int,
            stride: int = 1,
            expand_ratio: int = 6,
            layout: TensorLayout = TensorLayout.NHWC
    ):
        self.blocks = []

        # First block may have stride > 1
        self.blocks.append(
            InvertedResidual(
                in_channels, out_channels, stride, expand_ratio, layout
            )
        )

        # Remaining blocks have stride = 1
        for _ in range(1, num_blocks):
            self.blocks.append(
                InvertedResidual(
                    out_channels, out_channels, 1, expand_ratio, layout
                )
            )


    def forward(self, x: EdgeTensor) -> EdgeTensor:
        for block in self.blocks:
            x = block.forward(x)
        return x
    

    def __repr__(self) -> str:
        return f"BottleneckStack(num_blocks={len(self.blocks)})"