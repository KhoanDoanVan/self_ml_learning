import numpy as np
from typing import Tuple, Optional
from enum import Enum


class TensorLayout(Enum):
    """
    Tensor memory layout formats
    """
    NCHW = "NCHW" # Batch, Channels, Height, Width
    NHWC = "NHWC" # Batch, Height, Width, Channels



class EdgeTensor:
    """
    Layout-Aware tensor for edge inference

    Maintains explicit layout information and provides memory-contiguous
    operations critical for CPU cache efficiency
    """

    def __init__(self, data: np.ndarray, layout: TensorLayout):
        if not data.flags["C_CONTIGUOUS"]:
            data = np.ascontiguousarray(data)

        self.data = data
        self.layout = layout
        self._validate_shape(data)


    def _validate_shape(self):
        if len(self.data.shape) != 4:
            raise ValueError(f"Expected 4D Tensor, go shape {self.data.shape}")
        

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return self.data.shape
    
    
    @property
    def batch_size(self) -> int:
        return self.data.shape[0]
    

    @property
    def channels(self) -> int:
        if self.layout == TensorLayout.NCHW:
            return self.data.shape[1]
        else:
            return self.data.shape[3]
        

    @property
    def height(self) -> int:
        if self.layout == TensorLayout.NCHW:
            return self.data.shape[2]
        else:
            return self.data.shape[1]
        

    @property
    def width(self) -> int:
        if self.layout == TensorLayout.NCHW:
            return self.data.shape[3]
        else:
            return self.data.shape[2]
        

    def get_nchw_shape(self) -> Tuple[int, int, int, int]:
        return (self.batch_size, self.channels, self.height, self.width)
    

    def to_layout(self, target_layout: TensorLayout) -> 'EdgeTensor':
        if self.layout == target_layout:
            return self
        
        # Transpose axes to convert between layouts
        if self.layout == TensorLayout.NCHW and target_layout == TensorLayout.NHWC:
            transposed = np.transpose(self.data, (0, 2, 3, 1))
        else:
            transposed = np.transpose(self.data, (0, 3, 1 ,2))

        # Force contiguous memory layout - critical for performance
        contigous = np.ascontiguousarray(transposed)

        return EdgeTensor(contigous, target_layout)
    

    def copy(self) -> 'EdgeTensor':
        return EdgeTensor(self.data.copy, self.layout)
    

    def __repr__(self) -> str:
        nchw_shape = self.get_nchw_shape()
        return (f"EdgeTensor(shape={self.shape}, layout={self.layout.value}, "
                f"NCHW={nchw_shape}, contiguous={self.data.flags['C_CONTIGUOUS']})")