from tensor import EdgeTensor
import numpy as np

def relu(x: EdgeTensor) -> EdgeTensor:
    return EdgeTensor(np.maximum(0, x.data), x.layout)
