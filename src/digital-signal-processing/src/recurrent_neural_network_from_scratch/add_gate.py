import numpy as np


# This is adding node in Computational Graph
class AddGate:

    def forward(
            self,
            x1,
            x2
    ):
        return x1 + x2
    

    def backward(
            self,
            x1,
            x2,
            dz
    ):
        # create 1 tensors values have shape like x1,x2
        dx1 = dz * np.ones_like(x1)
        dx2 = dz * np.ones_like(x2)
        return dx1, dx2