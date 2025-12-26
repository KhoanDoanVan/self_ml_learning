import numpy as np



class MultiplyGate:
    
    def forward(
            self,
            W,
            x
    ):
        return np.dot(W, x)
    


    def backward(
            self,
            W,
            x,
            dz
    ):
        
        # dW = ∂L / ∂W = ∂L/∂z · ∂z/∂W
        dW = np.asarray(
            np.dot(
                np.transpose(np.asmatrix(dz)),
                np.asmatrix(x)
            )
        )

        # dx = ∂L / ∂x = ∂L/∂z · ∂z/∂x
        dx = np.dot(
            np.transpose(W),
            dz
        )

        return dW, dx