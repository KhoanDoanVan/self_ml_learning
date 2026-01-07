
import numpy as np



class MLP(object):

    def __init__(
            self,
            num_inputs,
            hidden_layers,
            num_outputs
    ):
        """
        Constructor for Multiple Layer Perceptron
        :param num_inputs: int
        :param hidden_layers: list int
        :param num_outputs: int
        """

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs


        # Create a generic representation of the layers
        layers = [num_inputs] + hidden_layers + [num_outputs]

        print("layers: ", layers)

        # Create random connection weights for the layers
        weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            weights.append(w)

        self.weights = weights


    def forward(
            self,
            inputs
    ):
        """
        Computes forward propagation of the network based on input signals
        :param inputs: ndarray
        """

        # the input layer activation is just the input itself
        activations = inputs

        # iterate through the network layers
        for w in self.weights:

            net_inputs = np.dot(activations, w)

            activations = self._sigmoid(net_inputs)

        return activations


    def _sigmoid(
            self,
            x
    ):
        """
        Sigmoid activation function
        :param x: float
        :return: float
        """

        return 1 / (1 + np.exp(-x))


if "__main__" == __name__:

    mlp = MLP(
        num_inputs=3,
        hidden_layers=[3, 5],
        num_outputs=2
    )

    inputs = np.random.rand(mlp.num_inputs)

    print("inputs: ", inputs)

    outputs = mlp.forward(inputs)

    print("outputs: ", outputs)