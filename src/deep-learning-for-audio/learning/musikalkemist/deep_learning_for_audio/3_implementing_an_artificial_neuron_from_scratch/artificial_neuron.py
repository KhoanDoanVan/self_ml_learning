import math

def sigmoid(
        h
):
    return 1.0 / (1.0 + math.exp(-h))


def activate(
        inputs,
        weights
):

    h = 0

    for x, y in zip(inputs, weights):
        h += x * y

    return sigmoid(h)


if "__main__" == __name__:

    inputs = [0.5, 0.7, 0.2]
    weights = [0.4, 0.3, 0.2]

    output = activate(inputs, weights)

    print(output)