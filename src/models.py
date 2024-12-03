from typing import Callable, List

from nn import Module, Neuron, Layer
from Value import Value


class MLP(Module):
    """Multi-layer perceptron class; a fully-connected neural network."""

    def __init__(
        self, n_inputs: int, n_outputs: List, activation_fn: Callable = lambda x: x
    ):
        sz = [n_inputs] + n_outputs

        self.layers = [
            Layer(sz[i], sz[i + 1], activation_fn=activation_fn)
            for i in range(len(n_outputs))
        ]

        self.activation_fn = activation_fn

    def __call__(self, x: List[Value]) -> List[Value]:
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self) -> List:
        return [param for layer in self.layers for param in layer.parameters()]

    def __repr__(self) -> str:
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"


# if __name__ == "__main__":
    # TODO: test MLP implementation
    # model = MLP(3, [4, 4, 1], activation=tanh)
