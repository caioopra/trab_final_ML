from abc import ABC, abstractmethod

from typing import Callable, List
from random import uniform

from Value import Value


class Module(ABC):
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    @abstractmethod
    def parameters(self) -> List[Value]: ...

    @abstractmethod
    def __call__(self, x: List[Value]) -> Value: ...


class Neuron(Module):
    """Class representing a single neuron from a neural network.

    Args:
        n_inputs (int): size of the input vector for the neuron.
        activation_fn (Callable, optional): activation function for the ouput of the neuro.
            Defaults to `lambda x: x` (which is the identity function).
    """

    def __init__(self, n_inputs: int, activation_fn: Callable = lambda x: x):
        self.weights = [Value(uniform(-1, 1)) for _ in range(n_inputs)]
        self.bias = Value(uniform(-1, 1))

        self.activation_fn = activation_fn

    def __call__(self, x: List[Value]) -> Value:
        """Computes the output of the neuron for a ginve vector of Value's"""
        output = self.bias
        for weight_i, x_i in zip(self.weights, x):
            output += weight_i * x_i

        return self.activation_fn(output)

    def parameters(self) -> List[Value]:
        """Returns the parameters of the neuron (vector with the weights and the bias as last element)"""
        return self.weights + [self.bias]
