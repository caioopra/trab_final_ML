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
    def __call__(self, x: List[Value]) -> Value | List[Value]: ...


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

class Layer(Module):
    def __init__(self, n_inputs: int, n_outputs: int, activation_fn: Callable = lambda x: x):
        self.neurons = [
            Neuron(n_inputs, activation=activation_fn) for _ in range(n_outputs)
        ]
        self.activation = activation_fn

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    

def relu(input: list, derivative: bool = False) -> list:
    if isinstance(input[0], Value):
        new_nodes = []
        for x in input:
            new = Value(max(0, x.data), (x,), "relu")

            def _backward():
                new.grad += 1 if x.data > 0 else 0

            new._backward = _backward

            new_nodes.append(new)

        return new_nodes

    if derivative:
        return [1 if x > 0 else 0 for x in input]

    return [max(0, x) for x in input]
