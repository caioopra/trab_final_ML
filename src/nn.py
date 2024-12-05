from abc import ABC, abstractmethod

from typing import Callable, List, Union
from random import uniform
from math import exp, tanh as math_tanh

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

    def __repr__(self) -> str:
        return f"Neuron with weights: {[w.data for w in self.weights]} and bias: {self.bias.data}; activation_fn={self.activation_fn}"


class Layer(Module):
    def __init__(
        self, n_inputs: int, n_outputs: int, activation_fn: Callable = lambda x: x
    ):
        self.neurons = [
            Neuron(n_inputs, activation_fn=activation_fn) for _ in range(n_outputs)
        ]
        self.activation_fn = activation_fn

    def __call__(self, x: List[Value]) -> Union[Value, List[Value]]:
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self) -> List[Value]:
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self) -> str:
        return f"Layer with {len(self.neurons)} neurons; activation_fn={self.activation_fn}"


def relu(input: List[Value], derivative: bool = False) -> List[Value]:
    if isinstance(input, Value):
        new = Value(data=max(0, input.data), operation="relu", children=(input,))

        def _backward():
            input.grad += new.grad * (1 if input.data > 0 else 0)

        new._backward = _backward

        return new

    if isinstance(input[0], Value):
        new_nodes = []
        for x in input:
            new = Value(data=max(0, x.data), operation="relu", children=(x,))

            def _backward():
                x.grad += new.grad * (1 if x.data > 0 else 0)

            new._backward = _backward
            new_nodes.append(new)

        return new_nodes

    if derivative:
        return [1 if x > 0 else 0 for x in input]

    return [max(0, x) for x in input]


def sigmoid(input: list, derivative: bool = False) -> Union[List[Value] | Value]:
    """
    Applies the Sigmoid activation function to a list of Value objects.
    """
    if isinstance(input, Value):
        sigmoid_value = 1 / (1 + exp(-input.data))
        new = Value(data=sigmoid_value, operation="sigmoid", children=(input,))

        def _backward():
            input.grad += sigmoid_value * (1 - sigmoid_value) * new.grad

        new._backward = _backward

        return new

    if isinstance(input[0], Value):
        new_nodes = []
        for x in input:
            sigmoid_value = 1 / (1 + exp(-x.data))
            new = Value(data=sigmoid_value, operation="sigmoid", children=(x,))

            def _backward():
                x.grad += sigmoid_value * (1 - sigmoid_value) * new.grad

            new._backward = _backward
            new_nodes.append(new)

        return new_nodes

    if derivative:
        return [1 / (1 + exp(-x)) * (1 - (1 / (1 + exp(-x)))) for x in input]

    return [1 / (1 + exp(-x)) for x in input]


def softmax(input: Union[Value, list], derivative: bool = False) -> Union[Value, list]:
    """
    Applies the Softmax activation function to a Value object or a list of Value objects.
    """
    if isinstance(input, Value):
        exp_value = exp(input.data)
        sum_exp_value = exp_value  # Since it's a single value, sum is the value itself
        softmax_value = exp_value / sum_exp_value

        new = Value(data=softmax_value, operation="softmax", children=(input,))

        def _backward():
            input.grad += new.grad * softmax_value * (1 - softmax_value)

        new._backward = _backward

        return new

    if isinstance(input[0], Value):
        exp_values = [exp(x.data) for x in input]
        sum_exp_values = sum(exp_values)
        softmax_values = [exp(x.data) / sum_exp_values for x in input]

        new_nodes = []

        for i, x in enumerate(input):
            new = Value(data=softmax_values[i], operation="softmax", children=(x,))

            def _backward():
                for j, _ in enumerate(softmax_values):
                    if i == j:
                        x.grad += new.grad * softmax_values[i] * (1 - softmax_values[i])
                    else:
                        x.grad += new.grad * -softmax_values[i] * softmax_values[j]

            new._backward = _backward
            new_nodes.append(new)

        return new_nodes

    exp_values = [exp(x) for x in input]
    sum_exp_values = sum(exp_values)
    softmax_values = [exp(x) / sum_exp_values for x in input]

    if derivative:
        grads = []
        for i in range(len(input)):
            grad = [0] * len(softmax_values)
            for j, _ in enumerate(softmax_values):
                if i == j:
                    grad[i] += softmax_values[i] * (1 - softmax_values[i])
                else:
                    grad[i] += -softmax_values[i] * softmax_values[j]
            grads.append(grad)
        return grads

    return softmax_values


def tanh(input: list, derivative: bool = False) -> Union[List[Value] | Value]:
    """
    Applies the hyperbolic tangent (tanh) activation function to a list of input values.
    """
    if isinstance(input, Value):
        t = math_tanh(input.data)

        new = Value(data=t, operation="tanh", children=(input,))

        def _backward():
            input.grad += (1 - t**2) * new.grad

        new._backward = _backward

        return new

    if isinstance(input[0], Value):
        new_nodes = []
        for x in input:
            t = math_tanh(x.data)
            new = Value(data=t, operation="tanh", children=(x,))

            def _backward():
                x.grad += (1 - t**2) * new.grad

            new._backward = _backward
            new_nodes.append(new)

        return new_nodes

    if derivative:
        return [1 - (math_tanh(x)) ** 2 for x in input]

    return [math_tanh(x) for x in input]


if __name__ == "__main__":
    # simple neuron test
    print("Simple neuron test:")
    n = Neuron(n_inputs=2)
    x = [Value(1), Value(2)]
    y = n(x)
    print(f"Output of the neuron: {y}")
    print(f"Parameters of the neuron: {n.parameters()}\n---------")

    # simple layer test
    print("Simple layer test:")
    l = Layer(n_inputs=2, n_outputs=3)
    x = [Value(1), Value(2)]
    y = l(x)
    print(f"Output of the layer: {y}")
    print(f"Amount of neurons in layer: {len(l.neurons)}")
    print(f"Parameters of the layer: {l.parameters()}\n---------")

    # activation functions test
    print("Activation functions test:")
    x = [Value(0), Value(1), Value(-2), Value(3)]
    print(f"Input: {[v.data for v in x]}")

    print("ReLU:")
    y = relu(x)
    print(f"Output: {[v for v in y]}")

    print("Sigmoid:")
    y = sigmoid(x)
    print(f"Output: {[v for v in y]}")
    print(f"Expected: {[1 / (1 + exp(-v.data)) for v in x]}")

    print("Softmax:")
    y = softmax(x)
    print(f"Output: {[v for v in y]}")
    print(f"Expected: {[exp(v.data) / sum([exp(v.data) for v in x]) for v in x]}")

    print("Tanh:")
    y = tanh(x)
    print(f"Output: {[v.data for v in y]}")
    print(f"Expected: {[math_tanh(v.data) for v in x]}")
