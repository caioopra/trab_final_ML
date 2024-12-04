from typing import Union
import math

class Value:
    """Class that stores a single value and it's gradient. Works 'similarly' to PyTorch's tensors."""

    def __init__(
        self,
        data: Union[int, float],
        operation: str = "",
        children: Union[tuple["Value"], tuple["Value", "Value"], tuple] = (),
    ):
        self.data = data
        self.grad = 0.0

        self._operation = operation
        self._previous = set(children)

        self._backward = lambda: None  # default backward function

    def __add__(self, other: Union[int, float, "Value"]) -> "Value":
        """Overloads the '+' operator to allow for Value + Value operations."""
        other = other if isinstance(other, Value) else Value(other)

        output = Value(self.data + other.data, operation="+", children=(self, other))

        def _backward():  # backpropagation function; partial derivative of the operation
            self.grad += 1.0 * output.grad
            other.grad += 1.0 * output.grad

        output._backward = _backward

        return output

    def __radd__(self, other: Union[int, float, "Value"]) -> "Value":
        return self.__add__(other)

    def __mul__(self, other: Union[int, float, "Value"]) -> "Value":
        """Overloads the '*' operator to allow for Value * Value operations."""
        other = other if isinstance(other, Value) else Value(other)

        output = Value(self.data * other.data, operation="*", children=(self, other))

        def _backward():  # partial derivative of the operation
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad

        output._backward = _backward

        return output

    def __rmul__(self, other: Union[int, float, "Value"]) -> "Value":
        return self.__mul__(other)

    def __pow__(self, other: Union[int, float]) -> "Value":
        """Overloads the '**' operator to allow for Value ** Value operations."""
        is_int_or_float = isinstance(other, (int, float))
        assert is_int_or_float

        output = Value(self.data**other, operation=f"**{other}", children=(self,))

        def _backward():
            self.grad += other * self.data ** (other - 1)
            self.grad *= output.grad

        output._backward = _backward

        return output

    def log(self) -> "Value":
        assert self.data > 0

        output = Value(math.log(self.data), operation="log", children=(self,))

        def _backward():
            self.grad += output.grad * (1 / self.data)  # Corrigido para apenas multiplicar pelo gradiente da saída

        output._backward = _backward
        return output
    
    def __sub__(self, other: Union[int, float, "Value"]) -> "Value":
        return self + (-other)

    def __rsub__(self, other: Union[int, float, "Value"]) -> "Value":
        return self.__sub__(other)

    def __neg__(self):
        return self * -1

    def __repr__(self) -> str:
        """Returns a string representation of the Value object."""
        return f"Value(data={self.data}, grad={self.grad})"

    def backward(self):
        """Runs the backpropagation algorithm to calculate the gradients of the values. Based on PyTorch's autograd."""
        topology = []
        visited_nodes = set()

        # auxiliary function to build the network topology (recursively)
        def _build_topology(node: "Value"):
            if node in visited_nodes:
                topology.append(node)
                return

            visited_nodes.add(node)

            for prev in node._previous:
                _build_topology(prev)

            topology.append(node)

        _build_topology(self)

        self.grad = 1.0  # Start with the gradient of the output node

        # run the backpropagation calculatios
        for node in topology[::-1]:
            node._backward()

    def __truediv__(self, other: Union[int, float, "Value"]) -> "Value":
        other = other if isinstance(other, Value) else Value(other)

        return self * other**-1

    def __rtruediv__(self, other: Union[int, float, "Value"]) -> "Value":
        return other * self**-1


if __name__ == "__main__":
    a = Value(-4)
    b = Value(5)
    c = Value(9)
    d = a + b
    e = d * c
    f = Value(-3)
    g = e * f
    g.backward()

    [print(node) for node in [a, b, c, d, e, f, g]]
