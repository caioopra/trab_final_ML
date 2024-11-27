from typing import Union


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

    def __add__(self, other: Union[int, float, "Value"]) -> "Value":
        other = other if isinstance(other, Value) else Value(other)

        output = Value(self.data + other.data, operation="+", children=(self, other))

        return output

    def __radd__(self, other: Union[int, float, "Value"]) -> "Value":
        return self.__add__(other)

    def __mul__(self, other: Union[int, float, "Value"]) -> "Value":
        other = other if isinstance(other, Value) else Value(other)

        output = Value(self.data * other.data, operation="*", children=(self, other))

        return output

    def __rmul__(self, other: Union[int, float, "Value"]) -> "Value":
        return self.__mul__(other)

    def __pow__(self, other: Union[int, float]) -> "Value":
        is_int_or_float = isinstance(other, (int, float))
        assert is_int_or_float

        output = Value(self.data**other, operation=f"**{other}", children=(self,))

        return output

    def __sub__(self, other: Union[int, float, "Value"]) -> "Value":
        return self + (-other)

    def __rsub__(self, other: Union[int, float, "Value"]) -> "Value":
        return self.__sub__(other)

    def __neg__(self):
        return self * -1

    def __repr__(self):
        return f"Value(data={self.data})"

    def backward(self):
        """Runs the backpropagation algorithm to calculate the gradients of the values. Based on PyTorch's autograd."""
        topology = []
        visited_nodes = set()

        # auxiliar function to build the network topology (recursively)
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
        print("Built topology: ", topology)
        # TODO: calculation of each node's gradient


if __name__ == "__main__":
    a = Value(1)
    b = Value(3)

    print(f"{1+a=}, {a+1=}, {a+b=}, {a*b=}, {a-b=}, {b - 1}, {3 * b}")
    a.backward()
    c = a + b
    c.backward()
    d = c * b
    d.backward()

