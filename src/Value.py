from typing import Union

class Value:
    """Class that stores a single value and it's gradient. Works 'similarly' to PyTorch's tensors."""

    def __init__(self, data: Union[int, float]):
        self.data = data
        self.grad = 0.0

    def __add__(self, other: Union[int, float, 'Value']) -> 'Value':
        other = other if isinstance(other, Value) else Value(other)

        output = Value(self.data + other.data)

        return output

    def __radd__(self, other: Union[int, float, 'Value']) -> 'Value':
        return self.__add__(other)

    def __mul__(self, other: Union[int, float, 'Value']) -> 'Value':
        other = other if isinstance(other, Value) else Value(other)

        output = Value(self.data * other.data)

        return output

    def __repr__(self):
        return f"Value(data={self.data})"


if __name__ == "__main__":
    a = Value(1)
    b = Value(3)

    print(f"a + b = {a + b}")
    print(f"a * b = {a * b}")
