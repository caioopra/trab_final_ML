from typing import Union


class Value:
    """Class that stores a single value and it's gradient. Works 'similarly' to PyTorch's tensors."""

    def __init__(self, data: Union[int, float], operation: str = ""):
        self.data = data
        self.grad = 0.0

        self._operation = operation

    def __add__(self, other: Union[int, float, "Value"]) -> "Value":
        other = other if isinstance(other, Value) else Value(other)

        output = Value(self.data + other.data, operation="+")

        return output

    def __radd__(self, other: Union[int, float, "Value"]) -> "Value":
        return self.__add__(other)

    def __mul__(self, other: Union[int, float, "Value"]) -> "Value":
        other = other if isinstance(other, Value) else Value(other)

        output = Value(self.data * other.data, operation="*")

        return output

    def __rmul__(self, other: Union[int, float, "Value"]) -> "Value":
        return self.__mul__(other)

    def __pow__(self, other: Union[int, float]) -> "Value":
        is_int_or_float = isinstance(other, (int, float))
        assert is_int_or_float

        output = Value(self.data**other, operation=f"**{other}")

        return output

    def __sub__(self, other: Union[int, float, "Value"]) -> "Value":
        return self + (-other)

    def __rsub__(self, other: Union[int, float, "Value"]) -> "Value":
        return self.__sub__(other)

    def __neg__(self):
        return self * -1

    def __repr__(self):
        return f"Value(data={self.data})"


if __name__ == "__main__":
    a = Value(1)
    b = Value(3)

    print(f"{1+a=}, {a+1=}, {a+b=}, {a*b=}, {a-b=}, {b - 1}, {3 * b}")

