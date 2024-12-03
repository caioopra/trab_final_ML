from typing import Callable, List

from nn import Module, Neuron, Layer, tanh
from Value import Value

from metrics import sse

from sklearn.metrics import precision_score, recall_score, f1_score


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


if __name__ == "__main__":
    model = MLP(3, [4, 4, 1], activation_fn=tanh)

    xs = [[3, 2, -1], [0.5, 1, 1], [1, 0, -1], [0, 3, -1]]
    ys = [1, -1, -1, 1]

    EPOCHS = 100
    LR = 0.01

    for epoch in range(EPOCHS):
        y_pred = [model(x) for x in xs]

        loss = sse(y_pred, ys)

        for p in model.parameters():
            p.grad = 0

        loss.backward()

        for p in model.parameters():
            p.data -= LR * p.grad

        if epoch % 5 == 0:
            print(f"Epoch {epoch} Loss: {loss.data}")

    print(y_pred)
    y_pred_labels = [1 if y.data > 0 else -1 for y in y_pred]

    # Calculate precision and recall
    precision = precision_score(ys, y_pred_labels, pos_label=1)
    recall = recall_score(ys, y_pred_labels, pos_label=1)

    print(f"Final Precision: {precision}")
    print(f"Final Recall: {recall}")
    print(y_pred)
