from typing import Callable, List

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score)

from metrics import sse
from nn import Layer, Module, relu, sigmoid, tanh
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


if __name__ == "__main__":
    model = MLP(3, [4, 4, 1], activation_fn=tanh)

    xs, ys = make_classification(
        n_samples=100,
        n_features=3,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )

    EPOCHS = 50 
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

    unique_preds = set(y_pred)
    print(f"Unique predictions: {unique_preds}")
    y_pred_labels = [1 if y.data > 0.5 else 0 for y in y_pred]  # for tanh 

    # Calculate precision and recall
    precision = precision_score(ys, y_pred_labels, pos_label=1)
    recall = recall_score(ys, y_pred_labels, pos_label=1)
    f1 = f1_score(ys, y_pred_labels, pos_label=1)

    print(f"Final Precision: {precision}")
    print(f"Final Recall: {recall}")
    print(f"Final F1: {f1}")

    cm = confusion_matrix(ys, y_pred_labels)
    print("Confusion Matrix:")
    print(cm)

    print("Layer1 ", model.layers[0])
    print("Layer2 ", model.layers[1])
    print("Layer 1 neurons ", model.layers[0].neurons)

    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(ys)), ys, color="blue", label="Ground Truth")
    plt.scatter(
        range(len(y_pred_labels)),
        y_pred_labels,
        color="red",
        label="Predictions",
        marker="x",
    )
    plt.xlabel("Sample Index")
    plt.ylabel("Class Label")
    plt.title("Ground Truth vs Predictions")
    plt.legend()
    plt.show()
