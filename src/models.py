from typing import Callable, List

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from metrics import sse
from nn import Layer, Module, tanh
from Value import Value


class MLP(Module):
    """Multi-layer perceptron class; a fully-connected neural network."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: List,
        activation_fn: Callable = lambda x: x,
        regression: bool = False,
    ):
        sz = [n_inputs] + n_outputs

        self.layers = [
            Layer(
                sz[i],
                sz[i + 1],
                activation_fn=(  # if it is a regression problem, the last layer shouldn't have an activation function
                    activation_fn
                    if not regression or i < len(n_outputs) - 1
                    else lambda x: x
                ),
            )
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
        n_samples=1200,
        n_features=3,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )
    xs_train, xs_test, ys_train, ys_test = train_test_split(xs, ys, test_size=0.25)

    EPOCHS = 50
    LR = 0.001

    # Training loop
    for epoch in range(EPOCHS):
        y_pred_train = [model(x) for x in xs_train]

        loss = sse(y_pred_train, ys_train)

        for p in model.parameters():
            p.grad = 0

        loss.backward()

        for p in model.parameters():
            p.data -= LR * p.grad

        if epoch % 5 == 0:
            print(f"Epoch {epoch} Loss: {loss.data}")

    # Testing loop
    y_pred_test = [model(x) for x in xs_test]
    y_pred_labels = [1 if y.data > 0.5 else 0 for y in y_pred_test]  # for tanh

    # Calculate precision and recall
    precision = precision_score(ys_test, y_pred_labels, pos_label=1)
    recall = recall_score(ys_test, y_pred_labels, pos_label=1)
    f1 = f1_score(ys_test, y_pred_labels, pos_label=1)

    print(f"Final Precision: {precision}")
    print(f"Final Recall: {recall}")
    print(f"Final F1: {f1}")

    cm = confusion_matrix(ys_test, y_pred_labels)
    print("Confusion Matrix:")
    print(cm)

    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(ys_test)), ys_test, color="blue", label="Ground Truth")
    plt.scatter(
        range(len(y_pred_labels)),
        y_pred_labels,
        color="red",
        label="Predictions",
        marker="x",
    )
    plt.legend()
    plt.show()
