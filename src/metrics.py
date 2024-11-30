from math import sqrt, log

from typing import List

from Value import Value


def categorical_cross_entropy(y_true: List[float], y_pred: List[float]) -> Value:
    """
    Calculates the categorical cross-entropy loss.
    """
    y_pred = [max(y_hat, 1e-15) for y_hat in y_pred]
    
    loss = -sum((y * log(y_hat) for y, y_hat in zip(y_true, y_pred)))
    
    return Value(loss)
