from math import log
from typing import List
from Value import Value
from typing import Union
import numpy as np
from sklearn.metrics import log_loss
from math import log

def categorical_cross_entropy(
    y_true: List[int], y_pred: List[List[Value]], value: bool = False
) -> Union[Value, float]:
    """
    Computes the categorical cross-entropy loss, given the true labels and predicted probabilities.
    """
    loss = Value(0.0)

    for yt, yp in zip(y_true, y_pred):
        prob = yp[yt]
        loss += -Value(log(prob.data))  
    
    loss = loss / len(y_true)

    if value:
        return loss.data
    return loss


def mse(y_true: float, y_pred: float) -> Value:
    """
    Calculates mean squared error, given two lists of values
    """
    loss = (y_true - y_pred) ** 2

    return Value(loss)


def rmse(y_true: float, y_pred: float) -> Value:
    """
    Calculates root mean squared error, given two lists of values
    """
    loss = ((y_true - y_pred) ** 2) ** 0.5

    return Value(loss)

if __name__ == "__main__":
    y_true = [0, 1, 2]  
    y_pred = [
        [Value(0.1), Value(0.7), Value(0.2)],  
        [Value(0.4), Value(0.4), Value(0.2)], 
        [Value(0.2), Value(0.2), Value(0.6)],  
    ]   

    loss_custom = categorical_cross_entropy(y_true, y_pred)
    loss_custom.backward()

    for pred in y_pred:
        print([v.grad for v in pred])

    y_pred_sklearn = np.array([[0.1, 0.7, 0.2], [0.4, 0.4, 0.2], [0.2, 0.2, 0.6]])  
    loss_sklearn = log_loss(y_true, y_pred_sklearn)

    print(f'Loss customizada: {loss_custom}')
    print(f'Loss sklearn: {loss_sklearn}')
