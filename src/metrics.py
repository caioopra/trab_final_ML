from typing import List
from Value import Value
from sklearn.metrics import mean_squared_error, root_mean_squared_error


def mse(y_true: List[float], y_pred: List[float]) -> Value:
    """
    Calculates mean squared error, given two lists of values
    """
    mse_val = sum([(y - y_hat) ** 2 for y, y_hat in zip(y_true, y_pred)], Value(0.0))
    mse_val = mse_val / len(y_true)

    if isinstance(mse_val, Value):
        return mse_val

    return Value(data=mse_val)


def rmse(y_true: List[float], y_pred: List[float]) -> Value:
    """
    Calculates root mean squared error, given two lists of values
    """
    mse_val = sum([(y - y_hat) ** 2 for y, y_hat in zip(y_true, y_pred)], Value(0.0))
    mse_val = mse_val / len(y_true)

    rmse_val = mse_val ** (0.5)

    if isinstance(rmse_val, Value):
        return rmse_val

    return Value(rmse_val)


def sse(y_true: List[float], y_pred: List[float]) -> Value:
    """
    Calculates sum of squared errors, given two lists of values
    """
    return Value(sum((yout - ygt) ** 2 for ygt, yout in zip(y_true, y_pred)))


if __name__ == "__main__":
    y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_pred = [1.1, 2.1, 2.9, 4.2, 5.2]

    print(mse(y_true, y_pred))
    print(rmse(y_true, y_pred))
    print(sse(y_true, y_pred))
    mse_sklearn = mean_squared_error(y_true, y_pred)
    rmse_sklearn = root_mean_squared_error(y_true, y_pred)
    sse_sklearn = mse_sklearn * len(y_true)

    print(f"Scikit-learn MSE: {mse_sklearn}")
    print(f"Scikit-learn RMSE: {rmse_sklearn}")
    print(f"Scikit-learn SSE: {sse_sklearn}")
