from typing import List, Union
from Value import Value
from sklearn.metrics import mean_squared_error, root_mean_squared_error


def mse(
    y_true: List[float], y_pred: List[float], value: bool = False
) -> Union[Value, float]:
    """
    Calculates mean squared error, given two lists of values
    """
    mse_val = sum([(y - y_hat) ** 2 for y, y_hat in zip(y_true, y_pred)], Value(0.0))
    mse_val = mse_val / len(y_true)

    if isinstance(mse_val, Value):
        if value:
            return mse_val.data
        return mse_val

    if value:
        return mse_val
    return Value(data=mse_val)


def rmse(
    y_true: List[float], y_pred: List[float], value: bool = False
) -> Union[Value, float]:
    """
    Calculates root mean squared error, given two lists of values
    """
    mse_val = sum([(y - y_hat) ** 2 for y, y_hat in zip(y_true, y_pred)], Value(0.0))
    mse_val = mse_val / len(y_true)

    rmse_val = mse_val ** (0.5)

    if isinstance(rmse_val, Value):
        if value:
            return rmse_val.data
        return rmse_val

    if value:
        return rmse_val

    return Value(rmse_val)


def sse(
    y_true: List[float], y_pred: List[float], value: bool = False
) -> Union[Value, float]:
    """
    Calculates sum of squared errors, given two lists of values
    """
    sse_val = sum((yout - ygt) ** 2 for ygt, yout in zip(y_true, y_pred))

    if isinstance(sse_val, Value):
        if value:
            return sse_val.data
        return sse_val

    if value:
        return sse_val

    return Value(sse_val)


if __name__ == "__main__":
    y_true = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    y_pred = [1.1, 2.1, 2.9, 4.2, 5.2, 5.8, 7.1, 8.3, 9.1, 10.2]

    # Compare the results with scikit-learn's implementation
    mse_val = mse(y_true, y_pred, value=True)
    rmse_val = rmse(y_true, y_pred, value=True)
    sse_val = sse(y_true, y_pred, value=True)

    print(f"Custom MSE: {mse_val}")
    print(f"Custom RMSE: {rmse_val}")
    print(f"Custom SSE: {sse_val}")
    
    mse_sklearn = mean_squared_error(y_true, y_pred)
    rmse_sklearn = root_mean_squared_error(y_true, y_pred)
    sse_sklearn = mse_sklearn * len(y_true)

    assert abs(mse_val - mse_sklearn) < 1e-6, "MSE values do not match!"
    assert abs(rmse_val - rmse_sklearn) < 1e-6, "RMSE values do not match!"
    assert abs(sse_val - sse_sklearn) < 1e-6, "SSE values do not match!"

    print(f"Scikit-learn MSE: {mse_sklearn}")
    print(f"Scikit-learn RMSE: {rmse_sklearn}")
    print(f"Scikit-learn SSE: {sse_sklearn}")
