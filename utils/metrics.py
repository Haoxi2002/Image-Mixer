import numpy as np


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def mean_var_mae(pred, true):
    bs, pred_len, features = pred.shape
    mae = np.mean(np.abs(pred - true), axis=1)
    mae = np.mean(mae, axis=1)
    mean_mae = np.mean(mae, axis=0)
    var_mae = np.var(mae, axis=0)

    bottom_20_indices = np.argsort(true, axis=1)[:, :pred_len // 5, :]
    top_20_indices = np.argsort(true, axis=1)[:, -(pred_len // 5):, :]

    top_20_true = np.take_along_axis(true, top_20_indices, axis=1)
    bottom_20_true = np.take_along_axis(true, bottom_20_indices, axis=1)
    top_20_pred = np.take_along_axis(pred, top_20_indices, axis=1)
    bottom_20_pred = np.take_along_axis(pred, bottom_20_indices, axis=1)

    top_mae = np.mean(np.abs(top_20_true - top_20_pred))
    bottom_mae = np.mean(np.abs(bottom_20_true - bottom_20_pred))
    peak_mae = (top_mae + bottom_mae) / 2
    return mean_mae, var_mae, top_mae, bottom_mae, peak_mae


def mean_var_mse(pred, true):
    bs, pred_len, features = pred.shape
    mse = np.mean((pred - true) ** 2, axis=1)
    mse = np.mean(mse, axis=1)
    mean_mse = np.mean(mse, axis=0)
    var_mse = np.var(mse, axis=0)
    bottom_20_indices = np.argsort(true, axis=1)[:, :pred_len // 5, :]
    top_20_indices = np.argsort(true, axis=1)[:, -(pred_len // 5):, :]

    top_20_true = np.take_along_axis(true, top_20_indices, axis=1)
    bottom_20_true = np.take_along_axis(true, bottom_20_indices, axis=1)
    top_20_pred = np.take_along_axis(pred, top_20_indices, axis=1)
    bottom_20_pred = np.take_along_axis(pred, bottom_20_indices, axis=1)

    top_mse = np.mean((top_20_true - top_20_pred) ** 2)
    bottom_mse = np.mean((bottom_20_true - bottom_20_pred) ** 2)
    peak_mse = (top_mse + bottom_mse) / 2
    return mean_mse, var_mse, top_mse, bottom_mse, peak_mse


def metric(pred, true):
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    mean_mse, var_mse, top_mse, bottom_mse, peak_mse = mean_var_mse(pred, true)
    mean_mae, var_mae, top_mae, bottom_mae, peak_mae = mean_var_mae(pred, true)
    return mean_mse, var_mse, top_mse, bottom_mse, peak_mse, mean_mae, var_mae, top_mae, bottom_mae, peak_mae, rmse, mape, mspe
