import numpy as np
from scipy.signal import find_peaks


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


def find_peaks_own(x):
    peaks_all = []
    for i in range(x.shape[0]):
        peaks, _ = find_peaks(x[i, :, 0], prominence=np.mean(x[i, :, 0]) / 10)
        peaks = expand(peaks, x.shape[1])
        peaks_all.append(peaks)
    return peaks_all


def expand(x, maxx, offset=5):
    expanded = set()
    for index in x:
        for i in range(-offset, offset + 1):
            if 0 <= index + i < maxx:
                expanded.add(index + i)
    sorted_expanded = sorted(expanded)
    return np.asarray(sorted_expanded)


def mean_var_mae(pred, true):
    bs, pred_len, features = pred.shape
    mae = np.mean(np.abs(pred - true), axis=1)
    mae = np.mean(mae, axis=1)
    mean_mae = np.mean(mae, axis=0)
    var_mae = np.var(mae, axis=0)

    # bottom_20_indices = np.argsort(true, axis=1)[:, :pred_len // 5, :]
    # top_20_indices = np.argsort(true, axis=1)[:, -(pred_len // 5):, :]
    #
    # top_20_true = np.take_along_axis(true, top_20_indices, axis=1)
    # bottom_20_true = np.take_along_axis(true, bottom_20_indices, axis=1)
    # top_20_pred = np.take_along_axis(pred, top_20_indices, axis=1)
    # bottom_20_pred = np.take_along_axis(pred, bottom_20_indices, axis=1)
    #
    # top_mae = np.mean(np.abs(top_20_true - top_20_pred))
    # bottom_mae = np.mean(np.abs(bottom_20_true - bottom_20_pred))

    top_indices = find_peaks_own(true)
    bottom_indices = find_peaks_own(-true)
    top_mae = []
    bottom_mae = []
    for i in range(true.shape[0]):
        top_true = np.take_along_axis(true[i, :, 0], top_indices[i], axis=0)
        bottom_true = np.take_along_axis(true[i, :, 0], bottom_indices[i], axis=0)
        top_pred = np.take_along_axis(pred[i, :, 0], top_indices[i], axis=0)
        bottom_pred = np.take_along_axis(pred[i, :, 0], bottom_indices[i], axis=0)
        top_mae.append(np.mean(np.abs(top_pred - top_true)))
        bottom_mae.append(np.mean(np.abs(bottom_pred - bottom_true)))
    top_mae = np.mean(np.array(top_mae))
    bottom_mae = np.mean(np.array(bottom_mae))
    peak_mae = (top_mae + bottom_mae) / 2
    return mean_mae, var_mae, top_mae, bottom_mae, peak_mae


def mean_var_mse(pred, true):
    bs, pred_len, features = pred.shape
    mse = np.mean((pred - true) ** 2, axis=1)
    mse = np.mean(mse, axis=1)
    mean_mse = np.mean(mse, axis=0)
    var_mse = np.var(mse, axis=0)
    # bottom_20_indices = np.argsort(true, axis=1)[:, :pred_len // 5, :]
    # top_20_indices = np.argsort(true, axis=1)[:, -(pred_len // 5):, :]
    # top_20_true = np.take_along_axis(true, top_20_indices, axis=1)
    # bottom_20_true = np.take_along_axis(true, bottom_20_indices, axis=1)
    # top_20_pred = np.take_along_axis(pred, top_20_indices, axis=1)
    # bottom_20_pred = np.take_along_axis(pred, bottom_20_indices, axis=1)

    top_indices = find_peaks_own(true)
    bottom_indices = find_peaks_own(-true)
    top_mse = []
    bottom_mse = []
    for i in range(true.shape[0]):
        top_true = np.take_along_axis(true[i, :, 0], top_indices[i], axis=0)
        bottom_true = np.take_along_axis(true[i, :, 0], bottom_indices[i], axis=0)
        top_pred = np.take_along_axis(pred[i, :, 0], top_indices[i], axis=0)
        bottom_pred = np.take_along_axis(pred[i, :, 0], bottom_indices[i], axis=0)
        top_mse.append(np.mean((top_pred - top_true) ** 2))
        bottom_mse.append(np.mean((bottom_pred - bottom_true) ** 2))
    top_mse = np.mean(np.array(top_mse))
    bottom_mse = np.mean(np.array(bottom_mse))
    # top_mse = np.mean((top_20_true - top_20_pred) ** 2)
    # bottom_mse = np.mean((bottom_20_true - bottom_20_pred) ** 2)
    peak_mse = (top_mse + bottom_mse) / 2
    return mean_mse, var_mse, top_mse, bottom_mse, peak_mse


def metric(pred, true):
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    mean_mse, var_mse, top_mse, bottom_mse, peak_mse = mean_var_mse(pred, true)
    mean_mae, var_mae, top_mae, bottom_mae, peak_mae = mean_var_mae(pred, true)
    return mean_mse, var_mse, top_mse, bottom_mse, peak_mse, mean_mae, var_mae, top_mae, bottom_mae, peak_mae, rmse, mape, mspe


if __name__ == '__main__':
    pred = np.random.random((4, 288, 1))
    true = pred + np.random.random((4, 288, 1))
    mean_mse, var_mse, top_mse, bottom_mse, peak_mse, mean_mae, var_mae, top_mae, bottom_mae, peak_mae, rmse, mape, mspe = metric(
        pred, true)
    print("Mean MSE:", mean_mse)
    print("Variance MSE:", var_mse)
    print("Top MSE:", top_mse)
    print("Bottom MSE:", bottom_mse)
    print("Peak MSE:", peak_mse)
