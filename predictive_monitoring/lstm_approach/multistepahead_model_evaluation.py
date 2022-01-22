from math import sqrt
import numpy as np

from sklearn.metrics import mean_squared_error

def rmse_perc(y_true, y_pred):
    loss = np.sqrt(mean_squared_error(y_true, y_pred))
    loss = loss / np.mean(y_true)
    
    return loss

def evaluate_forecast(actual, predicted, type='rmse'):
    scores = list()
    # calculate error for each day
    for i in range(actual.shape[1]):
        if type == 'rmse':
            # calculate mse
            mse = mean_squared_error(actual[:, i], predicted[:, i])
            # calculate rmse
            rmse = sqrt(mse)
            # store
            scores.append(rmse)
        elif type == 'rmse_p':
            rmse_p = rmse_perc(actual[:, i], predicted[:, i])
            scores.append(rmse_p)
        elif type == 'true_val':
            true_val = (predicted[:, i] - actual[:, i]).mean()
            scores.append(true_val)
        elif type == 'true_val_max':
            true_val_max = ((predicted[:, i] - actual[:, i])/actual[:, i]).max()
            scores.append(true_val_max*100)
        elif type == 'true_val_min':
            true_val_min = ((predicted[:, i] - actual[:, i])/actual[:, i]).min()
            scores.append(true_val_min*100)
        elif type == 'true_val_ratio_pos':
            t = (predicted[:, i] - actual[:, i])
            true_val_ratio_pos = (t >0).sum() / t.shape[0]
            scores.append(true_val_ratio_pos)
    # calculate overall error
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores


def walk_forward_validation(model, test_x, test_y, test_set, scaler_transf, n_out):
    predictions = list()
    real_values = list()
    start_val = 0
    for i, test_input_x in enumerate(test_x):
        test_input_x = test_input_x.reshape(
            (1, test_input_x.shape[0], test_input_x.shape[1])
        )
        yhat = model.predict(test_input_x, verbose=0)
        yhat_reshaped = np.array(yhat[0]).reshape((n_out, 1))
        end_val = start_val + n_out
        test_window = test_set[start_val:end_val, :-1]
        if test_window.shape[0] == yhat_reshaped.shape[0]:
            yhat_matrix = np.append(test_window, yhat_reshaped, axis=1)
            yhat_original_scale = scaler_transf.inverse_transform(yhat_matrix)
            inv_yhat = yhat_original_scale[:, -1]
            y_real = test_y[i, :]
            y_real_reshaped = y_real.reshape((n_out, 1))
            y_real_matrix = np.append(test_window, y_real_reshaped, axis=1)
            y_real_original_scale = scaler_transf.inverse_transform(y_real_matrix)
            inv_y = y_real_original_scale[:, -1]
            predictions.append(inv_yhat)
            real_values.append(inv_y)
        start_val += 1
    predictions = np.array(predictions)
    real_values = np.array(real_values)
    score, scores = evaluate_forecast(real_values, predictions)
    return score, scores, predictions, real_values
