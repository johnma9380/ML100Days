import numpy as np
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_absolute_percentage_error as MAPE

def regression_report(y_true, pred, verbose = False):
    mse = MSE(y_true, pred)
    mae = MAE(y_true, pred)
    rmse = np.sqrt(mse)
    mape = MAPE(y_true, pred)
    if verbose:
        print(f'mse = {mse:.4f}')
        print(f'mae = {mae:.4f}')
        print(f'rmse = {rmse:.4f}')
        print(f'mape = {mape:.4f}')
    re_list = [mse, mae,rmse,mape]

    return re_list



def dection_datatype(total):
    print(set(total.dtypes.values))
    int_features = []
    float_features = []
    object_features = []

    # .dtypes(欄位類型), .columns(欄位名稱) 是 DataFrame 提供的兩個方法
    for dtype, feature in zip(total.dtypes, total.columns):
        if dtype == 'float64':
            float_features.append(feature)
        elif dtype == 'int64':
            int_features.append(feature)
        else: # dtype == 'object':
            object_features.append(feature)

    print(f'{len(int_features)} Integer Features : {int_features}\n')
    print(f'{len(float_features)} Float Features : {float_features}\n')
    print(f'{len(object_features)} Object Features : {object_features}')
    return int_features, float_features, object_features