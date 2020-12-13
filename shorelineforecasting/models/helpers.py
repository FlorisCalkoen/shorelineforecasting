import os
import pandas as pd
import numpy as np
import torch

from sklearn import preprocessing
from torch.utils.data import DataLoader, TensorDataset
from shorelineforecasting.utils.forecasting_metrics import evaluate


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def configure_torch(seed=42):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device is:", device)
    seed_everything(seed)


def min_max_scale_columns(df, ranges=(-1, 1)):
    x = df.values
    scaler = preprocessing.MinMaxScaler(ranges)
    scaled = scaler.fit_transform(x)
    return pd.DataFrame(scaled, columns=df.columns, index=df.index), scaler


def ts_scaler(ts, return_as_df=False, ranges=(-1, 1)):
    x = ts.values.reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler(ranges)
    scaled = min_max_scaler.fit_transform(x).flatten()
    if return_as_df is True:
        return pd.DataFrame(scaled, columns=['dist'], index=ts.index)
    return pd.Series(scaled, index=ts.index)


def min_max_scale_df(df):
    x = df.values
    return pd.DataFrame((x - x.min()) / (x.max() - x.min()), columns=df.columns, index=df.index)


def min_max_scale_dfwise(df, ranges=(-1, 1)):
    x = df.values
    scaler = preprocessing.MinMaxScaler(ranges)
    reshaped = x.reshape([-1, 1])
    scaled = scaler.fit_transform(reshaped)
    scaled = scaled.reshape(x.shape)
    return pd.DataFrame(scaled, columns=df.columns, index=df.index), scaler


def transform_alike(df, scaler):
    x = df.values
    scaled = scaler.transform(x)
    return pd.DataFrame(scaled, columns=df.columns, index=df.index), scaler


def split_df(df, ratio, transpose=False):
    """Input dataframe and return 2 DataFrames which were randomly split."""

    if transpose is True:
        df = df.T

    shuffled_idx = np.random.permutation(range(len(df)))
    split_idx = int(len(df) * ratio)
    split1 = df.iloc[shuffled_idx[:split_idx]]
    split2 = df.iloc[shuffled_idx[split_idx:]]

    if transpose is True:
        return split1.T, split2.T

    return split1, split2



def get_scaled_splits(timeframe: pd.DataFrame, ratio=.8):
    train_raw, test_raw = split_df(timeframe, ratio=ratio)
    train, train_scaler = min_max_scale_dfwise(train_raw)
    test, test_scaler = transform_alike(test_raw, train_scaler)
    train, val = split_df(train, ratio=ratio)

    return train, val, test, test_raw, test_scaler


# def to_tensor_dataset(df, train_window):
#     # split at train window
#     features = df[:train_window].T
#     targets = df[train_window:].T
#
#     # cast to pt tensor
#     features = torch.tensor(features.values)
#     targets = torch.tensor(targets.values)
#
#     return TensorDataset(features, targets)


def to_tensor_dataset(tf, prediction_length, transpose=False):
    if transpose is True:
        tf = tf.T
    features = torch.tensor(tf.iloc[:, :-prediction_length].values)
    targets = torch.tensor(tf.iloc[:, -prediction_length:].values)
    return TensorDataset(features, targets)


def to_torch_dataloaders(train_ds, validation_ds, test_ds, prediction_length, batch_size):
    dataloaders = {
        'train': DataLoader(to_tensor_dataset(train_ds, prediction_length), batch_size=batch_size, shuffle=True, drop_last=True),
        'val': DataLoader(to_tensor_dataset(validation_ds, prediction_length), batch_size=batch_size, shuffle=True, drop_last=True),
        'test': DataLoader(to_tensor_dataset(test_ds, prediction_length), batch_size=batch_size, shuffle=False, drop_last=True),
    }
    trainiter = iter(dataloaders['train'])
    features, labels = next(trainiter)
    print(f"Dataloader: train shape: {features.shape}, target shape: {labels.shape}")
    return dataloaders


def get_lstm_configs(tsdf, configs):
    lstm_configs = configs['lstm']
    lstm_configs['output_size'] = len(tsdf) - lstm_configs['train_window']
    lstm_configs['forecast_size'] = lstm_configs['train_window'] + lstm_configs['output_size'] * lstm_configs['horizon']
    return lstm_configs


def to_dataframe(forecast_tensor, model_configs, target, inverse_scaling=True, test_scaler=None, transpose=False):
    df = pd.DataFrame(forecast_tensor.numpy()[:, :, 0])

    if transpose is True:
        df = df.T

    if inverse_scaling is True:
        df = pd.DataFrame(test_scaler.inverse_transform(df))

    if transpose is True:
        df.index = pd.RangeIndex(start=1984, stop=1984 + model_configs['forecast_size'], step=1)
        df.columns = target.columns[:len(df.columns)]

    else:
        df.index = target.index[:len(df)]
        df.columns = pd.RangeIndex(start=1984, stop=1984 + model_configs['forecast_size'], step=1)

    return df


def evaluate_performance(s_true, s_pred, model_configs):
    true_y = s_true[
             model_configs['train_window']: model_configs['train_window'] + model_configs['output_size']].to_numpy()
    pred_y = s_pred[
             model_configs['train_window']: model_configs['train_window'] + model_configs['output_size']].to_numpy()
    metrics = evaluate(true_y, pred_y, metrics=model_configs['evaluation'])
    return list(metrics.values())


if __name__ == "__main__":
    from shorelineforecasting.definitions import ROOT_DIR
    print(ROOT_DIR)