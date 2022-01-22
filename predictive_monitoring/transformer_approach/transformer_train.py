import os, sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gcd_data_manipulation import load_data, data_aggregation, scale_values, series_to_supervised
from pandas import Timestamp
from data_loader import LoadGoogleDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from transformer_model import Transformer
import torch
from plot_fn import *
from ray import tune
import pandas as pd
import numpy as np
# import time


def dummy_config():
    """
    A dummy configuration for the model.
    Mainly for debugging
    dim_val requires an even number
    :return:
    """
    config = {
        "random_seed": 1,
        "lr": 0.01,
        "lr_step": 2,
        "gamma": 0.99,
        "epochs": 2,
        "n_heads": 2,
        "dim_val": 12,
        "dim_att": 13,
        "encoder_layers": 9,
        "decoder_layers": 1,
        "batch_size": 4,
        "input_feat_enc": 60,
        "input_feat_dec": 1,
        "seq_len": 24,
        "prediction_step": 3
    }
    return config


def init_transformer(config, device):
    """
    Provides the initialization of the model
    :param config: Config dictionary with the model parameters
    :param device: Device where to deploy the model, usually cpu or cuda
    :return: The model initialized in the selected device
    """
    # Model hyperparams
    n_heads = config['n_heads']
    dim_val = config['dim_val']
    dim_attn = config['dim_att']
    n_decoder_layers = config['decoder_layers']
    n_encoder_layers = config['encoder_layers']

    # Data hyperparams
    input_feat_enc = config['input_feat_enc']
    input_feat_dec = config['input_feat_dec']
    seq_len = config['seq_len']
    prediction_step = config['prediction_step']

    # init network and optimizer
    t_model = Transformer(dim_val, dim_attn, input_feat_enc, input_feat_dec, seq_len, n_decoder_layers,
                          n_encoder_layers, n_heads, prediction_step, device)

    t_model.to(device)
    return t_model


def select_device(cuda_number=0):
    """
    Returns a device for the model
    :param cuda_number: Id of the cuda, if available.
    :return: a device for the model
    """
    cuda_name = "cuda:"+str(cuda_number)
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    return device


def q95(x):
    return x.quantile(0.95)


def q75(x):
    return x.quantile(0.75)


def q25(x):
    return x.quantile(0.25)


def prepare_data(input_path):  # , columns_file, selected_cols):
    """
    Function that prepares the selected job deployment for the model.
    It loads the data into a Pandas dataframe, it keeps the selected columns,
    then it removes the nans and scales each column with its max and min.
    Finally it averages the data through the tasks.
    :param input_path:
    :param columns_file:
    :param selected_cols:
    :return:
    """
    data_path = input_path
    # columns_json = columns_file

    # with open(columns_json) as f:
    #     columns_selection = json.load(f)
    columns_selection = [
        "end time",
        "CPU rate",
        "canonical memory usage",
        "assigned memory usage",
        "unmapped page cache",
        "total page cache",
        "maximum memory usage",
        "disk I/O time",
        "local disk space usage",
        "maximum CPU rate",
        "maximum disk IO time",
        "cycles per instruction",
        "memory accesses per instruction",
        # "CPU ratio usage",
        # "memory ratio usage",
        # "disk ratio usage",
        "Efficiency",  # target metric
    ]

    # df = load_data(input_path=data_path, selected_columns=columns_selection[selected_cols])
    df = load_data(input_path=data_path, selected_columns=columns_selection)

    df["datetime"] = timestamps_readings = [
        Timestamp(int(t / 1000000) + 1304233200, tz="US/Pacific", unit="s")
        for t in df["end time"].values
    ]
    df.set_index("datetime", inplace=True)
    df.interpolate(method="time", inplace=True)

    # start = time.time()
    readings_df_x = df.groupby('end time').agg(
        [('mean', 'mean'), ('median', 'median'), ('min', 'min'), ('max', 'max'), ('q25', q25), ('q75', q75),
         ('q95', q95), ('std', 'std')])
    # end = time.time()
    # print("The time of execution of above program is :", end-start)

    # start = time.time()
    # readings_df_x = df.groupby('end time').agg(
    #     [('mean', 'mean'), ('median', 'median'), ('min', 'min'), ('max', 'max'), ('std', 'std')])
    # end = time.time()
    # print("The time of execution of above program is :", end - start)

    readings_df = readings_df_x[columns_selection[1:-1]]
    readings_df["Efficiency"] = readings_df_x["Efficiency"]["mean"].values

    # values = readings_df.values
    # scaled, scaler = scale_values(values)

    #Remove nans
    readings_df.dropna(axis=0, inplace=True)  # remove rows with nans.
    #df.fillna(0., inplace=True)  # change nans by zeros
    
    if len(readings_df) > 50:
        # Scales data
        min_max_scaler = MinMaxScaler()
        # readings_df[readings_df.columns] = min_max_scaler.fit_transform(readings_df[readings_df.columns])
        try: 
            df_scaled = pd.DataFrame(min_max_scaler.fit_transform(readings_df), columns=readings_df.columns)
        except (ValueError):
            print("ValueError")
            return None, None
        #
        # df = data_aggregation(df, aggr_type="mean")  # Data aggregated by TS using the mean value
        return df_scaled, min_max_scaler
    else:
        return None, None


def transformer_train(config, df, device):
    """
    Trains a transformer model with a training and then a validation loop.
    It is prepared to run with Tune ray ASHA scheduler.
    :param config: Dictionary with the model parameters.
    :param df: Data to forecast
    :param device: usually cpu or cuda
    :return: It is prepared for the ASHA scheduler, it does not return anything but it saves the models.
    """
    torch.manual_seed(config["random_seed"])
    # model initialization
    model = init_transformer(config, device)
    # loss function selection
    loss_f = torch.nn.MSELoss()
    # optimizer selection
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    # scheduler selection
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["lr_step"], gamma=config["gamma"])
    # preparation of training dataset
    train_dataset = LoadGoogleDataset("train", seq_len=config["seq_len"], prediction_step=config["prediction_step"],
                                      data_frame=df)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False)
    # preparation of validation dataset
    validation_dataset = LoadGoogleDataset("validation", seq_len=config["seq_len"],
                                           prediction_step=config["prediction_step"], data_frame=df)
    validation_loader = DataLoader(validation_dataset, batch_size=config["batch_size"], shuffle=False)

    # Training loop
    for e in range(config["epochs"]):
        model.train(mode=True)
        for x_enc, x_dec, target in train_loader:
            optimizer.zero_grad()
            # x_dec = torch.ones(x_dec.shape)
            x_enc, x_dec, target = x_enc.to(device), x_dec.to(device), target.to(device)
            x_dec = x_dec.unsqueeze(-1)
            out = model.forward(x_enc.float(), x_dec.float(), training=True)
            loss = loss_f(out.double(), target.double())
            loss.backward()
            optimizer.step()

        # Validate model
        val_loss = 0.0
        val_steps = 0
        model.train(mode=False)
        for x_enc, x_dec, target in validation_loader:
            with torch.no_grad():
                x_enc, x_dec, target = x_enc.to(device), x_dec.to(device), target.to(device)
                x_dec = x_dec.unsqueeze(-1)
                out = model.forward(x_enc.float(), x_dec.float(), training=False)

                loss = loss_f(out.double(), target.double())
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(e) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)
        #
        tune.report(loss=(val_loss / val_steps))
        # print("epoch: " + str(e) + " validation loss: " + str(val_loss / val_steps))
        lr_scheduler.step()


def push_to_tensor(tensor, x):
    t = torch.Tensor([x])
    t = t.unsqueeze(-1)
    t = t.unsqueeze(0)
    return torch.cat((tensor[:, 1:, :], t), 1)


def transformer_test(model, df, df_scaler, device, config, save_dir, experiment_name):
    """
    Function for testing the model
    :param model: Model to be tested. Already initialized
    :param df: Dataset to forecast
    :param device: Usually cpu or cuda
    :param config: Dictionary with model parameters
    :param save_dir: Saving directory, for figures
    :param experiment_name: Saving name, for figures
    :return: Returns the mean loss of the test.
    """
    torch.manual_seed(config["random_seed"])
    save_dir = save_dir
    experiment_name = experiment_name
    # Loss function
    loss_f = torch.nn.MSELoss()
    loss_progress = list()
    if config["prediction_step"] > 1:
        outputs = dict()
        targets = dict()
        for ii in range(config["prediction_step"]):
            outputs[str(ii)] = list()
            targets[str(ii)] = list()
    else:
        outputs = list()
        targets = list()

    # Prepare test dataset
    test_dataset = LoadGoogleDataset("test", seq_len=config["seq_len"], prediction_step=config["prediction_step"],
                                      data_frame=df)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Test/forecast loop
    model.train(mode=False)
    decoder_input = torch.ones([1, 24, 1])
    first_it = 0
    last_output = 0
    for x_enc, x_dec, target in test_loader:
        with torch.no_grad():
            x_enc, x_dec, target = x_enc.to(device), x_dec.to(device), target.to(device)
            x_dec = x_dec.unsqueeze(-1)

            if first_it == 0:
                decoder_input = x_dec
                first_it = first_it + 1
            else:
                decoder_input = push_to_tensor(decoder_input, last_output)

            out = model.forward(x_enc.float(), decoder_input.float(), training=False)
            last_output = out[0][0].double()

            loss = loss_f(out.double(), target.double())

            if config["prediction_step"] > 1:
                for ii in range(config["prediction_step"]):
                    outputs[str(ii)].append(out.squeeze().cpu().detach().tolist()[ii])
                    targets[str(ii)].append(target.squeeze().cpu().detach().tolist()[ii])
            else:
                outputs.append(out.squeeze().cpu().detach().tolist())
                targets.append(target.squeeze().cpu().detach().tolist())

            loss_progress.append(loss.cpu().detach().tolist())

    if config["prediction_step"] > 1:
        mean_tgt = sum(targets[str(0)]) / len(targets[str(0)])
    else:
        mean_tgt = sum(targets) / len(targets)
    mean_loss = (sum(loss_progress) / mean_tgt) / len(loss_progress)

    # re-scale outputs
    minmax_scaler = df_scaler
    l_df = len(df["Efficiency"])
    df_computed = df

    values = dict()

    if config["prediction_step"] > 1:
        eff_out = dict()
        tgt_out = dict()
        for ii in range(config["prediction_step"]):
            real_eff = np.zeros(len(df["Efficiency"]))
            real_eff[l_df - len(outputs[str(ii)]):] = outputs[str(ii)]
            df_computed["Efficiency"] = real_eff
            df_unscaled = df_scaler.inverse_transform(df_computed)
            eff_out[str(ii)] = df_unscaled[l_df - len(outputs[str(ii)]):, -1]

            real_eff = np.zeros(len(df["Efficiency"]))
            real_eff[l_df - len(outputs[str(ii)]):] = targets[str(ii)]
            df_computed["Efficiency"] = real_eff
            df_unscaled = df_scaler.inverse_transform(df_computed)
            tgt_out[str(ii)] = df_unscaled[l_df - len(outputs[str(ii)]):, -1]

            values["eff_" + str(ii)] = eff_out[str(ii)].tolist()
        values["tgt"] = tgt_out["0"].tolist()
    else:
        real_eff = np.zeros(len(df["Efficiency"]))
        real_eff[l_df - len(outputs):] = outputs
        df_computed["Efficiency"] = real_eff
        df_unscaled = df_scaler.inverse_transform(df_computed)
        eff_out = df_unscaled[l_df - len(outputs):, -1]

        real_eff = np.zeros(len(df["Efficiency"]))
        real_eff[l_df - len(outputs):] = targets
        df_computed["Efficiency"] = real_eff
        df_unscaled = df_scaler.inverse_transform(df_computed)
        tgt_out = df_unscaled[l_df - len(outputs):, -1]

        values["eff"] = eff_out.tolist()
        values["tgt"] = tgt_out.tolist()

    # saving data
    out_file = open(save_dir + experiment_name + "_values.json", "w")
    json.dump(values, out_file, indent=4)

    # Plotting only relevant results.
    if mean_loss < 0.1:
        plot_loss(loss_progress, mean_loss, save_dir, experiment_name)
        if config["prediction_step"] > 1:
            cross_correlation = dict()
            for ii in range(config["prediction_step"]):
                plot_target(eff_out[str(ii)], tgt_out[str(ii)], mean_loss, save_dir, experiment_name + "_" + str(ii))
                cross_correlation[str(ii)] = np.correlate(eff_out[str(ii)], tgt_out[str(ii)], mode="same")
                ll = list()
                len_cc = len(cross_correlation[str(ii)])
                for value in range(len_cc):
                    ll.append(int(value - len_cc / 2))

                plt.figure(figsize=(20, 8))
                plt.scatter(ll[int(len_cc / 2) - 10:int(len_cc / 2) + 10],
                            cross_correlation[str(ii)][int(len_cc / 2) - 10:int(len_cc / 2) + 10])
                plt.vlines(0, min(cross_correlation[str(ii)][int(len_cc / 2) - 10:int(len_cc / 2) + 10]),
                           max(cross_correlation[str(ii)]))
                plt.savefig(save_dir + experiment_name + "_" + str(ii) + "_cross-correlation.png")
                plt.close()

        else:
            plot_target(eff_out, tgt_out, mean_loss, save_dir, experiment_name)

    return mean_loss
