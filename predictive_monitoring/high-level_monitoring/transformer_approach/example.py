from transformer_train import init_transformer, prepare_data, q25, q75, q95
from data_loader import LoadGoogleDataset
from torch.utils.data import DataLoader
from csv import writer
import numpy as np
import json
import torch

if __name__ == '__main__':
    # Set paths to data
    data_path = "../data/task-usage_job-ID-3418339_total.csv"
    results_path = "..."
    results_file = "...csv"
    # Prepare dataset
    df, scaler = prepare_data(data_path)
    # Set up a device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the model configuration
    with open("models/multistep/config.json") as jfile:
        config = json.load(jfile)

    # Initialize the model
    model = init_transformer(config, device)
    # Load the model
    model_state, _ = torch.load("multistep/models/model_data", map_location=device)
    model.load_state_dict(model_state)
    model.eval()
    # Select a loss function
    loss_f = torch.nn.MSELoss()
    # Prepare test dataset
    test_dataset = LoadGoogleDataset("test", seq_len=config["seq_len"], prediction_step=config["prediction_step"],
                                     data_frame=df)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Run test/forecast loop
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

    for x_enc, x_dec, target in test_loader:
        with torch.no_grad():
            # Send data to device and prepare dimensions
            x_enc, x_dec, target = x_enc.to(device), x_dec.to(device), target.to(device)
            x_dec = x_dec.unsqueeze(-1)
            # Forecast
            out = model.forward(x_enc.float(), x_dec.float(), training=False)
            # Compute loss
            loss = loss_f(out.double(), target.double())

            # Store results and target values
            if config["prediction_step"] > 1:
                for ii in range(config["prediction_step"]):
                    outputs[str(ii)].append(out.squeeze().cpu().detach().tolist()[ii])
                    targets[str(ii)].append(target.squeeze().cpu().detach().tolist()[ii])
            else:
                outputs.append(out.squeeze().cpu().detach().tolist())
                targets.append(target.squeeze().cpu().detach().tolist())

            # Keep loss values in a list
            loss_progress.append(loss.cpu().detach().tolist())

    # re-scale outputs
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
            df_unscaled = scaler.inverse_transform(df_computed)
            eff_out[str(ii)] = df_unscaled[l_df - len(outputs[str(ii)]):, -1]

            real_eff = np.zeros(len(df["Efficiency"]))
            real_eff[l_df - len(outputs[str(ii)]):] = targets[str(ii)]
            df_computed["Efficiency"] = real_eff
            df_unscaled = scaler.inverse_transform(df_computed)
            tgt_out[str(ii)] = df_unscaled[l_df - len(outputs[str(ii)]):, -1]

            values["eff_" + str(ii)] = eff_out[str(ii)].tolist()
            values["tgt_" + str(ii)] = tgt_out[str(ii)].tolist()

    else:
        real_eff = np.zeros(len(df["Efficiency"]))
        real_eff[l_df - len(outputs):] = outputs
        df_computed["Efficiency"] = real_eff
        df_unscaled = scaler.inverse_transform(df_computed)
        eff_out = df_unscaled[l_df - len(outputs):, -1]

        real_eff = np.zeros(len(df["Efficiency"]))
        real_eff[l_df - len(outputs):] = targets
        df_computed["Efficiency"] = real_eff
        df_unscaled = scaler.inverse_transform(df_computed)
        tgt_out = df_unscaled[l_df - len(outputs):, -1]

        values["eff"] = eff_out.tolist()
        values["tgt"] = tgt_out.tolist()

    # saving data
    with open(results_path + results_file, 'w') as f:
        dict_writer = writer(f)
        dict_writer.writerow(values.keys())
        dict_writer.writerows(zip(*values.values()))

