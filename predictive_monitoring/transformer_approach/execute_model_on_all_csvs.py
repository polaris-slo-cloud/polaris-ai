from os import walk
import os
import json
from csv import writer
import matplotlib.pyplot as plt
from transformer_train import init_transformer, transformer_test, prepare_data
import torch
from data_loader import LoadGoogleDataset
from torch.utils.data import DataLoader
import numpy as np


if __name__ == '__main__':

    #data_path = "/home/vcpujol/src/polaris-ai/predictive_monitoring/high-level_monitoring/data/task-usage_job-ID-3418339_total.csv"
    data_path = "/home/tuwien/data/cloud_data/Google-clusterdata-2011-2/processed_data/high-level_monitoring/scheduling_class_3_29days/"
    path_to_save = "/home/tuwien/vcasamayor/results/all_multistep/"
    path_to_model = "/home/tuwien/vcasamayor/src/multistep_test_all/predictive_monitoring/high-level_monitoring/transformer_approach/models/multistep"

    files_list = []
    for _, _, filenames in walk(data_path):
        files_list.extend(filenames)
        
    files_done = []
    for _, _, filenames in walk(path_to_save):
        files_done.extend(filenames)

    files_todo = []
    for csv_file in files_list:
        if csv_file not in files_done:
            files_todo.append(csv_file)

    for file_name in files_todo:
        print(file_name)
        df, scaler = prepare_data(data_path + file_name)
        if df is not None:

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            with open(path_to_model + "/config.json") as jfile:
                config = json.load(jfile)

            saved_model = path_to_model + "/model"
            model = init_transformer(config, device)
            model_state, _ = torch.load(saved_model, map_location=device)
            model.load_state_dict(model_state)

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

            for x_enc, x_dec, target in test_loader:
                with torch.no_grad():
                    x_enc, x_dec, target = x_enc.to(device), x_dec.to(device), target.to(device)
                    x_dec = x_dec.unsqueeze(-1)

                    out = model.forward(x_enc.float(), x_dec.float(), training=False)

                    loss = loss_f(out.double(), target.double())

                    if config["prediction_step"] > 1:
                        for ii in range(config["prediction_step"]):
                            outputs[str(ii)].append(out.squeeze().cpu().detach().tolist()[ii])
                            targets[str(ii)].append(target.squeeze().cpu().detach().tolist()[ii])
                    else:
                        outputs.append(out.squeeze().cpu().detach().tolist())
                        targets.append(target.squeeze().cpu().detach().tolist())

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
            with open(path_to_save + file_name, 'w') as f:
                dict_writer = writer(f)
                dict_writer.writerow(values.keys())
                dict_writer.writerows(zip(*values.values()))
        else:
            print("Too Nans")
