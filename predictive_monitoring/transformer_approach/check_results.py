from os import walk
import os
import json
from csv import DictReader
import matplotlib.pyplot as plt
from transformer_train import init_transformer, transformer_test, prepare_data
import torch


if __name__ == '__main__':

    data_path = "/home/vcpujol/src/polaris-ai/predictive_monitoring/high-level_monitoring/data/task-usage_job-ID-3418339_total.csv"

    path_to_save = "/data/results/vcpujol/transformers/single_deployment/google_traces/multistep_best_eff_computed/"

    df, scaler = prepare_data(data_path)  #, columns_file, columns_scheme)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device ='cpu'

    results_path = "/data/results/vcpujol/transformers/single_deployment/google_traces/"
    experiment = "multistep_test_eff/"
    path = results_path + experiment

    test_loss = dict()

    folders_list = []
    for _, dirnames, _ in walk(path):
        folders_list.extend(dirnames)
        break

    ii = 0
    for folder in folders_list:
        exp_path = path + folder

        with open(exp_path + "/params.json") as jfile:
            config = json.load(jfile)

        progress_csv = "/progress.csv"
        fieldnames = ["loss", "time_this_iter_s", "should_checkpoint", "done", "timesteps_total", "episodes_total",
                      "training_iteration", "experiment_id", "date", "timestamp", "time_total_s", "pid", "hostname",
                      "node_ip", "time_since_restore", "timesteps_since_restore", "iterations_since_restore",
                      "trial_id"]

        progress_data = DictReader(open(exp_path + progress_csv), delimiter=",", fieldnames=fieldnames)
        next(progress_data)
        loss_evolution = list()
        chkp_id = 0
        exp_id = None
        for d in progress_data:
            loss_evolution.append(float(d["loss"]))
            chkp_id = d["training_iteration"]
            exp_id = d["trial_id"]
        chkp_id = str(int(chkp_id) - 1)

        val_avg_loss = sum(loss_evolution)/len(loss_evolution)
        print("validation loss: " + str(val_avg_loss))
        if val_avg_loss < 0.05:
            plt.figure(figsize=(20, 8))
            plt.plot(loss_evolution, '-', color='indigo', label='Loss', linewidth=2)
            plt.legend()
            plt.title("Loss evolution in validation")
            plt.savefig(path_to_save + "/" + exp_id + "_validation_loss.png")
            plt.close()

            checkpoint = exp_path + "/checkpoint_" + chkp_id + "/checkpoint"
            model = init_transformer(config, 'cpu')
            model_state, _ = torch.load(checkpoint, map_location='cpu')
            model.load_state_dict(model_state)

            loss = transformer_test(model=model, df=df, df_scaler=scaler, device=device, config=config, save_dir=path_to_save, experiment_name=exp_id)
            test_loss[exp_id] = loss
            print("test loss: " + str(loss))

        ii = ii + 1
        print(ii)

    if test_loss is not None:
        # the json file where the output must be stored
        out_file = open(path_to_save + experiment.strip("/") +"loss.json", "w")

        json.dump(test_loss, out_file, indent=4)

        out_file.close()
