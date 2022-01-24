from scipy.stats import norm
import numpy as np

import matplotlib as mpl
from cycler import cycler
import matplotlib.pyplot as plt

def extract_jobs_comparison_plot(selected_df, others_df, feature):
    sel_vals = sorted(selected_df[feature].values.reshape((selected_df[feature].values.shape[0] * selected_df[feature].values.shape[1],)))
    pdf_sel = norm.pdf(sel_vals, 0, 1)

    others_vals = sorted(others_df[feature].values.reshape((others_df[feature].values.shape[0] * others_df[feature].values.shape[1],)))
    pdf_others_cpu = norm.pdf(others_vals, 0, 1)

    # How to obtain bars that sum up to unity
    weights_sel = np.ones_like(sel_vals) / len(sel_vals)
    weights_others = np.ones_like(others_vals) / len(others_vals)

    plt.clf()

    mpl.rcParams['axes.prop_cycle'] = cycler('color', ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']) # Swapped first with second

    plt.figure(figsize=(15,9))
    plt.hist(others_vals, weights=weights_others, bins=50, alpha = 0.5, label="Other jobs")
    plt.hist(sel_vals, weights=weights_sel, bins=50, alpha = 0.5, label="Selected job")

    #plt.xlim((0, 0.04))
    plt.xlabel("used {feature}".format(feature=feature), fontsize=22)
    plt.ylabel("Frequency", fontsize=22)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(fontsize=18)