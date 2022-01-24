from scipy.stats import norm
import numpy as np

import matplotlib as mpl
from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
    
## Cluster plots

def plot_silhouette_results(cluster_labels, silhouette_avg, sample_silhouette_values, n_clusters):
    plt.figure(figsize=(12, 12))

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    plt.xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    #plt.ylim([0, len(scaled) + (n_clusters + 1) * 10])

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i 

        color = cm.nipy_spectral(float(i) / n_clusters)
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10000  # 10 for the 0 samples

    plt.title("The silhouette plot for the various clusters.")
    plt.xlabel("The silhouette coefficient values")
    plt.ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")

    plt.yticks([])  # Clear the yaxis labels / ticks
    plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
    
    
def plot_feature_scatter(ax, data, clusterer, cluster_labels, feature, dim1, dim2, notation, n_clusters):
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax.scatter(data[:, dim1], data[:, dim2], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax.set_title("The visualization of the {feature} {notation}. K = {n_clusters}".format(feature=feature, notation=notation, n_clusters=n_clusters))
    ax.set_xlabel("Feature space for the {feature} - mean".format(feature=feature))
    ax.set_ylabel("Feature space for the {feature} - std".format(feature=feature))