# Machine Learning workload profiling

## Rationale
Our team has created a reference implementation of PolarisProfiler, which includes the main profiling processes. The primary purpose of this tool is to optimize the scheduling of Machine Learning (ML) workloads, using the [Alibaba Cluster Trace](https://github.com/alibaba/clusterdata). To achieve this, we have developed a profiling approach that focuses on assessing the duration of the workload. This duration is a crucial feature in planning and scheduling workloads, as it ensures efficient use of resources while also meeting SLOs.

![Profiling model overview](Figures/Profiling model.pdf)

## Understanding the code
The experiment were mainly conducted in two phases. 
1. The first phase is the Exploratory Data Analysis. In this regard, we provide [this notebook](polaris-ai/ml_data-profiling/alibaba_data-EDA-v0.3.ipynb) where we investigate the Alibaba dataset. To reproduce the analyses, please download the Alibaba Cluster Trace at its link. Furthermore, we extract compact and representative elements that we will use for our profiling work. We make them available in the [experiments](polaris-ai/ml_data-profiling/experiments) folder. We also perform and export the [clustering experiment](polaris-ai/ml_data-profiling/experiments/hdbscan_300_power_transform_euclidean.pkl), whose results we leveraged in the second phase.
2. The second phase consists in running the PolarisProfiler. The code can be found in [this notebook](polaris-ai/ml_data-profiling/alibaba_data-evaluation.ipynb). We use 100,001 samples that are available [here](polaris-ai/ml_data-profiling/experiments/100_001_sampled_workload_data.csv). Here, we load the clustering result, we perform the classification, and we use the classifier to predict the duration on [new, unseen workload](polaris-ai/ml_data-profiling/experiments/1_000_sampled_test_data.csv).