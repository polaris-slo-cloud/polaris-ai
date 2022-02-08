# Workload profiling

This code performs steps to generate precise and representative profile groups for workload, based on what we call static metadata.

Static metadata is all the information regarding the workload that doesn't change at runtime. Ideally, when users deploy some applications in the platform, they give information regarding the type of workload they intend to submit, details about the operating system, and the applications' priorities.
This data is essential to underline behavioral and design patterns for the users and their workload. However, this information is relevant in the measure that it characterizes specific workload execution schemes.

## Approach overview
Our approach follows a series of steps, given the premises mentioned beforehand, and starting from the assumption that the system doesn't have, initially, any profile label. Thus:
1. The first step is to perform unsupervised learning techniques to find similarities in workload execution. Here, we look at a few key features, namely:
    - CPU
    - Memory
    - Disk
    - Level of parallelization
    - Runtime length
2. Once the algorithm has extracted relevant groups, we can derive information regarding the static metadata, linking the workload to their static features and deducing patterns.
3. Finally, we create models of each of these profiles to let the system perform an automatic workload assignment to each group.

## Jupyter notebooks
Here we briefly describe the main focus of every notebook, showing at the same time the pipeline we followed.
1. [`01_preliminary_analyses.ipynb`](https://github.com/polaris-slo-cloud/polaris-ai/tree/main/profiling/01_preliminary_analyses.ipynb) performs an introductory exploration on the Google Cluster Dataset 2011, with the aim of uncovering information useful for profiling workloads.
2. [`02_workload_profiling-evaluation.ipynb`](https://github.com/polaris-slo-cloud/polaris-ai/tree/main/profiling/02_workload_profiling-evaluation.ipynb) deals with finding an unsupervised methodology for grouping together similar workload, starting from their key features (CPU, Memory, Disk, Tasks #, Runtime). Furthermore, it evaluates the outcome looking at unsupervised learning performance metrics and workload features patterns
3. [`03_profiling_alternative_methodologies.ipynb`](https://github.com/polaris-slo-cloud/polaris-ai/tree/main/profiling/03_profiling_alternative_methodologies.ipynb) explores alternative clustering methodologies for creating profile groups. 
4. [`04_metadata_analysis.ipynb`]() analyzes the results from the K-Means clustering, and analyzes the static metadata.
5. [`05_clusters_interpretability.ipynb`](https://github.com/polaris-slo-cloud/polaris-ai/tree/main/profiling/05_clusters_interpretability.ipynb) explores several techniques for extracting patterns from the clusters. In particular, we aim at extracting metadata patterns using TF-IDF, Random Forest, and XGBoost.
6. [`06_users_analysis.ipynb`](https://github.com/polaris-slo-cloud/polaris-ai/tree/main/profiling/06_users_analysis.ipynb) provides metrics and plots to evaluate users behavior. This study is useful to have a global picture of the work and to set the bases for testing the approach on a user.
7. [`07_single_user_profiling.ipynb`](https://github.com/polaris-slo-cloud/polaris-ai/tree/main/profiling/07_single_user_profiling.ipynb) analyzes the process chosen for profiling, with a single user. This study goes in the direction of exploring the potential of the proposed approach in extracting profiles in a hierarchical way.