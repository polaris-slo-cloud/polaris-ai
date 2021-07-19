# polaris-ai

This repository contains the tools developed in the framework of the polaris-slo-cloud project that belong to AI technologies.
In this regard, the architecture for the AI technologies of the polaris-slo-cloud project is presented below:
![polaris-ai architecture](https://raw.githubusercontent.com/vikcas/figures/main/Polaris-ai_architecture_scheme.png)

In the previous scheme, the white and grey boxes represent the input data. In blue, there are the AI technologies that will be researched or used. The purple circles represent the tools that this project will develop. These aim to perform the actions that are represented in green, to finally obtain a complete cloud management system.

At this point of development, this repository contains the tools to treat [cloud workload data](./data_extraction) and to create and test models for [high-level monitoring](./predictive_monitoring).

In particular, the [`data_extraction` folder](./data_extraction) contains the scripts to pre-process the input data. So far, we refer to the [Google cluster data (2011)](https://research.google/tools/datasets/cluster-workload-traces/) as our first primary source.

The [`high-level monitoring` folder](./predictive_monitoring) includes the means for generating and testing models. Specifically, we have so far the code to develop [LSTM](./predictive_monitoring/lstm_approach) and [transformer](./predictive_monitoring/transformer_approach) neural networks.
