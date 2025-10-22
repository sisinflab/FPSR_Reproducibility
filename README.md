# A Reproducible and Fair Evaluation of Partition-aware Collaborative Filtering

This is the official repository for the paper "_A Reproducible and Fair Evaluation of Partition-aware Collaborative Filtering_".

This repository provides all the necessary code, data splits, and configurations to reproduce the experiments presented in the paper. The experimental framework is built upon **Elliot**, a comprehensive and rigorous framework for reproducible recommender systems evaluation. We strongly suggest referring to the official Elliot [GitHub page](https://github.com/sisinflab/elliot) and [documentation](https://elliot.readthedocs.io/en/latest/).

## 📖 Overview

Similarity-based collaborative filtering (CF) models are known for their strong performance and simplicity. However, their scalability is often limited by the quadratic cost of item-item similarity matrices. Partitioning-based paradigms, such as the **Fine-tuning Partition-aware Similarity Refinement (FPSR)** framework, have emerged to balance effectiveness and efficiency.

This work presents a transparent, fully reproducible benchmark of FPSR and its extension FPSR+. We re-assess the original claims under controlled conditions, conduct a fair comparison against strong baselines like BISM, and perform a fine-grained analysis of their performance on long-tail items. Our investigation clarifies the trade-offs of partition-aware modeling and offers actionable guidance for scalable recommender system design.


### Our Contributions

Our work provides four key contributions to the field:

1. **Replicability Study:** We conduct a thorough replicability study of the original FPSR and FPSR+ papers, identifying where results align and systematically explaining discrepancies caused by unavailable data splits and implementation details.

2. **Fair Benchmarking:** We present a rigorous and fair benchmark of the FPSR family against strong baselines, including BISM and state-of-the-art models. This is performed under a unified protocol with consistent data splits and extensive, comparable hyperparameter optimization for all methods.

3. **Robustness to Partitioning:** We perform an in-depth analysis of the FPSR family's robustness to different partitioning strategies. We investigate the architectural impact of hub connectors and global signals, especially in scenarios with small or imbalanced item clusters.

4. **Long-Tail Analysis:** We provide a fine-grained long-tail analysis, breaking down model performance on head (popular) versus tail (niche) items. This clarifies how partition-aware models and their hub mechanisms influence recommendation quality across the entire item popularity spectrum.

---

## 📋 Table of Contents

- [Prerequisites](#-prerequisites)
- [Datasets](#-datasets)
- [Reproducing Experimental Results](#-reproducing-experimental-results)
  - [**RQ1**: Replicability Study (Table 2)](#rq1-replicability-study-table-2)
  - [**RQ2**: Fair Benchmarking (Table 3)](#rq2-fair-benchmarking-table-3)
  - [**RQ3**: Robustness to Partitioning (Table 4)](#rq3-robustness-to-partitioning-table-4)
  - [**RQ4**: Long-tail Analysis (Table 5)](#rq4-long-tail-analysis-table-5)
- [Citation](#-citation)
- [License](#-license)

---

## ⚙️ Prerequisites & Installation

We implemented and tested our experiments in a Python 3.8.20 environment.
For full reproducibility, we recommend using a machine with CUDA support.

1.  Create and activate a virtual environment:
```
# PYTORCH ENVIRONMENT (CUDA 12.4, cuDNN 8.9.2.26)
$ python3.8 -m venv venv
$ source venv/bin/activate
```



2.  Install the required packages from the \`requirements.txt\` file:
```
$ pip install --upgrade pip
$ pip install -r requirements.txt
```
---








## 📊 Datasets

The experiments are conducted on four public datasets: **Amazon-CDs**, **Douban**, **Gowalla**, and **Yelp2018**.

*   **Original Splits (for RQ1)**: For the replicability study, we use the original data splits provided by the FPSR authors where available.
For datasets where splits were not explicitly provided (e.g., Amazon-CDs and Douban), we instrumented the [original FPSR source code](https://github.com/Joinn99/FPSR/) to export the stochastically generated train, validation, and test sets.
In all cases, these splits are saved under the following generic paths:
    ```
    data/<dataset>/train_fpsr.tsv
    data/<dataset>/valid_fpsr.tsv
    data/<dataset>/test_fpsr.tsv
    ```
* **New Fair Splits (for RQ2, RQ3, RQ4)**: For our fair benchmarking, we generated new, consistent user-based hold-out splits.
To ensure perfect reproducibility, these splits are provided directly in this repository under the following generic paths:
    ```
    data/<dataset>/splitting/0/0/train.tsv
    data/<dataset>/splitting/0/0/val.tsv
    data/<dataset>/splitting/0/test.tsv
    data/<dataset>/splitting/0/test_head.tsv
    data/<dataset>/splitting/0/test_tail.tsv
    ```

N.B. All datasets are in a TSV format compatible with Elliot.

---














## 🚀 Reproducing Experimental Results

All experiments can be launched using the main Elliot runner script. The general command is:
```
python start_experiments.py --config <config_file_name>             # do not type ".yml" !!!!!!
```

Each research question corresponds to a set of configuration files that reproduce the results reported in the paper's tables.














### **RQ1**: Replicability Study (Table 2)

To replicate the results from the original FPSR/FPSR+ papers (Table 2), run the commands below.

Replace `<dataset_name>` with one of the following: `amazon_cds`, `douban`, `gowalla`, or `yelp2018`.

1. To run the base FPSR model:
    ```
    python start_experiments.py --config reproducibility_fpsr_<dataset_name>
    ```
2. To run both FPSR+ variants (FPSR+D and FPSR+F):
    ```
    python start_experiments.py --config reproducibility_fpsrplus_<dataset_name>
    ```

Important Note on Model Names:
* The configuration file `reproducibility_fpsrplus_<dataset_name>` is designed to run both FPSR+ variants sequentially.
* Throughout this repository's configuration files, the model names are mapped as follows:
  * `FPSRplus` stands for FPSR+(D) variant.
  * `FPSRplusF` stands for FPSR+(F) variant.
* There is no reproducibility for FPSR+(D) and FPSR+(F) on Yelp2018, as well in their original paper.
















### **RQ2**: Fair Benchmarking (Table 3)

To reproduce our main benchmark results from Table 3, you have two options.

Replace `<dataset_name>` with one of the following: `amazon_cds`, `douban`, `gowalla`, or `yelp2018`.

#### Option 1: Reproduce Final Results Directly (Recommended)

To directly obtain the results shown in Table 3, you can run the experiments using the pre-configured best hyperparameter settings for all models.
```
python start_experiments.py --config best_<dataset_name>
```

#### Option 2: Re-run the Full Hyperparameter Optimization
If you wish to replicate the entire hyperparameter search process (20 TPE trials for each model), you can use the exploration configuration files. **Note: This process is computationally expensive.**
```
python start_experiments.py --config explorations_<dataset_name>
```

N.B. In the folder `recs_ECIR_26`, we share all the recommendation lists needed for the also all the `significativity_<dataset_name>` experiments.









### **RQ3**: Robustness to Partitioning (Table 4)

The results for the robustness analysis (Table 4) can be generated by running the FPSR family models with fixed partition size ratio (\`τ\`) values on Douban dataset.
To reproduce this table specifically, you can use the following command:
```
python start_experiments.py --config tau_sensitivity_douban
```















# **RQ4**: Long-tail Analysis (Table 5)

To reproduce the fine-grained long-tail analysis results from Table 5, we have streamlined the entire process into a single script.

Replace `<dataset_name>` with one of the following: `amazon_cds`, `douban`, `gowalla`, or `yelp2018`.

```
python longtail_experiments.py --dataset <dataset_name>
```

Note that, within this script, you can accurately check the list of experiments that can be performed to faithfully reproduce the published results.










---











## ✒️ Citation

If you find this work useful for your research, please cite our paper:
```
to define
```
