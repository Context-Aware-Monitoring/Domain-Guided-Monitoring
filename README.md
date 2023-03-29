# Domain-Guided-Monitoring
Can we use domain knowledge to better monitor complex systems?

## Differences between this version and the original
This version of DomainML was created during the course of my thesis *Generating
Causal Domain Knowledge for Cloud Systems Monitoring*. Some corrections were
made and some new features were added. 

The most important differences are:
- You can now save and load knowledge via the
  `--experimentconfig_load_knowledge_df`,
  `--experimentconfig_serialize_knowledge_df`, and
  `--experimentconfig_knowledge_df_file` command options
- A variety of causal algorithms are now available via the [cdt package](https://fentechsolutions.github.io/CausalDiscoveryToolbox/html/index.html). To use them, simply change
  the `--experimentconfig_model_type` to causal_{your_algo}. The list of
  available causal algorithms can be found in
  `src/features/preprocessing/{your_dataset}`
- Timestamps can be removed from the Huawei dataset. By default, they will be
  removed, so if you don't want that to happen add use the command parameter
  `--no-huaweipreprocessorconfig_remove_dates_from_payload`
- The BGL and HDFS datasets now work with DomainML
- Some new notebooks to analyze the results and reproduce my results from the
  paper can be found under `notebooks`
- Note that if you want to run any notebooks and import some of the classes or
  functions from the source code, you will have to comment out
  `@dataclass_cli.add` in the source code, restart your Jupyter kernel, and then
  run the notebook cells. This is a weird bug.

## Repository Structure
This repo is structured as follows:
- `notebooks/` contains some jupyter notebooks used for simple exploration and experimentation.
- `data/` should be used for the data used for and created during training and preprocessing steps. 
- `artifacts/` is used for storing the outputs of experiment runs
- `src/` contains all the code used for our experiments:
  - `src/features` includes code handling the features of our experiments. It is separated into
    - `src/features/preprocessing`: any preprocessing steps necessary to use the raw data. will produce intermediate results so that we don't have to run this step every time,
    - `src/features/knowledge`: code for handling the **Expert Knowledge** part of the training,
    - `src/features/sequences`: code handling the sequences (eg big data part) used during training.
  - `src/training` contains code used for the actual training part
    - `src/training/models` defines the model structures used for training.
    - `src/training/analysis` defines training analysis such as printing learned embeddings.
- `environment.yml` defines the anaconda environment and dependencies used for running the experimentation code.

### Supported Datasets
For now, this repository supports two types of datasets:
- [MIMIC dataset](https://mimic.physionet.org/about/mimic/) and
- Huawei log dataset.
- [HDFS](https://www.kaggle.com/datasets/omduggineni/loghub-hadoop-distributed-file-system-log-data)
- [BGL](https://www.kaggle.com/datasets/omduggineni/loghub-bgl-log-data)

If you want to add a new type of dataset, look at the preprocessors implemented
in `src/features/preprocessing/` and put your own implementation there.

The HDFS and BGL raw log files can be preprocessed using `notebooks/logs_to_df.ipynb`.

### Supported Expert Knowledge
With this repository, the following types of expert knowledge are supported:
- **Hierarchical Expert Knowledge** (original idea see `CHOI, Edward, et al. GRAM: graph-based attention model for healthcare representation learning. In: Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining. 2017. S. 787-795.`)
- **Textual Expert Knowledge** (original idea see `MA, Fenglong, et al. Incorporating medical code descriptions for diagnosis prediction in healthcare. BMC medical informatics and decision making, 2019, 19. Jg., Nr. 6, S. 1-13.`)
- **Causal Expert Knowledge** (original idea see `YIN, Changchang, et al. Domain Knowledge guided deep learning with electronic health records. In: 2019 IEEE International Conference on Data Mining (ICDM). IEEE, 2019. S. 738-747.`)

## How to Run this Code
In order to run this code, you need Anaconda + Python >= 3.8. This repository contains a makefile that simplifies executing the code on a remote server using miniconda. Below are step-by-step descriptions of what to do in order to run the code from either the Makefile or manual setup.

### Run via Makefile
- **Create and activate conda environment**: run `make install` 
- **Get the data**: We don't include training data into this repo, so you have to download it yourself and move it to the `data/` folder. For now, we support training on the MIMIC dataset and on Huawei log data. 
  - **Get MIMIC**: In order to access MIMIC dataset, you need a credentialed account on physionet. You can request access [here](https://mimic.physionet.org/gettingstarted/access/). Run `make install_mimic PHYSIONET_USER="<yourphysionetusername>"` to download the required physionet files from their website. 
  - **Get Huawei Log Data**: Download the Huawei `concurrent data` dataset and move the file `concurrent data/logs/logs_aggregated_concurrent.csv` to the `data/` directory
- **Run the code**: To run the experiment, execute `make run ARGS="<yourargshere>"`. There are a bunch of commandline options, the most important ones are:
  -  `--experimentconfig_sequence_type`: dataset to use, for now valid values here are `mimic` and `huawei_logs`
  -  `--experimentconfig_model_type`: use this to choose the knowledge model you want to run; valid values are `simple`, `gram`, `text` and `causal`
  -  to see the full list of options run `python main.py -h`
- **Visualize Results**: Metrics, artefacts and parameters of an experiment run are logged in MLFlow. You can use the mlflow UI to get a good overview over the experiment results. Execute `make ui` to start mlflow UI on port 5000.

### Run manual SetUp
- **Create and activate conda environment**: run `conda env update -f environment.yml` to create (or update) an anaconda environment from the given `environment.yml` file. Activate the environment by running `conda activate healthcare-aiops`
- **Get the data**: We don't include training data into this repo, so you have to download it yourself and move it to the `data/` folder. For now, we support training on the MIMIC dataset and on Huawei log data. 
  - **Get MIMIC**: In order to access MIMIC dataset, you need a credentialed account on physionet. You can request access [here](https://mimic.physionet.org/gettingstarted/access/) or use the example MIMIC dataset available [here](https://mimic.physionet.org/gettingstarted/demo/). Once you have access, move `ADMISSIONS.csv` and `DIAGNOSES_ICD.csv` into the `data/` directory
  - **Get Huawei Log Data**: Download the Huawei `concurrent data` dataset and move the file `concurrent data/logs/logs_aggregated_concurrent.csv` to the `data/` directory
- **Run the code**: To run the experiment from the command line, execute `python main.py`. There are (amongst others) the following commandline options:
  -  `--experimentconfig_sequence_type`: dataset to use, for now valid values here are `mimic` and `huawei_logs`
  -  `--experimentconfig_model_type`: use this to choose the knowledge model you want to run; valid values are `simple`, `gram`, `text` and `causal`
  -  to see the full list of options run `python main.py -h`
- **Visualize Results**: Metrics, artefacts and parameters of an experiment run are logged in MLFlow. You can use the mlflow UI to get a good overview over the experiment results. Execute `mlflow ui` to start mlflow UI on port 5000.

## Run via Docker
Instead of running the experiments directly on your host machine, you can build
the Docker image described in `Dockerfile` and start a container for the
experiments. The run command for the docker container can be found in `docker_run.sh`