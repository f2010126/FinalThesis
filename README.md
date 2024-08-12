# FinalThesis_12Aug
Upload Again Attempt

# File Structure

<img width="617" alt="Schema" src="https://github.com/user-attachments/assets/7d2bdca3-f540-4bff-8892-d69ed5381233">

## Adapters
Code for Using PEFT type as part of the framework
## Cleaned Datasets
Core Datasets in format of sentence and label for easier processing. Raw datasets are available via Huggingface.
# HPO
## Ray Cluster Test
### MetaDataCreation
Relevant to training the surrogate. Contains code to make the performance matrix, training data, train, optimise and predict with the model. 

### RayCode
Test stuff for Ray, trying the library and available optimisers of Ray Tune.

### BoHBCode
#### HPO:
Adjust `bohb_ray_slurm_template.sh` as needed and execute it. It starts a RayCluster before starting the BoHB runner. The BoHB runner, bohb_runner.py starts multiple processes for HPO and results are stored. There is no Multiprocessing clash due to Ray having a separate environment. Parameter space is defined in `bohb_ray_cluster.py`

Search Space


| Name    | Type | Range/Value|
| -------- | ------- |------------|
| max seq length | categorical | 128, 256, 512 |
| train batch size | categorical | 4, 8, 16    |
| model name | categorical | Bert base uncased, Bert base multilingual cased 
| |             | Bert base german cased oldvocab,gottbert base  |
| |             | TinyBERT General 4L de, Alberti bert base multilingual cased |
| |             | Distilbert base german cased    |
| optimizer | categorical | Adam, AdamW, SGD , RAdam |
| learning rate | float | [2e-5 7e-5] log |
| scheduler | categorical | linear with warmup , cosine with warmup|
| | |cosine with hard restarts with warmup, polynomial decay with warmup|
| weight decay | categorical | [1e-5, 1e-3] log |
|warmup steps | categorical | 10, 100, 500 |
| gradient accumulation steps | categorical | 1, 4, 8, 16|
| adam epsilon | float | [1e-8, 1e-6]  log |






#### Surrogate Training
Run create_training_data.py to make the cost matrix, create_training_data.py to make. the training data for the model.

The files to train (`metamodel_train.py`) and optimise the model  (`metamodel_train.py`) are provided. `metamodel_predict.py` is used to test. 
 
![Final_perfmatrix_](https://github.com/user-attachments/assets/b1997e0b-a604-42d0-a76d-ef02bae3e244)

# TinyBERT
Code for distillation of TinyBERT. Consists of 2 stage distillation: 

General for the Embedded layers

![genDistill_](https://github.com/user-attachments/assets/0b64178f-7957-4d7c-a7bf-bab5bd0091e0)


Finetuning for the predicted layers.

![taskDistill_](https://github.com/user-attachments/assets/a2d941cb-dd6b-414d-a3c2-f4ba3942e3f3)

## Workflow
- Run the search for the incubent pipelines using `bohb_ray_slurm_template.sh`
- Create the performance matrix based on the results 
- Create the Training data using `create_training_data.py`
- Train, optimise and predict using `metamodel_train.py`, `metamodel_train.py` and `metamodel_predict.py`.

### Extension for new Datasets
- Add the dataset processing to `data_modules.py`.

### Extension for new Pipeline options
- Add the changes to the parameter space in `bohb_ray_cluster.py`

