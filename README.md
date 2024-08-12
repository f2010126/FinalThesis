# FinalThesis_12Aug
Upload Again Attempt

# File Structure

## Adapters
Code for Using PEFT type as part of the framework
## Cleaned Datasets
Core Datasets in format of sentence and label for easier processing. Raw datasets are available via Huggingface.
# HPO
## Ray Cluster Test
### MetaDataCreation
Relevant to training the surrogate

### RayCode
Test stuff for Ray, trying the library and available optimisers of Ray Tune.

### BoHBCode
#### HPO:
Adjust `bohb_ray_slurm_template.sh` as needed and execute it. It starts a RayCluster before starting the BoHB runner. The BoHB runner starts multiple processes for HPO and results are stored. There is no Multiprocessing clash due to Ray having a separate environment.

#### Surrogate Training
Run create_training_data.py to make the cost matrix, create_training_data.py to make. the training data for the model.

The files to train (`metamodel_train.py`) and optimise the model  (`metamodel_train.py`) are provided. `metamodel_predict.py` is used to test. 
 

# TinyBERT
Code for distillation of TinyBERT. Consists of 2 stage distillation: General for the Embedded layers, Finetuning for the predicted layers.

## Workflow
- Run the search for the incubent pipelines using `bohb_ray_slurm_template.sh`
- Create the performance matrix based on the results
- Create the Training data using `create_training_data.py`
- Train, optimise and predict using `metamodel_train.py`, `metamodel_train.py` and `metamodel_predict.py`.

### Extension for new Datasets
- Add the dataset processing to `data_modules.py`.

### Extension for new Pipeline options
- Add the changes to the parameter space in `bohb_ray_cluster.py`

