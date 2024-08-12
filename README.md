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
Test stuff for Ray

### BoHBCode
#### HPO:
Adjust bohb_ray_slurm_template as needed and execute it. It starts a RayCluster before starting the BoHB runner. The BoHB runner starts multiple processes for HPO. results are stored. 

#### Surrogate Training
Run create_training_data.py to make the cost matrix, create_training_data.py to make. the training data for the model.

The files to train (`metamodel_train.py`) and optimise the model  (`metamodel_optimise.py`) are provided. `metamodel_predict.py` is used to test. 
 






# TinyBERT
Code for distillation of TinyBERT

