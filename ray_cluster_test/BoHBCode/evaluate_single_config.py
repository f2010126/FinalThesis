import argparse
from ast import parse
from cgi import test
import logging
import os
import threading
import time
import traceback
from numpy import save
import yaml
import json
import torch
from lightning.pytorch import Trainer, seed_everything

from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from tqdm import trange
import sys
sys.path.append(os.path.abspath('/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test'))

from BoHBCode.data_modules import get_datamodule
from BoHBCode.train_module import PLMTransformer

MAX_GPU_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16


import os
import yaml

# ensure the value of 'aug' key is set correctly in all .yaml files in the given folder
def alter_aug_key(folder_path):
    for filename in os.listdir(folder_path):
        # Check if the file is a .yaml file and contains "1X" in its name
        if filename.endswith('.yaml') and '1X' in filename:
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                yaml_content = yaml.safe_load(file)  
            if 'aug' in yaml_content:
                yaml_content['aug'] = True
            else:
                yaml_content['aug'] = True
            with open(file_path, 'w') as file:
                yaml.safe_dump(yaml_content, file, default_flow_style=False)

# save metrics to a file
def save_metrics(file_path, data, key, overwrite=True):
    file_exists = os.path.exists(file_path)
    if file_exists and not overwrite:
        with open(file_path, 'r') as file:
            file_data = json.load(file)
    else:
        file_data = {}
    file_data[key] = data
    with open(file_path, 'w') as file:
        json.dump(file_data, file, indent=4)

# convert tensor values to float
def convert_tensor_values_to_float(data):
    return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in data.items()}

# train a single config
def train_single_config(config, task_name='gnad10', budget=1, data_dir='./cleaned_datasets', train=True):
    # dataset name  config['incumbent_for']
    eval_dir = os.path.join(os.getcwd(), 'IncumbentEvaluations', 'Logs',  config['incumbent_for'])
    os.makedirs(data_dir, exist_ok=True)
    model_dir = os.path.join(os.getcwd(), 'IncumbentEvaluations', 'Models', config['incumbent_for'])
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"budget aka epochs------> {budget} for task {task_name}")
    if torch.cuda.is_available():
        logging.debug("CUDA available, using GPU no. of device: {}".format(
            torch.cuda.device_count()))
    else:
        logging.debug("CUDA not available, using CPU")
    seed_everything(9)

    # data_dir should be the location of the tokenised dataset
    dm = get_datamodule(task_name=task_name, model_name_or_path=config['model_config']['model'],
                        max_seq_length=config['model_config']['dataset']['seq_length'],
                        train_batch_size=config['model_config']['dataset']['batch'],
                        eval_batch_size=config['model_config']['dataset']['batch'], data_dir=data_dir)
    dm.setup("fit")
    model_config = {'model_name_or_path': config['model_config']['model'],
                    'optimizer_name':config['model_config']['optimizer']['type'],
                    'learning_rate': config['model_config']['optimizer']['lr'],
                    'scheduler_name': config['model_config']['optimizer']['scheduler'],
                    'weight_decay': config['model_config']['optimizer']['weight_decay'],
                    'sgd_momentum': config['model_config']['optimizer']['momentum'],
                    'warmup_steps': config['model_config']['training']['warmup'],
                    }
    model = PLMTransformer(
        config=model_config, 
        num_labels=dm.task_metadata['num_labels'],)
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir, save_top_k=1, monitor="val_acc_epoch",filename=f"{config['incumbent_for']}_best", mode="max")
    metric_file = os.path.join(model_dir, f'{config["incumbent_for"]}_metrics.json')
    csv_logger = CSVLogger(save_dir=eval_dir)
    # set up trainer
    trainer = Trainer(
        max_epochs=int(budget),
        accelerator='cpu',
        #accelerator='auto',
        devices="auto",
        logger=[csv_logger],
        limit_train_batches=5,
        limit_val_batches=5,
        log_every_n_steps=0.5,
        val_check_interval=0.5,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=config['model_config']['training']['gradient_accumulation'],
    )
    if train:
        try:
            start = time.time()
            trainer.fit(model, datamodule=dm)
            print(f"Training completed for {config['incumbent_for']} epochs {trainer.default_root_dir}")
            
            end = time.time() - start
            train_metric=convert_tensor_values_to_float(trainer.callback_metrics)
            train_metric['time_taken'] = end
            save_metrics(metric_file, train_metric, key=f'training_metrics', overwrite=True)
        except Exception as e:
            print(f"Exception in training: with config {config['incumbent_for']} and task {config['incumbent_for']} ")
            print(e)
            traceback.print_exc()
    # test the model
    try:
        start = time.time()
        result = trainer.test(ckpt_path=os.path.join(model_dir,f"{config['incumbent_for']}_best.ckpt"),datamodule=dm) if train else trainer.test(model=model, datamodule=dm)
        end = time.time() - start
        test_metric=convert_tensor_values_to_float(trainer.callback_metrics)
        test_metric['time_taken'] = end
        save_metrics(os.path.join(model_dir, f"{config['incumbent_for']}_metrics.json"), 
                     test_metric, 
                     key=f'test_metrics', overwrite=False)
    except Exception as e:
        print(f"Exception in training: with config {config['incumbent_for']} and task {config['incumbent_for']} ")
        print(e)
        traceback.print_exc()

# for non augmented data
def evaluate(args):
    sample_config = {'adam_epsilon': 7.648065011196061e-08,
                     'gradient_accumulation_steps': 8,
                     'gradient_clip_algorithm': 'norm',
                     'learning_rate': 2.8307701958512803e-05,
                     'max_grad_norm': 1.9125507303302376, 'max_seq_length': 128,
                     'model_name_or_path': 'deepset/bert-base-german-cased-oldvocab',
                     'optimizer_name': 'SGD', 'per_device_eval_batch_size': 8,
                     'per_device_train_batch_size': 4, 'scheduler_name': 'cosine_with_warmup',
                     'warmup_steps': 500, 'weight_decay': 8.372735952480551e-05, 'sgd_momentum': 0.12143549900084782}
    
    sample_config = {"adam_epsilon": 1.2372243448105274e-07,
                     "gradient_accumulation_steps": 16,
                     "learning_rate": 3.277577722487855e-05,
                     "max_seq_length": 128,
                     "model_name_or_path": "dbmdz/distilbert-base-german-europeana-cased",
                     "optimizer_name": "Adam",
                     "per_device_train_batch_size": 4,
                     "scheduler_name": "cosine_with_warmup",
                     "warmup_steps": 10,
                     "weight_decay": 0.00011557671486497145,
                     
                     'gradient_clip_algorithm': 'norm', 'max_grad_norm': 1.9125507303302376, 
                     'per_device_eval_batch_size': 8,
                     }
    output = []
    for name in ['mtop_domain', 'tyqiangz', 'omp',"cardiff_multi_sentiment","swiss_judgment_prediction","hatecheck-german","german_argument_mining","tagesschau", 'gnad10']:
        config_file=os.path.join(f'/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test/IncumbentConfigs/{name}_incumbent.yaml')
        try:
            with open(config_file) as in_stream:
                metadata = yaml.safe_load(in_stream)
                sample_config = metadata
        except Exception as e:
            print(f"problem loading config file {config_file}")
            print(e)
            traceback.print_exc()

        print(f"Training for {name}")
        output.append(train_single_config(
        config=sample_config, task_name=name, budget=1, train=False))
    print(output)

    # write to file
    lock = threading.Lock()
    os.makedirs(os.path.join(os.getcwd(), 'SingleConfig',), exist_ok=True)
    output_file = os.path.join(
        os.getcwd(), 'SingleConfig', f'{args.model_name}_meta_dataset_time.txt')
    # create output file

    # open file for appendin
    with lock:
        with open(output_file, 'w') as file:
            # write text to data
            file.write(str(output))


def run_incumbent_aug(config=None, budget=1):
    if config is None:
            config={
    'seed': 42,
    'incumbent_for': 'miam_1X_10Labels',
    'model_config': {
        'model': 'bert-base-uncased',
        'optimizer': {
            'type': 'RAdam',
            'lr': 6.146670783169018e-05,
            'momentum': 0.9,
            'scheduler': 'cosine_with_warmup',
            'weight_decay': 6.265835646508776e-05,
            'adam_epsilon': 8.739737941142407e-08
        },
        'training': {
            'warmup': 100,
            'gradient_accumulation': 4
        },
        'dataset': {
            'name': 'miam_1X_10Labels',
            'seq_length': 512,
            'batch': 8,
            'num_training_samples': 9935,
            'average_text_length': 6.091092098641168,
            'num_labels': 10
        }
    },
    'aug': True,
    'run_info': [],}
    data_dir="/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/tokenized_data"
    # data dir is just tokenized data folder
    if 'aug' in config and config['aug']:
        data_dir= os.path.join(data_dir, 'Augmented')
        task_name = 'augemented'
    else:
        task_name = config['incumbent_for']
    data_dir=os.path.join(data_dir, config['incumbent_for'])
    
    return_val=train_single_config(config, task_name=task_name, budget=budget, data_dir=data_dir, train=True)
    print(return_val)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a Single Config')
    parser.add_argument('--budget', type=int, default=1, help='Budget')
    parser.add_argument('--config_loc', type=str, default='/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test/BoHBCode/IncumbentConfigs',
                        help='Location of the config file')
    args = parser.parse_args()
    # evaluate(args)
    yaml_files = []
    for filename in os.listdir(args.config_loc):
        if filename.endswith('.yaml'):
            file_path = os.path.join(args.config_loc, filename)
            yaml_files.append(file_path)

    for filename in yaml_files: 
        with open(filename, 'r') as file:
            incumbent_config = yaml.safe_load(file)
            dataset=incumbent_config['incumbent_for']
            # 'Bundestag-v2', 'mlsum', 
            dataset_list=['tagesschau', 'german_argument_mining', 'hatecheck-german', 
                          'financial_phrasebank_75agree_german', "gnad10","senti_lex","mtop_domain",
                          "tweet_sentiment_multilingual",'x_stance', 'swiss_judgment_prediction','miam']
            # next((string for string in dataset_list if string in filename), None) get the name
            # any(string in filename for string in dataset_list) 
            if any(string in filename for string in dataset_list):
                run_incumbent_aug(config=incumbent_config, budget=1)

    # alter_aug_key('/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test/BoHBCode/IncumbentConfigs')
# GNAD10 batch size 8 takes 115 s per epoch
# MTOP batch size 8 takes 264 s per epoch
# Cardiff batch size 8 takes 18 s per epoch
# Sentilex batch size 8 takes 34 s per epoch
# OMP batch size 8 takes 785 s per epoch
# Tyqiangz batch size 8 takes 19 s per epoch
# Amazon batch size 8 takes 3600 s per epoch
