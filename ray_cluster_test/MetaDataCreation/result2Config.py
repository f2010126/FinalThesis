# Created date: 2021-05-20
import argparse

from gettext import find
import os

import hpbandster.core.result as hpres
import yaml

from copy import deepcopy
import json
import pickle
import sys
import re
sys.path.append(os.path.abspath('/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test'))

DATASET_LIST_LOC='/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test/BoHBCode/datasets.txt'
"""
Read the result.pkl for the runs and convert the best config to a yaml file
The results location needs the configs, results jsons and the .pkl file.
:param working_dir: the directory with the results
"""

def extract_metadata(metadata_file_path):
    # Example of extracting metadata from a .pkl file
    with open(metadata_file_path, 'r') as f:
        data = json.load(f)

    metadata = {
        'task_name': data.get('task_name', 'default_task'),
        'num_labels': data.get('num_labels', 0),
        'average_text_length': data.get('average_text_length', 0),
        'num_training_samples': data.get('num_training_samples', 0)
    }
    return metadata

def read_dataset_names(dataset_list_loc):
    with open(dataset_list_loc, 'r') as file:
        dataset_names = [line.strip() for line in file.readlines()]
    return dataset_names


def incumbent_to_yaml(incumbent_config, default_config):
    # write the incumbent to the format as default
    mc = deepcopy(default_config)
    mc['model_config']['model'] = incumbent_config['model_name_or_path']
    mc['model_config']['optimizer']['type'] = incumbent_config['optimizer_name']
    mc['model_config']['optimizer']['lr']=incumbent_config['learning_rate']
    mc['model_config']['optimizer']['scheduler']=incumbent_config['scheduler_name']
    mc['model_config']['optimizer']['weight_decay']=incumbent_config['weight_decay']
    mc['model_config']['optimizer']['adam_epsilon']=incumbent_config['adam_epsilon']

    mc['model_config']['training']['warmup']=incumbent_config['warmup_steps']
    mc['model_config']['training']['gradient_accumulation']=incumbent_config['gradient_accumulation_steps']

    mc['model_config']['dataset']['seq_length'] = incumbent_config['max_seq_length']
    mc['model_config']['dataset']['batch']= incumbent_config['per_device_train_batch_size']

    return mc

def create_yaml(working_dir, dataset='gnad10', result_dir='IncumbentConfigs', metadata = None):
    if metadata is None:
        raise ValueError("Metadata is required and cannot be None")
    
    required_keys = ['task_name', 'num_labels', 'average_text_length', 'num_training_samples']
    for key in required_keys:
        if key not in metadata:
            raise KeyError(f"Metadata is missing required key: {key}")
    # load the example run from the log files
    result = hpres.logged_results_to_HBS_result(working_dir)
    id2conf = result.get_id2config_mapping()

    inc_id = result.get_incumbent_id()
    inc_runs = result.get_runs_by_id(inc_id)

    run_info = []
    for run in inc_runs:
        run_info.append({'budget': run.budget, 'info': run.info})
    inc_config = id2conf[inc_id]['config']

    # Read the default config
    default_config_path=os.path.join('/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test/MetaDataCreation','default.yaml')
    with open(default_config_path) as in_stream:
        default_config = yaml.safe_load(in_stream)
    
    format_incumbent = incumbent_to_yaml(inc_config, default_config)
    # add the metadata
    format_incumbent['incumbent_for'] = metadata['task_name'] # its from the datasetruns so HPO
    format_incumbent['model_config']['dataset']['name']=metadata['task_name'] # That it trained on. NOT THE SAME AS THE INCUMBENT where it is best.
    format_incumbent['model_config']['dataset']['num_labels'] = metadata['num_labels']
    format_incumbent['model_config']['dataset']['average_text_length'] = metadata['average_text_length']
    format_incumbent['model_config']['dataset']['num_training_samples'] = metadata['num_training_samples']
    format_incumbent['run_info']=json.dumps(run_info)
    # model and dataset are the last part of the name
    dataset_name = format_incumbent['model_config']['dataset']['name'].split('/')[-1]

    # create the output folder
    if not os.path.exists(os.path.join(os.getcwd(),result_dir)):
        os.makedirs(os.path.join(os.getcwd(),result_dir), exist_ok=True)    
    output_path = os.path.join(os.getcwd(),result_dir, f"{dataset_name}_incumbent.yaml")
    with open(output_path, "w+") as out_stream:
        yaml.dump(format_incumbent, out_stream)

def convert(result_dir):
    cleaned_dataset_loc='/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model'
    pkl_files =find_pkl_files(working_dir)
    dataset_strings=read_dataset_names(DATASET_LIST_LOC)
    for file_path in pkl_files:
        file_name = os.path.basename(file_path)
        folder_path = os.path.dirname(file_path)
        dataset_name = extract_dataset_name(file_name, dataset_strings)
        print(f"Dataset Name: {dataset_name}")
        aug_metadata_loc = args.metadata_loc + '/Augmented' if re.search(r'\d+X', file_name) else args.metadata_loc
        metadata_loc = os.path.join(cleaned_dataset_loc,aug_metadata_loc, dataset_name, 'metadata.json')
        metadata = extract_metadata(metadata_loc)
        metadata['task_name'] = dataset_name
        print(f"Metadata: {metadata}")
        try:
            create_yaml(working_dir=folder_path, dataset=metadata['task_name'], result_dir=result_dir, metadata=metadata)
            print(f"Created YAML for {metadata['task_name']}") 
        except FileNotFoundError as e:
            print(f" Some File was not Found, {e}" )
            exit(1)
        except Exception as e:
            print(f"Error Here {e}")


def read_pkls(working_dir, result_dir):
    """ 
    :param working_dir: the directory with the bohb run results as sub folders
    :param result_dir: the directory to save the yaml files
    """
    dataset_strings=read_dataset_names(DATASET_LIST_LOC)
    for folder_name, _, files in os.walk(working_dir):
        cleaned_dataset_loc='/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model'
        for file_name in files:
            if file_name.endswith('.pkl'):
                pkl_file_path = os.path.join(folder_name, file_name)
                dataset_name = longest_exact_match_in_filename(file_name, dataset_strings)
                if dataset_name is None:
                    continue

                # load the metadata
                metadata_loc = os.path.join(cleaned_dataset_loc,args.metadata_loc, dataset_name[0], 'metadata.json')
                metadata = extract_result_from_pkl(pkl_file_path)
                metadata['task_name'] = dataset_name
                
                try:
                    create_yaml(working_dir=folder_name, dataset=dataset_name[0], result_dir=result_dir, metadata=metadata)  
                except FileNotFoundError:
                    print(f"Metadata file not found at {metadata_loc}")
                    exit(1)
                except Exception as e:
                    print(f"Error Here {e}")


def extract_dataset_name(file_name, valid_dataset_names):
    # Extract the base name without the extension
    base_name = os.path.splitext(file_name)[0]
    
    # Find the longest valid dataset name within the base name
    dataset_name = ''
    for name in valid_dataset_names:
        if name in base_name and len(name) > len(dataset_name):
            dataset_name = name
    return dataset_name

def find_pkl_files(starting_directory):
    pkl_files = []
    for root, dirs, files in os.walk(starting_directory):
        # We are only interested in the first level subdirectories
        if root == starting_directory:
            for dir in dirs:
                subdir = os.path.join(root, dir)
                for file in os.listdir(subdir):
                    if file.endswith('.pkl'):
                        file_path = os.path.join(subdir, file)
                        pkl_files.append(file_path)
    return pkl_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Run to Yaml')
    parser.add_argument('--result_directory',type=str, 
                        help='Directory with run results pkl file',default='financial_phrasebank_Bohb_P2_25_5')
    parser.add_argument('--dataset',type=str, 
                        help='Dataset name',default='tweet_sentiment_multilingual')
    parser.add_argument('--metadata_loc',type=str, 
                        help='File where metadata is stored',default='cleaned_datasets')
    parser.add_argument('--save_location',type=str, 
                        help='Directory with run results pkl file',default='IncumbentConfigs')
    # store the result under IncumbentConfigs/dataset/file.yaml 
    args = parser.parse_args()

    working_dir = os.path.join(os.getcwd(),'../datasetruns')
    working_dir = os.path.join('/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test/BoHBCode/datasetruns')
    save_location = os.path.join('/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test/BoHBCode',args.save_location)
    convert(save_location)



####################################################################################################
    # redo financial_phrasebank_Bohb_P2_25_5
    # mlsum is missing


   