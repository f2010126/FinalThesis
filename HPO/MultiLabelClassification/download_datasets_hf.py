"""
Download a list of datasets from Hugging Face to a given location
"""

import os

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer


# Swiss judegment Test at https://huggingface.co/datasets/rcds/occlusion_swiss_judgment_prediction

# Download a list of datasets from Hugging Face to a given location
# parse with models and max_seq_length


def download_raw_datasets(dataset_list):
    # add documenation
    """
    Download a list of datasets from Hugging Face to a given location
    :param dataset_list: list of datasets to download
    """
    dowloadpath = os.path.join(os.getcwd(), "raw_datasets")

    # check for download path
    if not os.path.exists(dowloadpath):
        os.makedirs(dowloadpath)
    for dataset_name in dataset_list:
        print("Downloading dataset: ", dataset_name)
        try:
            # might need 'de', 'german', 'vm2', 
            dataset = load_dataset(dataset_name)
            print(dataset)
        except Exception as e:
            print("Error downloading dataset: ", dataset_name)
            print(e)
            continue

        dataset_folder = dataset_name.split("/")[-1]  # last part of the url
        print(f'Download paths---->: {dowloadpath} {dataset_folder}')
        dataset.save_to_disk(os.path.join(dowloadpath, dataset_folder))
        print("Done downloading dataset: ", dataset_name)


def tokenise_datasets():
    # add documenation
    """
    Tokenise a list of datasets using a list of models and max_seq_length
    """
    raw_data_path = os.path.join(os.getcwd(), "raw_datasets")
    data_folder = [dataset_name.split("/")[-1] for dataset_name in dataset_list]
    models = ["bert-base-uncased", "bert-base-multilingual-cased", "deepset/bert-base-german-cased-oldvocab",
              "uklfr/gottbert-base", "dvm1983/TinyBERT_General_4L_312D_de",
              "linhd-postdata/alberti-bert-base-multilingual-cased",
              "dbmdz/distilbert-base-german-europeana-cased"]
    max_seq_length = [128, 256, 512]
    for dataset_name in dataset_list:
        print("Tokenising dataset: ", dataset_name)
        dataset_folder = dataset_name.split("/")[-1]  # last part of the url
        dataset = load_from_disk(os.path.join(raw_data_path, dataset_folder))
        for model in models:
            for seq_length in max_seq_length:
                print(f"Tokenising dataset: {dataset_name} Model: {model} Max_seq_length: {seq_length}")
                tokenizer = AutoTokenizer.from_pretrained(model)
                tokenized_dataset = dataset.map(
                    lambda examples: tokenizer(examples['text'], padding='max_length', truncation=True,
                                               max_length=seq_length), batched=True)
                tokenized_dataset.save_to_disk(os.path.join(raw_data_path, f"{dataset_folder}_{model}_{seq_length}"))
                print(f"Done tokenising dataset: {dataset_name} Model: {model} Max_seq_length: {seq_length}")


if __name__ == "__main__":
    dataset_list = [
                    ]
    download_raw_datasets(dataset_list=['gwlms/germeval2018'])

    print("Done")
