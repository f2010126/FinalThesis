# Taken from https://github.com/j0st/german-paraphraser/blob/main/paraphraser.py
import random
import re
import torch
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
import argparse
import os
import datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T5ForConditionalGeneration.from_pretrained("seduerr/t5_base_paws_ger")
model = model.to(device)
tokenizer = T5Tokenizer.from_pretrained("t5-base")

def generate_paraphrase(attention_masks, input_ids,num_paraphrases=3):
    paraphrases = []
    beam_outputs = model.generate(
                input_ids=input_ids, attention_mask=attention_masks,
                do_sample=True,
                max_length=512,
                top_k=120,
                top_p=0.98,
                early_stopping=True,
                num_return_sequences=num_paraphrases
            )
    for i, line in enumerate(beam_outputs):
        paraphrase = tokenizer.decode(line, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        paraphrases.append(paraphrase)
    
    return paraphrases


def generate_paraphrases_for_entry(entry, generate_paraphrase, num_paraphrases=3):
    encoding = tokenizer.encode_plus(entry['sentence'], pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
    label = entry['labels']
    
    # Generate paraphrases
    paraphrases = generate_paraphrase(attention_masks, input_ids,num_paraphrases)
    
    # Prepare new entries
    new_entries = []
    for paraphrase in paraphrases:
        new_entries.append({
            "sentence": paraphrase,
            "labels": label
        })
    
    return new_entries

def augment_with_paraphrases(dataset,  num_paraphrases= 3):
    # Use map function to generate new entries
    new_train_dataset = dataset['train'].map(
        lambda entry: generate_paraphrases_for_entry(entry, generate_paraphrase, num_paraphrases),
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    
    # Combine the new train dataset with the original validation and test sets
    new_dataset = DatasetDict({
        'train': new_train_dataset,
        'validation': dataset['validation'],
        'test': dataset['test']
    })
    
    return new_dataset


def augment_dataset(dataset_path,save_folder,num_paraphrases=3):
    # Load dataset
    dataset = datasets.load_from_disk(dataset_path)
    new_dataset = augment_with_paraphrases(dataset, num_paraphrases)    
    new_dataset.save_to_disk(save_folder)
    print(f"Saved augmented dataset to {save_folder}")

def start_aug(folder_path,num_paraphrases=3):
    subfolders = [os.path.join(folder_path, subfolder) for subfolder in os.listdir(folder_path)
                  if os.path.isdir(os.path.join(folder_path, subfolder)) and "1X" in subfolder]
    
    for subfolder in subfolders:
        dataset_path = subfolder  # Assuming the dataset is directly in the subfolder
        subfolder_name = os.path.basename(subfolder)
        new_subfolder_name = subfolder_name.replace('1X', '5X')
        save_path = os.path.join(folder_path, new_subfolder_name)
        
        # Create the new folder if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        augment_dataset(dataset_path, save_folder=save_path, num_paraphrases=num_paraphrases)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Augment dataset with paraphrases')
    parser.add_argument('--folder_directory',type=str, 
                        help='Directory with run results pkl file',default='/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/cleaned_datasets/Augmented')
    # store the result under IncumbentConfigs/dataset/file.yaml 
    args = parser.parse_args()

    # load the dataset with 1X in the name, this is the original dataset, we will augment it and save it to 5X folder
    start_aug(args.folder_directory,num_paraphrases=3)
