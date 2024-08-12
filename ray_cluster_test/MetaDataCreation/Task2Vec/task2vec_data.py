import os
import sys
import argparse
import datasets
from transformers import AutoTokenizer

sys.path.append(os.path.abspath('/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test'))


loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

max_seq_length = 512
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-uncased", do_lower_case=False)
def encode_batch(batch):
        """Encodes a batch of input data using the model tokenizer."""
        ## only the 'text' column is encoded
        tokenized_batch = tokenizer(batch["sentence"], max_length=max_seq_length, truncation=True,
                                         padding="max_length")
        # tokenized_batch["labels"] = [.str2int[label] for label in batch["labels"]]
        return tokenized_batch

def load_and_Prep_data(clean_data_path):
     # List of label names (str)
    
    dataset = datasets.load_from_disk(os.path.join(clean_data_path))
    label_names = label_cats.categories
    for split in dataset.keys():
        dataset[split] = dataset[split].map(encode_batch, batched=True)
        dataset[split] = dataset[split].remove_columns(['sentence'])

                    # Transform to pytorch tensors and only output the required columns
        columns = [c for c in dataset[split].column_names if c in loader_columns]
        dataset[split].set_format(type="torch", columns=columns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Task2Vec Data Generation')
    parser.add_argument('--task', type=str, help='The task to generate data for', default='germeval2018_fine')
    parser.add_argument('--clean_data_path', type=str, help='The location of the cleaned data', default='/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/cleaned_datasets/Bundestag-v2')
    args = parser.parse_args()
    load_and_Prep_data(args.clean_data_path)


    