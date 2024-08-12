import os
import matplotlib.pyplot as plt
from datasets import load_from_disk
import numpy as np

def load_datasets(directory):
    datasets = {}
    for dataset_name in os.listdir(directory): 
        dataset_path = os.path.join(directory, dataset_name)
        if os.path.isdir(dataset_path) and dataset_name not in ['Augmented', 'germeval2018Fine']:
            dataset = load_from_disk(dataset_path)
            label_map = dataset['train'].features['labels'].int2str if hasattr(dataset['train'].features['labels'], 'int2str') else None
            datasets[dataset_name] = {
                'dataset': dataset,
                'label_map': label_map
            }
    
    return datasets


def plot_class_distribution(datasets):
    for name, data in datasets.items():
        dataset = data['dataset']
        label_map = data['label_map']
        labels = dataset['train']['labels']  # Adjust this key if necessary
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Sort by frequency
        sorted_indices = np.argsort(counts)
        unique_labels = unique_labels[sorted_indices]
        counts = counts[sorted_indices]
        
        # Determine x-tick labels
        if label_map:
            x_labels = [label_map(i) for i in unique_labels]
        else:
            x_labels = unique_labels
        
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(unique_labels, counts, color=plt.cm.Paired(np.arange(len(unique_labels))), edgecolor='black')
        ax.set_title(f'Class Distribution in {name}')
        ax.set_xlabel('Class', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_xticks(unique_labels)
        ax.set_xticklabels(x_labels, rotation=45)
        plt.tight_layout()
        fig.savefig(f'appendix_class_distribution_{name}.png')
        

def plot_text_length_distribution(datasets):
    for name, dataset in datasets.items():
        text_lengths = [len(sentence) for sentence in dataset['train']['sentence']]
        

        plt.figure(figsize=(10, 5))
        plt.hist(text_lengths, bins=50, color='green', edgecolor='black')
        plt.title(f'Text Length Distribution in {name}')
        plt.xlabel('Text Length')
        plt.ylabel('Frequency')
        plt.show()

def all_datasets_average_text_length(datasets):
    dataset_names = []
    avg_text_lengths = []

    for name, data in datasets.items():
        dataset = data['dataset']
        text_lengths = [len(sentence) for sentence in dataset['train']['sentence']]
        avg_text_length = np.mean(text_lengths)
        dataset_names.append(name)
        avg_text_lengths.append(avg_text_length)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(dataset_names, avg_text_lengths, color='orange', edgecolor='black')
    ax.set_title('Average Text Length in Each Dataset')
    ax.set_xlabel('Dataset', fontweight='bold')
    ax.set_ylabel('Average Text Length', fontweight='bold')
    ax.set_xticks(np.arange(len(dataset_names)))
    ax.set_xticklabels(dataset_names, rotation=90)
    plt.tight_layout()
    fig.savefig('chap4_datasets_avg_text_length.png')

def all_datasets_num_training_samples(datasets):
    dataset_names = []
    num_samples = []

    for name, data in datasets.items():
        dataset = data['dataset']
        dataset_names.append(name)
        num_samples.append(len(dataset['train']))

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(dataset_names, num_samples, color='blue', edgecolor='black')
    ax.set_title('Number of Training Samples in Each Dataset')
    ax.set_xlabel('Dataset', fontweight='bold')
    ax.set_ylabel('Number of Training Samples', fontweight='bold')
    ax.set_xticks(np.arange(len(dataset_names)))
    ax.set_xticklabels(dataset_names, rotation=90)
    plt.tight_layout()
    fig.savefig('chap4_datasets_num_training_samples.png')


if __name__ == "__main__":
    datasets_directory = '/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/cleaned_datasets'
    datasets = load_datasets(datasets_directory)
    all_datasets_average_text_length(datasets)
    all_datasets_num_training_samples(datasets)
    plot_class_distribution(datasets)