from datasets import load_dataset
from transformers import AutoTokenizer


class MultiLabelDataset():
    def preprocess_function(self, example):
        text = f"{example['title']}.\n{example['content']}"
        all_labels = example['all_labels']  # .split(', ')
        labels = [0. for i in range(len(self.classes))]
        for label in all_labels:
            label_id = self.class2id[label]
            labels[label_id] = 1.
        example = self.tokenizer(text, truncation=True, max_length=512)
        example['labels'] = labels
        return example

    def set_var(self):
        dataset = load_dataset(self.dataset_name)
        self.classes = [class_ for class_ in dataset['train'].features['label 1'].names if class_]
        self.class2id = {class_: id for id, class_ in enumerate(self.classes)}
        self.id2class = {id: class_ for class_, id in self.class2id.items()}
        self.tokenized_dataset = dataset.map(self.preprocess_function)

    def __init__(self, model_path, dataset_name):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.dataset_name = dataset_name
        self.set_var()


class KnowledgatorBioTech(MultiLabelDataset):
    def __init__(self, model_path):
        super().__init__(model_path=model_path,
                         dataset_name='knowledgator/events_classification_biotech')


def get_dataset(dataset_name, model_path):
    dataset_obj = MultiLabelDataset(dataset_name=dataset_name,
                                    model_path=model_path)
    return dataset_obj


if __name__ == '__main__':
    dataset_obj = get_dataset(dataset_name='knowledgator/events_classification_biotech',
                              model_path='microsoft/deberta-v3-small')
    print(dataset_obj.classes)
