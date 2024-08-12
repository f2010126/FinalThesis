from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# use export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python to avoid error
model_path = 'microsoft/deberta-v3-small'
clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
tokenizer = AutoTokenizer.from_pretrained(model_path)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = sigmoid(predictions)
    predictions = (predictions > 0.5).astype(int).reshape(-1)
    return clf_metrics.compute(predictions=predictions,
                               references=labels.astype(int).reshape(-1))


def train_multi_label():
    def preprocess_function(example):
        text = f"{example['title']}.\n{example['content']}"
        print(example['all_labels'])
        all_labels = example['all_labels']  # .split(', ')
        labels = [0. for i in range(len(classes))]
        for label in all_labels:
            label_id = class2id[label]
            labels[label_id] = 1.
        example = tokenizer(text, truncation=True)
        example['labels'] = labels
        return example

    def data_process():
        dataset = load_dataset('knowledgator/events_classification_biotech')
        classes = [class_ for class_ in dataset['train'].features['label 1'].names if class_]
        class2id = {class_: id for id, class_ in enumerate(classes)}
        id2class = {id: class_ for class_, id in class2id.items()}
        return dataset, classes, class2id, id2class

    dataset, classes, class2id, id2class = data_process()
    tokenized_dataset = dataset.map(preprocess_function)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                               num_labels=len(classes),
                                                               id2label=id2class,
                                                               label2id=class2id, )
    training_args = TrainingArguments(
        output_dir="my_awesome_model",
        learning_rate=2e-5,
        per_device_train_batch_size=3,
        per_device_eval_batch_size=3,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True, )

    trainer = Trainer(model=model, args=training_args,
                      train_dataset=tokenized_dataset["train"],
                      eval_dataset=tokenized_dataset["test"],
                      tokenizer=tokenizer, data_collator=data_collator,
                      compute_metrics=compute_metrics,
                      )
    trainer.train()


if __name__ == '__main__':
    train_multi_label()
