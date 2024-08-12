# Copyright (c) [2024] [Dipti Sengupta]
# Licensed under the CC0 1.0 Universal See LICENSE file in the project root for full license information.


from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
import torch
from torch.optim import Adam, AdamW
from transformers import AutoConfig, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, \
    get_polynomial_decay_schedule_with_warmup, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup, \
    get_cosine_with_hard_restarts_schedule_with_warmup, TrainingArguments, Trainer

# local
from data_modules import get_dataset

clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = sigmoid(predictions)
    predictions = (predictions > 0.5).astype(int).reshape(-1)
    return clf_metrics.compute(predictions=predictions,
                               references=labels.astype(int).reshape(-1))


class ModelTrainer:
    def __init__(self, dataset_name, config=None):
        self.config = config
        self.model_path = config['model_path']
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_path'])
        self.dataset = get_dataset(dataset_name, config['model_path'])
        self.scheduler_name = config['scheduler_name']
        self.optimizer_name = config['optimizer_name']

    def set_scheduler(self, optimizer, num_training_steps):
        if self.scheduler_name == "linear_with_warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config['warmup_steps'],
                num_training_steps=num_training_steps,
            )
        elif self.scheduler_name == "cosine_with_warmup":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config['warmup_steps'],
                num_training_steps=num_training_steps,
            )

        elif self.scheduler_name == "constant_with_warmup":
            scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config['warmup_steps'],
            )
        elif self.scheduler_name == "polynomial_decay_with_warmup":
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config['warmup_steps'],
                num_training_steps=num_training_steps,
            )
        elif self.scheduler_name == "cosine_with_hard_restarts_with_warmup":
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config['warmup_steps'],
                num_training_steps=num_training_steps,
            )
        else:
            raise ValueError(f"Invalid scheduler {self.scheduler_name} config: {self.config}")

        return scheduler

    def set_optimiser_scheduler(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config['weight_decay'],
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.optimizer_name == "AdamW":
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.config['learning_rate'],
                              )
        elif self.optimizer_name == "Adam":
            optimizer = Adam(optimizer_grouped_parameters, lr=self.config['learning_rate'],
                             )
        elif self.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=self.config['learning_rate'],
                                        momentum=self.config['sgd_momentum'])
        elif self.optimizer_name == "RAdam":
            optimizer = torch.optim.RAdam(optimizer_grouped_parameters, lr=self.config['learning_rate'],
                                          )
        else:
            raise ValueError(f"Invalid optimizer {self.optimizer_name} config: {self.config}")

        num_training_steps = len(self.dataset.tokenized_dataset['train']) // self.config['batch_size'] * self.config[
            'num_epochs']
        scheduler = self.set_scheduler(optimizer, num_training_steps)

        return optimizer, scheduler

    def train_model(self):
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.config['model_path'],
                                                                        num_labels=len(self.dataset.classes),
                                                                        id2label=self.dataset.id2class,
                                                                        label2id=self.dataset.class2id,
                                                                        )
        optimiser, scheduler = self.set_optimiser_scheduler()
        training_args = TrainingArguments(
            output_dir="my_awesome_model",
            learning_rate=2e-5,
            per_device_train_batch_size=3,
            per_device_eval_batch_size=3,
            num_train_epochs=2,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )

        trainer = Trainer(model=self.model, args=training_args,
                          train_dataset=self.dataset.tokenized_dataset["train"],
                          eval_dataset=self.dataset.tokenized_dataset["test"],
                          tokenizer=self.tokenizer, data_collator=data_collator,
                          compute_metrics=compute_metrics,
                          optimizers=(optimiser, scheduler),
                          )
        trainer.train()


if __name__ == '__main__':
    config = {'model_path': 'microsoft/deberta-v3-small',
              'dataset_name': 'knowledgator/events_classification_biotech',
              'batch_size': 3,
              'num_epochs': 2,
              'learning_rate': 2e-5,
              'scheduler_name': 'linear_with_warmup',
              'optimizer_name': 'AdamW',
              'weight_decay': 0.01,
              'warmup_steps': 0,
              'max_seq_length': 512, }
    model_trainer = ModelTrainer('knowledgator/events_classification_biotech', config=config)
    model_trainer.train_model()
