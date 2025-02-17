import logging
from typing import Optional

import evaluate
import torch
import torchmetrics
from lightning import LightningModule
from statistics import mean
from torch.optim import Adam, AdamW
from transformers import AutoConfig, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, \
    get_polynomial_decay_schedule_with_warmup, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup, \
    get_cosine_with_hard_restarts_schedule_with_warmup

# import local file
try:
    from .experiment_utilities import remove_checkpoint_files
except ImportError:
    from experiment_utilities import remove_checkpoint_files

class PLMTransformer(LightningModule):
    def __init__(
            self,
            config,
            num_labels: int,
            **kwargs,
    ):
        super(PLMTransformer, self).__init__()

        # access validation outputs, save them in-memory as instance attributes
        self.validation_step_outputs = []
        self.training_step_outputs = []

        self.task = 'binary' if num_labels == 2 else 'multiclass'
        self.config = config

        self.save_hyperparameters()
        self.accuracy = torchmetrics.Accuracy(task=self.task, num_classes=num_labels)

        self.model_config = AutoConfig.from_pretrained(config['model_name_or_path'], num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(config['model_name_or_path'],
                                                                        config=self.model_config)
        # self.metric = evaluate.load(
        #     "glue", self.hparams.task_name, experiment_id=datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        # )
        self.accuracy = torchmetrics.Accuracy(task=self.task, num_classes=num_labels)
        self.optimizer_name = config['optimizer_name']
        self.scheduler_name = config['scheduler_name']
        self.train_acc = evaluate.load('accuracy')
        self.train_f1 = evaluate.load('f1')
        # self.train_bal_acc = evaluate.load('hyperml/balanced_accuracy')

        self.prepare_data_per_node = True

    # Training
    def forward(self, **inputs):
        return self.model(**inputs)

    def on_fit_start(self) -> None:
        pass

    def evaluate_step(self, batch, batch_idx, stage='val'):
        outputs = self(**batch)
        loss, logits = outputs[:2]

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        # calculate pred
        labels = batch["labels"]

        acc = self.accuracy(preds, labels)
        self.train_f1.add_batch(predictions=preds, references=labels)
        self.train_acc.add_batch(predictions=preds, references=labels)
        # self.train_bal_acc.add_batch(predictions=preds, references=labels)
        f1 = self.train_f1.compute(average='weighted')['f1']
        # bal_acc = self.train_bal_acc.compute()['balanced_accuracy']
        train_acc = self.train_acc.compute()['accuracy']

        self.log(f'{stage}_acc', acc, sync_dist=True, on_step=True)
        self.log(f'{stage}_loss', loss, sync_dist=True, on_step=True)
        self.log(f'{stage}_f1', f1, sync_dist=True, on_step=True)
        # self.log(f'{stage}_bal_acc', bal_acc, sync_dist=True, on_step=True)
        return {f"loss": loss, f"accuracy": acc, f"f1": f1, }

    def training_step(self, batch, batch_idx, dataloader_idx=0, print_str="train"):
        result = self.evaluate_step(batch, batch_idx, stage='train')
        self.training_step_outputs.append({"train_loss": result["loss"], "train_accuracy": result["accuracy"],
                                           "train_f1": result["f1"],  # "train_bal_acc": result["bal_acc"]
                                           })

        return result

    def validation_step(self, batch, batch_idx, dataloader_idx=0, print_str="val"):
        result = self.evaluate_step(batch, batch_idx, stage='val')
        self.validation_step_outputs.append({"val_loss": result["loss"], "val_accuracy": result["accuracy"],
                                             "val_f1": result["f1"],  # "val_bal_acc": result["bal_acc"]
                                             })
        return result

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        values = self.evaluate_step(batch, batch_idx, stage='test')
        return values

    def on_train_epoch_end(self):
        outputs = self.training_step_outputs
        avg_loss = torch.stack([x["train_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["train_accuracy"] for x in outputs]).mean()
        avg_f1 = mean([x["train_f1"] for x in outputs])
        # avg_bal_acc = mean([x["train_bal_acc"] for x in outputs])

        self.log("ptl/train_loss", avg_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("ptl/train_accuracy", avg_acc, on_step=False, on_epoch=True, logger=True,
                 sync_dist=True)
        self.log("ptl/train_f1", avg_f1, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        #  self.log("ptl/train_bal_acc", avg_bal_acc, on_step=False, on_epoch=True,  logger=True, sync_dist=True)

        return {"loss": avg_loss, "acc": avg_acc, "f1": avg_f1, }

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        avg_f1 = mean([x["val_f1"] for x in outputs])
        # avg_bal_acc = mean([x["val_bal_acc"] for x in outputs])

        self.log("ptl/val_loss", avg_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("ptl/val_accuracy", avg_acc, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("ptl/val_f1", avg_f1, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        # self.log("ptl/val_bal_acc", avg_bal_acc, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        logging.debug("on_validation_epoch_end--->")
        return {"loss": avg_loss, "acc": avg_acc, "f1": avg_f1, }

    def on_validation_end(self):
        # last hook that's used by Trainer in ray.
        logging.debug("on_validation_end")

    # Optimizers
    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
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

        if self.scheduler_name == "linear_with_warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config['warmup_steps'],
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        elif self.scheduler_name == "cosine_with_warmup":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config['warmup_steps'],
                num_training_steps=self.trainer.estimated_stepping_batches,
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
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        elif self.scheduler_name == "cosine_with_hard_restarts_with_warmup":
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config['warmup_steps'],
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        else:
            raise ValueError(f"Invalid scheduler {self.scheduler_name} config: {self.config}")
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    # data
