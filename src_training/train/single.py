from model.model import ClassificationModel
from dataset.datamodule import AnnotatedDataModule
import pytorch_lightning as pl
import numpy as np
from copy import copy
import json
from os import path
from train.dimension_lists_py import dimension_splits, label_dims
import torch


def get_model(out_size):
    return ClassificationModel(out_size=out_size)


def get_trainer():
    callback_es = pl.callbacks.EarlyStopping(monitor="val_f1_macro", patience=3)

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        auto_select_gpus=True,
        max_epochs=10,
        callbacks=[callback_es],
    )
    return trainer


def single_training(folds=5):
    threshold = 0.15
    datamodule = AnnotatedDataModule(
        folds=folds, preprocessing_args=(threshold, "mean"), label_columns=label_dims
    )
    datamodule.prepare_data()
    datamodule.setup(dev_fold=-1)

    model = get_model(len(label_dims))
    trainer = get_trainer()

    trainer.fit(model=model, datamodule=datamodule)
    metrics = model.last_metrics

    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            metrics[k] = v.tolist()
    fname = f"single_training_{threshold}.json"
    save_metrics(metrics, fname)


def save_metrics(metrics, name):
    json_ = json.dumps(metrics)
    filename = path.join("metrics", name)
    with open(filename, "w") as f:
        json.dump(json_, f, indent=4)
