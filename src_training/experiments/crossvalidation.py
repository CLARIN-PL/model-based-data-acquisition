import pytorch_lightning as pl
from model.model import ClassificationModel
from dataset.datamodule import AnnotatedDataModule, AnnotatedDataset
from experiments.B_dataset import BDataset
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader


class CrossValidation:
    def __init__(
        self,
        k_folds=10,
        datamodule_class=AnnotatedDataModule,
        dataset_class=AnnotatedDataset,
        name="doccano1",
        transformer_name="allegro/herbert-base-cased",
    ) -> None:
        self.k = k_folds
        self.datamodule_class = datamodule_class
        self.dataset_class = dataset_class
        self.name = name
        self.transformer_name = transformer_name

    def get_model(self, out_size=None):
        if out_size is None:
            out_size = len(self.dataset_class.LABEL_COLUMNS)
        return ClassificationModel(
            out_size=out_size, transformer_name=self.transformer_name
        )

    def get_trainer(self):
        callback_es = pl.callbacks.EarlyStopping(monitor="val_f1_macro", patience=3)

        trainer = pl.Trainer(
            enable_checkpointing=False,
            accelerator="auto",
            devices=1,
            auto_select_gpus=True,
            max_epochs=10,
            callbacks=[callback_es],
        )
        return trainer

    def one_A_iteration(self, dev_fold, preprocessing_args=(0.15, "mean")):
        datamodule = self.datamodule_class(
            folds=self.k, preprocessing_args=preprocessing_args, dev_fold=dev_fold
        )
        if dev_fold == 0:
            datamodule.prepare_data()
        # print("iter fold", dev_fold)
        datamodule.setup()
        model = self.get_model()
        trainer = self.get_trainer()
        trainer.fit(model=model, datamodule=datamodule)
        trainer.test(model=model, datamodule=datamodule)
        test_data = datamodule.data[datamodule.data["split"] == "test"]
        # print(test_data["fold"].value_counts())
        # print(datamodule.data[datamodule.data["split"] == "train"]["fold"].value_counts())
        # print(datamodule.data[datamodule.data["split"] == "dev"]["fold"].value_counts())

        originals = model.last_metrics["true"]
        predictions = model.last_metrics["predicted"]
        texts = test_data["text"].to_list()
        folds = test_data["fold"].to_list()
        label_names = datamodule.label_columns
        return texts, originals, predictions, label_names, folds

    def one_B_iteration(self, dev_fold, data):
        test_fold = (dev_fold + 1) % self.k
        train_data = data[data["fold"] != test_fold]
        train_data = train_data[train_data["fold"] != test_fold]
        dev_data = train_data[train_data["fold"] == dev_fold]
        test_data = data[data["fold"] == test_fold]

        train_loader = DataLoader(
            BDataset(train_data, "train"), shuffle=True, batch_size=16
        )
        
        dev_loader = DataLoader(BDataset(dev_data, "train"), batch_size=16)
        test_loader = DataLoader(BDataset(test_data, "test"), batch_size=16)

        model = self.get_model()
        trainer = self.get_trainer()

        trainer.fit(
            model=model, train_dataloaders=train_loader, val_dataloaders=dev_loader
        )
        trainer.test(model=model, dataloaders=test_loader)

        A_labels = test_data["originals"].to_list()
        B_labels = test_data["predicted"].to_list()
        predictions = model.last_metrics["predicted"]
        texts = test_data["text"].to_list()
        folds = test_data["fold"].to_list()
        return texts, A_labels, B_labels, predictions, folds

    def run_A_B(self, run_A=True, run_B=False):
        if run_A:
            self.new_data = {"text": [], "originals": [], "predicted": [], "fold": []}
            for i in tqdm(range(self.k)):
                texts, originals, predictions, labels, folds = self.one_A_iteration(i)
                self.new_data["text"].extend(texts)
                self.new_data["originals"].extend(originals)
                self.new_data["predicted"].extend(predictions)
                self.new_data["fold"].extend(folds)
            self.new_data = pd.DataFrame(data=self.new_data)
            self.new_data.to_csv(f"data/train/B_{self.name}.csv", index=False)
        self.new_data = pd.read_csv(f"data/train/B_{self.name}.csv")
        if run_B:
            self.C_data = {
                "text": [],
                "A_label": [],
                "B_label": [],
                "predicted": [],
                "fold": [],
            }
            for i in tqdm(range(self.k)):
                texts, A_labels, B_labels, predictions, folds = self.one_B_iteration(
                    i, self.new_data
                )
                self.C_data["text"].extend(texts)
                self.C_data["A_label"].extend(A_labels)
                self.C_data["B_label"].extend(B_labels)
                self.C_data["predicted"].extend(predictions)
                self.C_data["fold"].extend(folds)
            self.C_data = pd.DataFrame(data=self.C_data)
            self.C_data.to_csv(f"data/train/C_{self.name}.csv", index=False)

    def one_increment_iteration(self, dev_fold, preprocessing_args=(0.15, "mean")):
        datamodule = self.datamodule_class(
            folds=self.k, preprocessing_args=preprocessing_args, dev_fold=dev_fold
        )
        if dev_fold == 0:
            datamodule.prepare_data()
        datamodule.setup()
        train_folds = [
            i for i in range(self.k) if i != dev_fold and i != (dev_fold + 1) % self.k
        ]
        results = {k + 1: None for k in range(len(train_folds))}
        for i in range(len(train_folds)):
            curr_train_folds = train_folds[: (i + 1)]
            train_data = datamodule.data[datamodule.data["fold"].isin(curr_train_folds)]
            train_loader = DataLoader(
                self.dataset_class(train_data, datamodule.label_columns),
                batch_size=datamodule.batch_size,
                shuffle=True,
            )
            model = self.get_model()
            trainer = self.get_trainer()
            trainer.fit(
                model=model,
                train_dataloaders=train_loader,
                val_dataloaders=datamodule.val_dataloader(),
            )
            trainer.test(model=model, dataloaders=datamodule.test_dataloader())
            if "original" not in results:
                results["original"] = model.last_metrics["true"]
            results[len(curr_train_folds)] = model.last_metrics["predicted"]

        test_fold = (dev_fold + 1) % self.k
        df_results = pd.DataFrame(data=results)
        df_results["test_fold"] = test_fold

        return df_results

    def run_increment(self):
        df = None
        for i in tqdm(range(self.k)):
            curr_df = self.one_increment_iteration(dev_fold=i)
            if df is None:
                df = curr_df
            else:
                df = pd.concat([df, curr_df], axis=0)
        df.to_csv(f"data/train/increment_{self.name}.csv", index=False)

    def classic_iteration(self, dev_fold, datamodule):
        if dev_fold == 0:
            datamodule.prepare_data()
        model = self.get_model(out_size=len(datamodule.label_columns))
        trainer = self.get_trainer()
        trainer.fit(model=model, datamodule=datamodule)
        trainer.test(model=model, datamodule=datamodule)
        test_labels = model.last_metrics["true"]
        test_predicted = model.last_metrics["predicted"]
        test_texts = datamodule.get_split("test").data["text"].tolist()
        df = pd.DataFrame(
            data={
                "labels": test_labels,
                "predictions": test_predicted,
                "text": test_texts,
            }
        )
        df["fold"] = (datamodule.dev_fold + 1) % self.k
        return df

    def run_threshold(self):
        thresholds = [0.1, 0.15, 0.20, 0.25]
        df = None
        for th in tqdm(thresholds):
            for i in range(self.k):
                datamodule = self.datamodule_class(
                    folds=self.k, preprocessing_args=(th, "mean"), dev_fold=i
                )
                results = self.classic_iteration(i, datamodule)
                results["threshold"] = th
                if df is None:
                    df = results
                else:
                    df = pd.concat([df, results], axis=0)
        df.to_csv(f"data/train/thresholds_{self.name}.csv", index=False)

    def run_single(self):
        df = None
        for label in tqdm(self.dataset_class.LABEL_COLUMNS):
            curr_labels = [label]
            for i in range(self.k):
                datamodule = self.datamodule_class(
                    folds=self.k,
                    preprocessing_args=(0.15, "mean"),
                    dev_fold=i,
                    label_columns=curr_labels,
                )
                results = self.classic_iteration(i, datamodule)
                results["label"] = label
                if df is None:
                    df = results
                else:
                    df = pd.concat([df, results], axis=0)
        df.to_csv(f"data/train/single_{self.name}.csv", index=False)
