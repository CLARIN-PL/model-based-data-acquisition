import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer
from torchmetrics import F1Score, Accuracy, Precision, Recall
import pytorch_lightning as pl


class ClassificationModel(pl.LightningModule):
    def __init__(
        self,
        out_size: int,
        transformer_name: str = "allegro/herbert-base-cased",
        max_length: int = 512,
        lr: float = 1e-5,
        cuda: bool = True,
    ) -> None:
        super().__init__()
        self.is_cuda = cuda and torch.cuda.is_available()
        self.transformer = AutoModel.from_pretrained(transformer_name)
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_name)

        embedding_size = self.transformer.config.hidden_size
        self.out_size = out_size
        self.linear = nn.Linear(embedding_size, self.out_size)
        if self.is_cuda:
            self.transformer = self.transformer.to("cuda")
            self.linear = self.linear.to("cuda")
        # self.sigmoid = nn.Sigmoid()
        self.max_length = max_length

        self.loss_function = nn.CrossEntropyLoss()

        self.metrics = {
            "val": self._prepare_metrics(),
            "test": self._prepare_metrics(),
        }
        self.predictions = {"true": [], "predicted": []}

        self.last_metrics = {}
        self.lr = lr
        self.threshold = 0.0

    def _prepare_metrics(self):
        return {
            "acc_micro": Accuracy(
                task="multilabel", average="micro", num_classes=self.out_size
            ),
            "f1_micro": F1Score(
                task="multilabel", average="micro", num_classes=self.out_size
            ),
            "prec_micro": Precision(
                task="multilabel", average="micro", num_classes=self.out_size
            ),
            "rec_micro": Recall(
                task="multilabel", average="micro", num_classes=self.out_size
            ),
            "acc_macro": Accuracy(
                task="multilabel", average="macro", num_classes=self.out_size
            ),
            "f1_macro": F1Score(
                task="multilabel", average="macro", num_classes=self.out_size
            ),
            "prec_macro": Precision(
                task="multilabel", average="macro", num_classes=self.out_size
            ),
            "rec_macro": Recall(
                task="multilabel", average="macro", num_classes=self.out_size
            ),
            "acc_class": Accuracy(
                task="multilabel", average="none", num_classes=self.out_size
            ),
            "f1_class": F1Score(
                task="multilabel", average="none", num_classes=self.out_size
            ),
            "prec_class": Precision(
                task="multilabel", average="none", num_classes=self.out_size
            ),
            "rec_class": Recall(
                task="multilabel", average="none", num_classes=self.out_size
            ),
        }

    def _prepare_input(self, texts):
        if isinstance(texts, str):
            return [texts]
        return list(texts)

    def forward(self, X):
        return self._shared_step(X)

    def _shared_step(self, X):
        tokens = self.tokenizer.batch_encode_plus(
            self._prepare_input(X),
            padding="longest",
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if self.is_cuda:
            tokens = tokens.to("cuda")

        embeds = self.transformer(**tokens).pooler_output
        out = self.linear(embeds)
        return out

    def training_step(self, batch, batch_idx):
        texts, labels = batch
        # print(labels)
        out = self._shared_step(texts)
        loss = self.loss_function(out, labels)
        return loss

    # def get_predictions_from_output(self, output):
    #     preds = torch.sigmoid(output)
    #     return preds

    def update_metrics(self, stage, labels, preds):
        labels = labels.type(torch.int16)
        for k, metric in self.metrics[stage].items():
            metric.update(preds, labels)

    def calculate_metrics(self, stage):
        metric_values = {}
        for k, metric in self.metrics[stage].items():
            value = metric.compute()
            metric.reset()
            metric_values[f"{stage}_{k}"] = value

        self.last_metrics.update(**metric_values)

        return metric_values

    def validation_step(self, batch, batch_idx):
        texts, labels = batch
        preds = self._shared_step(texts)
        # preds = self.get_predictions_from_output(preds)
        if not self.trainer.sanity_checking:
            self.update_metrics("val", labels.to("cpu"), preds.to("cpu"))

    def validation_epoch_end(self, outputs) -> None:
        if self.trainer.sanity_checking:
            return None

        metrics = self.calculate_metrics("val")
        self.log_dict(metrics)

    def binarization_prediction(self, out):
        preds = torch.zeros_like(out)
        mask = out >= self.threshold
        preds[mask] = 1
        return preds

    def update_predictions(self, labels, preds):
        self.predictions["true"].append(labels)
        predictions = self.binarization_prediction(preds)
        self.predictions["predicted"].append(predictions)

    def test_step(self, batch, batch_idx):
        texts, labels = batch

        # curr = labels[labels == 0.0].sum()
        # print("number of zeros", curr)

        preds = self._shared_step(texts)
        # preds = self.get_predictions_from_output(preds)

        self.update_metrics("test", labels.to("cpu"), preds.to("cpu"))
        self.update_predictions(labels.to("cpu"), preds.to("cpu"))

    def test_epoch_end(self, outputs):
        metrics = self.calculate_metrics("test")

        self.last_metrics["true"] = torch.cat(self.predictions["true"], dim=0).tolist()
        self.last_metrics["predicted"] = torch.cat(
            self.predictions["predicted"], dim=0
        ).tolist()
        self.predictions["true"] = []
        self.predictions["predicted"] = []
        self.log_dict(metrics)

    def configure_optimizers(self):
        params = self.parameters()
        wd = 0.001
        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=wd)
        return optimizer
