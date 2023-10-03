from predicting.model import ClassificationModel
from utils.config import LABEL_DIMS
from torch import load
import pandas as pd
from tqdm import tqdm
from os.path import join


class TextPredictor:
    def __init__(
        self,
        input_file,
        output_file,
        ckpt_file="lightning_logs/version_2/checkpoints/epoch=3-step=152.ckpt",
        batch_size=16,
    ) -> None:
        self.ckpt_file = ckpt_file
        self.output_file = output_file
        self.batch_size = batch_size
        self.input_file = input_file

    def load_data(self):
        self.data = pd.read_csv(self.input_file)
        return self

    def load_model(self):
        self.model = ClassificationModel(out_size=len(LABEL_DIMS))
        self.model.load_state_dict(load(self.ckpt_file)["state_dict"])
        self.model.eval()
        return self

    def predict_batch(self, text_batch):
        out = self.model(text_batch)
        preds = self.model.binarization_prediction(out).cpu()
        return preds

    def save_prediction(self, batch, batch_idx):
        if batch_idx == 0:
            batch.to_csv(self.output_file, index=False)
            return
        batch.to_csv(
            self.output_file,
            index=False,
            mode="a",
            header=False,
        )

    def predict(self):
        size = self.data.shape[0]
        for idx in tqdm(range(0, size, self.batch_size)):
            curr_batch = self.data.iloc[idx : idx + self.batch_size]
            preds = self.predict_batch(curr_batch["text"].tolist())
            curr_batch[LABEL_DIMS] = preds
            self.save_prediction(curr_batch, idx)
        return self

    def run(self):
        return self.load_model().load_data().predict()
