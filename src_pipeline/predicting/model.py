import torch
from transformers import AutoModel, AutoTokenizer


class ClassificationModel(torch.nn.Module):
    def __init__(
        self,
        out_size: int,
        transformer_name: str = "allegro/herbert-base-cased",
        max_length: int = 512,
        cuda: bool = True,
    ) -> None:
        super().__init__()
        self.is_cuda = cuda and torch.cuda.is_available()
        self.transformer = AutoModel.from_pretrained(transformer_name)
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_name)

        embedding_size = self.transformer.config.hidden_size
        self.out_size = out_size
        self.linear = torch.nn.Linear(embedding_size, self.out_size)
        if self.is_cuda:
            self.transformer = self.transformer.to("cuda")
            self.linear = self.linear.to("cuda")
        self.max_length = max_length
        self.threshold = 0.0

    def _prepare_input(self, texts):
        if isinstance(texts, str):
            return [texts]
        return list(texts)

    def forward(self, X):
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

    def binarization_prediction(self, out):
        preds = torch.zeros_like(out)
        mask = out >= self.threshold
        preds[mask] = 1
        return preds
