import re
import pandas as pd
from tqdm import tqdm
from abc import ABC
from cleaning.filter import ParagraphFilter, CommentFilter

tqdm.pandas()


class TextCleaner(ABC):
    def __init__(self, text_filter, input_file, output_file=None) -> None:
        self.input_file = input_file
        self.output_file = output_file
        if not self.output_file:
            self.output_file = self.input_file.replace(".csv", "_cleaned.csv")

        self.text_filter = text_filter

    def read_file(self):
        self.df = pd.read_csv(self.input_file)
        return self

    def save_file(self):
        self.df.to_csv(self.output_file, index=False)
        return self

    def remove_whitespaces(self):
        self.df["text"] = self.df["text"].progress_apply(
            lambda text: re.sub(r"\s+", " ", str(text))
        )
        return self

    def clean(self):
        self.remove_whitespaces()
        return self

    def filter_texts(self):
        filtered = self.text_filter.filter_texts(self.df)
        return filtered

    def run(self):
        return self.read_file().clean().save_file()


class ParagraphCleaner(TextCleaner):
    def __init__(
        self,
        input_file="data/paragraphs.csv",
        output_file=None,
        paragraph_count_min=3,
        paragraph_count_max=10,
        filter_args={},
    ) -> None:
        super().__init__(ParagraphFilter(**filter_args), input_file, output_file)

        self.paragraph_count_min = paragraph_count_min
        self.paragraph_count_max = paragraph_count_max

    def clean(self):
        super().clean()
        self.df = self.paragraph_count()
        self.df = self.filter_texts()
        self.df = self.paragraph_count()
        return self

    def _count_inside(self, x):
        count = x.count()
        return self.paragraph_count_min <= count <= self.paragraph_count_max

    def paragraph_count(self):
        article_ids = (
            self.df[["article_id", "id_inside_article"]]
            .groupby("article_id")
            .agg(self._count_inside)
            .reset_index()
        )
        article_ids = article_ids[article_ids["id_inside_article"]]["article_id"]
        df_new = self.df[self.df.article_id.isin(article_ids.tolist())]
        return df_new


class CommentCleaner(TextCleaner):
    def __init__(
        self, input_file="data/comments.csv", output_file=None, filter_args={}
    ) -> None:
        super().__init__(CommentFilter(**filter_args), input_file, output_file)

    def clean(self):
        super().clean()
        self.df = self.filter_texts()
        return self
