from utils.text_type import TextType
from utils.config import LABEL_DIMS
from clustering.clustering_model import ClusteringModel
from clustering.tuner import ClusterTuner
import pandas as pd
from os.path import join


class TextClusterizer:
    def __init__(
        self,
        paragraph_file="data/predicted_paragraphs_cleaned.csv",
        comment_file="data/predicted_comments_cleaned.csv",
        output_file="data/cluster_id.csv",
        min_comments_for_article=10,
        testing_model_output_dir="data",
        comment_k=2,
        article_k=2,
    ) -> None:
        self.paragraph_file = paragraph_file
        self.comment_file = comment_file
        self.output_file = output_file
        self.min_comments_for_article = min_comments_for_article

        self.testing_model_output_dir = testing_model_output_dir
        self.cluster_no = {TextType.COMMENT: comment_k, TextType.ARTICLE: article_k}

    def read_comments(self):
        comments = pd.read_csv(self.comment_file)
        return comments

    def _agg_bool_sum(self, x):
        s = x.sum()
        return 1 if s > 0 else 0

    def read_articles(self):
        articles = pd.read_csv(self.paragraph_file)
        articles = (
            articles[["article_id"] + LABEL_DIMS]
            .groupby(by="article_id")
            .agg(self._agg_bool_sum)
            .reset_index()
        )
        return articles

    def _remove_unpopular_articles(self):
        grouped = (
            self.data[TextType.COMMENT][["text", "article_id"]]
            .groupby("article_id")
            .agg("count")
        )
        grouped["article_id"] = grouped.index
        saved = grouped[grouped["text"] >= self.min_comments_for_article]
        self.data[TextType.ARTICLE] = self.data[TextType.ARTICLE][
            self.data[TextType.ARTICLE]["article_id"].isin(
                list(saved["article_id"].unique())
            )
        ]
        return self.data[TextType.ARTICLE]

    def read_data(self):
        self.data = {
            TextType.COMMENT: self.read_comments(),
            TextType.ARTICLE: self.read_articles(),
        }
        self._remove_unpopular_articles()
        return self

    def prepare_one_model(self, text_type):
        model = ClusteringModel(k=self.cluster_no[text_type])
        train_data = self.data[text_type][LABEL_DIMS].drop_duplicates()
        model.set_data(train_data)
        model.train()
        return model

    def setup_models(self):
        self.models = {
            TextType.COMMENT: self.prepare_one_model(TextType.COMMENT),
            TextType.ARTICLE: self.prepare_one_model(TextType.ARTICLE),
        }
        return self

    def prepare_one_tuner(self, text_type):
        filepath = join(
            self.testing_model_output_dir,
            f"cluster_results_{text_type.name.lower()}.json",
        )
        tuner = ClusterTuner(filepath)
        tuner.set_data(self.data[text_type])
        return tuner

    def setup_tuners(self):
        self.tuners = {
            TextType.COMMENT: self.prepare_one_tuner(TextType.COMMENT),
            TextType.ARTICLE: self.prepare_one_tuner(TextType.ARTICLE),
        }
        return self

    def set_data(self, data, text_type):
        self.data[text_type] = data

    def predict_data(self, text_type):
        data = self.data[text_type]
        if text_type == TextType.ARTICLE:
            data["id_inside_article"] = 0
        data["text_type"] = text_type
        data["cluster_id"] = self.models[text_type].model.predict(data[LABEL_DIMS])
        data = data[["article_id", "id_inside_article", "text_type", "cluster_id"]]
        self.set_data(data, text_type)
        return self

    def save_data(self):
        self.processed_data = pd.concat(list(self.data.values()), axis=0)
        self.processed_data.to_csv(self.output_file, index=False)
        return self

    def find_k(self, comment_ks=range(2, 20), article_ks=range(2, 20)):
        self.tuners[TextType.COMMENT].find_k_for_entire_data(ks=comment_ks)
        self.tuners[TextType.ARTICLE].find_k_for_entire_data(ks=article_ks)
        return self

    def run_tuning_k(self, comment_ks=range(2, 20), article_ks=range(2, 20)):
        return self.read_data().setup_tuners().find_k(comment_ks, article_ks)

    def run_prediction(self):
        return (
            self.read_data()
            .setup_models()
            .predict_data(TextType.ARTICLE)
            .predict_data(TextType.COMMENT)
            .save_data()
        )
