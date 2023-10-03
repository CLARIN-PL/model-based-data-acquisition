from utils.text_type import TextType
from sampling.project_handler import ProjectHandler
from sampling.used_handler import UsedTextsHandler
from sampling.inside_clusterizer import ClusterizerInsideArticle
from tqdm import tqdm
import pandas as pd

tqdm.pandas()


class TextSampler:
    def __init__(
        self,
        clusters_file="data/cluster_id.csv",
        paragraphs_file="data/paragraphs_cleaned_predicted.csv",
        comments_file="data/comments_cleaned_predicted.csv",
        used_articles_file="data/used_articles.csv",
        used_comments_file="data/used_comments.csv",
        texts_file="data/texts.csv",
        projects_file="data/projects.csv",
        clustering_article_comments=False,
        clustering_article_comments_args=dict(max_k=3, sampling_method="balanced"),
    ):
        self.clusters_file = clusters_file
        self.paragraphs_file = paragraphs_file
        self.comments_file = comments_file
        self.used_articles_file = used_articles_file
        self.used_comments_file = used_comments_file
        self.projects_file = projects_file
        self.texts_file = texts_file
        self.clustering_article_comments = clustering_article_comments
        self.clustering_article_comments_args = clustering_article_comments_args

    def prepare_handlers(self):
        self.project_handler = ProjectHandler(self.texts_file, self.projects_file)
        self.used_texts_handler = UsedTextsHandler(
            self.used_articles_file, self.used_comments_file
        )

    def prepare_clusters(self, text_type):
        self.cluster_data = self.cluster_data[
            self.cluster_data.text_type == text_type.value
        ]
        return self.cluster_data

    def load_data(self, text_type):
        self.cluster_data = pd.read_csv(self.clusters_file)
        self.prepare_clusters(text_type)

        self.comments_data = pd.read_csv(self.comments_file)
        self.used_texts_handler.read_used_comments()
        if text_type == TextType.ARTICLE:
            self.paragraphs_data = pd.read_csv(self.paragraphs_file)
            self.used_texts_handler.read_used_articles()
        return self

    def get_curr_cluster_id(self, used_data):
        if used_data.shape[0] > 0:
            return (used_data.tail(1)["cluster_id"].item() + 1) % self.no_clusters
        return 0

    def get_data_from_cluster(self, text_type):
        if text_type == TextType.COMMENT:
            data = self.comments_data
            used_data = self.used_texts_handler.used_comments_data
        elif text_type == TextType.ARTICLE:
            data = self.cluster_data
            used_data = self.used_texts_handler.used_articles_data
        else:
            raise ValueError("TextType does not exist")

        self.no_clusters = data["cluster_id"].max() + 1
        cluster_id = self.get_curr_cluster_id(used_data)
        curr_df = data[data["cluster_id"] == cluster_id]
        counter = 0
        while curr_df.shape[0] == 0:
            if counter > self.no_clusters:
                print(f"NO MORE {text_type.name}S!")
                return None
            counter += 1
            cluster_id = (cluster_id + 1) % self.no_clusters
            curr_df = data[data["cluster_id"] == cluster_id]

        return curr_df

    def sample_comment_package(self, N, curr_cluster=None, batched=True, curr_pkg=0):
        if not batched:
            self.load_data(text_type=TextType.COMMENT)
            self.comments_data = self.used_texts_handler.remove_used_comments(
                self.comments_data
            )
            self.comments_data = pd.merge(
                self.comments_data,
                self.cluster_data[["article_id", "id_inside_article", "cluster_id"]],
                on=["article_id", "id_inside_article"],
            )
            curr_cluster = self.get_data_from_cluster(TextType.COMMENT)
            if curr_cluster is None:
                return None
            sampled = curr_cluster.sample(N)
        else:
            sampled = curr_cluster.iloc[curr_pkg * N : (curr_pkg + 1) * N]
        sampled["text_type"] = TextType.COMMENT

        self.used_texts_handler.add_used_comments(sampled)
        self.project_handler.add_project(sampled, TextType.COMMENT)

    def sample_batch_comments(self, batch_size, N):
        def compute_sample_amount(row, dict_):
            diff = dict_["M"] // dict_["n"]
            if row // N < dict_["value"] + diff:
                diff = row // N - dict_["value"]
            dict_["value"] += diff
            dict_["M"] -= diff * dict_["n"]
            dict_["n"] -= 1
            return dict_["value"]

        self.load_data(text_type=TextType.COMMENT)
        self.comments_data = self.used_texts_handler.remove_used_comments(
            self.comments_data
        )
        self.comments_data = pd.merge(
            self.comments_data,
            self.cluster_data[["article_id", "id_inside_article", "cluster_id"]],
            on=["article_id", "id_inside_article"],
        )
        clusters_amounts = self.cluster_data["cluster_id"].value_counts(ascending=True)
        memory_dict = {"M": batch_size, "n": clusters_amounts.shape[0], "value": 0}
        sampling_amounts = clusters_amounts.progress_apply(
            lambda x: compute_sample_amount(x, memory_dict)
        )

        for i, (cluster_id, amount) in enumerate(sampling_amounts.items()):
            curr_cluster = self.comments_data[
                self.comments_data["cluster_id"] == cluster_id
            ]
            curr_sampled = curr_cluster.sample(amount * N)
            for i in tqdm(range(amount)):
                self.sample_comment_package(N, curr_sampled, batched=True, curr_pkg=i)

    def get_comments_to_article(self, article_id, N):
        comments = self.comments_data[self.comments_data.article_id == article_id]
        if self.clustering_article_comments:
            clusterizer = ClusterizerInsideArticle(
                comments, N, **self.clustering_article_comments_args
            )
            comments = clusterizer.run()
        comments["text_type"] = TextType.COMMENT
        return comments.head(N)

    def sample_article_package(self, N, sampled_article=None, batched=True):
        if not batched:
            self.load_data(TextType.ARTICLE)
            self.cluster_data = self.used_texts_handler.remove_used_articles(
                self.cluster_data
            )
            curr_cluster = self.get_data_from_cluster(TextType.ARTICLE)
            if curr_cluster is None:
                return None
            sampled_article = curr_cluster.sample(1)
        else:
            sampled_article = pd.DataFrame(
                data={k: [v] for k, v in sampled_article.items()},
                columns=list(sampled_article.index),
            )
        sampled_id = sampled_article["article_id"].item()

        sampled = self.paragraphs_data[self.paragraphs_data.article_id == sampled_id]
        sampled["text_type"] = TextType.ARTICLE

        sampled_comments = self.get_comments_to_article(sampled_id, N)

        sampled = pd.concat([sampled, sampled_comments], axis=0)
        self.used_texts_handler.add_used_articles(sampled_article)
        self.project_handler.add_project(sampled, TextType.ARTICLE)
        return None

    def sample_batch_articles(self, batch_size, N_com):
        def compute_sample_amount(row, dict_):
            diff = dict_["M"] // dict_["n"]
            if row < dict_["value"] + diff:
                diff = row - dict_["value"]
            dict_["value"] += diff
            dict_["M"] -= diff * dict_["n"]
            dict_["n"] -= 1
            return dict_["value"]

        self.load_data(TextType.ARTICLE)
        self.cluster_data = self.used_texts_handler.remove_used_articles(
            self.cluster_data
        )

        clusters_amounts = self.cluster_data["cluster_id"].value_counts(ascending=True)
        memory_dict = {"M": batch_size, "n": clusters_amounts.shape[0], "value": 0}
        sampling_amounts = clusters_amounts.progress_apply(
            lambda x: compute_sample_amount(x, memory_dict)
        )

        for i, (cluster_id, amount) in enumerate(sampling_amounts.items()):
            curr_cluster = self.cluster_data[
                self.cluster_data["cluster_id"] == cluster_id
            ]
            curr_sampled = curr_cluster.sample(amount)
            curr_sampled.progress_apply(
                (lambda article: self.sample_article_package(N_com, article)), axis=1
            )

    def run(
        self,
        N_art=10,
        N_com=20,
        iteration_num=50,
        text_type=None,
        articles_batched=True,
        comments_batched=True,
    ):
        assert text_type in [
            None,
            TextType.ARTICLE.name.lower(),
            TextType.COMMENT.name.lower(),
        ]
        self.prepare_handlers()
        article = text_type == TextType.ARTICLE.name.lower() or text_type is None
        comment = text_type == TextType.COMMENT.name.lower() or text_type is None
        if article:
            if articles_batched:
                self.sample_batch_articles(iteration_num, N_art)
            else:
                for _ in tqdm(range(iteration_num)):
                    self.sample_article_package(N_art, batched=False)

        if comment:
            if comments_batched:
                self.sample_batch_comments(iteration_num, N_com)
            else:
                for _ in tqdm(range(iteration_num)):
                    self.sample_comment_package(N_com, batched=False)
