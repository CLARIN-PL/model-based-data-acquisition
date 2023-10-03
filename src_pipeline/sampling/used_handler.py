import pandas as pd
from os.path import isfile


class UsedTextsHandler:
    def __init__(self, used_articles_file, used_comments_file):
        self.used_articles_file = used_articles_file
        self.used_comments_file = used_comments_file

    def read_used_articles(self):
        if isfile(self.used_articles_file):
            self.used_articles_data = pd.read_csv(self.used_articles_file)
        else:
            self.used_articles_data = pd.DataFrame(columns=["article_id", "cluster_id"])
        return self.used_articles_data

    def read_used_comments(self):
        if isfile(self.used_comments_file):
            self.used_comments_data = pd.read_csv(self.used_comments_file)
        else:
            self.used_comments_data = pd.DataFrame(
                columns=["article_id", "id_inside_article", "cluster_id", "text"]
            )
        return self.used_comments_data

    def add_used_comments(self, just_used):
        self.used_comments_data = pd.concat(
            [
                self.used_comments_data,
                just_used[["text", "article_id", "id_inside_article", "cluster_id"]],
            ],
            axis=0,
        ).drop_duplicates()
        self.used_comments_data.to_csv(self.used_comments_file, index=False)
        return self.used_comments_data

    def add_used_articles(self, just_used):
        self.used_articles_data = pd.concat(
            [self.used_articles_data, just_used[["article_id", "cluster_id"]]],
            axis=0,
        ).drop_duplicates()
        self.used_articles_data.to_csv(self.used_articles_file, index=False)
        return self.used_articles_data

    def remove_used_comments(self, data):
        return (
            pd.concat([data, self.used_comments_data, self.used_comments_data], axis=0)
            .drop_duplicates(subset=["article_id", "id_inside_article"], keep=False)
            .drop(columns=["cluster_id"])
        )

    def remove_used_articles(self, data):
        return pd.concat(
            [data, self.used_articles_data, self.used_articles_data], axis=0
        ).drop_duplicates(subset=["article_id"], keep=False)
