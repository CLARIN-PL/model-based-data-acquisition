import json
from utils.text_type import TextType
import pandas as pd
from tqdm import tqdm


class TextReader:
    def __init__(
        self,
        input_file,
        paragraph_output_file="data/paragraphs.csv",
        comment_output_file="data/comments.csv",
    ):
        self.input_file = input_file
        self.paragraph_output_file = paragraph_output_file
        self.comment_output_file = comment_output_file
        self.paragraphs, self.paragraphs_id = [], []
        self.comments, self.comments_id = [], []

    def load_data(self):
        with open(self.input_file, "r") as f:
            lines = f.readlines()
        self.data = []
        for id_, line in tqdm(enumerate(lines)):
            curr = json.loads(line)
            self.data.append(
                (
                    curr["article"]["content"],
                    curr["article"]["comments"]
                    if "comments" in curr["article"]
                    else [],
                    id_,
                )
            )
        return self

    def extract_data(self):
        for art, com, id_ in tqdm(self.data):
            curr = []
            curr_paragraph_id = 0
            for a in art:
                if "text" in a:
                    curr.append(a["text"])
                    self.paragraphs_id.append((id_, curr_paragraph_id))
                    curr_paragraph_id += 1
                else:
                    print(a)
            self.paragraphs.extend(curr)
            curr_comment_id = 0
            for c in com:
                if "message" in c:
                    self.comments.append(c["message"])
                    self.comments_id.append((id_, curr_comment_id))
                    curr_comment_id += 1
                elif "content" in c:
                    self.comments.append(c["content"])
                    self.comments_id.append((id_, curr_comment_id))
                    curr_comment_id += 1
                else:
                    print(c)
        return self

    def save_data(self, text_type):
        texts, ids, filepath = self.get_data_by_type(text_type, return_filepath=True)
        self.data = pd.DataFrame(
            data={
                "article_id": list(map(lambda x: x[0], ids)),
                "id_inside_article": list(map(lambda x: x[1], ids)),
                "text": texts,
            }
        )
        self.data.to_csv(filepath, index=False)
        return self

    def get_data_by_type(self, text_type, return_filepath=False):
        if text_type == TextType.COMMENT:
            if return_filepath:
                return self.comments, self.comments_id, self.comment_output_file
            return self.comments, self.comments_id
        if text_type == TextType.ARTICLE:
            if return_filepath:
                return self.paragraphs, self.paragraphs_id, self.paragraph_output_file
            return self.paragraphs, self.paragraphs_id
        raise ValueError("Text type does not exist")

    def run(self):
        return (
            self.load_data()
            .extract_data()
            .save_data(TextType.ARTICLE)
            .save_data(TextType.COMMENT)
        )
