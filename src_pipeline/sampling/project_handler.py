from utils.config import LABEL_DIMS, CONDITION_LABEL_DIMS
import pandas as pd
from os.path import isfile


class ProjectHandler:
    def __init__(
        self, texts_file="data/texts.csv", projects_file="data/projects.csv"
    ) -> None:
        self.texts_file = texts_file
        self.projects_file = projects_file

    def get_dimensions(self, project):
        for exist_col, new_col in CONDITION_LABEL_DIMS.items():
            project[new_col] = project[exist_col]
        return project

    def add_project(self, sampled, text_type):
        if isfile(self.projects_file):
            projects = pd.read_csv(self.projects_file)
        else:
            projects = pd.DataFrame(columns=["project_id", "project_type"] + LABEL_DIMS)
        curr_proj_id = 0
        if projects.shape[0] > 0:
            curr_proj_id = projects["project_id"].max() + 1

        average_dimensions = (
            sampled[LABEL_DIMS].apply((lambda x: int(bool(x.sum()))), axis=0).tolist()
        )

        self.save_project(projects, curr_proj_id, average_dimensions, text_type)
        self.save_texts(sampled[["text", "text_type"]], curr_proj_id)

    def save_project(self, projects, curr_proj_id, average_dimensions, text_type):
        curr_project = pd.DataFrame(
            data={"project_id": [curr_proj_id], "project_type": [text_type]},
            columns=projects.columns,
        )
        curr_project.loc[0, LABEL_DIMS] = average_dimensions
        curr_project = self.get_dimensions(curr_project)
        projects = pd.concat([projects, curr_project], axis=0)
        projects.to_csv(self.projects_file, index=False)

    def save_texts(self, texts, proj_id):
        sampled_texts = pd.DataFrame(
            columns=["project_id", "order_in_project", "text_type", "text"]
        )
        sampled_texts["text"] = texts["text"]
        sampled_texts["project_id"] = proj_id
        sampled_texts["order_in_project"] = list(range(texts.shape[0]))
        sampled_texts["text_type"] = texts["text_type"]

        #! --------------
        # sampled_texts["vector"] = texts["vector"]
        # sampled_texts["centroid"] = texts["centroid"]
        # sampled_texts["cluster_id"] = texts["cluster_id"]

        #!------------
        if isfile(self.texts_file):
            sampled_texts.to_csv(self.texts_file, mode="a", index=False, header=False)
        else:
            sampled_texts.to_csv(self.texts_file, index=False)
