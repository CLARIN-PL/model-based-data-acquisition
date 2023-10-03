import pandas as pd
from tqdm import tqdm


def filtering(df_old, df_new):
    columns = ["text", "article_id", "id_inside_article"]
    print(df_old.shape)
    print(df_new.shape)
    df = pd.DataFrame(columns=columns)
    for i in tqdm(df_new.article_id.unique()):
        curr = df_old[df_old.article_id == i]
        curr_new = df_new[df_new.article_id == i]
        out = curr[~curr.id_inside_article.isin(curr_new.id_inside_article.tolist())][
            columns
        ]
        df = pd.concat([df, out], axis=0)
    # df_filtered = df_old[~df_old["article_id"].isin(df_new["article_id"].tolist()) ][columns]
    # print(df_filtered["text"].isin(df_new["text"].tolist()).value_counts())
    print(df.shape)
    return df


def group(df, aggr):
    grouped = (
        df[["article_id", "id_inside_article"]]
        .groupby(by="article_id")
        .agg(aggr)
        .reset_index()
    )
    return grouped


def comparison(df_old, df_new, aggr):
    df_old_agg = group(df_old, aggr)
    df_new_agg = group(df_new, aggr)
    df_agg = pd.merge(df_old_agg, df_new_agg, on="article_id")
    print(df_agg.shape)
    comp = df_agg[df_agg["id_inside_article_x"] != df_agg["id_inside_article_y"]]
    return comp


def distribution(df):
    grouped = group(df, "count")
    return (
        grouped["id_inside_article"].value_counts().reset_index().sort_values("index")
    )


if __name__ == "__main__":
    paragraph_old_path = "data/predicted_paragraphs.csv"
    paragraph_path = "data/predicted_paragraphs_cleaned.csv"
    df_old = pd.read_csv(paragraph_old_path)
    df_new = pd.read_csv(paragraph_path)

    # df_max = comparison(df_old, df_new, "max")
    # print(df_max)
    # df_count = comparison(df_old, df_new, "count")
    # print(df_count)
    # dist = distribution(df_new)
    # print(dist.head(10))
    f = filtering(df_old, df_new)
    f.to_csv("data/out_paragraphs.csv", index=False)
    print(f)

    f = pd.read_csv("data/out_paragraphs.csv")
    print(f["id_inside_article"].value_counts())
