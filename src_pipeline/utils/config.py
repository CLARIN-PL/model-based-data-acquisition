DATA_DATE = "18-4-23"
VERSION = 3

PIPELINE_INIT_ARGUMENTS = dict(
    reader_args=dict(
        input_file="data/articles_2023-04-18.json",
        paragraph_output_file=f"data/{DATA_DATE}/paragraphs.csv",
        comment_output_file=f"data/{DATA_DATE}/comments.csv",
    ),
    paragraph_cleaner_args=dict(
        input_file=f"data/{DATA_DATE}/paragraphs.csv",
        output_file=f"data/{DATA_DATE}/paragraphs_cleaned.csv",
        paragraph_count_min=3,
        paragraph_count_max=10,
        filter_args=dict(
            has_enough_words_threshold=5,
        ),
    ),
    comment_cleaner_args=dict(
        input_file=f"data/{DATA_DATE}/comments.csv",
        output_file=f"data/{DATA_DATE}/comments_cleaned.csv",
        filter_args=dict(
            has_enough_words_threshold=4,
            has_enough_long_words_count=3,
            has_enough_long_words_length=4,
            unique_words_percent=0.5,
            same_char_words_percent=0.5,
        ),
    ),
    paragraph_predictor_args=dict(
        input_file=f"data/{DATA_DATE}/paragraphs_cleaned.csv",
        output_file=f"data/{DATA_DATE}/paragraphs_cleaned_predicted.csv",
        ckpt_file="lightning_logs/version_2/checkpoints/epoch=3-step=152.ckpt",
        batch_size=16,
    ),
    comment_predictor_args=dict(
        input_file=f"data/{DATA_DATE}/comments_cleaned.csv",
        output_file=f"data/{DATA_DATE}/comments_cleaned_predicted.csv",
        ckpt_file="lightning_logs/version_2/checkpoints/epoch=3-step=152.ckpt",
        batch_size=16,
    ),
    clusterizer_args=dict(
        paragraph_file=f"data/{DATA_DATE}/paragraphs_cleaned_predicted.csv",
        comment_file=f"data/{DATA_DATE}/comments_cleaned_predicted.csv",
        output_file=f"data/{DATA_DATE}/cluster_id.csv",
        min_comments_for_article=10,
        testing_model_output_dir="data",
        comment_k=5,
        article_k=4,  # old 3
    ),
    comment_cluster_tuner_args=dict(
        test_output_file=f"data/{DATA_DATE}/cluster_results_comments_per_article.json",
        clusters_file=f"data/{DATA_DATE}/cluster_id.csv",
        comments_file=f"data/{DATA_DATE}/comments_cleaned_predicted.csv",
    ),
    sampler_args=dict(
        clusters_file=f"data/{DATA_DATE}/cluster_id.csv",
        paragraphs_file=f"data/{DATA_DATE}/paragraphs_cleaned_predicted.csv",
        comments_file=f"data/{DATA_DATE}/comments_cleaned_predicted.csv",
        used_articles_file="data/projects/used_articles.csv",
        used_comments_file="data/projects/used_comments.csv",
        texts_file=f"data/projects/v{VERSION}/texts.csv",
        projects_file=f"data/projects/v{VERSION}/projects.csv",
        clustering_article_comments=True,
        clustering_article_comments_args=dict(
            max_k=3,
            sampling_method="balanced",  # ["balanced", "most_numerous"]
        ),
    ),
)

PIPELINE_RUN_ARGUMENTS = dict(
    reader=False,
    paragraph_cleaner=False,
    comment_cleaner=False,
    paragraph_predictor=False,
    comment_predictor=False,
    clusterizer_prediction=False,
    clusterizer_tuning=False,
    clusterizer_tuning_args=dict(comment_ks=range(2, 15), article_ks=range(2, 15)),
    comment_cluster_tuning=False,
    comment_cluster_tuning_args=dict(min_amount=10, max_k=10),
    sampler=True,
    sampler_args=dict(
        N_art=10,
        N_com=20,
        iteration_num=1000,
        text_type=None,  # ["article", "comment", None]
        articles_batched=True,
        comments_batched=True,
    ),
    all_true=False,
    verbose=True,
)

LABEL_DIMS = [
    "Pozytywne",
    "Negatywne",
    "Radość",
    "Zachwyt",
    "Inspiruje",
    "Spokój",
    "Zaskoczenie",
    "Współczucie",
    "Strach",
    "Smutek",
    "Wstręt",
    "Złość",
    "Ironiczny",
    "Żenujący",
    "Wulgarny",
    "Polityczny",
    "Interesujący",
    "Zrozumiały",
    "Potrzebuję więcej informacji, aby ocenić ten tekst",
    "Obraża mnie",
    "Może kogoś atakować / obrażać / lekceważyć",
    "Mnie bawi/śmieszy?",
    "Może kogoś bawić?",
]

CONDITION_LABEL_DIMS = {
    "Może kogoś bawić?": "Mnie bawi/śmieszy?",
    "Może kogoś atakować / obrażać / lekceważyć": "Obraża mnie",
}
