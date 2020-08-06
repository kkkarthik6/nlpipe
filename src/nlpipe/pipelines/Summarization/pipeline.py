from kedro.pipeline import Pipeline, node

from .nodes import Summarizer

summarizer=Summarizer()

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                summarizer.summarize_corpus,
                ["train_x"],
                "summaries"
            )
        ]
    )