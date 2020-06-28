

from kedro.pipeline import Pipeline, node

from .nodes import SpacyBulk

spacy_preprocess=SpacyBulk()


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                spacy_preprocess.get_DF,
                ["train_x", "params:preprocess_out"],
                "df_sents",
            ),
            node(
                spacy_preprocess.get_DF,
                ["test_x", "params:preprocess_out"],
                "df_sents_test",
            )
        ]
    )

