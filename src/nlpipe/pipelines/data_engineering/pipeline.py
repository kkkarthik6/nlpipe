


from kedro.pipeline import Pipeline, node

from .nodes import InputAbstraction

input_abstraction = InputAbstraction(sample_data_ratio=0.01)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                input_abstraction.split_data,
                ["steam_reviews", "params:target", "params:columns"],
                dict(
                    train_x="train_x",
                    train_y="train_y",
                    test_x="test_x",
                    test_y="test_y",
                ),
            )
        ]
    )
