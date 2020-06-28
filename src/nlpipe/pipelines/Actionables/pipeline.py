


from kedro.pipeline import Pipeline, node

from .nodes import gmm_train, model_predict, umap_iplot, report_gen, RF_train


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                RF_train,
                ["corex_pooled", "train_y"],
                "RF_model"
            ),
            node(model_predict,
                 ["RF_model", "corex_pooled_t"],
                 "predictions"),
            node(report_gen,
                 ["test_y", "predictions"],
                 "report"),
            node(gmm_train,
                 ["corex_pooled", "params:n_topics"],
                 "gmm"),
            node(model_predict,
                 ["gmm", "corex_pooled_t"], "cluster_out"),
            node(umap_iplot,
                 ["corex_pooled_t", "test_x", "cluster_out"], "umap_vecs")
        ]
    )

