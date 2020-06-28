


from kedro.pipeline import Pipeline, node

from .nodes import TopicVectorizer, InterFeatureEncoder, IntraFeaturePooling, SentenceEmbeddings

sentence_encode=SentenceEmbeddings(sentence_col='sentences', unique_id='textID')


topic_vectorizer=TopicVectorizer()
inter_vectorizer=InterFeatureEncoder(latent_dim=64)
intra_pool=IntraFeaturePooling()

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(topic_vectorizer.train_Corex,
                 ['params:n_topics', 'df_sents'],
                 dict(
                     corex_model='corex_model',
                     countvectorizer='countvectorizer'
                 )
            ),
            node(
                topic_vectorizer.predict_Corex,
                ["df_sents", "corex_model", "countvectorizer"],
                dict(
                    corex_vecs_1="corex_vecs_1",
                    corex_vecs_2="corex_vecs_2",
                    textID="textID"
                )
            ),
            node(
                sentence_encode.get_precBert,
                ["df_sents"], "distill_vec"
            ),
            node( inter_vectorizer.autoencoder,
                  ["corex_vecs_2", "distill_vec"],
                  dict(ae='auto_encoder_model')

            ),
            node(inter_vectorizer.ae_predict,
                 ["corex_vecs_2", "distill_vec", "textID", 'auto_encoder_model'],
                 dict(vec="corex_encoded",
                      uniqueID="textIDA")),
            node(intra_pool.get_mean_pooling,
                 ["corex_encoded", "textIDA"],
                 "corex_pooled"),

            node(
                topic_vectorizer.predict_Corex,
                ["df_sents_test", "corex_model", "countvectorizer"],
                dict(
                    corex_vecs_1="corex_vecs_1_t",
                    corex_vecs_2="corex_vecs_2_t",
                    textID="textID_t"
                )
            ),
            node( sentence_encode.get_precBert,
                  ["df_sents_test"], "distill_vec_t"),
            node(inter_vectorizer.ae_predict,
                 ["corex_vecs_2_t", "distill_vec_t", "textID_t", 'auto_encoder_model'],
                 dict(vec="corex_encoded_t",
                      uniqueID="textIDA_t")),
            node(intra_pool.get_mean_pooling,
                 ["corex_encoded_t", "textIDA_t"],
                 "corex_pooled_t"),
        ]
    )

