# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.


from kedro.pipeline import Pipeline, node

from .nodes import TopicVectorizer, InterFeatureEncoder, IntraFeaturePooling, SentenceEmbeddings

sentence_encode=SentenceEmbeddings(sentence_col='sentences', unique_id='textID')

import pickle

'''with open('/Users/karthik/nlpipe/nlpipe/data/06_models/count_vectorizer.pkl', 'rb') as f:
    countvectorizer=pickle.load(f)
'''
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

