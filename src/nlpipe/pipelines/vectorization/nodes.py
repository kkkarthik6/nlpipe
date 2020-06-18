'''# Copyright 2020 QuantumBlack Visual Analytics Limited
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
"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.

PLEASE DELETE THIS FILE ONCE YOU START WORKING ON YOUR OWN PROJECT!
"""

from typing import Any, Dict

import pandas as pd


def split_data(data: pd.DataFrame, example_test_data_ratio: float) -> Dict[str, Any]:
    """Node for splitting the classical Iris data set into training and test
    sets, each split into features and labels.
    The split ratio parameter is taken from conf/project/parameters.yml.
    The data and the parameters will be loaded and provided to your function
    automatically when the pipeline is executed and it is time to run this node.
    """
    data.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "target",
    ]
    classes = sorted(data["target"].unique())
    # One-hot encoding for the target variable
    data = pd.get_dummies(data, columns=["target"], prefix="", prefix_sep="")

    # Shuffle all the data
    data = data.sample(frac=1).reset_index(drop=True)

    # Split to training and testing data
    n = data.shape[0]
    n_test = int(n * example_test_data_ratio)
    training_data = data.iloc[n_test:, :].reset_index(drop=True)
    test_data = data.iloc[:n_test, :].reset_index(drop=True)

    # Split the data to features and labels
    train_data_x = training_data.loc[:, "sepal_length":"petal_width"]
    train_data_y = training_data[classes]
    test_data_x = test_data.loc[:, "sepal_length":"petal_width"]
    test_data_y = test_data[classes]

    # When returning many variables, it is a good practice to give them names:
    return dict(
        train_x=train_data_x,
        train_y=train_data_y,
        test_x=test_data_x,
        test_y=test_data_y,
    )
'''

'''Vectorization'''
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from scipy.sparse import csr_matrix
from corextopic import corextopic as ct
from sentence_transformers import SentenceTransformer
import pandas as pd
import keras
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
import warnings
import numpy as np
warnings.filterwarnings('ignore')


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class PipelineError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message
class ClassicVectorizationTrain:
    def __init__(self, tfidf=0,countvectorizer=0):
        self.tfidf=tfidf
        self.countvectorizer=countvectorizer
        self.text_corpus=None
        self.count_vectorizer=None
        self.tfidf_vectorizer=None
        self.sparse_rep=None
        if self.tfidf:
            self.tfidf_vectorizer=self.get_tfidfVectorizer()
        if self.countvectorizer:
            self.count_vectorizer=self.get_countVectorizer()
    def get_countVectorizer(self,text_corpus=None):
        self.text_corpus = text_corpus
        if len(self.text_corpus)==0:
            raise PipelineError('Please provide text corpus', 'This object provides advanced corex vectors.')

        print(self.text_corpus)
        if self.count_vectorizer==None:
            self.count_vectorizer=CountVectorizer(stop_words='english',max_df=0.3,max_features=10000)
            self.count_vectorizer.fit(self.text_corpus)
        return self.count_vectorizer
    def get_tfidfVectorizer(self):
        self.text_corpus = text_corpus
        if self.count_vectorizer==None:
            self.tfidf_vectorizer=TfidfVectorizer(stop_words='english',max_df=0.3,max_features=10000)
            self.tfidf_vectorizer.fit(self.text_corpus)
        return self.tfidf_vectorizer
    def get_sparseRepresentation(self):
        self.text_corpus = text_corpus
        if self.count_vectorizer:
            transformed=self.count_vectorizer.transform(self.text_corpus)
        if self.tfidf_vectorizer:
            transformed=self.tfidf_vectorizer.transform(self.text_corpus)
        self.sparse_rep=csr_matrix(transformed)
        return self.sparse_rep

class TopicVectorizer:
    def __init__(self,count_vectorizer=None,sentence_col='sentences',unique_id='textID'):
        self.corex=None
        self.lda=None
        self.hdp=None
        self.sentence_col = sentence_col
        self.uniqueID = unique_id
        #self.text_corpus=None
        self.countvectorizer=count_vectorizer
    def train_Corex(self,n_topics,data_df):
        #self.countvectorizer=countvectorizer
        self.data_df=data_df
        text_corpus=self.data_df[self.sentence_col].values.tolist()
        print(len(text_corpus))
        if len(text_corpus)==0:
            raise PipelineError('Please provide text corpus', 'This object provides advanced corex vectors.')
        if self.countvectorizer:
            doc_word=self.countvectorizer.transform(text_corpus)
        else:
            countvectorizer_obj=ClassicVectorizationTrain(countvectorizer=1)
            self.countvectorizer=countvectorizer_obj.get_countVectorizer(text_corpus=text_corpus)
            doc_word=self.countvectorizer.transform(text_corpus)
        doc_word=csr_matrix(doc_word)
        words = list(np.asarray(self.countvectorizer.get_feature_names()))
        not_digit_inds = [ind for ind,word in enumerate(words) if not word.isdigit()]
        doc_word = doc_word[:,not_digit_inds]
        words = [word for ind,word in enumerate(words) if not word.isdigit()]
        # Train the CorEx topic model with 50 topics
        self.corex = ct.Corex(n_hidden=n_topics, words=words, max_iter=200, verbose=False, seed=1)
        self.corex.fit(doc_word, words=words)
        return dict(corex_model=self.corex, countvectorizer= self.countvectorizer)
    def predict_Corex(self,data_df,corex,countvectorizer):
        self.data_df = data_df
        if countvectorizer:
            self.countvectorizer=countvectorizer
        text_corpus = self.data_df[self.sentence_col].values.tolist()
        #print(text_corpus)
        if len(text_corpus)==0:
            raise PipelineError('Please provide text corpus', 'This object provides advanced corex vectors.')
        if corex:
            self.corex=corex
        if self.countvectorizer==None:
            self.countvectorizer=CountVectorizer(stop_words='english', max_df=0.3, max_features=10000).fit(text_corpus)
        if self.corex:
            doc_word=self.countvectorizer.transform(text_corpus)
            doc_word=csr_matrix(doc_word)
            words = list(np.asarray(self.countvectorizer.get_feature_names()))
            not_digit_inds = [ind for ind, word in enumerate(words) if not word.isdigit()]
            doc_word = doc_word[:,not_digit_inds]
            #words= [word for ind,word in enumerate(words) if not word.isdigit()]
            vec=np.array(self.corex.predict_proba(doc_word))
        else:
            raise PipelineError('Please Input/Train the model to make predictions', 'Use train_Corex to train your corex model')
        return dict(
            corex_vecs_1 = vec[0, :, :],
            corex_vecs_2 = vec[1, :, :],
            textID=self.data_df[self.uniqueID].values.tolist()
        )


class SentenceEmbeddings:
    def __init__(self,sentence_col='sentences',unique_id='textID'):
        #self.text_corpus=[]
        #self.sentence_level=sentence_level
        self.data_df=None
        self.bert_fast=None
        self.bert_high_precision=None
        self.bert_mid_precision=None
        self.sentence_col=sentence_col
        self.uniqueID=unique_id
    def get_distillBert(self,df=pd.DataFrame()):
        self.data_df = df
        model=SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        embeddings=model.encode(self.data_df[self.sentence_col].values.tolist())
        self.bert_fast=pd.DataFrame({self.uniqueID:self.data_df[self.uniqueID].values.tolist(),self.sentence_col:self.data_df[self.sentence_col].values.tolist(),'embeddings':embeddings})
        return np.asarray(embeddings)
    def get_precBert(self,df=pd.DataFrame()):
        self.data_df = df
        model=SentenceTransformer('bert-base-nli-stsb-mean-tokens')
        embeddings=model.encode(self.data_df[self.sentence_col].values.tolist())
        self.bert_high_precision=pd.DataFrame({self.uniqueID:self.data_df[self.uniqueID].values.tolist(),self.sentence_col:self.data_df[self.sentence_col].values.tolist(),'embeddings':embeddings})
        return self.bert_high_precision
    def get_meanBert(self,df=pd.DataFrame()):
        self.data_df = df
        model=SentenceTransformer('bert-base-nli-mean-tokens')
        embeddings=model.encode(self.data_df[self.sentence_col].values.tolist())
        self.bert_mid_precision=pd.DataFrame({self.uniqueID:self.data_df[self.uniqueID].values.tolist(),self.sentence_col:self.data_df[self.sentence_col].values.tolist(),'embeddings':embeddings})
        return self.bert_mid_precision

class IntraFeaturePoolingDF:
    def __init__(self, vector_coulumn='embeddings', unique_id='textID'):
        self.df = df
        self.vec_column=vector_coulumn
        self.uniqueID=unique_id
    def get_mean_pooling(self, df=pd.DataFrame()):
        self.df = df
        func=lambda grp:np.mean(grp)
        self.mean_pooling=np.vstack(self.df.groupby(self.uniqueID)[self.vec_column].apply(func).values)
        return self.mean_pooling
    def get_median_pooling(self, df=pd.DataFrame()):
        self.df = df
        func=lambda grp:np.median(grp)
        self.median_pooling=np.vstack(self.df.groupby(self.uniqueID)[self.vec_column].apply(func).values)
        return self.median_pooling
class IntraFeaturePooling:
    def __init__(self):
        self.vec=[]
        self.uniqueID=[]
        self.df=pd.DataFrame()
    def get_mean_pooling(self, vec, uniqueID):
        self.vec=vec
        self.uniqueID = uniqueID
        self.df = pd.DataFrame({'uniqueID': uniqueID, 'vec_column': list(vec)})
        print(self.df.shape)
        func = lambda grp: np.mean(grp)
        self.mean_pooling=np.vstack(self.df.groupby('uniqueID')['vec_column'].apply(func).values)
        print(self.mean_pooling.shape)
        return self.mean_pooling
    def get_median_pooling(self, vec, uniqueID):
        self.vec=vec
        self.uniqueID=uniqueID
        self.df= pd.DataFrame({'uniqueID': uniqueID, 'vec_column':vec})
        func=lambda grp:np.median(grp)
        self.median_pooling=np.vstack(self.df.groupby('uniqueID')['vec_column'].apply(func).values)
        return self.median_pooling

class Autoencoder:
    """
    Autoencoder for learning latent space representation
    architecture simplified for only one hidden layer
    """

    def __init__(self, latent_dim=768, activation='relu', epochs=50, batch_size=128):
        self.latent_dim = latent_dim
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.his = None

    def _compile(self, input_dim):
        """
        compile the computational graph
        """
        input_vec = Input(shape=(input_dim,))
        encoded = Dense(self.latent_dim, activation=self.activation)(input_vec)
        decoded = Dense(input_dim, activation=self.activation)(encoded)
        self.autoencoder = Model(input_vec, decoded)
        self.encoder = Model(input_vec, encoded)
        encoded_input = Input(shape=(self.latent_dim,))
        decoder_layer = self.autoencoder.layers[-1]
        self.decoder = Model(encoded_input, self.autoencoder.layers[-1](encoded_input))
        self.autoencoder.compile(optimizer='adam', loss=keras.losses.mean_squared_error)

    def fit(self, X):
        if not self.autoencoder:
            self._compile(X.shape[1])
        X_train, X_test = train_test_split(X)
        self.his = self.autoencoder.fit(X_train, X_train,
                                        epochs=200,
                                        batch_size=128,
                                        shuffle=True,
                                        validation_data=(X_test, X_test), verbose=0)


class InterFeatureEncoder:
    def __init__(self, latent_dim=100):
        self.latent_dim = latent_dim
        self.feature_lst= None
        self.AE = None

    def concatenate_features(self, feature_1=[], feature_2=[]):
        feature_lst=[]
        if len(feature_1)>0:
            feature_lst.append(feature_1)
        if len(feature_2) > 0:
            feature_lst.append(feature_2)
        '''if len(feature_3) > 0:
            feature_lst.append(feature_3)'''
        self.feature_lst = feature_lst
        cu = self.feature_lst[0]
        print(cu.shape)
        for i, m in enumerate(self.feature_lst):
            if i > 0:
                print(m.shape)
                cu = np.hstack([cu, m])
        self.feature_input = cu

    def autoencoder(self,feature_1=[], feature_2=[]):
        self.concatenate_features(feature_1, feature_2)
        if not self.AE:
            print(self.feature_input.shape)
            self.AE = Autoencoder(latent_dim=self.latent_dim)
            print('Fitting Autoencoder ...')
            self.AE.fit(self.feature_input)
            print('Fitting Autoencoder Done!')
        return dict(ae=self.AE.encoder)
    def ae_predict(self,feature_1=[], feature_2=[],uniqueID=[],model=None):
        if model:
            self.AE.encoder=model
        self.concatenate_features(feature_1, feature_2)
        vec = self.AE.encoder.predict(self.feature_input)
        return dict(vec=vec, uniqueID=uniqueID)