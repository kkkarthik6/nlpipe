
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
    def __init__(self, stopwords='english', max_df=0.3, max_features=10000, ngram_range=(1, 2)):
        self.stopwords=stopwords
        self.max_df=max_df
        self.max_features=max_features
        self.ngram_range=ngram_range

    def get_countVectorizer(self, text_corpus):
        if text_corpus==None:
            raise PipelineError('Please provide text corpus', 'This function needs text corpus to generate count vectors')
        #print(self.text_corpus)
        count_vectorizer=CountVectorizer(stop_words=self.stopwords, max_df=self.max_df, max_features=self.max_features, ngram_range=self.ngram_range)
        count_vectorizer.fit(text_corpus)
        return count_vectorizer
    def get_tfidfVectorizer(self,text_corpus):
        if text_corpus==None:
            raise PipelineError('Please provide text corpus', 'This function needs text corpus to generate tfidf vectors')
        tfidf_vectorizer=TfidfVectorizer(stop_words=self.stopwords, max_df=self.max_df, max_features=self.max_features, ngram_range=self.ngram_range)
        tfidf_vectorizer.fit(text_corpus)
        return tfidf_vectorizer
    def get_sparseRepresentation(self,text_corpus,vectorizer):
        if text_corpus==None:
            raise PipelineError('Please provide text corpus', 'This function needs text corpus to generate sparse representation')
        if vectorizer:
            transformed=vectorizer.transform(text_corpus)
        else:
            raise PipelineError('Please provide vectorizer', 'This function needs count/tfidf vectorizer to generate sparse representation')
        sparse_rep = csr_matrix(transformed)
        return sparse_rep

class TopicVectorizer:
    def __init__(self,count_vectorizer=None,sentence_col='sentences', unique_id='textID'):
        self.corex=None
        self.lda=None
        self.hdp=None
        self.sentence_col = sentence_col
        self.uniqueID = unique_id
        self.text_corpus=None
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
            countvectorizer_obj=ClassicVectorizationTrain()
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
        data_df = df
        model=SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        embeddings=model.encode(data_df[self.sentence_col].values.tolist())
        self.bert_fast=pd.DataFrame({self.uniqueID: data_df[self.uniqueID].values.tolist(),self.sentence_col: data_df[self.sentence_col].values.tolist(),'embeddings':embeddings})
        return np.asarray(embeddings)
    def get_precBert(self,df=pd.DataFrame()):
        data_df = df
        model=SentenceTransformer('bert-base-nli-stsb-mean-tokens')
        embeddings=model.encode(data_df[self.sentence_col].values.tolist())
        self.bert_high_precision=pd.DataFrame({self.uniqueID: data_df[self.uniqueID].values.tolist(),self.sentence_col: data_df[self.sentence_col].values.tolist(),'embeddings':embeddings})
        return np.asarray(embeddings)
    def get_meanBert(self,df=pd.DataFrame()):
        data_df = df
        model=SentenceTransformer('bert-base-nli-mean-tokens')
        embeddings=model.encode(data_df[self.sentence_col].values.tolist())
        self.bert_mid_precision=pd.DataFrame({self.uniqueID: data_df[self.uniqueID].values.tolist(),self.sentence_col: data_df[self.sentence_col].values.tolist(),'embeddings':embeddings})
        return np.asarray(embeddings)

class IntraFeaturePoolingDF:
    def __init__(self, vector_coulumn='embeddings', unique_id='textID'):
        self.vec_column=vector_coulumn
        self.uniqueID=unique_id
    def get_mean_pooling(self, df=pd.DataFrame()):
        func=lambda grp:np.mean(grp)
        self.mean_pooling=np.vstack(df.groupby(self.uniqueID)[self.vec_column].apply(func).values)
        return self.mean_pooling
    def get_median_pooling(self, df=pd.DataFrame()):
        func=lambda grp:np.median(grp)
        self.median_pooling=np.vstack(df.groupby(self.uniqueID)[self.vec_column].apply(func).values)
        return self.median_pooling

class IntraFeaturePooling:
    def __init__(self):
        self.vec=[]
        self.uniqueID=[]
        self.df=pd.DataFrame()
    def get_mean_pooling(self, vec, uniqueID):
        self.vec=vec
        self.uniqueID = uniqueID
        df = pd.DataFrame({'uniqueID': uniqueID, 'vec_column': list(vec)})
        print(df.shape)
        func = lambda grp: np.mean(grp)
        self.mean_pooling=np.vstack(df.groupby('uniqueID')['vec_column'].apply(func).values)
        print(self.mean_pooling.shape)
        return self.mean_pooling
    def get_median_pooling(self, vec, uniqueID):
        self.vec=vec
        self.uniqueID=uniqueID
        df= pd.DataFrame({'uniqueID': uniqueID, 'vec_column': vec})
        func = lambda grp:np.median(grp)
        self.median_pooling=np.vstack(df.groupby('uniqueID')['vec_column'].apply(func).values)
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
        gamma=5
        cu = np.hstack([gamma*feature_1, feature_2])
        print(cu.shape)
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