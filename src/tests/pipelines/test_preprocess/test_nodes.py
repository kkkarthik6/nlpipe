from pathlib import Path

import pytest
import pandas as pd
from nlpipe.pipelines.data_engineering.nodes import InputAbstraction
from nlpipe.pipelines.Preprocess.nodes import SpacyBulk,Spacyize,CustomTokenParser,PipelineError

@pytest.fixture
def spacy_bulk():
    return SpacyBulk()

@pytest.fixture
def spacyize():
    return Spacyize()

@pytest.fixture
def custom_token_parser():
    return CustomTokenParser()

@pytest.fixture
def input_abstraction():
    return InputAbstraction()

@pytest.fixture
def split_data(input_abstraction):
    test_data_source_df = pd.read_csv(r'data/01_raw/iris.csv')
    test_target = ['species']
    test_columns = ['sepal_length','sepal_width','petal_length','petal_width']
 
    test_result = input_abstraction.split_data(test_data_source_df, test_target, test_columns)

    return test_result

class TestPreprocessNodes_SpacyBulk:
    
    def test_defaults(self, spacy_bulk):
        assert spacy_bulk.ner == 1
        assert spacy_bulk.parser == 1
        assert spacy_bulk.tagger == 1
        assert len(spacy_bulk.disable) == 0

    def test_defaults_disabled_success(self,spacy_bulk):
        test_ner = 0
        test_parser = 0
        test_tagger = 0

        test_spacy_bulk = SpacyBulk(ner=test_ner, parser=test_parser, tagger=test_tagger)

        assert len(test_spacy_bulk.disable) == 3

    def test_get_df_empty_text_faliure(self, spacy_bulk, split_data):
        test_text_corpus = []
        test_preprocess_out = []

        with pytest.raises(PipelineError) as pipelineError:
            spacy_bulk.get_DF(test_text_corpus,test_preprocess_out)
        assert str(pipelineError.value) == "('Input text cannot be None', 'This object extract sentences, parts of speech, named entity recognition, dependencies')"