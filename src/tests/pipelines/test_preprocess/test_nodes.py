from pathlib import Path

import pytest
import pandas as pd
from nlpipe.pipelines.Preprocess.nodes import SpacyBulk,Spacyize,CustomTokenParser

@pytest.fixture
def spacy_bulk():
    return SpacyBulk()

@pytest.fixture
def spacyize():
    return Spacyize()

@pytest.fixture
def custom_token_parser():
    return CustomTokenParser()

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

    def test_split_data_nodata_failure(self, spacy_bulk):
        assert 1 == 1