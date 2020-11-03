from pathlib import Path

import pytest
import pandas as pd
from nlpipe.pipelines.data_engineering.nodes import InputAbstraction, PipelineError

@pytest.fixture
def input_abstraction():
    return InputAbstraction()

class TestDataEngineeringNodes:
    def test_defaults(self, input_abstraction):
        assert input_abstraction.sample_data_ratio == 0.02 
        assert input_abstraction.test_data_ratio == 0.2
        assert input_abstraction.drop_nan == 1
        assert input_abstraction.numeric_labels == 0

    def test_split_data_nodata_failure(self, input_abstraction):
        test_data_source_df = pd.DataFrame()
        test_target = []
        test_columns = []

        pytest.raises(PipelineError,input_abstraction.split_data,test_data_source_df, test_target, test_columns)
    
    def test_split_data_nodata_withtarget_failure(self, input_abstraction):
        test_data_source_df = pd.DataFrame()
        test_target = ['species']
        test_columns = []

        pytest.raises(PipelineError,input_abstraction.split_data,test_data_source_df, test_target, test_columns)
    
    def test_split_data_nodata_withcolumns_failure(self, input_abstraction):
        test_data_source_df = pd.DataFrame()
        test_target = []
        test_columns = ['sepal_length']

        pytest.raises(PipelineError,input_abstraction.split_data,test_data_source_df, test_target, test_columns)

    def test_split_data_withdata_nocolumns_failure(self, input_abstraction):
        test_data_source_df = pd.read_csv(r'data/01_raw/iris.csv')
        test_target = ['species']
        test_columns = []

        pytest.raises(PipelineError,input_abstraction.split_data,test_data_source_df, test_target, test_columns)
    
    def test_split_data_withdata_success(self, input_abstraction):
        test_data_source_df = pd.read_csv(r'data/01_raw/iris.csv')
        test_target = ['species']
        test_columns = ['sepal_length','sepal_width','petal_length','petal_width']
        input_abstraction.sample_data_ratio = 1
        
        test_result = input_abstraction.split_data(test_data_source_df, test_target, test_columns)
        
        assert len(test_result['train_x']) == 120
        assert len(test_result['train_y']) == 120
        assert len(test_result['test_x']) == 30
        assert len(test_result['test_y']) == 30

    def test_split_data_withdata_multitarget_success(self, input_abstraction):
        test_data_source_df = pd.read_csv(r'data/01_raw/iris.csv')
        test_target = ['species','species']
        test_columns = ['sepal_length','sepal_width','petal_length','petal_width']
        input_abstraction.sample_data_ratio = 1
        
        test_result = input_abstraction.split_data(test_data_source_df, test_target, test_columns)
        
        assert len(test_result['train_x']) == 120
        assert len(test_result['train_y']) == 120
        assert len(test_result['test_x']) == 30
        assert len(test_result['test_y']) == 30

    def test_split_data_withdata_numericlabels_success(self, input_abstraction):
        test_data_source_df = pd.read_csv(r'data/01_raw/iris.csv')
        test_target = ['species']
        test_columns = ['sepal_length','sepal_width','petal_length','petal_width']
        test_numericlabels = [0,1,2]
        input_abstraction.sample_data_ratio = 1
        input_abstraction.numeric_labels = 1

        test_result = input_abstraction.split_data(test_data_source_df, test_target, test_columns)
        
        assert len(test_result['train_x']) == 120
        assert len(test_result['train_y']) == 120
        assert len(test_result['test_x']) == 30
        assert len(test_result['test_y']) == 30
        assert set(test_result['train_y']) == set(test_numericlabels)
        assert set(test_result['test_y']) == set(test_numericlabels)

    def test_get_metadata_nodata(self, input_abstraction):
        test_data_source_df = pd.DataFrame()
        
        pytest.raises(PipelineError,input_abstraction.get_metadata,test_data_source_df)