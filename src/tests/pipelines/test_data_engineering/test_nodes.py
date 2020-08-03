from pathlib import Path

import pytest
import pandas as pd
from nlpipe.pipelines.data_engineering.nodes import InputAbstraction

@pytest.fixture
def input_abstraction():
    return InputAbstraction()

class TestDataEngineeringNodes:
    def test_defaults(self, input_abstraction):
        assert input_abstraction.sample_data_ratio == 0.02 
        assert input_abstraction.test_data_ratio == 0.2
        assert input_abstraction.drop_nan == 1
        assert input_abstraction.numeric_labels == 1

    def test_split_data_nodata_failure(self, input_abstraction):
        test_data_source_df = pd.DataFrame()
        test_target = []
        test_columns = []

        #with pytest.raises(ValueError) as valueError:
        #    input_abstraction.split_data(test_data_source_df, test_target, test_columns)
        
        #assert str(valueError.value) == 'The source dataframe cannot be empty or None'

        pytest.raises(ValueError,input_abstraction.split_data,test_data_source_df, test_target, test_columns)
    
    def test_split_data_nodata_withtarget_failure(self, input_abstraction):
        test_data_source_df = pd.DataFrame()
        test_target = ['species']
        test_columns = []

        pytest.raises(ValueError,input_abstraction.split_data,test_data_source_df, test_target, test_columns)
    
    def test_split_data_nodata_withcolumns_failure(self, input_abstraction):
        test_data_source_df = pd.DataFrame()
        test_target = []
        test_columns = ['sepal_length']

        pytest.raises(ValueError,input_abstraction.split_data,test_data_source_df, test_target, test_columns)

    def test_split_data_withdata_notarget_failure(self, input_abstraction):
        test_data_source_df = pd.read_csv(r'data/01_raw/iris.csv')
        test_target = []
        test_columns = []

        #with pytest.raises(ValueError) as valueError:
        #    input_abstraction.split_data(test_data_source_df, test_target, test_columns)
        #assert str(valueError.value) == 'target cannot be empty' 
        pytest.raises(ValueError,input_abstraction.split_data,test_data_source_df, test_target, test_columns)
    
    def test_split_data_withdata_nocolumns_failure(self, input_abstraction):
        test_data_source_df = pd.read_csv(r'data/01_raw/iris.csv')
        test_target = ['species','species']
        test_columns = []

        #with pytest.raises(ValueError) as valueError:
        #    input_abstraction.split_data(test_data_source_df, test_target, test_columns)
        #assert str(valueError.value) == 'target cannot be empty' 
        #pytest.raises(ValueError,input_abstraction.split_data,test_data_source_df, test_target, test_columns)
        assert type(input_abstraction.split_data(test_data_source_df, test_target, test_columns)) == dict
    
    def test_split_data_withdata_success(self, input_abstraction):
        test_data_source_df = pd.read_csv(r'data/01_raw/iris.csv')
        test_target = ['species']
        test_columns = []
        input_abstraction.sample_data_ratio = 1
        
        #with pytest.raises(ValueError) as valueError:
        #    input_abstraction.split_data(test_data_source_df, test_target, test_columns)
        #assert str(valueError.value) == 'target cannot be empty' 
        #pytest.raises(ValueError,input_abstraction.split_data,test_data_source_df, test_target, test_columns)
        test_result = input_abstraction.split_data(test_data_source_df, test_target, test_columns)
        
        assert len(test_result['train_x']) == 120
        assert len(test_result['train_y']) == 120
        assert len(test_result['test_x']) == 30
        assert len(test_result['test_y']) == 30

    def test_get_metadata(self, input_abstraction):
        test_data_source_df = pd.read_csv(r'data/01_raw/iris.csv')
        test_target = ['species','species']
        test_columns = []

        #with pytest.raises(ValueError) as valueError:
        #    input_abstraction.split_data(test_data_source_df, test_target, test_columns)
        #assert str(valueError.value) == 'target cannot be empty' 
        #pytest.raises(ValueError,input_abstraction.split_data,test_data_source_df, test_target, test_columns)
        
        test_metadata = input_abstraction.get_metadata()
        

        assert type(input_abstraction.split_data(test_data_source_df, test_target, test_columns)) == dict