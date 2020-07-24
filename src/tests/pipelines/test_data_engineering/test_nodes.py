from pathlib import Path

import pytest
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
        test_data_source_df = None
        test_target = []
        test_columns = []

        pytest.raises(AttributeError,input_abstraction.split_data,test_data_source_df, test_target, test_columns)