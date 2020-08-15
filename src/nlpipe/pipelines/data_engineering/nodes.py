
import warnings
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Any, Dict



SAMPLING_THRESHOLD=0.2  #generates a warning if sample_data_ratio is higher than this number


class LimitDataLoadWarning(UserWarning):
    pass

class NoTargetWarning(UserWarning):
    pass

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



class InputAbstraction:
    def __init__(self, sample_data_ratio=0.02, test_data_ratio= 0.2, drop_nan=1, numeric_labels=0):
        self.sample_data_ratio=sample_data_ratio
        self.test_data_ratio=test_data_ratio
        self.drop_nan=drop_nan
        self.numeric_labels=numeric_labels
    def split_data(self,data=pd.DataFrame(),target=[],columns=[]) -> Dict[str, Any]:
        self.columns = columns
        self.target = target
        # Arguments check
        if data.empty:
            raise PipelineError('Please provide data', 'data argument empty, Cannot proceed without data')
        if not self.columns:
            raise PipelineError('Please provide Input Column Name which has to be subjected to analysis', 'columns argument empty, Cannot proceed with out column name')
        if not self.target:
            data['target']=[1]*data.shape[0]
            warnings.warn("No target column provided, All data points will have same target value, Please use target argument to provide target labels", NoTargetWarning)
        elif len(self.target) > 1:
            data = data.dropna(subset=self.target)
            data['target'] = data[self.target].agg('. '.join, axis=1)
            classes = list(set(data['target'].values.tolist()))
        else:
            data = data.dropna(subset=self.target)
            data['target'] = data[self.target]
            classes = list(set(data['target'].values.tolist()))

        # Shuffle and sample the data
        if self.sample_data_ratio > SAMPLING_THRESHOLD:
            warnings.warn("'This data is huge',Please lower the sample_data_ratio value", LimitDataLoadWarning)
        data = data.sample(frac=self.sample_data_ratio).reset_index(drop=True)
        # split data to train and test sets with labels
        train_x, test_x, train_y, test_y = train_test_split(data[self.columns].values.tolist(), data['target'].values.tolist(), test_size=self.test_data_ratio)
        if self.numeric_labels:
            train_y = list(map(lambda x: classes.index(x), train_y))
            test_y = list(map(lambda x: classes.index(x), test_y))
        # When returning many variables, it is a good practice to give them names:
        return dict(
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            test_y=test_y,
        )
    def get_metadata(self,data=pd.DataFrame()):
        if data.empty:
            raise PipelineError('Please provide data', 'Cannot proceed without data')
        return {'columns': data.columns, 'columnTypes': data.dtypes, 'shape': data.shape}