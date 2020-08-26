
import warnings
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Any, Dict


SAMPLING_THRESHOLD=0.2  #generates a warning if sample_data_ratio is higher than this number
class LimitDataLoadWarning(UserWarning):
    pass
class InputAbstraction:
    def __init__(self, sample_data_ratio=0.02, test_data_ratio= 0.2, drop_nan=1, numeric_labels=1):
        self.sample_data_ratio=sample_data_ratio
        self.test_data_ratio=test_data_ratio
        self.drop_nan=drop_nan
        self.numeric_labels=numeric_labels
    def split_data(self,data=pd.DataFrame(),target=[],columns=[]) -> Dict[str, Any]:

        self.data = data
        if self.drop_nan:
            self.data = self.data.dropna(subset=target)
        self.columns = columns
        self.target = target
        # Shuffle all the data

        if self.sample_data_ratio>SAMPLING_THRESHOLD:
            self.data = self.data.sample(frac=self.sample_data_ratio).reset_index(drop=True)
            warnings.warn("'This data is huge',Please lower the sample_data_ratio value", LimitDataLoadWarning)
        else:
            self.data = self.data.sample(frac=self.sample_data_ratio).reset_index(drop=True)

        if len(self.target) > 1:
            self.data = self.data.dropna(subset=self.target)
            self.data['target'] = self.data[self.target].agg('. '.join, axis=1)
            classes = list(set(self.data['target'].values.tolist()))
        else:
            self.data['target'] = self.data[self.target]
            classes = list(set(self.data['target'].values.tolist()))
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.data[self.columns].values.tolist(),self.data['target'].values.tolist(), test_size=self.test_data_ratio)
        if self.numeric_labels:
            self.train_y = list(map(lambda x: classes.index(x), self.train_y))
            self.test_y = list(map(lambda x: classes.index(x), self.test_y))
        # When returning many variables, it is a good practice to give them names:
        return dict(
            train_x=self.train_x,
            train_y=self.train_y,
            test_x=self.test_x,
            test_y=self.test_y,
        )
    def get_metadata(self):
        return {'columns': self.data.columns, 'columnTypes': self.data.dtypes, 'shape': self.data.shape}