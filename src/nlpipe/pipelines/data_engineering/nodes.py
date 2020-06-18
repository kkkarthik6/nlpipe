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

'''Data Abstraction'''
import warnings
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Any, Dict
class LimitDataLoadWarning(UserWarning):
    pass
class InputAbstraction:
    def __init__(self, sample_data_ratio=0.02, test_data_ratio= 0.2,drop_nan=1,numeric_labels=1):
        self.sample_data_ratio=sample_data_ratio
        self.test_data_ratio=test_data_ratio
        self.drop_nan=drop_nan
        self.numeric_labels=numeric_labels
    def split_data(self,data=pd.DataFrame(),target=[],columns=[]) -> Dict[str, Any]:
        """
        The split ratio parameter is taken from conf/project/parameters.yml.
        The data and the parameters will be loaded and provided to your function
        automatically when the pipeline is executed and it is time to run this node.
        """
        self.data = data
        if self.drop_nan:
            self.data=self.data.dropna(subset=target)
        self.columns = columns
        self.target=target
        # Shuffle all the data
        threshold=0.2
        if self.sample_data_ratio>threshold:
            self.data = self.data.sample(frac=self.sample_data_ratio).reset_index(drop=True)
            warnings.warn("'This data is huge',Please lower the sample_data_ratio value",LimitDataLoadWarning)
        else:
            self.data = self.data.sample(frac=self.sample_data_ratio).reset_index(drop=True)

        if len(self.target)>1:
            self.data=self.data.dropna(subset=self.target)
            self.data['target']=self.data[self.target].agg('. '.join, axis=1)
            classes=list(set(self.data['target'].values.tolist()))
        else:
            self.data['target']=self.data[self.target]
            classes=list(set(self.data['target'].values.tolist()))
        self.train_x,self.test_x,self.train_y,self.test_y=train_test_split(self.data[self.columns].values.tolist(),self.data['target'].values.tolist(),test_size=self.test_data_ratio)
        if self.numeric_labels:
            self.train_y=list(map(lambda x: classes.index(x),self.train_y))
            self.test_y = list(map(lambda x: classes.index(x), self.test_y))
        # When returning many variables, it is a good practice to give them names:
        return dict(
            train_x=self.train_x,
            train_y=self.train_y,
            test_x=self.test_x,
            test_y=self.test_y,
        )
    def get_metadata(self):
        return {'columns':self.data.columns,'columnTypes':self.data.dtypes,'shape':self.data.shape}