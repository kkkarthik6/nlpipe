

'''Data Preprocessing'''
import spacy
import numpy as np
import pandas as pd

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


class MakeDataframe():
    def __init__(self, column_names: list, column_values: list):
        self.column_names=column_names
        self.column_values=column_values
    def run(self):
        if self.column_names and self.column_values:
            assert len(self.column_names)==len(self.column_values), 'number column names and column values provided do not match'
            it_values = iter(self.column_values)
            the_len = len(next(it_values))
            if not all(len(l) == the_len for l in it_values):
                raise PipelineError('Not all in column_values lists have same length!',\
                                    'Please makes sure number of rows are same for all columns')
            data=pd.DataFrame(dict(zip(self.column_names,self.column_values)))
            del self.column_values
            del self.column_names
            return data
        else:
            raise PipelineError('Cannot process empty lists', 'Please provide column_names and column_values')


class DataOperations():
    '''
    1)rename columns
    2)drop duplicates
    3)drop nulls
    4)sample data
    5)select columns
    6)delete columns
    7)slice
    8)select columns and number of rows
    9)joins/merges
    10)concats
    11)apply lambda functions on column
    12)combine columns
    '''
    def __init__(self,pyspark:bool):
        self.pyspark=pyspark
    def rename_col(self,df:pd.DataFrame(),dict_conversion:dict):
        if df.empty():
            raise PipelineError('The data frame provided was empty','Please check the dataframe argument')
        for k in list(dict_conversion.keys()):
            if k not in self.df.columns:
                raise PipelineError('Please check the old column names', 'The provided column/columns do not exit')
        df=df.rename(cols=dict_conversion)
        return df
    def drop_duplicates(self, df:pd.DataFrame(), column_names:list, retain:bool):
        if df.empty():
            raise PipelineError('The data frame provided was empty','Please check the dataframe argument')
        if len(column_names)<1:
            return df.drop_duplicates(keep=retain)
        else:
            return df.drop_duplicates(subset=column_names, keep=retain)
    def drop_nans(self, df:pd.DataFrame(), column_names:list, drop_columns:bool):
        if df.empty():
            raise PipelineError('The data frame provided was empty', 'Please check the dataframe argument')
        if len(column_names)<1:
            return df.dropna(axis=drop_columns)
        else:
            return df.dropna(subset=column_names, axis=drop_columns)
    def random_sample_data(self,df:pd.DataFrame(),quantity:float,isratio:bool):
        if df.empty():
            raise PipelineError('The data frame provided was empty', 'Please check the dataframe argument')
        if isratio and quantity<1:
            return df.sample(frac=quantity,random_state=2)
        elif isratio=False and quantity>1:
            return df.sample(int(quantity),random_state=2)
        else:
            raise PipelineError('It can either be ratio or a quantity','Please check the isratio flag to match with type of sampling')
    def stratified_sample_data(self,df:pd.DataFrame(),group_cols:list,ratio:float):
        if df.empty():
            raise PipelineError('The data frame provided was empty', 'Please check the dataframe argument')
        if len(group_cols)<1:
            raise PipelineError('Please provide the stratification group', 'group_cols argument cannot be empty')
        groups= df.groupby(group_cols)
        grp_list=[]
        for i in groups.indices:
            grp_list.append(groups.get_group(i).sample(frac=ratio))
        df_stratified=pd.concat(grp_list)
        return df_stratified
    def select_columns(self,df:pd.DataFrame(),col_list:list):
        if df.empty():
            raise PipelineError('The data frame provided was empty', 'Please check the dataframe argument')
        if len(col_list)<1:
            raise PipelineError('Please provide the columns that are wanted','col_list arfument cannot be empty')
        try:
            df=df[col_list]
        except:
            raise PipelineError('One or more columns that were provided do not exist in dataframe', 'Please check col_list to see if all there are missing columns from original dataframe')
        return df
    def delete_columns(self,df:pd.DataFrame(),col_list:list):
        if df.empty():
            raise PipelineError('The data frame provided was empty', 'Please check the dataframe argument')
        if len(col_list)<1:
            raise PipelineError('Please provide the columns that needs to be deleted','col_list arfument cannot be empty')
        for i in col_list:
            try:
                del df[i]
            except:
                print('The column does not exist : '+str(i))
        return df
    def slice_rows(self, df:pd.DataFrame(),x_ind:tuple):
        if df.empty():
            raise PipelineError('The data frame provided was empty', 'Please check the dataframe argument')
        if len(x_ind)==2 and max(x_ind)<=df.shape[0]:
            df_slice=df.iloc[x_ind[0]:x_ind[1],:]
        elif len(x_ind)==1:
            df_slice=df.iloc[x_ind[0]:,:]
        else:
            raise PipelineError('Please check the slice index','Provide a valid slice index for x_ind argument')
        return df_slice
    def join_data(self,df1:pd.DataFrame(),df2:pd.DataFrame(),join_type:str,unique_id:str,right_unique_id:str,left_unique_id:str):
        if df1.empty() or df2.empty():
            raise PipelineError('One or more dataframes provided are empty', 'Please provide 2 valid dataframes you would like to merge')
        if unique_id=='' and right_unique_id=='' and left_unique_id=='':
            raise PipelineError('Unique key not provided to merge','Please provide one or more unique keys to merge the dataframes on')
        if join_type not in ['inner','outer','right','left']:
            raise PipelineError('Join type unknown', "Please provide a join type from one the following list ['inner','outer','right','left']")
        if unique_id:
            df_merge=df1.merge(df2,how=join_type,on=unique_id )
        elif right_unique_id and left_unique_id:
            df_merge=df1.merge(df2,how=join_type,right_on=right_unique_id,left_on=left_unique_id)
        else:
            raise PipelineError("Could not join", 'Please provide unique_id fields to be merged on in both dataframes')
        return df_merge
    def concatenate(self, df_lists:list):
        if len(df_lists)>0:
            return pd.concat(df_lists)
    def apply_functions(self, df:pd.DataFrame(),column:str,return_coumn:str,function:object):
        if df.empty():
            raise PipelineError('The data frame provided was empty', 'Please check the dataframe argument')
        if column!='' and return_coumn!='':
            raise PipelineError('One or more column names were not provided','Please provide the column name where the lambda function has to be applied and also column name where results has to be stored')
        if function:
            df[return_coumn] = df[column].apply(lambda x: function(x))
            return df
        else:
            raise PipelineError('Please provide the function that has to be applied on dataframe','Function not provided')
    def combine_columns(self,df:pd.DataFrame(),combined_column:str,columns:list):
        if df.empty():
            raise PipelineError('The data frame provided was empty', 'Please check the dataframe argument')
        if combined_column!='':
            raise PipelineError('Combined column name was not provided','Please provide the combined column name where results have to be stored')
        if len(columns)<2:
            raise PipelineError('One or more column names were not provided',\
                                'Please provide the column names that needs to be combined as one ')
        elif len(columns)==2:
            df[combined_column]=df[columns[0]]+' '+df[columns[1]]
        else:
            df[combined_column] = df[columns[0]] + ' ' + df[columns[1]]
            for i in columns[2:]:
                df[combined_column]=df[combined_column]+ ' '+df[i]
        return df















