"""
Module to preprocess data. Partial requirement for course
D7062E - Artificial Intelligence and Pattern Recognition
LTU - Fall 2022 - Group 8:
- Alejandro Oliveros
- Emmanouil Manouselis
- Georgios Savvidis
- Johan Bini
- Rafael Silva
"""
import numpy as np
import pandas as pd


def get_dataframe_with_column_names(input_csv_file):
    col_names = []
    for x in range(1, 241):
        col_names.append('Column' + str(x))

    col_names.append('Label_as_string')
    col_names.append('Label_as_number')

    dataframe = pd.read_csv(input_csv_file, header=None)
    dataframe.columns = col_names
    return dataframe


"""
Preprocesses the input data by dropping rows with missing values.
"""


def preprocess_data_dropping_missing_values(input_csv_file, columns_to_drop):
    # Read the input file into a pandas dataframe
    dataframe = get_dataframe_with_column_names(input_csv_file)

    # Remove unecessary columns
    for x in range(len(columns_to_drop)):
        dataframe = dataframe.drop(
            dataframe.columns[[columns_to_drop[x]]], axis=1)

    # HANDLING MISSING VALUES (Dropping rows with missing values)
    # https://builtin.com/machine-learning/how-to-preprocess-data-python
    dataframe = dataframe.dropna()

    # return the result dataframe
    return dataframe


"""
Preprocesses the input data by filling out rows with mean values of group.
"""


def preprocess_data_filling_out_missing_values(
        input_csv_file, label_to_group_by, columns_to_drop):
    dataframe = get_dataframe_with_column_names(input_csv_file)

    # HANDLING MISSING VALUES
    # https://stackoverflow.com/questions/19966018/pandas-filling-missing-values-by-mean-in-each-group

    subset_df = dataframe.loc[:, dataframe.isnull().any()]
    for x in range(len(subset_df.columns)):
        dataframe[subset_df.columns[x]] = dataframe.groupby([label_to_group_by])[
            subset_df.columns[x]].transform(lambda x: x.fillna(x.mean()))

    # Remove unecessary columns
    for x in range(len(columns_to_drop)):
        dataframe = dataframe.drop(
            dataframe.columns[[columns_to_drop[x]]], axis=1)

    return dataframe


"""
Entrypoint.
"""
if __name__ == "__main__":
    """ Running the method by filling out the missing values with the mean of the group """
    INPUT_FILE = 'train-final.csv'
    preprocessed_data = preprocess_data_filling_out_missing_values(
        INPUT_FILE, 'Label_as_string', [240])
    print(preprocessed_data)
    # print(preprocessed_data.info())
    # print(preprocessed_data["Column9"].loc[12:15])

    """ Running the method of discarding all rows that has a NaN value
    INPUT_FILE = 'train-final.csv'
    preprocessed_data = preprocess_data_dropping_missing_values(INPUT_FILE, [240])
    print(preprocessed_data)
    # print(preprocessed_data.info())
    # print(preprocessed_data["Column9"].loc[12:15])
    """
