"""
Module to preprocess data. Partial requirement for course
D7062E - Artificial Intelligence and Pattern Recognition
LTU - Fall 2022 - Group 8
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

"""
Function to preprocess the input data by dropping rows
with missing values
"""


def preprocess_data_dropping_missing_values(input_csv_file):
    # Read the input file into a pandas dataframe
    dataframe = pd.read_csv(input_csv_file)

    # Remove unecessary columns
    dataframe = dataframe.drop(dataframe.columns[[240]], axis=1)

    # HANDLING MISSING VALUES (Dropping rows with missing values)
    # https://builtin.com/machine-learning/how-to-preprocess-data-python
    dataframe = dataframe.dropna()

    # return the result
    return dataframe


"""
Function to preprocess the input data by dropping filling out
missing values with the mean of the values for a given column.
"""


def preprocess_data_filling_out_missing_values(input_csv_file):
    dataframe = pd.read_csv(input_csv_file)

    # Remove unecessary columns
    dataframe = dataframe.drop(dataframe.columns[[240]], axis=1)

    # HANDLING MISSING VALUES
    # https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer = imputer.fit(dataframe)
    new_dataframe = pd.DataFrame(imputer.transform(dataframe))

    # return the result
    return new_dataframe


if __name__ == "__main__":
    INPUT_FILE = 'train-final.csv'
    """ Uncomment until line 57 to run the method of dropping rows with missing values
    preprocessed_data = preprocess_data_dropping_missing_values(INPUT_FILE)
    print(preprocessed_data.iloc[11, 0:10].values)
    print(preprocessed_data.info())
    """

    """ Uncomment until line 63 to run the method of filling out missing values """
    preprocessed_data = preprocess_data_filling_out_missing_values(INPUT_FILE)
    print(preprocessed_data.iloc[11, 0:10].values)
    print(preprocessed_data.info())
