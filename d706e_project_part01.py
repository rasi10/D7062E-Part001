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
    col_names = [
        "Column01",
        "Column02",
        "Column03",
        "Column04",
        "Column05",
        "Column06",
        "Column07",
        "Column08",
        "Column09",
        "Column10",
        "Column12",
        "Column12",
        "Column13",
        "Column14",
        "Column15",
        "Column16",
        "Column17",
        "Column18",
        "Column19",
        "Column20",
        "Column22",
        "Column22",
        "Column23",
        "Column24",
        "Column25",
        "Column26",
        "Column27",
        "Column28",
        "Column29",
        "Column30",
        "Column32",
        "Column32",
        "Column33",
        "Column34",
        "Column35",
        "Column36",
        "Column37",
        "Column38",
        "Column39",
        "Column40",
        "Column42",
        "Column42",
        "Column43",
        "Column44",
        "Column45",
        "Column46",
        "Column47",
        "Column48",
        "Column49",
        "Column50",
        "Column52",
        "Column52",
        "Column53",
        "Column54",
        "Column55",
        "Column56",
        "Column57",
        "Column58",
        "Column59",
        "Column60",
        "Column62",
        "Column62",
        "Column63",
        "Column64",
        "Column65",
        "Column66",
        "Column67",
        "Column68",
        "Column69",
        "Column70",
        "Column72",
        "Column72",
        "Column73",
        "Column74",
        "Column75",
        "Column76",
        "Column77",
        "Column78",
        "Column79",
        "Column80",
        "Column82",
        "Column82",
        "Column83",
        "Column84",
        "Column85",
        "Column86",
        "Column87",
        "Column88",
        "Column89",
        "Column90",
        "Column92",
        "Column92",
        "Column93",
        "Column94",
        "Column95",
        "Column96",
        "Column97",
        "Column98",
        "Column99",
        "Column100",
        "Column101",
        "Column102",
        "Column103",
        "Column104",
        "Column105",
        "Column106",
        "Column107",
        "Column108",
        "Column109",
        "Column110",
        "Column112",
        "Column112",
        "Column113",
        "Column114",
        "Column115",
        "Column116",
        "Column117",
        "Column118",
        "Column119",
        "Column120",
        "Column122",
        "Column122",
        "Column123",
        "Column124",
        "Column125",
        "Column126",
        "Column127",
        "Column128",
        "Column129",
        "Column130",
        "Column132",
        "Column132",
        "Column133",
        "Column134",
        "Column135",
        "Column136",
        "Column137",
        "Column138",
        "Column139",
        "Column140",
        "Column142",
        "Column142",
        "Column143",
        "Column144",
        "Column145",
        "Column146",
        "Column147",
        "Column148",
        "Column149",
        "Column150",
        "Column152",
        "Column152",
        "Column153",
        "Column154",
        "Column155",
        "Column156",
        "Column157",
        "Column158",
        "Column159",
        "Column160",
        "Column162",
        "Column162",
        "Column163",
        "Column164",
        "Column165",
        "Column166",
        "Column167",
        "Column168",
        "Column169",
        "Column170",
        "Column172",
        "Column172",
        "Column173",
        "Column174",
        "Column175",
        "Column176",
        "Column177",
        "Column178",
        "Column179",
        "Column180",
        "Column182",
        "Column182",
        "Column183",
        "Column184",
        "Column185",
        "Column186",
        "Column187",
        "Column188",
        "Column189",
        "Column190",
        "Column192",
        "Column192",
        "Column193",
        "Column194",
        "Column195",
        "Column196",
        "Column197",
        "Column198",
        "Column199",
        "Column200",
        "Column201",
        "Column202",
        "Column203",
        "Column204",
        "Column205",
        "Column206",
        "Column207",
        "Column208",
        "Column209",
        "Column210",
        "Column212",
        "Column212",
        "Column213",
        "Column214",
        "Column215",
        "Column216",
        "Column217",
        "Column218",
        "Column219",
        "Column220",
        "Column222",
        "Column222",
        "Column223",
        "Column224",
        "Column225",
        "Column226",
        "Column227",
        "Column228",
        "Column229",
        "Column230",
        "Column232",
        "Column232",
        "Column233",
        "Column234",
        "Column235",
        "Column236",
        "Column237",
        "Column238",
        "Column239",
        "Column240",
        "Label_as_string",
        "Label_as_number"]
    dataframe = pd.read_csv(input_csv_file, header=None)
    dataframe.columns = col_names
    return dataframe


"""
Preprocesses the input data by dropping rows with missing values.
"""


def preprocess_data_dropping_missing_values(input_csv_file):
    # Read the input file into a pandas dataframe
    dataframe = get_dataframe_with_column_names(input_csv_file)

    # Remove unecessary columns
    dataframe = dataframe.drop(dataframe.columns[[240]], axis=1)

    # HANDLING MISSING VALUES (Dropping rows with missing values)
    # https://builtin.com/machine-learning/how-to-preprocess-data-python
    dataframe = dataframe.dropna()

    # return the result dataframe
    return dataframe


"""
Preprocesses the input data by filling out rows with mean values of group.
"""


def preprocess_data_filling_out_missing_values(input_csv_file):
    dataframe = get_dataframe_with_column_names(input_csv_file)

    # HANDLING MISSING VALUES
    # https://stackoverflow.com/questions/19966018/pandas-filling-missing-values-by-mean-in-each-group
    dataframe["Column08"] = dataframe.groupby(
        ['Label_as_string'])['Column08'].transform(
        lambda x: x.fillna(
            x.mean()))
    dataframe["Column09"] = dataframe.groupby(
        ['Label_as_string'])['Column09'].transform(
        lambda x: x.fillna(
            x.mean()))
    dataframe["Column10"] = dataframe.groupby(
        ['Label_as_string'])['Column10'].transform(
        lambda x: x.fillna(
            x.mean()))
    dataframe["Column15"] = dataframe.groupby(
        ['Label_as_string'])['Column15'].transform(
        lambda x: x.fillna(
            x.mean()))
    dataframe["Column16"] = dataframe.groupby(
        ['Label_as_string'])['Column16'].transform(
        lambda x: x.fillna(
            x.mean()))
    dataframe["Column17"] = dataframe.groupby(
        ['Label_as_string'])['Column17'].transform(
        lambda x: x.fillna(
            x.mean()))

    # Remove unecessary columns
    dataframe = dataframe.drop(dataframe.columns[[240]], axis=1)

    # return the result dataframe
    return dataframe


"""
Entrypoint.
"""
if __name__ == "__main__":
    """ Running the method by filling out the missing values with the mean of the group """
    INPUT_FILE = 'train-final.csv'
    preprocessed_data = preprocess_data_filling_out_missing_values(INPUT_FILE)
    print(preprocessed_data.info())
    print(preprocessed_data)
    # print(preprocessed_data["Column09"].loc[12:15])

    """ Running the method of discarding all rows that has a NaN value
    INPUT_FILE = 'train-final.csv'
    preprocessed_data = preprocess_data_dropping_missing_values(INPUT_FILE)
    print(preprocessed_data.iloc[11, 0:10].values)
    print(preprocessed_data.info())
    print(preprocessed_data)
    """
