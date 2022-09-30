# -*- coding: utf-8 -*-

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

import pandas as pd
import matplotlib.pyplot as plt


def get_dataframe_with_column_names(input_csv_file):
    col_names = []
    for x in range(1, 241):
        col_names.append('Column' + str(x))

    col_names.append('Label_as_string')
    col_names.append('Label_as_number')

    dataframe = pd.read_csv(input_csv_file, header=None)
    dataframe.columns = col_names
    return dataframe


def pre_process(myframe: pd.core.frame.DataFrame, mystrategy: int, centering: bool, scaling: bool):
    # create a copy of input dataframe
    df2 = myframe.copy()

    # define columns containing x,y,z coordinates
    colsx = df.columns[0::3][0:20]
    colsy = df.columns[1::3][0:20]
    colsz = df.columns[2::3][0:20]

    if centering:
        # Center all gesture datapoints
        # Calculate centroid of each datapoint and save its cx,cy,cz in the pre-processed dataframe
        df2['cx'] = df2[colsx].mean(axis=1).tolist()
        df2['cy'] = df2[colsy].mean(axis=1).tolist()
        df2['cz'] = df2[colsz].mean(axis=1).tolist()
        # For each row, subtract centroid from each point
        df2[colsx] = df2[colsx].sub(df2['cx'], axis=0)
        df2[colsy] = df2[colsy].sub(df2['cy'], axis=0)
        df2[colsz] = df2[colsz].sub(df2['cz'], axis=0)

    if scaling:
        # Scale all gesture datapoints
        # Calculate the width, height and depth of each datapoint and store value in columns w,h,d
        df2['w'] = (df2[colsx].max(axis=1) - df2[colsx].min(axis=1)).tolist()
        df2['h'] = (df2[colsy].max(axis=1) - df2[colsy].min(axis=1)).tolist()
        df2['d'] = (df2[colsz].max(axis=1) - df2[colsz].min(axis=1)).tolist()
        # Scale the coordinates of each point based on the above values
        df2[colsx] = df2[colsx].div(df2['w'], axis=0)
        df2[colsy] = df2[colsy].div(df2['h'], axis=0)
        df2[colsz] = df2[colsz].div(df2['d'], axis=0)
    
    if mystrategy == 1:
        # Drop all lines that contain at least 1 NaN value
        df2 = myframe.dropna()

    elif mystrategy == 2:
        # Drop all lines in which numerucal label is missing
        df2 = df.dropna(subset=[df.columns[-1]])
        # Fill missing value with the mean of the column for all but last two
        # columns
        df2.iloc[:, :-2] = df2.iloc[:, :-2].fillna(df2.iloc[:, :-2].mean())

    elif mystrategy == 3:
        # find column indexes with at least one null value
        df2 = myframe.copy()
        c = df2.columns[df2.isnull().any(axis=0)]
        for ci in c:
            df2[ci] = df2[ci].fillna(df2.groupby(240)[ci].transform('mean'))

    return df2


# Defining a function to vizualize a single row in a 2-D graph
# (Z-dimennsion is depth)
def viz_gesture(
        vizframe: pd.core.frame.DataFrame,
        points: int,
        row2viz: int,
        links: list):
    g = vizframe.iloc[row2viz]
    xx = g[::3][0:int(points)].tolist()
    yy = g[1::3][0:int(points)].tolist()
    zz = g[2::3][0:int(points)].tolist()

    plt.scatter(xx, yy)

    for j1 in range(0, points):
        for j2 in range(0, points):
            if links[j1][j2] == 1:
                plt.plot([xx[j1], xx[j2]], [yy[j1], yy[j2]], 'r-')

    plt.title(g[4 * 3 * points])
    plt.show()
    return 0


# Defining function (process) to vizualize a gesture by label from the
# groupped dataframe
def viz_gesture_group(
        vizframe: pd.core.frame.DataFrame,
        points: int,
        links: list,
        gesture: str):
    gg = vizframe[vizframe[240] == gesture]
    viz_gesture(gg, points, 0, links)


# links array is used to code-in the interconnection between the different
# joints of the body
def get_interconnection_list():
    links = [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]
    return links


if __name__ == "__main__":
    # Constant with the path to the input file
    INPUT_TRAIN_FILE = 'train-final.csv'
    # Import the dataset.
    df = pd.read_csv(filepath_or_buffer=INPUT_TRAIN_FILE, header=None)

    # Get the interconnection list
    links = get_interconnection_list()

    # Lines which contain at least one NaN
    null_data = df[df.isnull().any(axis=1)]

    # pre-processing
    dfpp = pre_process(df, 3)

    # grouping by label string
    dfppg = dfpp.groupby([240], as_index=False).mean()
    dfppg.insert(240, 240, dfppg.pop(240))

    # try viz_gesture function  for row 6
    viz_gesture(dfpp, 20, 6, links)

    # try viz_gesture_group function
    viz_gesture_group(dfppg, 20, links, 'love')
