# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt



def pre_process(myframe: pd.core.frame.DataFrame, mystrategy: int):
    if mystrategy == 1:
        # Drop all lines that contain at least 1 NaN value
        df2 = myframe.dropna()

    elif mystrategy == 2:
        # Drop all lines in which numerucal label is missing
        df2 = df.dropna(subset=[df.columns[-1]])
        # Fill missing value with the mean of the column for all but last two columns
        df2.iloc[:, :-2] = df2.iloc[:, :-2].fillna(df2.iloc[:, :-2].mean())

    elif mystrategy == 3:
        # find column indexes with at least one null value
        df2 = myframe.copy()
        c = df2.columns[df2.isnull().any(axis=0)]
        for ci in c:
            df2[ci] = df2[ci].fillna(df2.groupby(240)[ci].transform('mean'))

    else:
        df2 = myframe.copy()

    return df2


# Defining a function to vizualize a single row in a 2-D graph (Z-dimennsion is depth)
def viz_gesture(vizframe: pd.core.frame.DataFrame, points: int, row2viz: int, links: list):
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


# Defining function (process) to vizualize a gesture by label from the groupped dataframe
def viz_gesture_group(vizframe: pd.core.frame.DataFrame, points: int, links: list, gesture: str):
    gg = vizframe[vizframe[240] == gesture]
    viz_gesture(gg, points, 0, links);


# links array is used to code-in the interconnection between the different joints of the body
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

# Import the dataset.
df = pd.read_csv(filepath_or_buffer="train-final.csv", header=None)

# Lines which contain at least one NaN
null_data = df[df.isnull().any(axis=1)]

# pre-processing
dfpp = pre_process(df, 3)

# grouping by label string
dfppg = dfpp.groupby([240], as_index=False).mean()
dfppg = dfppg.reindex(columns=sorted(dfppg.columns))

# try viz_gesture function  for row 6
viz_gesture(dfpp, 20, 6, links);

# try viz_gesture_group function
viz_gesture_group(dfppg, 20, links, 'love')
