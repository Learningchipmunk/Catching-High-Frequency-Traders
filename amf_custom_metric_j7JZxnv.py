# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 15:52:37 2020

"""

from sklearn import metrics


def micro_average_f1_score(dataframe_y_true, dataframe_y_pred):
    """
    Args
        dataframe_y_true: Pandas Dataframe
            Dataframe containing the true values of y.
            This dataframe was obtained by reading a csv file with following instruction:
            dataframe_y_true = pd.read_csv(CSV_1_FILE_PATH, index_col=0, sep=',')

        dataframe_y_pred: Pandas Dataframe
            This dataframe was obtained by reading a csv file with following instruction:
            dataframe_y_pred = pd.read_csv(CSV_2_FILE_PATH, index_col=0, sep=',')

    Returns
        score: Float
            The metric evaluated with the two dataframes. This must not be NaN.
    """

    score = metrics.f1_score(dataframe_y_true["type"], dataframe_y_pred["type"], average = "micro")

    return score


if __name__ == '__main__':
    import pandas as pd
    CSV_FILE_Y_TRUE = '--------.csv'
    CSV_FILE_Y_PRED = '--------.csv'
    df_y_true = pd.read_csv(CSV_FILE_Y_TRUE, index_col=0, sep=',')
    df_y_pred = pd.read_csv(CSV_FILE_Y_PRED, index_col=0, sep=',')
    print(micro_average_f1_score(df_y_true, df_y_pred))