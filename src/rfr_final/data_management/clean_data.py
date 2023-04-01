"""Function(s) for cleaning the data set(s)."""

import pandas as pd


def task_process_data(df):

    # set week + id as the time series index
    df = df.assign(weekid=df['week']*1000000000000+df['id'])
    df.set_index('weekid', inplace=True)


    # save the processed data to an excel file
    return df