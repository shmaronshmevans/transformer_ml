import sys

sys.path.append("..")
import pandas as pd
import numpy as np
from processing import hrrr_data
from processing import nysm_data
from processing import get_error
from processing import normalize
from processing import get_flag


def columns_drop(df):
    df = df.drop(
        columns=[
            "level_0",
            "index_x",
            "index_y",
            "lead time",
            "lsm",
            "station_y",
            "lat",
            "lon",
        ]
    )
    df = df.rename(columns={"station_x": "station"})
    return df


def create_data_for_model():
    """
    This function creates and processes data for a LSTM machine learning model.

    Args:
        station (str): The station identifier for which data is being processed.

    Returns:
        new_df (pandas DataFrame): A DataFrame containing processed data.
        df_train (pandas DataFrame): A DataFrame for training the machine learning model.
        df_test (pandas DataFrame): A DataFrame for testing the machine learning model.
        features (list): A list of feature names.
        forecast_lead (int): The lead time for the target variable.
    """
    # load nysm data
    nysm_df = nysm_data.load_nysm_data()
    nysm_df.reset_index(inplace=True)
    nysm_df = nysm_df.rename(columns={"time_1H": "valid_time"})

    # load hrrr data
    hrrr_df = hrrr_data.read_hrrr_data()

    # Filter NYSM data to match valid times from HRRR data and save it to a CSV file.
    mytimes = hrrr_df["valid_time"].tolist()
    nysm_df = nysm_df[nysm_df["valid_time"].isin(mytimes)]

    # Filter data by NY climate division
    nysm_cats_path = "/home/aevans/nwp_bias/src/landtype/data/nysm.csv"
    nysm_cats_df = pd.read_csv(nysm_cats_path)
    nysm_cats_df = nysm_cats_df[
        nysm_cats_df["climate_division_name"] == "Western Plateau"
    ]
    stations = nysm_cats_df["stid"].tolist()
    nysm_df = nysm_df[nysm_df["station"].isin(stations)]
    hrrr_df = hrrr_df[hrrr_df["station"].isin(stations)]

    # merge dataframes so that each row is hrrr + nysm data for the same time step
    # do this for each station individually
    for station in stations:
        nysm_df1 = nysm_df[nysm_df["station"] == station]
        hrrr_df1 = hrrr_df[hrrr_df["station"] == station]

        master_df = hrrr_df1.merge(nysm_df1, on="valid_time")
        master_df = master_df.drop_duplicates(
            subset=["valid_time", "t2m"], keep="first"
        )
        master_df = columns_drop(master_df)

        # Calculate the error using NWP data.
        master_df = get_error.nwp_error("t2m", master_df)
        # encode for day_of_year
        master_df = normalize.encode(master_df, "day_of_year", 366)
        # get flag for non-consecutive time steps
        master_df = get_flag.get_flag(master_df)

        cols_to_carry = ["valid_time", "station", "latitude", "longitude", "flag"]

        new_df = master_df.drop(columns=cols_to_carry)

        new_df, features = normalize.normalize_df(new_df)

        # Split the data into training and testing sets.
        length = len(new_df)
        test_len = int(length * 0.8)
        df_train = new_df.iloc[:test_len].copy()
        df_test = new_df.iloc[test_len:].copy()
        print("Test Set Fraction", len(df_test) / len(new_df))

        # Reintegrate the specified columns back into the training and testing DataFrames.
        for c in cols_to_carry:
            df_train[c] = master_df[c]
            df_test[c] = master_df[c]

    return df_train, df_test, features
