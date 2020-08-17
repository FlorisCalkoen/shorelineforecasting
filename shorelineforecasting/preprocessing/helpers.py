import os
import pandas as pd
from typing import List
from tqdm.auto import tqdm
import numpy as np
from datetime import timedelta, datetime
from shapely.geometry import Point


def get_sample(filepath: str = "./data/input/sds.csv", n: int = 1000):
    """Input filepath, sample size and return sample."""
    data = pd.read_csv(filepath)
    splitext = os.path.splitext(filepath)
    sample = data.sample(n)
    sample.to_csv(f"{splitext[0]}_sample{splitext[1]}", index=False)


def optimize_floats(df: pd.DataFrame) -> pd.DataFrame:
    """Input dataframe and return floats-optimized dataframe """
    floats = df.select_dtypes(include=['float64']).columns.tolist()
    df[floats] = df[floats].apply(pd.to_numeric, downcast='float')
    return df


def optimize_ints(df: pd.DataFrame) -> pd.DataFrame:
    """Input dataframe and return int-optimized dataframe """
    ints = df.select_dtypes(include=['int64']).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast='integer')
    return df


def optimize_objects(df: pd.DataFrame, ignore_features: List[str]) -> pd.DataFrame:
    """Input dataframe and return object-optimized dataframe """
    for col in df.select_dtypes(include=['object']):
        if col not in ignore_features:
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if float(num_unique_values) / num_total_values < 0.5:
                df[col] = df[col].astype('category')
    return df


def optimize(df: pd.DataFrame, ignore_features: List[str] = []):
    """Input dataframe and return fully optimized dataframe"""
    return optimize_floats(optimize_ints(optimize_objects(df, ignore_features)))


def tokenize(string_of_list):
  return string_of_list[1:-1].split(', ')


def str2flt(string_of_list):
    try:
        return [float(x) for x in string_of_list[1:-1].split(', ')]
    except:
        return 'NotConverted'


def create_tokenized_tsdf(df):
    tqdm.pandas()
    df['dt'] = df['dt'].progress_apply(str2flt)
    df['dist'] = df['dist'].progress_apply(str2flt)
    return df


def unnesting(df, explode):
    idx = df.index.repeat(df[explode[0]].str.len())
    df1 = pd.concat([
        pd.DataFrame({x: np.concatenate(df[x].values)}) for x in explode], axis=1)
    df1.index = idx

    return df1.join(df.drop(explode, 1), how='left')


def partial2date(number, reference_year=1984):
    tqdm.pandas()
    year = reference_year + int(number)
    d = timedelta(days=(reference_year + number - year)*365)
    day_one = datetime(year, 1, 1)
    date = d + day_one
    return date

def merge2point(lon, lat):
    return Point(lon, lat)


def add_geometry(df):
  df['geometry'] = list(map(merge2point, df['Intersect_lon'], df['Intersect_lat']))
  return df



if __name__ == "__main__":
    pass