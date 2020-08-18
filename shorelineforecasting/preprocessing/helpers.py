import os
import pandas as pd
from typing import List
from tqdm.auto import tqdm
import numpy as np
from datetime import timedelta, datetime
from shapely.geometry import Point
import geopandas as gpd

def get_sample(filepath: str = "./data/input/sds.csv", n: int = 1000):
    """
    Get sample of satellite-derived shoreline (SDS) positions. Input filepath of
    data in csv-format. Saves SDS-positions; metadata; and, outliers dataframe in
    pkl-format to data directory. Returns none.

    :param filepath: string
        Filepath in string of satellite-derived shoreline positions in csv format.
    :param n: integer
        Sample size.
    :return: None
    """

    data = pd.read_csv(filepath)
    splitext = os.path.splitext(filepath)
    sample = data.sample(n)
    sample.to_csv(f"{splitext[0]}_sample{splitext[1]}", index=False)

    transects = sample['transect_id'].unique()
    for i in ["sds_compressed.pkl", "tsdf.pkl", "df_outliers.pkl"]:
        data = pd.read_pickle(f"./data/input/{i}")
        selection = data.loc[data['transect_id'].isin(transects)]
        selection = selection.reset_index(drop=True)
        splitext = os.path.splitext(f"./data/input/{i}")
        selection.to_pickle(f"{splitext[0]}_sample{splitext[1]}")
        print(f"Saved sample from {i} as {splitext[0]}_sample{splitext[1]}")


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
    """Input string of list and return tokenized list."""
    return string_of_list[1:-1].split(', ')


def str2flt(string_of_list):
    """Input string of lists with floats and return list of floats."""
    try:
        return [float(x) for x in string_of_list[1:-1].split(', ')]
    except ValueError as e:
        print(f"ValueError: {e}")
        return "NotConverted"


def create_tokenized_tsdf(df):
    """Input df with dates, sds in a string and return as tokenized floats."""
    tqdm.pandas()
    df['dt'] = df['dt'].progress_apply(str2flt)
    df['dist'] = df['dist'].progress_apply(str2flt)
    return df


def unnesting(df, explode):
    """Input df with nested dates, sds and return unnested exploded df."""
    idx = df.index.repeat(df[explode[0]].str.len())
    df1 = pd.concat([
        pd.DataFrame({x: np.concatenate(df[x].values)}) for x in explode], axis=1)
    df1.index = idx

    return df1.join(df.drop(explode, 1), how='left')


def partial2date(number, reference_year=1984):
    """Input partial date, reference year and return datetime object."""
    tqdm.pandas()
    year = reference_year + int(number)
    d = timedelta(days=(reference_year + number - year)*365)
    day_one = datetime(year, 1, 1)
    date = d + day_one
    return date

def merge2point(lon, lat):
    """Input longitude, latitude and return GeoPandas Point."""
    return Point(lon, lat)


def add_geometry(df):
    """Input df with longitude, latiude and return df with GeoPandas geometries."""
    df['geometry'] = list(map(merge2point, df['Intersect_lon'], df['Intersect_lat']))
    return df


def df2gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Input pd.DataFrame and return gpd.GeoDataFrame."""
    crs = {"init": "epsg:4326"}
    return gpd.GeoDataFrame(df, crs=crs, geometry=df['geometry'])


def drop_empty_geometries(df: pd.DataFrame) -> pd.DataFrame:
    """Input pd.DataFrame and return pd.DataFrame without empty geometries."""
    gdf = df2gdf(df)
    idx = gdf.loc[gdf['geometry'].is_empty == False]['transect_id'].to_list()
    df = df.loc[df['transect_id'].isin(idx)]
    return df


def drop_by_index(s: pd.Series, outlier_dict: dict) -> pd.Series:
    """Input series, dictionary with drop-indices and return cleaned series."""
    idx = outlier_dict[s.name]
    mask = np.ones(len(s), dtype=bool)
    mask[idx] = False
    return s[mask]


if __name__ == "__main__":
    pass
