import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.logger import get_tsdf_stats_metadata, get_stats_tsdf
from visualization.plots import plot_nans_per_year, plot_nans_per_transect


logger = logging.getLogger(__name__)


def get_metadata_filter(metadata, tsdf, configs):
    """
    Input metadata DataFrame and return list of transects which should be included
    in the processing according to the settings specified in the configuration file.
    """
    print(f"Transects original df: {len(metadata['transect_id'].unique())}")
    for k, v in configs['selection']['metadata'].items():
        if k == 'flag_sandy' and v == True:
            metadata = metadata.loc[metadata['flag_sandy'] == True]
            logger.debug(f"{k}={v}:, {get_tsdf_stats_metadata(metadata, tsdf)}")
        if k == 'changerate_unc' and v is not None:
            metadata = metadata.loc[metadata['changerate_unc'] < float(v)]
            logger.debug(f"{k}={v}:, {get_tsdf_stats_metadata(metadata, tsdf)}")
        elif v is True:
            metadata = metadata.loc[metadata[k] != 0]
            logger.debug(f"{k}={v}:, {get_tsdf_stats_metadata(metadata, tsdf)}")
    print(f"Transects filtered df: {len(metadata['transect_id'].unique())}")
    return metadata['transect_id'].unique()


def drop_indices(s, outlier_dict):
    """Input series, outlier-dictionary and return series without outliers."""
    idx = outlier_dict[s.name]
    mask = np.ones(len(s), dtype=bool)
    mask[idx] = False
    return s[mask]


def filter_tsdf_by_metadata(tsdf, configs, outliers, transect_filter=None):
    """

    :param tsdf:
        Time-series DataFrame consisting dates (partial), shoreline positions (dist), transect_id and timestamp.
    :param configs:
        Dictionry parsed from YAML-configuraiton file.
    :param transect_filter:
        Optionally also input a selection of transects which should be kept.
    :param outliers: pd.DataFrame
        Outliers in pandas dataframe object.
    :return: tsdf
        Filtered time-series dataframe.
    """
    # set transect_id as index to optimize processing speed when filtering.
    tsdf = tsdf.set_index(['transect_id'])

    # filter according to metadata filter
    if transect_filter.any():
        tsdf = tsdf[tsdf.index.isin(transect_filter)]

    # optionally take sample
    if configs['run']['sample'] is not None:
        keep = np.random.choice(tsdf.index.unique(), size=configs['run']['sample'])
        tsdf = tsdf[tsdf.index.isin(keep)]

    # # handle outliers
    tsdf = tsdf.reset_index()
    outliers1 = outliers.set_index('transect_id')['outliers_1_as_int'].to_dict()
    outliers2 = outliers.set_index('transect_id')['outliers_2_as_int'].to_dict()
    if configs['selection']['stats']['drop_outliers_1'] is True:
        print('Dropping outliers 1...')
        tsdf = tsdf.groupby('transect_id').progress_apply(lambda x: drop_indices(x, outliers1))
        tsdf = tsdf.droplevel('transect_id')
    if configs['selection']['stats']['drop_outliers_2'] is True:
        print('Dropping outliers 2...')
        tsdf = tsdf.groupby('transect_id').progress_apply(lambda x: drop_indices(x, outliers2))
        tsdf = tsdf.droplevel('transect_id')

    return tsdf


def filter_tsdf_by_nans(tsdf: pd.DataFrame, configs):
    # # parse configs
    nans_per_yr_lt = configs['selection']['stats']['nans_per_year_lt']
    nans_per_transect_lt = configs['selection']['stats']['nans_per_transect_lt']
    # show_plot = configs['run']['show_plot']
    # log_stats = configs['default']['log_stats']

    #  yearly averages
    yearly = tsdf.groupby(tsdf.index.get_level_values('ts').year).mean()
    filtered1 = yearly[yearly.isnull().mean(axis=1) < nans_per_yr_lt]
    filtered2 = filtered1.loc[:, filtered1.isnull().mean() < nans_per_transect_lt]

    # # log stats
    # labels = [f"Before NaN filter",
    #           f"Filter 1: NaN's per year > {nans_per_yr_lt * 100} %",
    #           f"Filter 1: NaN's per year > {nans_per_transect_lt * 100} %"]
    # for i, j in zip([yearly, filtered1, filtered2], labels):
    #     logger.debug(f"{j}: {get_stats_tsdf(i)}")

    keep_years = filtered1.index
    keep_transects = filtered2.columns
    keep_rows = tsdf.index.get_level_values('ts').year.isin(keep_years)

    if configs['run']['show_plot'] is True:
        plot_nans_per_year(yearly, filtered1, filtered2, nans_per_yr_lt, nans_per_transect_lt)
        plot_nans_per_transect(yearly, filtered1, filtered2, nans_per_yr_lt, nans_per_transect_lt)

    return tsdf[keep_rows][keep_transects], keep_years, keep_transects


