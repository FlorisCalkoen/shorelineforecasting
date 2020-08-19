import logging
import pandas as pd
import numpy as np

from utils.logger import get_tsdf_stats_metadata, get_stats_tsdf

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


def filter_tsdf(tsdf, configs, outliers, transect_filter=None):
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
    # set transect_id as index to optimize processing speed when indexing.
    tsdf = tsdf.set_index(['transect_id'])

    # filter according to metadata filter
    if transect_filter.any():
        tsdf = tsdf[tsdf.index.isin(transect_filter)]
        tsdf = tsdf.reset_index()
        logger.debug(f"Transect filter: {get_stats_tsdf(tsdf)}")


    # optionally take sample
    if configs['run']['sample'] is not None:
        keep = np.random.choice(tsdf.index.unique(), size=configs['run']['sample'])
        tsdf = tsdf[tsdf.index.isin(keep)]

    # # handle outliers
    # tsdf = tsdf.reset_index()
    # outliers1 = outliers.set_index('transect_id')['outliers_1_as_int'].to_dict()
    # outliers2 = outliers.set_index('transect_id')['outliers_2_as_int'].to_dict()
    # if configs['default']['drop_outliers_1'] is True:
    #     print('Dropping outliers 1...')
    #     df = df.groupby('transect_id').progress_apply(lambda x: drop_indices(x, outliers1))
    #     df = df.droplevel('transect_id')
    # if configs['default']['drop_outliers_2'] is True:
    #     print('Dropping outliers 2...')
    #     df = df.groupby('transect_id').progress_apply(lambda x: drop_indices(x, outliers2))
    #     df = df.droplevel('transect_id')

