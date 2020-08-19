import logging
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)




def get_stats_tsdf(df):
    try:
        df = df.groupby(df.index.get_level_values('ts').year).mean()
    except:
        pass
    nans = df.isna().values.sum()
    observations = df.count().values.sum()
    p_nans = nans/observations
    n_transects = len(df.columns)
    timespan = np.mean(df.apply(lambda x: x.count()))

    return nans, observations, p_nans, n_transects, timespan


    # nans = df.isna().values.sum()
    # total = df.count().values.sum()
    # temp = pd.DataFrame()
    # temp['operation'] = [label]
    # temp['transects'] = [len(df.columns)]
    # temp['timespan'] = [np.mean(df.apply(lambda x: x.count()))]
    # temp['nans'] = [nans]
    # temp['p_nans'] = [nans / total]
    # temp['idx'] = [self.idx]
    # temp = temp.set_index('idx')
    #
    # self.res = pd.concat([self.res, temp])
    # self.idx += 1


def get_metadata_filter(df, configs):
    metadata_configs = configs['metadata_filter']
    log_stats = configs['default']['log_stats']

    print(f"Transects original df: {len(df['transect_id'].unique())}")
    if log_stats is True:
        logger.get_all_stats_metadata(df, data, label='raw_data')

    # loop through configs and filter frame
    for k, v in metadata_configs.items():
        if k == 'flag_sandy' and v == True:
            df = df.loc[df['flag_sandy'] == True]
        if k == 'changerate_unc' and v is not None:
            df = df.loc[df['changerate_unc'] < v]
        elif v is True:
            df = df.loc[df[k] != 0]
        if log_stats is True:
            logger.get_all_stats_metadata(df, data, label=k)

    print(f"Transects filtered df: {len(df['transect_id'].unique())}")
    return df['transect_id'].unique()