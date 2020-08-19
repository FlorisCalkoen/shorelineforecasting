import numpy as np
import pandas as pd
from datetime import datetime
import logging
import csv
import io

from preprocessing.helpers import partial2date


class CsvFormatter(logging.Formatter):
    """Input logging.Formatter object and return a CSV-formatted logging object."""
    def __init__(self):
        super().__init__()
        self.output = io.StringIO()
        self.writer = csv.writer(self.output, quoting=csv.QUOTE_ALL)

    def format(self, record):
        self.writer.writerow([record.levelname, record.msg])
        data = self.output.getvalue()
        self.output.truncate(0)
        self.output.seek(0)
        return data.strip()


def get_logger(configs):
    """
    Input configuration object, return logger object and write to output file.

    When no valid logging level is provided, level is set numerically set to WARNING.
    Output is formatted using CsvFormatter and written to text file in csv-format.

    :param configs:
        Configuration object which is created by parsing the configuration file.
    :return:class 'logging.Logger
        Logger object writing output to file.
    """
    level = getattr(logging, configs['run']['logLevel'].upper(), 30)
    timestr = datetime.now().strftime("%Y%m%d%H%M%S")
    fpath = f"./data/output/log_{timestr}.csv"
    logging.basicConfig(filename=fpath, level=level, )
    logger = logging.getLogger(__name__)
    logging.root.handlers[0].setFormatter(CsvFormatter())

    return logger


def get_stats_tsdf(tsdf):
    """Input time-series DataFrame and return several statistics time-series DataFrame."""
    tsdf = tsdf.groupby(tsdf.index.get_level_values('ts').year).mean()
    nans = tsdf.isna().values.sum()
    n_obs = tsdf.count().values.sum()
    n_transects = len(tsdf.columns)
    timespan = np.mean(tsdf.apply(lambda x: x.count()))
    p_nans = nans/n_obs
    return f"{nans}, {n_obs}, {n_transects}, {timespan}, {p_nans}"


def get_tsdf_stats_metadata(metadata, tsdf):
    """ Input metadata DataFrame and return time-series statistics of transects included in metadata."""
    transects = metadata['transect_id'].unique()
    tsdf = tsdf.loc[tsdf['transect_id'].isin(transects)]
    tsdf = tsdf.pivot(index='dt', columns='transect_id', values='dist')
    tsdf['ts'] = tsdf.index.map(partial2date)
    tsdf = tsdf.set_index(['ts', tsdf.index])
    return get_stats_tsdf(tsdf)





