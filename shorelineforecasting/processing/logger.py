import numpy as np
import pandas as pd
from datetime import timedelta, datetime

def partial2date(number, reference_year=1984):
  year = reference_year + int(number)
  d = timedelta(days=(reference_year + number - year)*365)
  day_one = datetime(year, 1, 1)
  date = d + day_one
  return date

def date2partial(date, reference_year=1984):
    year = date.year
    tt = date.timetuple()
    d = tt.tm_yday / 365
    return (year - reference_year) + d

class DataFrameLogger(object):
    def __init__(self):

        self.res = pd.DataFrame(columns=['idx', 'operation', 'transects', 'nans', 'p_nans'])
        self.res = self.res.set_index('idx')
        self.idx = 0

    def get_stats_metadata(self, df, label):
        temp = pd.DataFrame()
        temp['operation'] = [label]
        temp['transects'] = [len(df['transect_id'].unique())]
        temp['timespan'] = [df['timespan'].mean()]
        temp['nans'] = [np.nan]
        temp['p_nans'] = [np.nan]
        temp['idx'] = [self.idx]
        temp = temp.set_index('idx')

        self.res = pd.concat([self.res, temp])
        self.idx += 1

    def get_all_stats_metadata(self, metadata, data, label):
        # prepare data input according to metadata filter
        transects = metadata['transect_id'].unique()
        data = data.loc[data['transect_id'].isin(transects)]
        data = data.pivot(index='dt', columns='transect_id', values='dist')
        data['ts'] = data.index.map(partial2date)
        data = data.set_index(['ts', data.index])

        # get stats
        self.get_stats_tsdf(data, label)

    def get_stats_tsdf(self, df, label):
        try:
            df = df.groupby(df.index.get_level_values('ts').year).mean()
        except:
            pass
        nans = nan = df.isna().values.sum()
        total = df.count().values.sum()
        temp = pd.DataFrame()
        temp['operation'] = [label]
        temp['transects'] = [len(df.columns)]
        temp['timespan'] = [np.mean(df.apply(lambda x: x.count()))]
        temp['nans'] = [nans]
        temp['p_nans'] = [nans / total]
        temp['idx'] = [self.idx]
        temp = temp.set_index('idx')

        self.res = pd.concat([self.res, temp])
        self.idx += 1

if __name__ == "__main__":
    print('yay')