from preprocessing.helpers import get_sample, optimize, create_tokenized_tsdf, drop_non_tokenizable, unnesting, partials2dates, add_geometry, drop_empty_geometries, split_metadata_tsdf
from utils.configs import get_yaml_configs
from utils.logger import get_logger
from processing.filter import get_stats_tsdf
import pandas as pd

def clean_data(data):
    """
    Input Pandas DataFrame with satellite-derived shoreline positions in Deltares ShorelineMonitor format
    and return unnested Pandas DataFrame with shoreline positions and Pandas DataFrame with metadata about
    the sites where shoreline positions were observed.

    First pd.DataFrame dtypes are optimized to reduce memory size. Satellite-derived shoreline-positions and
    dates are hold in the DataFrame as one string per string. These are tokenized. Sites with no observations
    are empty and hence cannot be tokenized; these are dropped. The DataFrame is split into a time-series
    DataFrame, containing transect_id, date, and distance; and a metadata DataFrame containing all other data
    except dates and distances. Shoreline positions in the time-series dataframe are unnested. The dates are
    transformed from partials to datetime objects. Finally, geometry objects are added to the metadata.

    :param data: pd.DataFrame
        Pandas DataFrame with satellite-derived shoreline positions in Deltares ShorelineMonitor format.
    :return:
        tsdf: pd.DataFrame
        Pandas DataFrame unnested satellite-derived shoreline positions.

        metadata: pd.DataFrame
        Pandas DataFrame with metadata about sites where shoreline positions were derived.
    """

    data = optimize(data, ignore_features=['dt', 'dist', 'outliers_1', 'outliers_2'])
    data = create_tokenized_tsdf(data)
    data = drop_non_tokenizable(data)
    metadata, tsdf = split_metadata_tsdf(data)
    tsdf = unnesting(tsdf, explode=['dt', 'dist'])
    tsdf['dt'] = partials2dates(tsdf['dt'])
    metadata = add_geometry(metadata)
    metadata = drop_empty_geometries(metadata)
    return tsdf, metadata


if __name__ == "__main__":

    # read configurations and initialize logger
    configs = get_yaml_configs()
    logger = get_logger(configs)


    # Start with csv (sample) data. Optionally start with readily available data.
    # get_sample() # get (fresh) sample data
    data = pd.read_csv("./data/input/sds_sample.csv")
    tsdf, metadata = clean_data(data)


    # # Optionally just start with reading in these files.
    # metadata = pd.read_pickle("./data/input/sds_compressed_sample.pkl")
    # tsdf = pd.read_pickle("./data/input/tsdf_sample.pkl")
    # tsdf['dt'] = partials2dates(tsdf['dt'])
    # metadata = add_geometry(metadata)
    # metadata = drop_empty_geometries(metadata)

    # Read in outliers data.
    outliers = pd.read_pickle("./data/input/df_outliers_sample.pkl")


    # # # basic pre-processing
    # # metadata = drop_empty_geometries(metadata)
    #
    #
    # # get_sample()
    # #
    # # data = pd.read_csv("./data/input/sds_sample.csv")
    # # print(data)
    # # example = optimize(data, ['dt', 'dt2', 'dist', 'dist2', 'outliers_1', 'outliers_2'])
    # # example = create_tokenized_tsdf(example)
    # # print(f"Transects original df: {len(example['transect_id'].unique())}")
    # # example = example[(example['dt'] != 'NotConverted') & (example['dist'] != 'NotConverted')]
    # # print(f"Transects with observations: {len(example['transect_id'].unique())}")
    # # example = example.loc[example['flag_sandy'] == True]  # keep only sandy transects
    # # print(f"Transects flag sandy df: {len(example['transect_id'].unique())}")
    # # example = example.loc[example['changerate_unc'] < 0.5]  # keep transects with relatively constant trends
    # # print(f"Transects changerate_unc < 0.5 df : {len(example['transect_id'].unique())}")
    # # print(example['dt'].iloc[0][0])
    # # example = unnesting(example, ['dt', 'dist'])
    # # print(example.shape)
    # # example['ts'] = example['dt'].progress_apply(partial2date)
    # # example = add_geometry(example)
    # #
    # # example = example.set_index(['transect_id', 'ts'])
    # # print(example.head())
    #

