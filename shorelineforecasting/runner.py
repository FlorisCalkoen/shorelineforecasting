from preprocessing.helpers import get_sample, optimize, create_tokenized_tsdf, drop_non_tokenizable, unnesting, partial2date, partials2dates, add_geometry, drop_empty_geometries, split_metadata_tsdf, format_tsdf, pivot_tsdf, interpolate_nans, save_preprocessed_data, load_preprocessed_data
from utils.configs import get_yaml_configs
from utils.logger import get_logger, get_tsdf_stats_metadata
from processing.filter import get_metadata_filter, filter_tsdf_by_metadata, filter_tsdf_by_nans
import pandas as pd

def reshape_data(data):
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
    tsdf = format_tsdf(tsdf)
    metadata = add_geometry(metadata)
    metadata = drop_empty_geometries(metadata)
    return tsdf, metadata


def filter_data(tsdf, metadata, configs):
    """
    Input time-series DataFrame, metadata of the transects included in the time-series and
    configurations. The time-series data will be filtered according to settings specified
    in the YAML-configuration file. Metadata is also filtered according to the time-series
    which are left in the selection. The time-series, metadata and configuration settings
    are saved as a dictionary in pkl-format in the output directory.

    :param metadata:
    :param tsdf:
    :param configs:
    :return: None
    """
    metadata_filter = get_metadata_filter(metadata, tsdf, configs)
    tsdf = filter_tsdf_by_metadata(tsdf, configs, outliers, metadata_filter)
    tsdf = pivot_tsdf(tsdf)
    tsdf, _, _ = filter_tsdf_by_nans(tsdf, configs)
    tsdf = interpolate_nans(tsdf)
    save_preprocessed_data(tsdf, metadata, configs)


if __name__ == "__main__":

    # read configurations and initialize logger
    configs = get_yaml_configs()
    logger = get_logger(configs)
    logger.critical(f"Configs: {configs.items()}")

    # Start with csv (sample) data. Optionally skip "get_sample()" function and
    # start with readily available sample data. Also load the outliers dataframe.
    # get_sample() # get (fresh) sample data
    data = pd.read_csv("./data/input/sds_sample.csv")
    outliers = pd.read_pickle("./data/input/df_outliers_sample.pkl")

    # reshape and re-format data.
    # tsdf, metadata = reshape_data(data)

    # clean data and save preprocessed data in output directory.
    # filter_data(tsdf, metadata, configs)

    # load data
    configs, tsdf, metadata = load_preprocessed_data(filename="sample_1597922102.pkl")











