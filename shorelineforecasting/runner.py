from preprocessing.helpers import get_sample, optimize, create_tokenized_tsdf, drop_non_tokenizable, unnesting, partial2date, partials2dates, add_geometry, drop_empty_geometries, split_metadata_tsdf, format_tsdf
from utils.configs import get_yaml_configs
from utils.logger import get_logger, get_tsdf_stats_metadata
from processing.filter import get_metadata_filter, filter_tsdf
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
    tsdf = format_tsdf(tsdf)
    metadata = add_geometry(metadata)
    metadata = drop_empty_geometries(metadata)
    return tsdf, metadata


if __name__ == "__main__":

    # read configurations and initialize logger
    configs = get_yaml_configs()
    logger = get_logger(configs)
    logger.critical(f"Configs: {configs.items()}")

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

    metadata_filter = get_metadata_filter(metadata, tsdf, configs)
    print(metadata_filter)
    filtered_tsdf = filter_tsdf(tsdf, configs, outliers, metadata_filter)
    print(filtered_tsdf)







