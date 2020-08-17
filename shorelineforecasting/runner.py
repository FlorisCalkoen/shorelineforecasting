from preprocessing.helpers import get_sample, optimize, create_tokenized_tsdf, unnesting, partial2date, add_geometry
import pandas as pd


if __name__ == "__main__":
    # get_sample()

    data = pd.read_csv("./data/input/sds_sample.csv")
    print(data)
    example = optimize(data, ['dt', 'dt2', 'dist', 'dist2', 'outliers_1', 'outliers_2'])
    example = create_tokenized_tsdf(example)
    print(f"Transects original df: {len(example['transect_id'].unique())}")
    example = example[(example['dt'] != 'NotConverted') & (example['dist'] != 'NotConverted')]
    print(f"Transects with observations: {len(example['transect_id'].unique())}")
    example = example.loc[example['flag_sandy'] == True]  # keep only sandy transects
    print(f"Transects flag sandy df: {len(example['transect_id'].unique())}")
    example = example.loc[example['changerate_unc'] < 0.5]  # keep transects with relatively constant trends
    print(f"Transects changerate_unc < 0.5 df : {len(example['transect_id'].unique())}")
    # print(example)
    print(example['dt'].iloc[0][0])
    example = unnesting(example, ['dt', 'dist'])
    print(example.shape)
    example['ts'] = example['dt'].progress_apply(partial2date)
    example = add_geometry(example)

    example = example.set_index(['transect_id', 'ts'])
    print(example.head())

