import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_nans_per_year(yearly, filtered1, filtered2, nans_per_yr_lt, nans_per_transect_lt):
    """"""
    # function to calculate nan proportion
    f = lambda x: x.isnull().mean(axis=1) * 100

    # dataframe to save results
    res = pd.DataFrame(columns=['ts', 'type', 'p_nans'])
    res = res.set_index('ts')

    # calculate proportions
    label = 0
    for i in [yearly, filtered1, filtered2]:
        temp = pd.DataFrame(f(i), columns=['p_nans'])
        temp['type'] = label
        label += 1
        res = pd.concat([res, temp])

    # plot results
    fig, ax = plt.subplots(figsize=(16, 8))
    res = res.pivot(columns='type', values='p_nans')
    res.plot(kind="bar", stacked=False, ax=ax, rot=45, width=.7)

    # format graph
    ax.set_title("Filtering dataset by proportion of NaN's")
    ax.set_ylabel("Proportion NaN's per transect (%)")
    ax.set_xlabel("Time (yrs)")
    l1 = ax.legend([f'Original data',
                    f"Filter 1: NaN's per year > {nans_per_yr_lt * 100} ",
                    f"Filter 2: NaN's per transect > {nans_per_transect_lt * 100}"], title='Data selection')
    l2 = ax.legend(
        [f"Transects: {len(yearly.columns)}; years: {len(yearly.index)}; NaN's: {yearly.isna().values.sum()}",
         f"Transects: {len(filtered1.columns)}; years: {len(filtered1.index)}; NaN's: {filtered1.isna().values.sum()}",
         f"Transects: {len(filtered2.columns)}; years: {len(filtered2.index)}; NaN's: {filtered2.isna().values.sum()}"],
        loc="center right", title='Statistics')
    plt.gca().add_artist(l1)
    plt.gca().add_artist(l2)
    plt.show()


def plot_nans_per_transect(yearly, filtered1, filtered2, nans_per_yr_lt, nans_per_transect_lt):
    """"""
    # function to calculate observations per transect
    f = lambda x: np.count_nonzero(~np.isnan(x))

    # dataframe to save results
    res = pd.DataFrame(columns=['idx', 'type', 'p_n_values'])
    res = res.set_index('idx')

    # calculate proportions
    label = 0
    for i in [yearly, filtered1, filtered2]:
        temp = i.apply(f).value_counts(normalize=True)
        temp = pd.DataFrame(temp, columns=['p_n_values'])
        temp['type'] = label
        label += 1
        res = pd.concat([res, temp])

    # plot results
    fig, ax = plt.subplots(figsize=(16, 8))
    res = res.pivot(columns='type', values='p_n_values')
    res.plot(kind="bar", stacked=False, ax=ax, rot=45, width=.7)

    # format graph
    ax.set_title("Observations per transcents (proportionally) ")
    ax.set_ylabel("Number of transects (%)")
    ax.set_xlabel("Number of observations")
    ax.legend([f'Original data',
               f"Filter 1: NaN's per year < {nans_per_yr_lt * 100} ",
               f"Filter 2: NaN's per transect < {nans_per_transect_lt * 100}"], title='Data selection',
              loc="upper left")
    plt.show()


def plot_forecast(test, forecast, model_configs, transect_id=None):
    """

    :param test:
    :param forecast:
    :param model_configs:
    :param transect_id:
    :return:
    """
    if transect_id is None:
      transect_id = np.random.choice(forecast.columns)

    ts = test[transect_id]
    fcast = forecast[transect_id]

    fig, ax = plt.subplots()

    ax.plot(ts.index, ts.values, label=ts.name, marker='o')
    ax.plot(fcast.index, fcast.values, label="LSTM forecast", color='#ff7f0e')
    ax.axvline(x=1984 + model_configs['train_window'], c='r', linestyle='dashed', zorder=0)
    ax.axvspan(1984+model_configs['train_window'], 1984+len(ts)-1, alpha=0.3)

    ax.set(xlabel='Time (yrs)', ylabel='Relative shoreline position (m)')
    ax.legend()
    ax.set_axisbelow(True)
    plt.grid()
    plt.show()