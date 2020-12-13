import pandas as pd

# tf = pd.read_csv("/media/storage/data/shorelines/time-series-gluonts-prepared.csv")
# sites = pd.read_csv("/media/storage/data/shorelines/sites.csv")

def runner():
    # tf = pd.read_csv("gdrive/My Drive/data/shorelines/time-series-gluonts-prepared.csv")
    # metadata = pd.read_pickle("gdrive/My Drive/data/shorelines/sites-compressed.pkl")
    pass




if __name__ == "__main__":
    # print(tf.shape)
    #
    # # tf = pd.read_csv("/media/storage/data/shorelines/time-series-gluonts-prepared.csv")
    # # sites = pd.read_csv("/media/storage/data/shorelines/sites.csv")
    # # print(pd.__version__)

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt


    y = np.random.rand(100)
    x = range(len(y))

    z = []
    for i in x[:-1]:
        i = i+1
        tmp = y[i] - y[(i - 1)]
        z.append(tmp)
    print(len(z))

    fig, ax = plt.subplots(2, 1, figsize=(10, 12))
    ax[0].plot(x, y)
    ax[1].plot(range(len(z)), z)
    plt.show()


