# Traditional vs. Machine-Learning Methods for Forecasting Sandy Shoreline Evolution Using Historic Satellite-Derived Shorelines

Floris Calkoen, 2020, MSc Thesis Project Information Studies (University of Amsterdam) at
Deltares (Delft), which was continued september 2020 - december 2020 to write an article about
the results.  

Contributors: Floris Calkoen, Arjen Luijendijk, Cristian Rodriguez Rivero, Etienne Kras, and Fedor
Baart.  


Note, the code has not been maintained since 2020. Most likely there will be some dependency
issues. Also the data which was used has been updated since then, so the preprocessing examples are problably also broken.

## Configurations

```shell script
cd shorelineforecasting
python3.7 -m venv venv 
source venv/bin/activate 
(venv) pip install -r requirements.txt 
```

## Recommendations

Explore and understand code and date with the example notebook (example.ipynb). If you just want to run everything
at once, run ```python runner.py``` from the shorelineforecasting directory. Alternatively, run directly from root using
```sh scripts/run.sh```

## Citation

```
@Article{rs13050934,
AUTHOR = {Calkoen, Floris and Luijendijk, Arjen and Rivero, Cristian Rodriguez and Kras, Etienne and Baart, Fedor},
TITLE = {Traditional vs. Machine-Learning Methods for Forecasting Sandy Shoreline Evolution Using Historic Satellite-Derived Shorelines},
JOURNAL = {Remote Sensing},
VOLUME = {13},
YEAR = {2021},
NUMBER = {5},
ARTICLE-NUMBER = {934},
URL = {https://www.mdpi.com/2072-4292/13/5/934},
ISSN = {2072-4292},
ABSTRACT = {Forecasting shoreline evolution for sandy coasts is important for sustainable coastal management, given the present-day increasing anthropogenic pressures and a changing future climate. Here, we evaluate eight different time-series forecasting methods for predicting future shorelines derived from historic satellite-derived shorelines. Analyzing more than 37,000 transects around the globe, we find that traditional forecast methods altogether with some of the evaluated probabilistic Machine Learning (ML) time-series forecast algorithms, outperform Ordinary Least Squares (OLS) predictions for the majority of the sites. When forecasting seven years ahead, we find that these algorithms generate better predictions than OLS for 54% of the transect sites, producing forecasts with, on average, 29% smaller Mean Squared Error (MSE). Importantly, this advantage is shown to exist over all considered forecast horizons, i.e., from 1 up to 11 years. Although the ML algorithms do not produce significantly better predictions than traditional time-series forecast methods, some proved to be significantly more efficient in terms of computation time. We further provide insight in how these ML algorithms can be improved so that they can be expected to outperform not only OLS regression, but also the traditional time-series forecast methods. These forecasting algorithms can be used by coastal engineers, managers, and scientists to generate future shoreline prediction at a global level and derive conclusions thereof.},
DOI = {10.3390/rs13050934}
}
