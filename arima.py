# IMPORTANT NOTICE
# In order to run this script, package 'statsmodels 0.8.0' is needed. However, the latest release available in the
# repositories is 0.6.1. To install the needed version (tested on Mac OSX 10.11, Python 3.5.2 installed):
#       git clone git://github.com/statsmodels/statsmodels.git
#       cd statsmodels
#       python3 setup.py install
# (see https://github.com/statsmodels/statsmodels for further information and details)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from collections import Counter
import geopy as gp
import geopy.distance as gpd
import math

plt.style.use('ggplot')

PREDICTION_DAYS = 10  # number of days over which a prediction is to be performed
MAX_DISTANCE = 80  # maximum distance to consider for 'near' areas
FACTOR = 2  # correction factor

# import dataset, extract unique values for zones, areas, subareas, products
date_parse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
df = pd.read_csv('./data/dataset.csv', parse_dates=['Data'], index_col='Data', date_parser=date_parse)
zones = Counter(df['Zona']).keys()
areas = Counter(df['Area']).keys()
subareas = Counter(df['Sottoarea']).keys()
products = Counter(df['Categoria_prodotto']).keys()
# TODO Sort zones, areas, subareas, products (for presentation purposes)
# this doesn't work properly, because Zona_1 < Zona_10 < Zona_100 < Zona_11
subareas = sorted(subareas)
products = sorted(products)


def clean_result(x):
    if math.isnan(x):
        return x
    elif x < 0:
        return 0
    return round(x)


def mean_absolute_percentage_error(y_true, y_pred):
    mean_actual = np.mean(y_true)
    if mean_actual == 0:
        # TODO check whether it is appropriate
        return np.nan
    return np.mean(np.abs((y_true - y_pred) / mean_actual)) * 100


# model generation
output = {}  # dictionary
error_sarimax = pd.DataFrame(np.nan, index=subareas, columns=products)  # DataFrame
for subarea in subareas:
    print("\n" + ('-' * 80))
    print("Starting model generation: " + subarea)
    output[subarea] = {}

    for product in products:
        print('\t' + product)
        output[subarea][product] = {}

        # data preparation
        current = df.copy()
        current = current[current.Sottoarea == subarea]
        current = current[current.Categoria_prodotto == product]
        current_size = current.shape[0]
        if current_size == 0:
            continue

        # learning phase: seasonal ARIMA model
        mod = sm.tsa.statespace.SARIMAX(current['Vendite'].ix[:-PREDICTION_DAYS], trend='n', order=(0, 1, 0),
                                        seasonal_order=(1, 1, 1, 7))
        results = mod.fit(disp=0)
        # print(results.summary())
        current['forecast'] = results.predict(start=(current_size - PREDICTION_DAYS), end=current_size, dynamic=True)

        # result elaboration
        current['forecast'] = [clean_result(x) for x in current['forecast']]

        # error evaluation
        y_true = current['Vendite'].ix[-PREDICTION_DAYS:]
        y_pred = current['forecast'].ix[-PREDICTION_DAYS:]
        current_error = mean_absolute_percentage_error(y_true, y_pred)

        # saving data
        output[subarea][product] = current
        error_sarimax[product][subarea] = current_error

# import GPS coordinates and compute distance matrix
gps = pd.read_csv('./data/gps.csv', index_col='Sottoarea', dtype={'LATITUDINE': float, 'LONGITUDINE': float})
dm = pd.DataFrame(0.0, index=subareas, columns=subareas)
for sa1 in subareas:
    for sa2 in subareas:
        if dm[sa1][sa2] == 0 and sa1 != sa2:
            coords_sa1 = gps.ix[sa1]
            coords_sa2 = gps.ix[sa2]
            pt1 = gp.Point(coords_sa1[0], coords_sa1[1])
            pt2 = gp.Point(coords_sa2[0], coords_sa2[1])
            dm[sa1][sa2] = round(gpd.vincenty(pt1, pt2).kilometers)
            dm[sa2][sa1] = dm[sa1][sa2]

# model correction, according to distance
error_locality = pd.DataFrame(np.nan, index=subareas, columns=products)
for subarea in subareas:
    print("\n" + ('-' * 80))
    print("Starting model correction with locality: " + subarea)

    # compute locality correction, for each product
    for product in products:
        current = output[subarea][product]

        # find areas within MAX_DISTANCE distance from the current one
        near = []
        for sa in subareas:
            dist = dm[subarea][sa]
            if dist <= MAX_DISTANCE and dist != 0 and sa != subarea:
                near.append(sa)
        if len(near) == 0:
            # error_locality must take into account also the errors corresponding to subareas with no other area nearby
            error_locality[product][subarea] = error_sarimax[product][subarea]
            continue

        # compute the correction
        weighted_sum = 0
        weights = 0
        for sa in near:
            dist = dm[subarea][sa]
            # the nearest the area, the greatest weight in the weighted average
            weighted_sum += (current['forecast'] - output[sa][product]['forecast']) * (MAX_DISTANCE - dist)
            weights += (MAX_DISTANCE - dist)
        weighted_average = (weighted_sum / weights)
        current['forecast_locality'] = current['forecast'] + (weighted_average / FACTOR)
        current['forecast_locality'] = [clean_result(x) for x in current['forecast_locality']]

        # evaluate error
        y_true = current['Vendite'].ix[-PREDICTION_DAYS:]
        y_pred = current['forecast_locality'].ix[-PREDICTION_DAYS:]
        current_error = mean_absolute_percentage_error(y_true, y_pred)
        error_locality[product][subarea] = current_error

print("\n\n" + ('-' * 80))
print("Done\n")

# final results: average MAPE
average_MAPE_original_pr1 = np.nanmean(error_sarimax['Prodotto_1'])
average_MAPE_original_pr2 = np.nanmean(error_sarimax['Prodotto_2'])
average_MAPE_corrected_pr1 = np.nanmean(error_locality['Prodotto_1'])
average_MAPE_corrected_pr2 = np.nanmean(error_locality['Prodotto_2'])

print("Average MAPE on SARIMAX prediction: ")
print("\t" + "product 1: " + str(average_MAPE_original_pr1))
print("\t" + "product 2: " + str(average_MAPE_original_pr2))

print("Average MAPE on corrected prediction: ")
print("\t" + "product 1: " + str(average_MAPE_corrected_pr1))
print("\t" + "product 2: " + str(average_MAPE_corrected_pr2))
