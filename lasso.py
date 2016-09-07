import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import geopy as gp
import geopy.distance as gpd
import math

plt.style.use('ggplot')

PREDICTION_DAYS = 10  # number of days over which a prediction is to be performed
MAX_DISTANCE = 75  # maximum distance to consider for 'near' areas

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


# return a clean value (0 if the prediction is negative, the rounded value if it is positive)
def clean_value(x):
    if math.isnan(x):
        return x
    elif x < 0:
        return 0
    return round(x)


# compute the mean absolute percentage error (MAPE), corrected to avoid null denominators
def mean_absolute_percentage_error(y_true, y_pred):
    mean_actual = np.mean(y_true)
    if mean_actual == 0:
        # TODO check whether the following is appropriate
        return np.nan
    return np.mean(np.abs((y_true - y_pred) / mean_actual)) * 100


# model generation
output = {}  # dictionary
error = pd.DataFrame(np.nan, index=subareas, columns=products)  # DataFrame
for subarea in subareas:
    print("\n" + ('-' * 80))
    print("Starting model generation: " + subarea)
    output[subarea] = {}

    # find areas within MAX_DISTANCE distance from the current one
    near = []
    for sa in subareas:
        dist = dm[subarea][sa]
        if dist <= MAX_DISTANCE and sa != subarea and dist != 0:
            near.append(sa)

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

        # feature creation
        current['year'] = current.index.year
        current['month'] = current.index.month
        current['day'] = current.index.day
        current['day_of_week'] = current.index.weekday + 1
        current['num'] = np.arange(0, current_size).transpose() + 1

        # append sales volumes of near areas
        for sa in near:
            near_sa = df.copy()
            near_sa = near_sa[near_sa.Sottoarea == sa]
            near_sa = near_sa[near_sa.Categoria_prodotto == product]
            current[str(sa)] = near_sa[['Vendite']]

        # learning phase: Lasso method
        X_train = (current.loc[:, 'year':]).ix[:-PREDICTION_DAYS]
        X_test = (current.loc[:, 'year':]).ix[-PREDICTION_DAYS:]
        y_train = current.Vendite.ix[:-PREDICTION_DAYS]
        y_test = current.Vendite.ix[-PREDICTION_DAYS:]
        from sklearn import linear_model

        model = linear_model.Lasso(alpha=0.1)
        model.fit(X_train, y_train)
        result = model.predict(X_test).transpose()
        # print(results.summary())

        # learning phase: regression trees (discarded as non-deterministic and perceptively worse)
        # from sklearn.tree import ExtraTreeRegressor
        # clf = ExtraTreeRegressor()
        # clf.fit(X_train, y_train)
        # result = clf.predict(X_test).transpose()

        # result elaboration
        f = np.empty((current_size - PREDICTION_DAYS, 1)) * np.nan
        f = np.append(f, result)
        current['forecast'] = f.copy()
        current['forecast'] = [clean_value(x) for x in current['forecast']]

        # error evaluation
        y_true = current['Vendite'].ix[-PREDICTION_DAYS:]
        y_pred = current['forecast'].ix[-PREDICTION_DAYS:]
        current_error = mean_absolute_percentage_error(y_true, y_pred)

        # save data
        output[subarea][product] = current
        error[product][subarea] = current_error

print("\n\n" + ('-' * 80))
print("Done\n")

average_sarimax_pr1 = np.nanmean(error['Prodotto_1'])
average_sarimax_pr2 = np.nanmean(error['Prodotto_2'])

print("Average MAPE on prediction: ")
print("\t" + "product 1: " + str(average_sarimax_pr1))
print("\t" + "product 2: " + str(average_sarimax_pr2))
