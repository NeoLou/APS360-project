import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import datetime
import time

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split

import csv
import json

import os

np.random.seed(0)

def make_json(csvFilePath, jsonFilePath):
    data = {}
    with open(csvFilePath, encoding='utf-8') as csvf:
        csvReader = csv.DictReader(csvf)
        for rows in csvReader:
            key = rows['id']
            data[key] = rows
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
        jsonf.write(json.dumps(data, indent=4))

def string_to_float(val):
    if val != '':
        return float(val)
    else:
        return 0

def day_to_float(val):
    if val == 'monday':
        return 1.0
    elif val == 'tuesday':
        return 2.0
    elif val == 'wednesday':
        return 3.0
    elif val == 'thursday':
        return 4.0
    elif val == 'friday':
        return 5.0
    elif val == 'saturday':
        return 6.0
    elif val == 'sunday':
        return 7.0
    else:
        return 0
        
def time_to_seconds(val):
    if val != "":
        x = time.strptime(val,'%H:%M')
        return datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min).total_seconds()
    else:
        return 0

if __name__ == "__main__":

    print("Script Running: ")
    print()
    
    os.chdir('C:\\Users\\Luke Yang\\Documents\\100_Luke\\100_School\\130_University_of_Toronto\\2023_2024\\APS360\\Anime-popularity-predictor\\baseline_model')
    
    csvFilePath = r'animes_data_max_rank=5000.csv'
    jsonFilePath = r'animes_data_max_rank=5000.json'
    
    make_json(csvFilePath, jsonFilePath)

    jsonFile = open(jsonFilePath, 'r')
    jsonData = json.load(jsonFile)

    allAnimeData = []
    allAnimePopularity = []
    for id in jsonData:
        anime = jsonData[id]
        individualAnimeData = [string_to_float(anime["genres_0_id"]),
                               string_to_float(anime["genres_1_id"]),
                               string_to_float(anime["genres_2_id"]),
                               string_to_float(anime["genres_3_id"]),
                               string_to_float(anime["genres_4_id"]),
                               string_to_float(anime["genres_5_id"]),
                               string_to_float(anime["genres_6_id"]),
                               string_to_float(anime["genres_7_id"]),
                               string_to_float(anime["genres_8_id"]),
                               string_to_float(anime["genres_9_id"]),
                               string_to_float(anime["genres_10_id"]),
                               string_to_float(anime["num_episodes"]),
                               day_to_float(anime["broadcast_day_of_the_week"]),
                               time_to_seconds(anime["broadcast_start_time"]),
                               string_to_float(anime["average_episode_duration"]),
                               string_to_float(anime["studios_0_id"])
                            ]
        individualAnimePopularity = string_to_float(anime["popularity"])
        allAnimeData.append(individualAnimeData)
        allAnimePopularity.append(individualAnimePopularity)
    
    for i in range(len(allAnimePopularity)):
        allAnimePopularity[i] = (allAnimePopularity[i] - min(allAnimePopularity)) / (max(allAnimePopularity) - min(allAnimePopularity))
    
    allAnimeDataArray=np.array([np.array(xi) for xi in allAnimeData])
    allAnimeDataPopularityArray=np.array([np.array(xi) for xi in allAnimePopularity])

    x_train, x_test, y_train, y_test = train_test_split(allAnimeDataArray, allAnimeDataPopularityArray, test_size=0.2, random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

    scaler = StandardScaler()
    scaler.fit_transform(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    rmse_val = []
    listN = [*range(1, 100)]
    for n in listN:
        knn = KNeighborsRegressor(n_neighbors=n)
        knn.fit(x_train, y_train)

        y_pred = knn.predict(x_val)
        error = np.sqrt(mean_squared_error(y_val, y_pred))
        rmse_val.append(error)
        
    curve = pd.DataFrame(rmse_val)
    curve.plot()
    plt.xlabel("N")
    plt.ylabel("RMSE")
    plt.title("RMSE vs N")
    plt.show()

    index_min = np.argmin(rmse_val)
    best_n = listN[index_min]
    print("Best N: " + str(best_n))
    
    y_pred = knn.predict(x_test)
    error = np.sqrt(mean_squared_error(y_test, y_pred))
    print("Test RMSE: " + str(error))

import matplotlib.pyplot as plt
plt.plot(y_test, y_pred, 'bo')
plt.plot(y_test, y_test, 'ro')
plt.xlim(0.00, 1.00)
plt.ylim(-0.00, 1.00)
plt.title("test_predictions for popularity ranks")
plt.xlabel("True popularity rank (normalized)")
plt.ylabel("Predicted popularity rank (normalized)")
plt.show()
