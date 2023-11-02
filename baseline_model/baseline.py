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
    match val:
        case 'monday':
            return 1.0
        case 'tuesday':
            return 2.0
        case 'wednesday':
            return 3.0
        case 'thursday':
            return 4.0
        case 'friday':
            return 5.0
        case 'saturday':
            return 6.0
        case 'sunday':
            return 7.0
        case _:
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
    
    csvFilePath = r'anime_data.csv'
    jsonFilePath = r'anime_data.json'
    
    # make_json(csvFilePath, jsonFilePath)

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
                               string_to_float(anime["average_episode_duration"])
                            ]
        individualAnimePopularity = string_to_float(anime["popularity"])
        allAnimeData.append(individualAnimeData)
        allAnimePopularity.append(individualAnimePopularity)
    
    allAnimeDataArray=np.array([np.array(xi) for xi in allAnimeData])
    allAnimeDataPopularityArray=np.array([np.array(xi) for xi in allAnimePopularity])

    x_train, x_test, y_train, y_test = train_test_split(allAnimeDataArray, allAnimeDataPopularityArray, test_size=0.2, random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

    scaler = StandardScaler()
    scaler.fit_transform(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)

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
