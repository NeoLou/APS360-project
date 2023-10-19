# Note: Make sure you run this file in the folder you want the excel data file to be created!

import requests
import json
import requests_cache
import pandas as pd
from flatten_json import flatten
import time

# import certifi
# change certificate settings: add myanimelist certificate chain to certificate list
# print(certifi.where())

'''
Available API requests:

Get list of anime by search term
    resp = requests.get("https://api.myanimelist.net/v2/anime?q=one&limit=2", headers=headers)

Get anime details by anime id
    resp = requests.get("https://api.myanimelist.net/v2/anime/30230?fields=id,title,main_picture,alternative_titles,start_date,end_date,synopsis,mean,rank,popularity,num_list_users,num_scoring_users,nsfw,created_at,updated_at,media_type,status,genres,my_list_status,num_episodes,start_season,broadcast,source,average_episode_duration,rating,pictures,background,related_anime,related_manga,recommendations,studios,statistics", headers=headers)

Get top ranked anime by type
    resp = requests.get("https://api.myanimelist.net/v2/anime/ranking?ranking_type=all&limit=2", headers=headers)

Get anime details by season
    resp = requests.get("https://api.myanimelist.net/v2/anime/season/2023/summer?sort=anime_num_list_users&fields=id,title,start_date,end_date,mean,rank,popularity,num_list_users,num_scoring_users,nsfw,media_type,status,genres,my_list_status,num_episodes,start_season,broadcast,source,average_episode_duration,rating,background,related_anime,related_manga,recommendations,studios,statistics&limit=6", headers=headers)
'''

# Helper functions
# Function to print json
def json_print(json_obj):
    str = json.dumps(json_obj, sort_keys=True, indent=2)
    print(str)

# API authentication info (key, secret, header)
client_id = "6a3dc42a0c201194413f0f02733ae033"
#client_secret = "b24d9ab0be8b47325e5ff0ac8c29527553447a0e9e13e9aa7ea0b01cbcd30721"
headers = {
    'X-MAL-CLIENT-ID': client_id
}

# API query settings
base_url = "https://api.myanimelist.net/v2/anime/season/" # URL
sort_by_num_users = "?sort=anime_num_list_users" # Sort before getting data
limit = "&limit=40" # Set limit for num animes to get per season
nsfw = "&nsfw=true" # Include nsfw animes in query

# Selected fields
fields = "&fields=id,title,start_date,end_date,synopsis,mean,rank,popularity,num_list_users,\
        num_scoring_users,nsfw,media_type,status,genres,num_episodes,start_season,\
        broadcast,source,average_episode_duration,rating,background,studios,statistics"

'''
Fields options: (by default->id, main_picture, title)
    - synopsis: string of long synopsis paragraph
    - popularity: rank of popularity
    - media_type: should be tv for anime?
    - status: airing, finished, etc.
    - start_season: seaon and year that anime started
    - broadcast: what day and time of week that it broadcasts
    - source: light novel, manga, etc.
    - average_episode_duration: duration in seconds
    - background: ?
    - related_anime & related_manga: don't work for some reason?
    - studios: give name and id of studio(s)
    - statistics: ?

All fields:
fields = "&fields=id,title,main_picture,alternative_titles,start_date,end_date,synopsis,mean,\
          rank,popularity,num_list_users,num_scoring_users,nsfw,created_at,updated_at,media_type,\
          status,genres,my_list_status,num_episodes,start_season,broadcast,source,average_episode_duration,\
          rating,pictures,background,related_anime,related_manga,recommendations,studios,statistics"
'''

# Calling the API

# Install local cache to cache API calls (to avoid repeated calls)
requests_cache.install_cache()

# Loop variables
is_df_created = False # Var to track if df has been created (on 1st iter of loop)
seasons = ["winter", "spring", "summer", "fall"] # Arr for looping through anime seasons

# Collect data by looping through years and seasons, as many as we need
# Loop through years, from latest to oldest
for year in range(2023, 2000, -1):
    print(year)
    # Loop through seasons
    for i in range(4):
        season = seasons[i]
        print(season)
        try:
            # Call API to get data
            resp = requests.get(base_url + str(year) + "/" + season + sort_by_num_users + fields + limit + nsfw, headers=headers)
            resp.raise_for_status() # Check for exceptions

        # Resolving exceptions
        except requests.exceptions.HTTPError as err:
            print("HTTPError")
            print(err.response.text)
            raise SystemExit(err)
        except requests.exceptions.ConnectionError as err:
            print("ConnectionError:\n")
            print(err.response.text)
            raise SystemExit(err)
        except requests.exceptions.Timeout as err:
            print("Timeout:\n")
            print(err.response.text)
            raise SystemExit(err)
        except requests.exceptions.TooManyRedirects as err:
            print("TooManyRedirects:\n")
            print(err.response.text)
            raise SystemExit(err)
        except requests.exceptions.RequestException as err:
            print("Oops, something else:\n")
            print (err.response.text)
            raise SystemExit(err)
        
        r_json = resp.json() # Convert response to json format
        cur_season = r_json['season'] # Get cur season of data

        for r in r_json['data']: # Iterate through all animes that end in cur season
            root = r['node']
            if root['media_type'] != 'tv': # Check if it is an anime
                continue
            if root['start_season'] != cur_season:  # Check if anime did not start in cur season
                continue                            # This is to ensure that anime is seasonal
            
            if root['status'] != 'finished_airing': # Check if anime has finished airing
                continue
            # At this point we only have seasonal animes of the current season

            temp_df = pd.DataFrame(flatten(root), index=[0]) # Flatten json

            # Uncomment 3 lines below if we don't want to collect pictures, genre ids, studio ids
            #cols_dropped = ['main_picture_medium', 'main_picture_large']
            #regex_dropped = "^(genres_\d_id|studios_\d_id)$"
            #temp_df = df.drop(columns=cols_dropped).drop(df.filter(regex=regex_dropped).columns, axis = 1)
            
            # If dj_json has not been created, create it and set var to True
            if is_df_created == False:
                df_json = temp_df
                is_df_created = True
            # If df_json alr created, then just append to it
            else:
                df_json = pd.concat([df_json, temp_df], ignore_index=True)
        #time.sleep(0.25) # If we need to rate limit for api

# Once all data is in df, convert it to excel in current folder
df_json.to_excel('anime_data.xlsx', index=False)
print("\nSuccessfully collected data into 'anime_data.xlsx'\n")



