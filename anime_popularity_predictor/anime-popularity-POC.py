import requests
import json
import requests_cache
import pandas as pd
from flatten_json import flatten
# import certifi

# change certificate settings: add myanimelist certificate chain to certificate list
# print(certifi.where())

# Fn to print json
def json_print(json_obj):
    str = json.dumps(json_obj, sort_keys=True, indent=2)
    print(str)

# Install local cache to cache API calls (to avoid repeated calls)
requests_cache.install_cache()

# API documentation: https://myanimelist.net/apiconfig/references/api/v2

# client id for api key
client_id = "6a3dc42a0c201194413f0f02733ae033"

# client secret for api key (not useful?)
#client_secret = "b24d9ab0be8b47325e5ff0ac8c29527553447a0e9e13e9aa7ea0b01cbcd30721"

headers = {
    'X-MAL-CLIENT-ID': client_id
}

# Get list of anime by search term
#resp = requests.get("https://api.myanimelist.net/v2/anime?q=one&limit=2", headers=headers)

# Get anime details by anime id
#resp = requests.get("https://api.myanimelist.net/v2/anime/30230?fields=id,title,main_picture,alternative_titles,start_date,end_date,synopsis,mean,rank,popularity,num_list_users,num_scoring_users,nsfw,created_at,updated_at,media_type,status,genres,my_list_status,num_episodes,start_season,broadcast,source,average_episode_duration,rating,pictures,background,related_anime,related_manga,recommendations,studios,statistics", headers=headers)

# Get top ranked anime by type
#resp = requests.get("https://api.myanimelist.net/v2/anime/ranking?ranking_type=all&limit=2", headers=headers)

# Get anime details by season
#resp = requests.get("https://api.myanimelist.net/v2/anime/season/2023/summer?sort=anime_num_list_users&fields=id,title,start_date,end_date,mean,rank,popularity,num_list_users,num_scoring_users,nsfw,media_type,status,genres,my_list_status,num_episodes,start_season,broadcast,source,average_episode_duration,rating,background,related_anime,related_manga,recommendations,studios,statistics&limit=6", headers=headers)

base_url = "https://api.myanimelist.net/v2/anime/season/"

seasons = ["winter", "spring", "summer", "fall"]
sort_by_num_users = "?sort=anime_num_list_users"
fields = "&fields=id,title,start_date,end_date,mean,rank,popularity,num_list_users,num_scoring_users,nsfw,media_type,status,genres,num_episodes,start_season,broadcast,source,average_episode_duration,rating,background,studios,statistics"

# Fields options: (by default->id, main_picture, title)
# popularity: rank of popularity
# media_type: should be tv for anime?
# status: airing, finished, etc.
# start_season: seaon and year that anime started
# broadcast: what day and time of week that it broadcasts
# source: light novel, manga, etc.
# average_episode_duration: duration in seconds
# background: ?
# related_anime & related_manga: don't work for some reason?
# studios: give name and id of studio(s)
# statistics: ?

limit = "&limit=30"

# don't know if these params are necessary
nsfw = "&nsfw=true"
params = {
    'nsfw': True
}

first = True
for year in range(2023, 2005, -1):
    for i in range(4):
        season = seasons[i]
        #print("----------------------------------")
        #print(str(year), season)
        try:
            resp = requests.get(base_url + str(year) + "/" + season + sort_by_num_users + fields + limit + nsfw, headers=headers, params=params)
            resp.raise_for_status()
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
        r_json = resp.json() #convert response to json format
        cur_season = r_json['season'] #get cur season of data
        #print("Current season:")
        #print(cur_season)
        #print("\n")

        for r in r_json['data']: # Iterate through all animes that end in cur season
            root = r['node']
            if root['media_type'] != 'tv': # Check if it is an anime
                continue
            if root['start_season'] != cur_season: # Check if anime did not start in cur season
                continue
            if root['status'] != 'finished_airing': # Check if anime has finished airing
                continue
            # At this point we only have seasonal animes of the current season
            #json_print(root)
            #cols_dropped = ['main_picture_medium', 'main_picture_large']
            regex_dropped = "^(genres_\d_id|studios_\d_id)$"
            df = pd.DataFrame(flatten(root), index=[0])
            if first:
                #df_json = df.drop(columns=cols_dropped).drop(df.filter(regex=regex_dropped).columns, axis = 1)
                df_json = df.drop(df.filter(regex=regex_dropped).columns, axis = 1)
                first = False
            else:
                #df_json = pd.concat([df_json, df.drop(columns=cols_dropped).drop(df.filter(regex=regex_dropped).columns, axis = 1)], ignore_index=True)
                df_json = pd.concat([df_json, df.drop(df.filter(regex=regex_dropped).columns, axis = 1)], ignore_index=True)
        #print("------------------------------------------------")
        #time.sleep(0.25) #if we need to rate limit for api
df_json.to_excel('anime_data.xlsx', index=False)



