import requests
import json
# import certifi

# change certificate settings: add myanimelist certificate chain to certificate list
# print(certifi.where())

def json_print(json_obj):
    str = json.dumps(json_obj, sort_keys=True, indent=2)
    print(str)

client_id = "6a3dc42a0c201194413f0f02733ae033"

# client_secret = "b24d9ab0be8b47325e5ff0ac8c29527553447a0e9e13e9aa7ea0b01cbcd30721"

headers = {
    'X-MAL-CLIENT-ID': client_id
}

parameters = {
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
fields = "&fields=id,title,start_date,end_date,mean,rank,popularity,num_list_users,num_scoring_users,nsfw,media_type,status,genres,num_episodes,start_season,broadcast,source,average_episode_duration,rating,background,related_anime,related_manga,studios,statistics"

# Fields options: (default->id, main_picture, title)
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

limit = "&limit=2"
# don't know if this param is necessary
params = {
    'nsfw': True
}


for year in range(2023, 2021, -1):
    for i in range(2):
        season = seasons[i]
        print(str(year), season, "\n")
        try:
            resp = requests.get(base_url + str(year) + "/" + season + sort_by_num_users + fields + limit, headers=headers, params=params)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as err:
            print("HTTPError:\n")
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
        #print(resp.status_code)
        json_print(resp.json())




