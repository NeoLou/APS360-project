import requests as rq
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

resp = rq.get("https://api.myanimelist.net/v2/anime?q=one&limit=2", headers=headers)

print(resp.status_code)
json_print(resp.json())

resp = rq.get("https://api.myanimelist.net/v2/anime/30230?fields=id,title,main_picture,alternative_titles,start_date,end_date,synopsis,mean,rank,popularity,num_list_users,num_scoring_users,nsfw,created_at,updated_at,media_type,status,genres,my_list_status,num_episodes,start_season,broadcast,source,average_episode_duration,rating,pictures,background,related_anime,related_manga,recommendations,studios,statistics", headers=headers)

print(resp.status_code)
json_print(resp.json())


