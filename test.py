import requests
import pandas as pd
from flatten_json import flatten

# Static data for testing
json = {
  "average_episode_duration": 1555,
  "broadcast": {
    "day_of_the_week": "tuesday",  
    "start_time": "00:30"
  },
  "end_date": "2023-06-20",
  "genres": [
    {
      "id": 1,
      "name": "Action"
    },
    {
      "id": 2,
      "name": "Adventure"
    }
  ],
  "id": 49387,
  "main_picture": {
    "large": "https://cdn.myanimelist.net/images/anime/1170/124312l.jpg",
    "medium": "https://cdn.myanimelist.net/images/anime/1170/124312.jpg"
  },
  "mean": 8.81,
  "media_type": "tv",
  "nsfw": "white",
  "num_episodes": 24,
  "num_list_users": 509892,
  "start_date": "2023-01-10",
  "start_season": {
    "season": "winter",
    "year": 2023
  },
  "status": "finished_airing",
  "studios": [
    {
      "id": 569,
      "name": "MAPPA"
    }
  ],
  "title": "Vinland Saga Season 2"
}

# API authentication info (key, secret, header)
client_id = "6a3dc42a0c201194413f0f02733ae033"
#client_secret = "b24d9ab0be8b47325e5ff0ac8c29527553447a0e9e13e9aa7ea0b01cbcd30721"
headers = {
    'X-MAL-CLIENT-ID': client_id
}

# API query settings
base_url = "https://api.myanimelist.net/v2/anime/season/" # URL
sort_by_num_users = "?sort=anime_num_list_users" # Sort before getting data
limit = "&limit=20" # Set limit for num animes to get per season
nsfw = "&nsfw=true" # Include nsfw anime in query

# Selected fields
fields = "&fields=id,title,start_date,end_date,synopsis,mean,rank,popularity,num_list_users,\
        num_scoring_users,nsfw,media_type,status,genres,num_episodes,start_season,\
        broadcast,source,average_episode_duration,rating,background,studios,statistics"

# Loop variables
is_df_created = False # Var to track if df has been created (on 1st iter of loop)
seasons = ["winter", "spring", "summer", "fall"] # Arr for looping through anime seasons

# Collect data by looping through years and seasons, as many as we need
# Loop through years, from latest to oldest
for year in range(2023, 2022, -1):
    print(year)
    # Loop through seasons
    for i in range(2):
        season = seasons[i]
        print(season)
        # Call API to get data
        resp = requests.get(base_url + str(year) + "/" + season + sort_by_num_users + fields + limit + nsfw, headers=headers) 
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
            # If dj_json has not been created, create it and set var to True
            if is_df_created == False:
                df_json = temp_df
                is_df_created = True
            # If df_json alr created, then just append to it
            else:
                df_json = pd.concat([df_json, temp_df], ignore_index=True)

# Test for static data
# df_json = pd.json_normalize(json)
# df_json = pd.concat([df_json, df_json], ignore_index=True)

# Once all data is in df, convert it to excel in current folder
df_json.to_excel('test.xlsx', index=False)
print("Test complete")