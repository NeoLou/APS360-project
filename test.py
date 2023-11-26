# import requests
# import json
# import requests_cache
# import pandas as pd
# from flatten_json import flatten
# import matplotlib.pyplot as plt
# import os

# client_id = "6a3dc42a0c201194413f0f02733ae033"
# fields = ("&fields=id,title,start_date,end_date,media_type,studios,status")

# # Print json in readable format
# def json_print(json_obj):
#     str = json.dumps(json_obj, sort_keys=True, indent=2)
#     print(str)

# # Error check for get request
# def get_request_and_check(query, headers={'X-MAL-CLIENT-ID': client_id}):
#     # Check for exceptions
#     try:
#         resp = requests.get(query, headers=headers)
#         resp.raise_for_status() # Check for exceptions
#     # Resolving exceptions
#     except requests.exceptions.HTTPError as err:
#         print("HTTPError")
#         print(err.response.text)
#         raise SystemExit(err)
#     except requests.exceptions.ConnectionError as err:
#         print("ConnectionError:\n")
#         print(err.response.text)
#         raise SystemExit(err)
#     except requests.exceptions.Timeout as err:
#         print("Timeout:\n")
#         print(err.response.text)
#         raise SystemExit(err)
#     except requests.exceptions.TooManyRedirects as err:
#         print("TooManyRedirects:\n")
#         print(err.response.text)
#         raise SystemExit(err)
#     except requests.exceptions.RequestException as err:
#         print("Oops, something else:\n")
#         print (err.response.text)
#         raise SystemExit(err)
#     return resp

# # Convert get response to dataframe
# def convert_resp_to_df(resp, animes_df):
#     # Convert response to json
#     resp_json = resp.json() # Convert response to json
#     print("---------------------------------------------------")
#     json_print(resp_json)
#     for animes in resp_json['data']: # Iterate through all animes that end in cur season
#         anime = animes['node']
#         if anime['media_type'] != 'tv': # Check if it is an anime
#             continue
#         if anime['status'] != 'finished_airing': # Check if anime has finished airing
#             continue
#         if anime['studios'] == []: # Check if anime has no studio
#             continue
#         # At this point we only have seasonal animes of the current season
#         flat_json = flatten(anime) # Flatten json
#         cur_anime_df = pd.DataFrame(flat_json, index=[0]) # Flatten json, then convert it to dataframe
        
#         # Uncomment 3 lines below if we don't want to collect pictures, genre ids, studio ids
#         #cols_dropped = ['main_picture_medium', 'main_picture_large']
#         #regex_dropped = "^(genres_\d_id|studios_\d_id)$"
#         #temp_df = df.drop(columns=cols_dropped).drop(df.filter(regex=regex_dropped).columns, axis = 1)
        
#         # If dj_json has not been created, create it and set var to True
#         if animes_df.empty:
#             animes_df = cur_anime_df
#         # If df_json alr created, then just append to it
#         else:
#             animes_df = pd.concat([animes_df, cur_anime_df], ignore_index=True)
#     return animes_df

# limit = 10
# animes_df = pd.DataFrame()
# for i in range(2):
#   offset = str(i * limit)
#   query = ("https://api.myanimelist.net/v2/anime/"
#   f"ranking?ranking_type=bypopularity&limit={str(limit)}&offset={offset}&fields={fields}")
#   resp = get_request_and_check(query)
#   animes_df = convert_resp_to_df(resp, animes_df)
# print(animes_df)
# # Once all data is in df, convert it to excel in current folder
# animes_df.to_excel('test.xlsx', index=False)
# print("Test complete")

import numpy as np
for i in range(4, 16, 2):
    print(i)
    print(np.linspace(8, 64, i))