import pandas as pd
from openpyxl import load_workbook

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
    },
    {
      "id": 8,
      "name": "Drama"
    },
    {
      "id": 58,
      "name": "Gore"
    },
    {
      "id": 13,
      "name": "Historical"
    },
    {
      "id": 42,
      "name": "Seinen"
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
  "num_scoring_users": 234159,
  "popularity": 414,
  "rank": 34,
  "rating": "r",
  "source": "manga",
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

df = pd.json_normalize(json)
df = pd.concat([df, df], ignore_index=True)
df.to_excel("test.xlsx")


