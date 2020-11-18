import spotipy
import pandas as pd
import numpy as np

CLIENT_ID = "b665f59008ab415c927afbc0f00b44b1"
CLIENT_SECRET = "84f8a61e253a4a0db91a1f50e8a0a9f1"

token = spotipy.oauth2.SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
access_token = token.get_access_token()
sp = spotipy.Spotify(access_token)

# list of necessary features
playlist_features_list = ["artist", "track_name",  "track_id", "danceability", "energy", "key", "loudness",
                          "mode", "speechiness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms", "time_signature"]

playlist_df = pd.DataFrame(columns=playlist_features_list)

# Make a dataset of songs that come from
classical_playlist = sp.playlist('37i9dQZF1DWWEJlAGA9gs0')['tracks']['items']

playlist_ids = []

for track in classical_playlist:
  playlist_ids.append(track['track']['id'])

track_features = sp.audio_features(playlist_ids)
print(track_features[0])




