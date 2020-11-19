import spotipy
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score

# Hey bri guy

CLIENT_ID = "b665f59008ab415c927afbc0f00b44b1"
CLIENT_SECRET = "84f8a61e253a4a0db91a1f50e8a0a9f1"

token = spotipy.oauth2.SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
access_token = token.get_access_token()
sp = spotipy.Spotify(access_token)

# list of necessary features
playlist_features_list = ["danceability", "energy", "key", "loudness",
                          "mode", "speechiness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms", "time_signature"]


def get_playlist_features(playlistID, genre):
  # Make a dataset of songs that come from a playlist
  playlist = sp.playlist(playlistID)['tracks']['items']
  playlist_ids = []

  for track in playlist:
    track_id = track['track']['id']
    if track_id: playlist_ids.append(track_id)

  track_features = sp.audio_features(playlist_ids)

  tracks = []
  for track_feature in track_features:
    track = []
    for feature in playlist_features_list:
      track.append(track_feature[feature])

    tracks.append(track)

  return np.array(tracks)


def format_data(all_class_ids):
  classes = ['classical', 'rap', 'edm']
  x = []
  y = []

  for i, ids in enumerate(all_class_ids):
    features = []
    for playlist_id in ids:
      if len(features) > 0:
        features = np.concatenate((features, get_playlist_features(playlist_id, classes[i])))
      else:
        features = get_playlist_features(playlist_id, classes[i])

    y.extend([classes[i]]*len(features))

    if len(x) > 0:
      x = np.concatenate((x, features))
    else:
      x = features

  return x, y
# Marcin EDM 2020 spotify:playlist:6DHt1ugZdDCkPe3PNf69PQ

# Rap song: 3wsYSS0TEI5ERTuDNfiU7t
# EDM song: 5gneV1qwaIndUU4aZ3uhXP
# Deadmaus ghost n stuff: 3ezkJgagRPZ39KCTrKcSI7
# Classical song: 67TCAXIe154ZGDNaWceqxC

classical_train_ids = ['37i9dQZF1DWYkztttC1w38', '0DOYw5K9vybLVN1lOUO9b5', '4o6d2y91Us7AfsIzCj5uwr']
rap_train_ids = ['37i9dQZF1DX186v583rmzp', '4gdyJJFph3i2oMdpRnCONw', '5xNWwWNTIHp21TauVPWaPk']
edm_train_ids = ['6DHt1ugZdDCkPe3PNf69PQ', '09T8BRorjn8It7gCCKuT3U', '5mhb5QRMXgufiqEuHL74gi']

classical_test_ids = ['6wObnEPQ63a4kei1sEcMdH', '1ZJpJahEFst7u8njXeGFyv']
rap_test_ids = ['2nJsRFJkr7BegSfKpG2d7O', '7pf7okzvbnPdobKmjHJSRl']
edm_test_ids = ['37i9dQZF1DWZ5Se2LB1C5h', '5tkXZ9sMI2xAGsJXb3EuoK']

all_train_ids = [classical_train_ids, rap_train_ids, edm_train_ids]
all_test_ids = [classical_test_ids, rap_test_ids, edm_test_ids]

x_train, y_train = format_data(all_train_ids)
x_test, y_test = format_data(all_train_ids)

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

# precision = precision_score(y_train, y_pred)
# recall = recall_score(y_train, y_pred)
accuracy = accuracy_score(y_test, y_pred)
# roc_auc = roc_auc_score(y_train, nb.scores)

# print("f1: " + str(f1))
# print("precision: " + str(precision))
# print("recall: " + str(recall))
print("accuracy: " + str(accuracy))
# print("roc auc: " + str(roc_auc))

