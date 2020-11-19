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


# Marcin EDM 2020 spotify:playlist:6DHt1ugZdDCkPe3PNf69PQ

# Rap song: 3wsYSS0TEI5ERTuDNfiU7t
# EDM song: 5gneV1qwaIndUU4aZ3uhXP
# Deadmaus ghost n stuff: 3ezkJgagRPZ39KCTrKcSI7
# Classical song: 67TCAXIe154ZGDNaWceqxC

classical_test_ids = ['37i9dQZF1DWYkztttC1w38', '0DOYw5K9vybLVN1lOUO9b5', '4o6d2y91Us7AfsIzCj5uwr']
test_classical = []

for track_id in classical_test_ids:
  if len(test_classical) > 0:
    test_classical = np.concatenate((test_classical, get_playlist_features(track_id, 'classical')))
  else:
    test_classical = get_playlist_features(track_id, 'classical')

print(len(test_classical))
test_rap = get_playlist_features('37i9dQZF1DX186v583rmzp', 'rap')
print(len(test_rap))
test_edm = get_playlist_features('3Di88mvYplBtkDBIzGLiiM', 'edm')
print(len(test_edm))

x_test = np.concatenate((np.array(test_classical), np.array(test_rap)))
x_test = np.concatenate((x_test, np.array(test_edm)))


classical_features = get_playlist_features('37i9dQZF1DWWEJlAGA9gs0', 'classical')
rap_features = get_playlist_features('5xNWwWNTIHp21TauVPWaPk', 'rap')
edm_features = get_playlist_features('6DHt1ugZdDCkPe3PNf69PQ', 'edm')

x_train = np.array(classical_features)
x_train = np.concatenate((x_train, np.array(rap_features)))
x_train = np.concatenate((x_train, np.array(edm_features)))

y_train = np.array(['classical'] * 100)
y_train = np.concatenate((y_train, np.array(['rap'] * 100)))
y_train = np.concatenate((y_train, np.array(['edm'] * 100)))

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)
print(len(y_pred))

# precision = precision_score(y_train, y_pred)
# recall = recall_score(y_train, y_pred)
accuracy = accuracy_score(y_train, y_pred)
# roc_auc = roc_auc_score(y_train, nb.scores)

# print("f1: " + str(f1))
# print("precision: " + str(precision))
# print("recall: " + str(recall))
print("accuracy: " + str(accuracy))
# print("roc auc: " + str(roc_auc))

