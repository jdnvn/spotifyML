import spotipy
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score

# Spotify track genre classifier

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
    track_obj = track['track']
    if track_obj:
      track_id = track_obj['id']
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

  return x, np.array(y)

# Marcin EDM 2020 spotify:playlist:6DHt1ugZdDCkPe3PNf69PQ

# Rap song: 3wsYSS0TEI5ERTuDNfiU7t
# EDM song: 5gneV1qwaIndUU4aZ3uhXP
# Deadmaus ghost n stuff: 3ezkJgagRPZ39KCTrKcSI7
# Classical song: 67TCAXIe154ZGDNaWceqxC


classical_train_ids = ['37i9dQZF1DWYkztttC1w38', '0DOYw5K9vybLVN1lOUO9b5',
                       '4o6d2y91Us7AfsIzCj5uwr', '37i9dQZF1DX4P0ijJK5lUv', '37i9dQZF1DWZnzwzLBft6A',
                       '1h0CEZCm6IbFTbxThn6Xcs']
rap_train_ids = ['37i9dQZF1DX186v583rmzp', '4gdyJJFph3i2oMdpRnCONw',
                 '5xNWwWNTIHp21TauVPWaPk', '37i9dQZF1DWYGxBNe4qojI', '37i9dQZF1DWT6MhXz0jw61',
                 '37i9dQZF1DX7Mq3mO5SSDc', '5zVTjqiHYpNyF8EzJ96Aqb']
edm_train_ids = ['6DHt1ugZdDCkPe3PNf69PQ', '09T8BRorjn8It7gCCKuT3U',
                 '5mhb5QRMXgufiqEuHL74gi', '37i9dQZF1DX0hvSv9Rf41p', '37i9dQZF1DX5Q27plkaOQ3',
                 '37i9dQZF1DX6J5NfMJS675']

classical_test_ids = ['6wObnEPQ63a4kei1sEcMdH', '1ZJpJahEFst7u8njXeGFyv']
rap_test_ids = ['2nJsRFJkr7BegSfKpG2d7O', '7pf7okzvbnPdobKmjHJSRl']
edm_test_ids = ['37i9dQZF1DWZ5Se2LB1C5h', '5tkXZ9sMI2xAGsJXb3EuoK']

all_train_ids = [classical_train_ids, rap_train_ids, edm_train_ids]
all_test_ids = [classical_test_ids, rap_test_ids, edm_test_ids]

x_train, y_train = format_data(all_train_ids)
x_test, y_test = format_data(all_test_ids)

print("Train size")
print(len(x_train))
print("Test size")
print(len(x_test))

# KNN ################
# knn = KNeighborsClassifier(n_neighbors=300)
# knn.fit(x_train, y_train)
# y_pred = knn.predict(x_test)

# accuracy = accuracy_score(y_test, y_pred)
# print("accuracy: " + str(accuracy))

# Logistic Regression
lr = LogisticRegression(multi_class='multinomial')
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Logistic Regression")
print("accuracy: " + str(accuracy))


# Random Forest ############
rf = RandomForestClassifier(max_depth=11, max_features='sqrt')
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

# precision = precision_score(y_train, y_pred)
# recall = recall_score(y_train, y_pred)
accuracy = accuracy_score(y_test, y_pred)
# roc_auc = roc_auc_score(y_train, nb.scores)

print("Random Forest Classifier")
print("accuracy: " + str(accuracy))

kf = KFold(n_splits=5)
kf.get_n_splits(x_train)

score = 0

for i, (train_index, test_index) in enumerate(kf.split(x_train)):
    X_train, X_test = x_train[train_index], x_train[test_index]
    Y_train, Y_test = y_train[train_index], y_train[test_index]

    rf = RandomForestClassifier(max_depth=11, max_features='sqrt')
    rf.fit(x_train, y_train)
    Y_pred = rf.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred, labels=['classical', 'rap', 'edm'], average='micro')
    print("fold " + str(i) + ": " + str(accuracy))
    score += f1

print("K-Fold with Random Forest")
print("avg f1 score: " + str(score/5))
