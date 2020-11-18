import spotipy
import spotipy.util as util

CLIENT_ID = "b665f59008ab415c927afbc0f00b44b1"
CLIENT_SECRET = "84f8a61e253a4a0db91a1f50e8a0a9f1"

token = util.oauth2.SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
cache_token = token.get_access_token()
sp = spotipy.Spotify(cache_token)

