import os
import time
from datetime import datetime
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

# โหลด Environment Variables 
base_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(base_dir, ".env"))

client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')

if not client_id or not client_secret:
    raise ValueError("Missing Spotify API credentials in environment variables.")

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

PLAYLISTS = {
    "ฉันฟังเพลงไทย": "1dnaNNnFcdle7DB9mSmian",
    "T-Pop Now": "654i1Fk51pcrk5gykqCLhp",
    "ROCK CODE": "08xy6NAWYPOlblS10xXo0i",
    "ฮิปฮอป R.E.A.L.": "1lnqGcFfHx8M0qMNfODv7c",
    "อินดี้ศาสตร์ Indieology": "6VOR5XrWWPtFz15VD1ULMI",
}

RUN_DATE = datetime.now().strftime('%Y-%m-%d')
records = []
unique_artist_ids = set()

def fetch_playlist_data():
    print(f"[{RUN_DATE}] Initializing data collection...")
    for name, playlist_id in PLAYLISTS.items():
        try:
            results = sp.playlist_tracks(playlist_id, limit=50)
            for idx, item in enumerate(results.get('items', [])):
                track = item.get('track')
                if not track or not track.get('id'):
                    continue

                artists = track.get('artists', [])
                artist_name = artists[0]['name'] if artists else "Unknown"
                artist_id = artists[0]['id'] if artists else None

                records.append({
                    'track_id': track['id'],
                    'track_name': track.get('name'),
                    'artist_name': artist_name,
                    'collection_date': RUN_DATE,
                    'playlist_name': name,
                    'playlist_rank': idx + 1,
                    'track_popularity': track.get('popularity'),
                    'release_date': track.get('album', {}).get('release_date'),
                    'artist_id': artist_id
                })
                if artist_id:
                    unique_artist_ids.add(artist_id)
            print(f" -> Successfully fetched {len(results.get('items', []))} tracks from: {name}")
            time.sleep(0.5)
        except Exception as e:
            print(f" -> Error processing playlist '{name}': {str(e)}")

def fetch_artist_metadata():
    metadata = {}
    artists_list = list(filter(None, unique_artist_ids))
    print(f"Fetching metadata for {len(artists_list)} unique artists...")
    
    batch_size = 50
    for i in range(0, len(artists_list), batch_size):
        batch = artists_list[i:i + batch_size]
        try:
            response = sp.artists(batch)
            for artist in response.get('artists', []):
                if artist and artist.get('id'):
                    metadata[artist['id']] = {
                        'artist_popularity': artist.get('popularity'),
                        'artist_followers': artist.get('followers', {}).get('total', 0),
                        'artist_genres': ', '.join(artist.get('genres', []))
                    }
            time.sleep(0.3)
        except Exception as e:
            print(f" -> Error fetching artist batch at index {i}: {str(e)}")
            time.sleep(1)
    return metadata

if __name__ == "__main__":
    fetch_playlist_data()
    
    artist_meta = fetch_artist_metadata() if records else {}
    df_tracks = pd.DataFrame(records)
    
    if not df_tracks.empty and artist_meta:
        df_artists = pd.DataFrame.from_dict(artist_meta, orient='index').reset_index().rename(columns={'index': 'artist_id'})
        df_merged = pd.merge(df_tracks, df_artists, on='artist_id', how='left')
        
        # บันทึกไฟล์ข้อมูลรายวัน
        output_file = os.path.join(base_dir, f'Spotify_Data_{RUN_DATE}.csv')
        df_merged.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"Pipeline completed. Output saved to: {output_file}")
    else:
        print("Pipeline aborted: No data retrieved.")
