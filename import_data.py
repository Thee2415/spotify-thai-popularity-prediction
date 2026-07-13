import spotipy
from spotipy.oauth2 import SpotifyClientCredentials 
from dotenv import load_dotenv
import os
import pandas as pd
import time
from datetime import datetime

# --- 1. CONFIGURATION & AUTHENTICATION ---
# Use relative paths instead of hardcoded drives for better environment flexibility
base_path = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(base_path, ".env"))

client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')

if not client_id or not client_secret:
    raise ValueError("Missing Spotify API credentials. Please check your .env file.")

# Initialize Spotify API connection
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

# 5 Target official playlists representing the Thai music market
PLAYLIST_CONFIG = {
  "ฉันฟังเพลงไทย": "1dnaNNnFcdle7DB9mSmian",
  "T-Pop Now": "654i1Fk51pcrk5gykqCLhp",
  "ROCK CODE": "08xy6NAWYPOlblS10xXo0i",
  "ฮิปฮอป R.E.A.L.": "1lnqGcFfHx8M0qMNfODv7c",
  "อินดี้ศาสตร์ Indieology": "6VOR5XrWWPtFz15VD1ULMI",
}

COLLECTION_DATE = datetime.now().strftime('%Y-%m-%d')
ALL_RECORDS = []
ALL_ARTIST_IDS = set()

# --- 2. DATA ACQUISITION (ETL) ---
def get_tracks_and_artists_from_playlists():
    """Extract top 50 tracks from each target playlist daily."""
    print(f"Starting data collection for date: {COLLECTION_DATE}")
    for playlist_name, playlist_id in PLAYLIST_CONFIG.items():
        try:
            results = sp.playlist_tracks(playlist_id, limit=50)
            for i, item in enumerate(results['items']):
                track = item.get('track', {})
                if not track or not track.get('id'): continue

                ALL_RECORDS.append({
                    'track_id': track['id'],
                    'track_name': track.get('name'),
                    'artist_name': track['artists'][0]['name'] if track.get('artists') else "Unknown",
                    'collection_date': COLLECTION_DATE,
                    'playlist_name': playlist_name,
                    'playlist_rank': i + 1,  # Capture track position on the playlist
                    'track_popularity': track.get('popularity'),
                    'release_date': track['album'].get('release_date') if track.get('album') else None,
                    'artist_id': track['artists'][0]['id'] if track.get('artists') else None
                })
                if track['artists'][0]['id']: 
                    ALL_ARTIST_IDS.add(track['artists'][0]['id'])
                    
            print(f"Successfully processed {len(results['items'])} tracks from: {playlist_name}")
            time.sleep(0.5) # API Rate-limiting safeguard
        except Exception as e:
            print(f"Error fetching tracks from '{playlist_name}': {e}")
            continue

def get_artist_metadata():
    """Fetch profile features (followers, popularity) for unique artists in batches of 50."""
    artist_metadata = {}
    artist_list = list(filter(None, ALL_ARTIST_IDS))
    batch_size = 50
    print(f"Fetching metadata for {len(artist_list)} unique artists...")
    for i in range(0, len(artist_list), batch_size):
        batch = artist_list[i:i + batch_size]
        try:
            artists_data = sp.artists(batch)
            if artists_data and 'artists' in artists_data:
                for artist in artists_data['artists']:
                    if artist and artist.get('id'):
                        artist_metadata[artist['id']] = {
                            'artist_popularity': artist.get('popularity'),
                            'artist_followers': artist.get('followers', {}).get('total', 0),
                            'artist_genres': ', '.join(artist.get('genres', []))
                        }
            time.sleep(0.3) 
        except Exception as e:
            print(f"Error fetching artist batch: {e}")
            time.sleep(1)
    return artist_metadata

# --- 3. PIPELINE AUTOMATION ---
if __name__ == "__main__":
    get_tracks_and_artists_from_playlists()
    
    artist_data = get_artist_metadata() if ALL_RECORDS else {}
    df_final = pd.DataFrame(ALL_RECORDS)
    df_artist_lookup = pd.DataFrame.from_dict(artist_data, orient='index').reset_index().rename(columns={'index': 'artist_id'})

    # Merge track data with artist profile data
    df_merged = pd.merge(df_final, df_artist_lookup, on='artist_id', how='left') if not df_artist_lookup.empty else df_final

    # Export daily snapshot to folder
    export_folder = os.path.join(base_path, "DATA_5playist")
    os.makedirs(export_folder, exist_ok=True)

    filename = f"spotify_data_{COLLECTION_DATE}_{datetime.now().strftime('%H%M')}.csv"
    df_merged.to_csv(os.path.join(export_folder, filename), index=False, encoding='utf-8-sig')
    print(f"Daily snapshot saved to: {filename}")
    
    # Auto-append to Master Dataset for centralized data warehouse storage
    master_path = os.path.join(base_path, 'Master_Raw_Data.csv')
    if os.path.exists(master_path):
        df_merged.to_csv(master_path, mode='a', header=False, index=False, encoding='utf-8-sig')
    else:
        df_merged.to_csv(master_path, index=False, encoding='utf-8-sig')
    print(f"Pipeline complete. Master Database updated at: {master_path}")
