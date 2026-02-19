import pandas as pd
import numpy as np
from ast import literal_eval
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def process_data():
    print("Loading data...")
    # Load datasets
    credits = pd.read_csv('data/raw/tmdb_5000_credits.csv')
    movies = pd.read_csv('data/raw/tmdb_5000_movies.csv')

    # Merge datasets on 'id'
    credits.rename(columns={'movie_id': 'id'}, inplace=True)
    df = movies.merge(credits, on='id')

    print(f"Initial shape: {df.shape}")

    # --- 2. Minimal Safe Cleaning ---
    print("Cleaning data...")
    # Drop duplicates on ID
    df = df.drop_duplicates(subset=['id'])

    # Drop rows where budget or revenue is <= 0 (unknowns)
    df = df[(df['budget'] > 0) & (df['revenue'] > 0)]

    # Filter for runtime (strictly feature films: 40 to 240 mins)
    df = df[(df['runtime'] >= 40) & (df['runtime'] <= 240)]
    
    print(f"Shape after cleaning: {df.shape}")

    # --- 3. Define Label (Hit vs Flop) ---
    print("Creating labels...")
    # Calculate ROI
    df['roi'] = df['revenue'] / df['budget']

    # Define Label
    def get_label(roi):
        if roi >= 2.0:
            return 'Hit'
        elif roi < 1.0:
            return 'Flop'
        else:
            return 'Exclude'

    df['label'] = df['roi'].apply(get_label)

    # Filter out the 'Exclude' band
    df = df[df['label'] != 'Exclude']
    print(f"Shape after label creation: {df.shape}")
    print(df['label'].value_counts())

    # --- 4. Feature Engineering ---
    print("Creating features...")
    # A) Dates
    df['release_date'] = pd.to_datetime(df['release_date'])
    df['release_month'] = df['release_date'].dt.month

    # B) Parse JSON columns
    df['genres'] = df['genres'].apply(literal_eval)
    df['genre_names'] = df['genres'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

    df['production_companies'] = df['production_companies'].apply(literal_eval)
    df['company_names'] = df['production_companies'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

    # Multi-hot encode Genres
    # We need to handle potential index issues if genres are empty
    # Explode genres, get dummies, sum by original index
    df_genres_exploded = df['genre_names'].apply(pd.Series).stack()
    if not df_genres_exploded.empty:
        genres_dummies = pd.get_dummies(df_genres_exploded).groupby(level=0).sum()
        df = pd.concat([df, genres_dummies], axis=1)
    
    # Top Production Companies (Top 20)
    all_companies = [c for sublist in df['company_names'] for c in sublist]
    if all_companies:
        top_companies = pd.Series(all_companies).value_counts().head(20).index.tolist()
        for company in top_companies:
            df[f'company_{company}'] = df['company_names'].apply(lambda x: 1 if company in x else 0)
    else:
        top_companies = []

    # C) Log Transform Budget
    df['log_budget'] = np.log1p(df['budget'])

    # --- 5. Remove Leakage ---
    print("Removing leakage...")
    leakage_cols = [
        'revenue', 'vote_average', 'vote_count', 'popularity', 'roi', 
        'status', 'original_title', 'title_x', 'title_y', 
        'genres', 'genre_names', 'production_companies', 'company_names', 
        'overview', 'tagline', 'homepage', 'keywords', 'spoken_languages', 'production_countries'
    ]

    cols_to_drop = [c for c in leakage_cols if c in df.columns]
    df_clean = df.drop(columns=cols_to_drop)

    print(f"Final shape: {df_clean.shape}")

    # --- 6. Save Processed Data ---
    output_path = 'data/processed/train_processed.csv' # Path relative to project root
    df_clean.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    process_data()
