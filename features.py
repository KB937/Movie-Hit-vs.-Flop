import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.preprocessing import MultiLabelBinarizer

def build_features(df):
    """
    Builds features for the Movie Hit vs Flop classifier.
    
    Args:
        df (pd.DataFrame): Raw dataframe with necessary columns.
        
    Returns:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        feature_names (list): List of feature names.
    """
    df = df.copy()
    
    # --- 1. Label Encoding ---
    # Ensure label exists or create it if revenue/budget exist (for training)
    if 'label' not in df.columns and 'revenue' in df.columns and 'budget' in df.columns:
        df['roi'] = df['revenue'] / df['budget']
        df['label'] = df['roi'].apply(lambda x: 'Hit' if x >= 2.0 else ('Flop' if x < 1.0 else 'Exclude'))
        df = df[df['label'] != 'Exclude']
    
    # If we are in inference mode and no label/revenue, we skip label generation
    if 'label' in df.columns:
        y = df['label'].map({'Hit': 1, 'Flop': 0})
    else:
        y = None

    # --- 2. Numeric Features ---
    # Log Budget
    if 'budget' in df.columns:
        df['log_budget'] = np.log1p(df['budget'])
    
    # Release Month
    if 'release_date' in df.columns:
        df['release_date'] = pd.to_datetime(df['release_date'])
        df['release_month'] = df['release_date'].dt.month
    
    # --- 3. Categorical Features (One-Hot / Multi-Hot) ---
    
    # Genres
    if 'genres' in df.columns:
        # Handle stringified JSON
        if isinstance(df['genres'].iloc[0], str):
            df['genres'] = df['genres'].apply(literal_eval)
        
        # Extract names
        df['genre_names'] = df['genres'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
        
        # Multi-hot encode
        mlb_genres = MultiLabelBinarizer()
        genres_encoded = mlb_genres.fit_transform(df['genre_names'])
        genres_df = pd.DataFrame(genres_encoded, columns=[f"genre_{c}" for c in mlb_genres.classes_], index=df.index)
        df = pd.concat([df, genres_df], axis=1)

    # Production Companies (Top 20)
    # Note: In a real pipeline, the top 20 should be learned from training data and applied to test.
    # Here we approximate by computing it on the passed df, assuming it's the training set.
    # For a strictly correct pipeline, we'd need a separate fit/transform step.
    if 'production_companies' in df.columns:
         # Handle stringified JSON
        if isinstance(df['production_companies'].iloc[0], str):
            df['production_companies'] = df['production_companies'].apply(literal_eval)
            
        df['company_names'] = df['production_companies'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
        
        all_companies = [c for sublist in df['company_names'] for c in sublist]
        top_20_companies = pd.Series(all_companies).value_counts().head(20).index.tolist()
        
        for company in top_20_companies:
            # Clean company name for column name
            safe_name = "".join(e for e in company if e.isalnum())
            df[f'company_{safe_name}'] = df['company_names'].apply(lambda x: 1 if company in x else 0)
            
        # "Other" feature? 
        # df['company_Other'] = df['company_names'].apply(lambda x: 1 if not any(c in top_20_companies for c in x) else 0)

    # --- 4. Select Features ---
    # Drop leakage and identifier columns
    exclude_cols = [
        'id', 'original_title', 'title', 'status', 'release_date', 
        'revenue', 'vote_average', 'vote_count', 'popularity', 'roi', 
        'genres', 'genre_names', 'production_companies', 'company_names',
        'overview', 'tagline', 'homepage', 'keywords', 'spoken_languages', 'production_countries',
        'budget', 'label' # label is y, budget is replaced by log_budget
    ]
    
    # Keep numeric and encoded columns
    feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    X = df[feature_cols]
    feature_names = X.columns.tolist()
    
    return X, y, feature_names
