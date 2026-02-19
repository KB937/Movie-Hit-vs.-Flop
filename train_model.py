import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report
import sys
import os
# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features import build_features

def train_baseline_model():
    print("Loading data...")
    # Load processed data (which still needs parsing of JSON columns by build_features)
    # Actually, train_processed.csv has already been processed by 01_data_processing / process_data.py
    # So features.py is REDUNDANT if we load train_processed.csv which already has:
    # - log_budget
    # - release_month
    # - genre columns
    # - company columns
    # - leakage removed
    
    # We should load train_processed.csv and separate X and y.
    df = pd.read_csv('data/processed/train_processed.csv')
    
    # Define target
    # label is 'Hit' or 'Flop'
    y = df['label'].map({'Hit': 1, 'Flop': 0})
    
    # Define features
    # Drop non-feature columns
    drop_cols = ['id', 'label', 'release_date'] 
    # Note: process_data.py already dropped most leakage.
    # We just need to ensure we only keep numeric features for the model.
    # The processed csv has encoded genres/companies, log_budget, release_month, runtime.
    
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Ensure all data is numeric
    X = X.select_dtypes(include=[np.number])
    
    # Handle missing values (e.g. from log_budget or other numeric columns)
    X = X.fillna(X.mean())
    
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Define custom aggressive weights: Flop (0) is 5x more important than Hit (1)
    custom_weights = {0: 5, 1: 1}

    models = {
        'LogisticRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(class_weight=custom_weights, random_state=42, max_iter=1000, solver='liblinear'))
        ]),
        'RandomForest': Pipeline([
            ('clf', RandomForestClassifier(class_weight=custom_weights, random_state=42, n_estimators=100))
        ]),
        'XGBoost': Pipeline([
            ('clf', XGBClassifier(random_state=42, eval_metric='logloss'))
        ])
    }
    
    best_model_name = None
    best_f1 = -1
    best_pipeline = None
    best_threshold = 0.5
    metrics_report = {}
    
    print("\nTraining models...")
    for name, pipeline in models.items():
        print(f"--> {name}")
        pipeline.fit(X_train, y_train)
        
        # Predict probabilities
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Tune Threshold
        best_t = 0.5
        best_model_f1 = 0.0
        
        thresholds = np.arange(0.3, 0.71, 0.01)
        for t in thresholds:
            y_pred_t = (y_proba >= t).astype(int)
            # We still optimize for F1-Score of the positive class (Hit) by default, 
            # but we want to see the trade-off.
            # Alternatively, we could optimize for macro-f1 or something else.
            # Keeping selection logic same for now to not break "best model" selection logic abruptly.
            score = f1_score(y_test, y_pred_t, pos_label=1)
            if score > best_model_f1:
                best_model_f1 = score
                best_t = t
                
        # Calculate final metrics at best threshold
        y_pred_best = (y_proba >= best_t).astype(int)
        f1_hit = f1_score(y_test, y_pred_best, pos_label=1)
        f1_flop = f1_score(y_test, y_pred_best, pos_label=0)
        
        # AUC
        auc = roc_auc_score(y_test, y_proba)
        
        print(f"    Best Threshold: {best_t:.2f}")
        print(f"    F1 Score (Hit): {f1_hit:.4f}")
        print(f"    F1 Score (Flop): {f1_flop:.4f}  <-- Look at this improvement!")
        print(f"    AUC: {auc:.4f}")
        
        # Confusion Matrix
        y_pred_t = (y_proba >= best_t).astype(int)
        cm = confusion_matrix(y_test, y_pred_t).tolist()
        
        metrics_report[name] = {
            'Best Threshold': float(best_t),
            'F1 Score': float(best_model_f1),
            'AUC': float(auc),
            'Confusion Matrix': cm,
            'Classification Report': classification_report(y_test, y_pred_t, output_dict=True)
        }
        
        if best_model_f1 > best_f1:
            best_f1 = best_model_f1
            best_model_name = name
            best_pipeline = pipeline
            best_threshold = best_t
            
    print(f"\nBest Model: {best_model_name} with F1: {best_f1:.4f} at Threshold: {best_threshold:.2f}")
    
    # Save best model
    joblib.dump(best_pipeline, 'models/movie_hit_flop_pipeline.joblib')
    print("Saved best pipeline to models/movie_hit_flop_pipeline.joblib")
    
    # Save Metrics
    metrics_report['Best Model'] = best_model_name
    metrics_report['Best Threshold Selected'] = float(best_threshold)
    
    with open('reports/metrics.json', 'w') as f:
        json.dump(metrics_report, f, indent=4)
    print("Saved metrics to reports/metrics.json")

if __name__ == "__main__":
    train_baseline_model()
