import sys
import os
import pandas as pd
# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict import load_resources, predict_result, load_resources

def test_prediction():
    print("Testing Prediction Logic...")
    try:
        pipeline, feature_cols = load_resources()
        print("Resources loaded successfully.")
        
        test_data = {
            'budget': 150000000,
            'runtime': 120,
            'release_date': '2023-07-21',
            'genres': ['Action', 'Adventure', 'Thriller'],
            'companies': ['Universal Pictures', 'Warner Bros.']
        }
        
        print(f"Test Input: {test_data}")
        
        pred, proba = predict_result(pipeline, feature_cols, test_data)
        
        print(f"Prediction: {pred}")
        print(f"Probability: {proba}")
        
        print("Test Passed!")
    except Exception as e:
        print(f"Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prediction()
