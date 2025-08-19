import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pickle

print("Script started...")

try:
    # 1. Load the dataset
    print("Loading data from heart.csv...")
    df = pd.read_csv('heart.csv')
    print("Data loaded successfully.")

    # 2. Define the 10 features your web form uses
    features = [
        'age', 'sex', 'cp', 'chol', 
        'thalach', 'exang', 'oldpeak', 
        'slope', 'ca', 'thal'
    ]
    target = 'target'
    print(f"Using {len(features)} features for training.")

    X = df[features]
    y = df[target]

    # 3. Create and train the pipeline
    print("Training the model...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipeline.fit(X, y)
    print("Model training complete.")

    # 4. Save the new pipeline, overwriting the old file
    with open('heart_pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

    print("\n✅ Success! A new 'heart_pipeline.pkl' has been created.")

except FileNotFoundError:
    print("\n❌ ERROR: 'heart.csv' not found. Please place it in the same folder.")
except KeyError as e:
    print(f"\n❌ ERROR: A required column was not found in the CSV file: {e}")
except Exception as e:
    print(f"\n❌ An unexpected error occurred: {e}")