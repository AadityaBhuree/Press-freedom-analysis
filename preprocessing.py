import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

def run_preprocessing(data_path="dataset.csv", output_dir="."):
    os.makedirs("models", exist_ok=True)
    print(f"--- Loading Dataset from '{data_path}' ---")
    df = pd.read_csv(data_path)
    
    # 1. Cleaning
    df = df.dropna(subset=['Country', 'Situation']).copy()
    df = df[df['Country'].str.strip() != 'OECS'].copy()
    
    # Clean text whitespace
    df['Region'] = df['Region'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
    df['Situation'] = df['Situation'].astype(str).str.strip()
    
    # 2. Feature Engineering
    print("Engineering features...")
    # Rank movement: Position 2021 minus Position 2022 (positive means improved rank)
    df['Position_Change'] = df['Position 2021'] - df['Position 2022']
    
    # Combined safety danger index
    df['Press_Danger_Index'] = (
        df['Journalist Killed'] + 
        df['Media Workers Killed'] + 
        df['Journalist Imprisoned'] + 
        df['Media Workers Imprisoned']
    )
    
    # Sub-score statistical features across the 5 dimensions
    subscore_cols = ['Politic Score', 'Economic Score', 'Legislative Score', 'Social Score', 'Security Score']
    df['Score_Variance'] = df[subscore_cols].var(axis=1)
    df['Score_Range'] = df[subscore_cols].max(axis=1) - df[subscore_cols].min(axis=1)
    df['Score_Min'] = df[subscore_cols].min(axis=1)

    # 3. Target Encoding
    situation_map = {
        'Good': 0,
        'Satisfactory': 1,
        'Problematic': 2,
        'Difficult': 3,
        'Very Serious': 4
    }
    df['target'] = df['Situation'].map(situation_map)
    
    # Check for any unmapped target values
    if df['target'].isnull().any():
        print("Warning: Unmapped Situation values found:", df[df['target'].isnull()]['Situation'].unique())
        df = df.dropna(subset=['target'])
    
    df['target'] = df['target'].astype(int)
    
    # Save label mapping reference
    label_mapping = {v: k for k, v in situation_map.items()}
    with open(os.path.join("models", "label_mapping.json"), "w") as f:
        json.dump(label_mapping, f, indent=4)
    print("Saved target label mapping to 'models/label_mapping.json'")

    # 4. Feature Selection & Categorical Encoding
    # Drop identifier columns, target column, and target-defining leakage columns
    # In RSF methodology, Situation is computed directly from Global Score, and Position is its rank.
    # To prevent trivial data leakage, we predict Situation purely from exogenous indicators.
    drop_cols = [
        'Country', 'ISO Code', 'Situation', 'target',
        'Global Score', 'Position 2022', 'Position 2021', 'Position_Change'
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols].copy()
    y = df['target'].copy()
    
    # Save feature names list for pipeline & inference consistency
    with open(os.path.join("models", "feature_names.json"), "w") as f:
        json.dump(feature_cols, f, indent=4)
    print("Saved feature names to 'models/feature_names.json'")
    
    # Encode categorical feature 'Region'
    region_le = LabelEncoder()
    X['Region'] = region_le.fit_transform(X['Region'])
    joblib.dump(region_le, os.path.join("models", "region_encoder.joblib"))
    
    print(f"Features ({len(X.columns)}):", list(X.columns))
    print(f"Target distribution:\n{y.value_counts().sort_index()}")
    
    # 5. Train-Test Split (80% Train, 20% Test, Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 6. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    joblib.dump(scaler, os.path.join("models", "scaler.joblib"))
    print("Saved feature scaler to 'models/scaler.joblib'")
    
    # 7. Save Processed Datasets
    X_train_scaled.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    
    print(f"\n--- Preprocessing Completed ---")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set:  {X_test.shape[0]} samples")
    print(f"Saved: X_train.csv, X_test.csv, y_train.csv, y_test.csv")

if __name__ == "__main__":
    run_preprocessing()
