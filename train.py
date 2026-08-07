import os
import json
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib

def load_data():
    if not (os.path.exists("X_train.csv") and os.path.exists("y_train.csv")):
        raise FileNotFoundError("Processed dataset files (X_train.csv, y_train.csv) not found! Please run 'python preprocessing.py' first.")
    
    X_train = pd.read_csv("X_train.csv")
    X_test = pd.read_csv("X_test.csv")
    y_train = pd.read_csv("y_train.csv").values.ravel()
    y_test = pd.read_csv("y_test.csv").values.ravel()
    
    return X_train, X_test, y_train, y_test

def get_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine": SVC(kernel='rbf', probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42, eval_metric='mlogloss')
    }
    return models

def run_training():
    os.makedirs("models", exist_ok=True)
    X_train, X_test, y_train, y_test = load_data()
    
    models = get_models()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    trained_models = {}
    
    print("========================================================================")
    print("                   TRAINING & CROSS-VALIDATION                          ")
    print("========================================================================")
    
    for name, model in models.items():
        print(f"\nTraining Model: {name}...")
        
        # 5-Fold Cross Validation on Training Data
        cv_scores = cross_validate(
            model, X_train, y_train, cv=skf,
            scoring=['accuracy', 'f1_macro', 'f1_weighted']
        )
        
        cv_acc = np.mean(cv_scores['test_accuracy'])
        cv_f1_macro = np.mean(cv_scores['test_f1_macro'])
        cv_f1_weighted = np.mean(cv_scores['test_f1_weighted'])
        
        # Fit on full training set & evaluate on test set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        test_acc = accuracy_score(y_test, y_pred)
        precision, recall, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
        _, _, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
        
        # Save model file
        model_filename = name.lower().replace(" ", "_") + ".joblib"
        model_path = os.path.join("models", model_filename)
        joblib.dump(model, model_path)
        
        results.append({
            "Model": name,
            "CV_Accuracy": float(cv_acc),
            "CV_F1_Macro": float(cv_f1_macro),
            "Test_Accuracy": float(test_acc),
            "Test_Precision_Macro": float(precision),
            "Test_Recall_Macro": float(recall),
            "Test_F1_Macro": float(f1_macro),
            "Test_F1_Weighted": float(f1_weighted),
            "Model_File": model_filename
        })
        
        trained_models[name] = model
        print(f"  -> CV Acc: {cv_acc:.4f} | Test Acc: {test_acc:.4f} | Test F1-Macro: {f1_macro:.4f}")
    
    # Save results to JSON
    with open(os.path.join("models", "model_results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    # Summary Table
    df_results = pd.DataFrame(results)
    print("\n" + "="*85)
    print("                               MODEL SUMMARY METRICS                            ")
    print("="*85)
    print(df_results[['Model', 'CV_Accuracy', 'Test_Accuracy', 'Test_Precision_Macro', 'Test_Recall_Macro', 'Test_F1_Macro']].to_string(index=False))
    print("="*85)
    
    best_model_row = df_results.loc[df_results['Test_F1_Macro'].idxmax()]
    print(f"\n[BEST] Best Performing Model (by Test F1-Macro): {best_model_row['Model']} (Acc: {best_model_row['Test_Accuracy']:.4f}, F1-Macro: {best_model_row['Test_F1_Macro']:.4f})")
    print(f"Models and metrics saved to 'models/' directory.")

if __name__ == "__main__":
    run_training()
