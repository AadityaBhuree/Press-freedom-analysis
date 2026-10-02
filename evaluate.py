import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import joblib

def run_evaluation(output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Check data files
    if not (os.path.exists("X_test.csv") and os.path.exists("y_test.csv")):
        raise FileNotFoundError("Test data files missing! Please run 'python preprocessing.py' first.")
        
    if not os.path.exists(os.path.join("models", "model_results.json")):
        raise FileNotFoundError("Model results missing! Please run 'python train.py' first.")
        
    X_test = pd.read_csv("X_test.csv")
    y_test = pd.read_csv("y_test.csv").values.ravel()
    
    with open(os.path.join("models", "label_mapping.json"), "r") as f:
        label_mapping = json.load(f)
    # Convert keys to int
    label_mapping = {int(k): v for k, v in label_mapping.items()}
    class_names = [label_mapping[i] for i in sorted(label_mapping.keys())]
    
    with open(os.path.join("models", "model_results.json"), "r") as f:
        results = json.load(f)
        
    sns.set_theme(style="whitegrid")
    
    # 1. Plot Model Comparison Bar Chart
    df_res = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    df_melted = df_res.melt(id_vars=['Model'], value_vars=['CV_Accuracy', 'Test_Accuracy', 'Test_F1_Macro'],
                            var_name='Metric', value_name='Score')
    
    ax = sns.barplot(data=df_melted, x='Model', y='Score', hue='Metric', palette='Blues_r')
    plt.title("Model Performance Metrics Comparison", fontsize=14, fontweight='bold')
    plt.ylim(0, 1.05)
    plt.ylabel("Score", fontsize=12)
    plt.xlabel("Classifier Model", fontsize=12)
    plt.legend(title="Metric", loc='lower right')
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f'{h:.2f}', (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=8, xytext=(0, 2), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "08_model_comparison_bar.png"), dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/08_model_comparison_bar.png")

    # 2. Confusion Matrices Subplot
    num_models = len(results)
    cols = 3
    rows = int(np.ceil(num_models / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.flatten()
    
    for idx, item in enumerate(results):
        model_name = item['Model']
        model_file = item['Model_File']
        model_path = os.path.join("models", model_file)
        
        if not os.path.exists(model_path):
            continue
            
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=class_names, yticklabels=class_names, cbar=False)
        axes[idx].set_title(f"{model_name}", fontsize=12, fontweight='bold')
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].set_ylabel("True Label")
        
        print(f"\n========================================================")
        print(f" Classification Report: {model_name}")
        print(f"========================================================")
        print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
        
    # Hide unused subplots
    for i in range(num_models, len(axes)):
        fig.delaxes(axes[i])
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "09_confusion_matrices.png"), dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/09_confusion_matrices.png")

    # 3. Feature Importance (Random Forest & XGBoost)
    plt.figure(figsize=(12, 6))
    rf_path = os.path.join("models", "random_forest.joblib")
    if os.path.exists(rf_path):
        rf_model = joblib.load(rf_path)
        importances = rf_model.feature_importances_
        feature_names = X_test.columns
        
        feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)
        
        ax = sns.barplot(data=feat_df, x='Importance', y='Feature', hue='Feature', palette='mako', legend=False)
        plt.title("Feature Importance (Random Forest Classifier)", fontsize=14, fontweight='bold')
        plt.xlabel("Gini Importance", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        for p in ax.patches:
            w = p.get_width()
            ax.annotate(f'{w:.3f}', (w, p.get_y() + p.get_height() / 2.),
                        ha='left', va='center', fontsize=9, xytext=(4, 0), textcoords='offset points')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "10_feature_importances.png"), dpi=300)
        plt.close()
        print(f"Saved: {output_dir}/10_feature_importances.png")

    # 4. Multiclass ROC Curve (One-vs-Rest for Best Model)
    best_model_name = df_res.loc[df_res['Test_F1_Macro'].idxmax()]['Model']
    best_model_file = df_res.loc[df_res['Test_F1_Macro'].idxmax()]['Model_File']
    best_model_path = os.path.join("models", best_model_file)
    
    if os.path.exists(best_model_path):
        best_model = joblib.load(best_model_path)
        if hasattr(best_model, "predict_proba"):
            y_test_bin = label_binarize(y_test, classes=sorted(label_mapping.keys()))
            n_classes = y_test_bin.shape[1]
            y_score = best_model.predict_proba(X_test)
            
            plt.figure(figsize=(9, 6))
            colors = sns.color_palette("Set2", n_classes)
            
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=colors[i], lw=2,
                         label=f'Class {class_names[i]} (AUC = {roc_auc:.2f})')
                         
            plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title(f'Multiclass ROC Curves - {best_model_name} (OvR)', fontsize=14, fontweight='bold')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "11_roc_curves.png"), dpi=300)
            plt.close()
            print(f"Saved: {output_dir}/11_roc_curves.png")

    print("\n--- Model Evaluation Completed ---")

if __name__ == "__main__":
    run_evaluation()
