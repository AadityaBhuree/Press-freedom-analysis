"""
Master execution entrypoint for Press Freedom Analysis ML pipeline.
Runs EDA, Preprocessing, Model Training & Cross-Validation, and Evaluation in sequence.
"""

from eda import run_eda
from preprocessing import run_preprocessing
from train import run_training
from evaluate import run_evaluation

def main():
    print("========================================================================")
    print("        STARTING PRESS FREEDOM ANALYSIS MACHINE LEARNING PIPELINE        ")
    print("========================================================================\n")
    
    # Step 1: Exploratory Data Analysis
    print(">>> STEP 1: Running Exploratory Data Analysis (eda.py)...")
    run_eda()
    
    # Step 2: Feature Engineering & Preprocessing
    print("\n>>> STEP 2: Running Data Preprocessing & Feature Engineering (preprocessing.py)...")
    run_preprocessing()
    
    # Step 3: Model Training & Cross Validation
    print("\n>>> STEP 3: Training & Cross-Validating Classifiers (train.py)...")
    run_training()
    
    # Step 4: Model Evaluation & Visualizations
    print("\n>>> STEP 4: Evaluating Models & Generating Plots (evaluate.py)...")
    run_evaluation()
    
    print("\n========================================================================")
    print("         PRESS FREEDOM ML PIPELINE EXECUTED SUCCESSFULLY!               ")
    print("========================================================================")

if __name__ == "__main__":
    main()
