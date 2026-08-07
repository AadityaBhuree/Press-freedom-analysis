# World Press Freedom Index — Machine Learning Analysis

An end-to-end Machine Learning pipeline for exploratory data analysis, feature engineering, and predictive modeling on the **World Press Freedom Index 2022** dataset (`dataset.csv`).

The goal is to predict a country's press freedom **`Situation`** classification (5 ordinal categories: *Good*, *Satisfactory*, *Problematic*, *Difficult*, *Very Serious*) based on political, economic, legislative, social, security, and safety indicator metrics.

---

## 📁 Repository Structure

```
Press-freedom-analysis/
├── dataset.csv              # Raw World Press Freedom Index 2022 dataset
├── requirements.txt         # Required Python packages
├── README.md                # Project documentation
├── .gitignore               # Git ignore configuration
├── eda.py                   # Exploratory Data Analysis & visual generation
├── preprocessing.py         # Data cleaning, feature engineering, and train/test split
├── train.py                 # Multi-model training and 5-fold cross-validation
├── evaluate.py              # Performance evaluation, confusion matrices & ROC curves
├── plots/                   # Saved visualization plots (generated)
├── models/                  # Saved trained models, encoders, and metrics (generated)
├── X_train.csv              # Preprocessed training features (generated)
├── X_test.csv               # Preprocessed testing features (generated)
├── y_train.csv              # Training target labels (generated)
└── y_test.csv               # Testing target labels (generated)
```

---

## 📊 Dataset Overview

- **Source**: World Press Freedom Index 2022 dataset (181 countries).
- **Features**:
  - `Global Score`: Overall press freedom score (0 to 100).
  - `Politic Score`, `Economic Score`, `Legislative Score`, `Social Score`, `Security Score`: Dimension sub-scores.
  - `Position 2022`, `Position 2021`: World ranking positions.
  - `Journalist Killed`, `Media Workers Killed`, `Journalist Imprisoned`, `Media Workers Imprisoned`: Safety statistics.
  - `Region`: Geographic region.
- **Target Variable**: `Situation`
  - Classes: `Good` (0), `Satisfactory` (1), `Problematic` (2), `Difficult` (3), `Very Serious` (4).

---

## 🛠️ Feature Engineering

`preprocessing.py` extracts new engineered signals:
1. `Position_Change`: `Position 2021` − `Position 2022` (positive values indicate rank improvement).
2. `Press_Danger_Index`: Total combined count of killed and imprisoned journalists and media workers.
3. `Score_Variance`: Variance across the 5 sub-score indicators per country (measures internal indicator imbalance).
4. `Score_Range` & `Score_Min`: Extreme values of sub-scores.

---

## 🤖 Models Compared

1. **Logistic Regression** (Linear baseline)
2. **K-Nearest Neighbors (KNN)** (Distance-based classification)
3. **Support Vector Classifier (SVC)** (RBF kernel)
4. **Random Forest Classifier** (Ensemble tree-based)
5. **XGBoost Classifier** (Gradient boosted decision trees)

---

## 🚀 How to Run the Project

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run Exploratory Data Analysis

Generates statistical summaries and saves 7 EDA charts in the `plots/` folder:

```bash
python eda.py
```

### Step 3: Run Data Preprocessing

Cleans data, engineers features, encodes target labels, scales features, and creates train/test splits (`X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`):

```bash
python preprocessing.py
```

### Step 4: Model Training & Cross-Validation

Trains all 5 models using 5-fold Stratified Cross Validation and saves model artifacts to `models/`:

```bash
python train.py
```

### Step 5: Model Evaluation & Visualizations

Generates model comparison metrics, confusion matrices, feature importance charts, and ROC curves in `plots/`:

```bash
python evaluate.py
```

---

## 📈 Outputs & Results

After running all scripts:
- **`plots/`** will contain all generated visualizations (`01_situation_distribution.png` through `11_roc_curves.png`).
- **`models/`** will contain saved `.joblib` model binaries, `scaler.joblib`, `region_encoder.joblib`, and `model_results.json`.
