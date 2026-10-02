import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(
    page_title="World Press Freedom ML Dashboard",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📰 World Press Freedom Index — Interactive ML Dashboard")
st.markdown("Predicting and analyzing press freedom classification across 181 countries using machine learning.")

@st.cache_data
def load_dataset():
    df = pd.read_csv("dataset.csv")
    df = df.dropna(subset=['Country']).copy()
    df = df[df['Country'].str.strip() != 'OECS'].copy()
    df['Region'] = df['Region'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
    df['Situation'] = df['Situation'].astype(str).str.strip()
    return df

@st.cache_resource
def load_models_and_encoders():
    models = {}
    model_files = {
        "Random Forest": "random_forest.joblib",
        "XGBoost": "xgboost.joblib",
        "Support Vector Machine": "support_vector_machine.joblib",
        "K-Nearest Neighbors": "k-nearest_neighbors.joblib",
        "Logistic Regression": "logistic_regression.joblib"
    }
    for name, file in model_files.items():
        path = os.path.join("models", file)
        if os.path.exists(path):
            models[name] = joblib.load(path)
            
    scaler = joblib.load(os.path.join("models", "scaler.joblib")) if os.path.exists(os.path.join("models", "scaler.joblib")) else None
    region_le = joblib.load(os.path.join("models", "region_encoder.joblib")) if os.path.exists(os.path.join("models", "region_encoder.joblib")) else None
    
    with open(os.path.join("models", "label_mapping.json"), "r") as f:
        label_map = json.load(f)
    label_map = {int(k): v for k, v in label_map.items()}
    
    return models, scaler, region_le, label_map

df = load_dataset()
models, scaler, region_le, label_map = load_models_and_encoders()

tab1, tab2, tab3, tab4 = st.tabs(["📊 Exploratory Data Analysis", "🔮 Live Country Predictor", "📈 Model Comparison", "📄 Dataset Viewer"])

# TAB 1: EDA
with tab1:
    st.header("Exploratory Data Analysis")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Countries", len(df))
    col2.metric("Mean Global Score", f"{df['Global Score'].mean():.2f}")
    col3.metric("Safest Country", df.loc[df['Global Score'].idxmax()]['Country'])
    col4.metric("Most Dangerous Country", df.loc[df['Global Score'].idxmin()]['Country'])
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Situation Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x='Situation', order=['Good', 'Satisfactory', 'Problematic', 'Difficult', 'Very Serious'], palette='Spectral_r', ax=ax)
        plt.title("Countries by Situation")
        plt.xticks(rotation=30)
        st.pyplot(fig)
        plt.close()
        
    with c2:
        st.subheader("Regional Score Averages")
        fig, ax = plt.subplots(figsize=(6, 4))
        avg_reg = df.groupby('Region')['Global Score'].mean().reset_index().sort_values('Global Score', ascending=False)
        sns.barplot(data=avg_reg, x='Global Score', y='Region', palette='viridis', ax=ax)
        plt.title("Mean Global Score by Region")
        st.pyplot(fig)
        plt.close()

# TAB 2: Live Predictor
with tab2:
    st.header("Real-Time Press Freedom Situation Predictor")
    st.write("Select a country or input custom metrics to predict its Press Freedom Situation classification.")
    
    mode = st.radio("Prediction Mode", ["Select Existing Country", "Custom Indicators Input"], horizontal=True)
    
    if mode == "Select Existing Country":
        selected_country = st.selectbox("Select Country", df['Country'].unique())
        country_data = df[df['Country'] == selected_country].iloc[0]
        
        st.subheader(f"Current Record for {selected_country}")
        st.write(f"**Actual Situation**: `{country_data['Situation']}` | **2022 Rank**: #{country_data['Position 2022']} | **Region**: {country_data['Region']}")
        
        pol_score = float(country_data['Politic Score'])
        econ_score = float(country_data['Economic Score'])
        leg_score = float(country_data['Legislative Score'])
        soc_score = float(country_data['Social Score'])
        sec_score = float(country_data['Security Score'])
        jk = float(country_data['Journalist Killed'])
        mwk = float(country_data['Media Workers Killed'])
        ji = float(country_data['Journalist Imprisoned'])
        mwi = float(country_data['Media Workers Imprisoned'])
        region = country_data['Region']
    else:
        c1, c2, c3 = st.columns(3)
        region = c1.selectbox("Region", sorted(df['Region'].unique()))
        pol_score = c2.slider("Politic Score", 0.0, 100.0, 60.0)
        econ_score = c3.slider("Economic Score", 0.0, 100.0, 50.0)
        
        c4, c5, c6 = st.columns(3)
        leg_score = c4.slider("Legislative Score", 0.0, 100.0, 65.0)
        soc_score = c5.slider("Social Score", 0.0, 100.0, 70.0)
        sec_score = c6.slider("Security Score", 0.0, 100.0, 75.0)
        
        c7, c8, c9, c10 = st.columns(4)
        jk = c7.number_input("Journalist Killed", 0, 50, 0)
        mwk = c8.number_input("Media Workers Killed", 0, 50, 0)
        ji = c9.number_input("Journalist Imprisoned", 0, 100, 0)
        mwi = c10.number_input("Media Workers Imprisoned", 0, 50, 0)

    # Derived Features
    press_danger = jk + mwk + ji + mwi
    subscores = [pol_score, econ_score, leg_score, soc_score, sec_score]
    score_var = float(np.var(subscores))
    score_range = float(np.max(subscores) - np.min(subscores))
    score_min = float(np.min(subscores))
    
    region_encoded = int(region_le.transform([region])[0]) if region_le else 0
    
    # Feature vector matching leak-free preprocessed training set (14 exogenous indicators)
    feature_names = [
        'Region', 'Politic Score', 'Economic Score', 'Legislative Score',
        'Social Score', 'Security Score', 'Journalist Killed',
        'Media Workers Killed', 'Journalist Imprisoned', 'Media Workers Imprisoned',
        'Press_Danger_Index', 'Score_Variance', 'Score_Range', 'Score_Min'
    ]
    input_vector = pd.DataFrame([[
        region_encoded, pol_score, econ_score, leg_score, soc_score, sec_score,
        jk, mwk, ji, mwi, press_danger, score_var, score_range, score_min
    ]], columns=feature_names)
    
    input_scaled = pd.DataFrame(scaler.transform(input_vector), columns=feature_names) if scaler else input_vector
    
    selected_model_name = st.selectbox("Select Classification Model", list(models.keys()))
    
    if st.button("🔮 Run Model Prediction", type="primary"):
        chosen_model = models[selected_model_name]
        pred_label_idx = chosen_model.predict(input_scaled)[0]
        predicted_situation = label_map[pred_label_idx]
        
        st.success(f"Predicted Press Freedom Situation ({selected_model_name}): **{predicted_situation}**")
        
        if hasattr(chosen_model, "predict_proba"):
            probs = chosen_model.predict_proba(input_scaled)[0]
            prob_df = pd.DataFrame({
                "Situation": [label_map[i] for i in range(len(probs))],
                "Probability": probs
            })
            st.write("Class Probabilities:")
            st.bar_chart(prob_df.set_index("Situation"))

# TAB 3: Model Comparison
with tab3:
    st.header("Model Evaluation & Comparison")
    if os.path.exists(os.path.join("models", "model_results.json")):
        with open(os.path.join("models", "model_results.json"), "r") as f:
            res = json.load(f)
        res_df = pd.DataFrame(res)
        st.dataframe(res_df[['Model', 'CV_Accuracy', 'Test_Accuracy', 'Test_Precision_Macro', 'Test_Recall_Macro', 'Test_F1_Macro']], use_container_width=True)
        
        st.subheader("Saved Plot Visualizations")
        p1, p2 = st.columns(2)
        if os.path.exists("plots/08_model_comparison_bar.png"):
            p1.image("plots/08_model_comparison_bar.png", caption="Model Comparison")
        if os.path.exists("plots/09_confusion_matrices.png"):
            p2.image("plots/09_confusion_matrices.png", caption="Confusion Matrices")
        
        p3, p4 = st.columns(2)
        if os.path.exists("plots/10_feature_importances.png"):
            p3.image("plots/10_feature_importances.png", caption="Feature Importances")
        if os.path.exists("plots/11_roc_curves.png"):
            p4.image("plots/11_roc_curves.png", caption="ROC Curves")

# TAB 4: Dataset Viewer
with tab4:
    st.header("Press Freedom Dataset 2022")
    st.dataframe(df, use_container_width=True)
