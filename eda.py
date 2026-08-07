import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(data_path="dataset.csv", output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"--- Loading Dataset from '{data_path}' ---")
    df = pd.read_csv(data_path)
    
    # Initial info
    print("\nDataset Shape (Raw):", df.shape)
    print("\nColumns & Data Types:")
    print(df.dtypes)
    
    # Cleaning steps
    # 1. Drop rows where Country is missing or invalid (e.g. OECS or empty rows)
    df = df.dropna(subset=['Country']).copy()
    df = df[df['Country'].str.strip() != 'OECS'].copy()
    
    # 2. Clean Region string column (fix 'Middle  East' double space)
    df['Region'] = df['Region'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # 3. Clean string columns whitespace
    df['Situation'] = df['Situation'].astype(str).str.strip()
    df['Country'] = df['Country'].astype(str).str.strip()
    df['ISO Code'] = df['ISO Code'].astype(str).str.strip()
    
    print("\nDataset Shape (Cleaned):", df.shape)
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    print("\nSummary Statistics of Numerical Features:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numeric_cols].describe().T[['mean', 'std', 'min', '50%', 'max']])
    
    # Set aesthetics
    sns.set_theme(style="whitegrid", palette="muted")
    
    # 1. Target Class Distribution
    plt.figure(figsize=(8, 5))
    situation_order = ['Good', 'Satisfactory', 'Problematic', 'Difficult', 'Very Serious']
    existing_order = [s for s in situation_order if s in df['Situation'].unique()]
    
    ax = sns.countplot(data=df, x='Situation', order=existing_order, palette='Spectral_r')
    plt.title("Distribution of Press Freedom Situations (2022)", fontsize=14, fontweight='bold')
    plt.xlabel("Situation", fontsize=12)
    plt.ylabel("Country Count", fontsize=12)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_situation_distribution.png"), dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/01_situation_distribution.png")

    # 2. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, square=True, linewidths=0.5)
    plt.title("Correlation Matrix of Numeric Indicators", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_correlation_heatmap.png"), dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/02_correlation_heatmap.png")

    # 3. Sub-scores Boxplot grouped by Situation
    score_cols = ['Politic Score', 'Economic Score', 'Legislative Score', 'Social Score', 'Security Score']
    df_melted = df.melt(id_vars=['Situation'], value_vars=score_cols, var_name='Score Type', value_name='Score')
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_melted, x='Score Type', y='Score', hue='Situation', hue_order=existing_order, palette='Spectral_r')
    plt.title("Distribution of Sub-scores across Situations", fontsize=14, fontweight='bold')
    plt.xlabel("Score Type", fontsize=12)
    plt.ylabel("Score (0 - 100)", fontsize=12)
    plt.legend(title="Situation", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_subscores_by_situation.png"), dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/03_subscores_by_situation.png")

    # 4. Global Score vs Security Score
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=df, x='Security Score', y='Global Score', hue='Situation', hue_order=existing_order,
                    style='Region', s=90, alpha=0.85, palette='Spectral_r')
    plt.title("Global Score vs Security Score by Region & Situation", fontsize=14, fontweight='bold')
    plt.xlabel("Security Score", fontsize=12)
    plt.ylabel("Global Score", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_global_vs_security_score.png"), dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/04_global_vs_security_score.png")

    # 5. Regional Scores Comparison
    plt.figure(figsize=(10, 6))
    regional_avg = df.groupby('Region')['Global Score'].mean().reset_index().sort_values(by='Global Score', ascending=False)
    ax = sns.barplot(data=regional_avg, x='Global Score', y='Region', palette='viridis')
    plt.title("Average Global Score by Region", fontsize=14, fontweight='bold')
    plt.xlabel("Mean Global Score", fontsize=12)
    plt.ylabel("Region", fontsize=12)
    for p in ax.patches:
        width = p.get_width()
        ax.annotate(f'{width:.2f}', (width, p.get_y() + p.get_height() / 2.),
                    ha='left', va='center', fontsize=10, xytext=(5, 0), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_regional_scores.png"), dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/05_regional_scores.png")

    # 6. Position 2022 Violin Plot by Situation
    plt.figure(figsize=(8, 5))
    sns.violinplot(data=df, x='Situation', y='Position 2022', order=existing_order, palette='Spectral_r', inner="quartile")
    plt.title("Position 2022 Distribution by Situation Group", fontsize=14, fontweight='bold')
    plt.xlabel("Situation", fontsize=12)
    plt.ylabel("Rank Position (Lower is Better)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "06_position_distribution_by_situation.png"), dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/06_position_distribution_by_situation.png")

    # 7. Safety Incidents (Killed & Imprisoned) by Region
    safety_cols = ['Journalist Killed', 'Media Workers Killed', 'Journalist Imprisoned', 'Media Workers Imprisoned']
    regional_safety = df.groupby('Region')[safety_cols].sum()
    
    regional_safety.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='YlOrRd')
    plt.title("Total Journalist Safety Incidents by Region", fontsize=14, fontweight='bold')
    plt.xlabel("Region", fontsize=12)
    plt.ylabel("Total Count", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Incident Type")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "07_journalists_safety_by_region.png"), dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/07_journalists_safety_by_region.png")

    print("\n--- EDA Completed Successfully ---")

if __name__ == "__main__":
    run_eda()
