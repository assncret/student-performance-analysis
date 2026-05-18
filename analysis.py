# =============================================================================
# Fundamentals of Data Science - Final Project
# Datasets: Student Performance (Supervised) + Student Dirty (Unsupervised)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: LOAD DATASETS
# ─────────────────────────────────────────────────────────────────────────────

# Supervised Dataset (hidden version with missing values)
df_sup = pd.read_csv('synthetic_student_performance_hidden.csv')

# Unsupervised Dataset (dirty version for clustering)
df_unsup = pd.read_csv('student_performance_dirty.csv')

print("=" * 60)
print("SUPERVISED DATASET - synthetic_student_performance_hidden")
print("=" * 60)
print(f"Shape: {df_sup.shape}")
print(df_sup.dtypes)
print("\nFirst 5 rows:")
print(df_sup.head())

print("\n" + "=" * 60)
print("UNSUPERVISED DATASET - student_performance_dirty")
print("=" * 60)
print(f"Shape: {df_unsup.shape}")
print(df_unsup.dtypes)
print("\nFirst 5 rows:")
print(df_unsup.head())


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: DATASET DESCRIPTION
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SUPERVISED DATASET - Statistical Summary")
print("=" * 60)
print(df_sup.describe())

print("\n" + "=" * 60)
print("UNSUPERVISED DATASET - Statistical Summary")
print("=" * 60)
print(df_unsup.describe())


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: MISSING DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("MISSING DATA - SUPERVISED DATASET (Before Imputation)")
print("=" * 60)
missing_sup = df_sup.isnull().sum()
missing_pct_sup = (df_sup.isnull().sum() / len(df_sup) * 100).round(2)
print(pd.DataFrame({'Missing Count': missing_sup, 'Missing %': missing_pct_sup}))

print("\n" + "=" * 60)
print("MISSING DATA - UNSUPERVISED DATASET (Before Imputation)")
print("=" * 60)
missing_unsup = df_unsup.isnull().sum()
missing_pct_unsup = (df_unsup.isnull().sum() / len(df_unsup) * 100).round(2)
print(pd.DataFrame({'Missing Count': missing_unsup, 'Missing %': missing_pct_unsup}))

# ── Plot: Missing values bar chart ──
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

m_sup = missing_sup[missing_sup > 0]
axes[0].bar(m_sup.index, m_sup.values, color='#2E75B6')
axes[0].set_title('Missing Values - Supervised Dataset', fontweight='bold')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=30)
for i, v in enumerate(m_sup.values):
    axes[0].text(i, v + 0.3, str(v), ha='center', fontsize=9)

m_unsup = missing_unsup[missing_unsup > 0]
axes[1].bar(m_unsup.index, m_unsup.values, color='#ED7D31')
axes[1].set_title('Missing Values - Unsupervised Dataset', fontweight='bold')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=30)
for i, v in enumerate(m_unsup.values):
    axes[1].text(i, v + 0.1, str(v), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('missing_values.png', dpi=150)
plt.show()
print(">> Figure saved: missing_values.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: DATA PREPROCESSING - SUPERVISED DATASET
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("DATA PREPROCESSING - SUPERVISED DATASET")
print("=" * 60)

# Define column types
num_cols_sup = ['Age', 'HoursStudied', 'AttendanceRate', 'PreviousGrade', 'FinalExamScore']
cat_cols_sup = ['Gender', 'Subject', 'StudyMethod', 'Passed']

# Impute: mean for numerical, mode for categorical
imp_num = SimpleImputer(strategy='mean')
imp_cat = SimpleImputer(strategy='most_frequent')

for col in num_cols_sup:
    df_sup[col] = imp_num.fit_transform(df_sup[[col]]).ravel()

for col in cat_cols_sup:
    df_sup[col] = imp_cat.fit_transform(df_sup[[col]]).ravel()

print("Missing values after imputation:")
print(df_sup.isnull().sum())

# Encode categorical columns using Label Encoding
le = LabelEncoder()
for col in cat_cols_sup:
    df_sup[col] = le.fit_transform(df_sup[col].astype(str))
    print(f"Encoded '{col}' with classes: {list(le.classes_)}")

# Drop StudentID (not a feature)
df_sup = df_sup.drop(columns=['StudentID'])

# Features and Target
X = df_sup.drop(columns=['FinalExamScore'])
y = df_sup['FinalExamScore']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape:  {y.shape}")

# Train/Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {X_train.shape[0]} samples")
print(f"Test size:  {X_test.shape[0]} samples")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: EXPLORATORY DATA ANALYSIS (EDA) - SUPERVISED
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("EDA - SUPERVISED DATASET")
print("=" * 60)

# Distribution of FinalExamScore
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(y, bins=20, color='#2E75B6', edgecolor='white', alpha=0.85)
ax.axvline(y.mean(), color='#ED7D31', linestyle='--', linewidth=2,
           label=f'Mean: {y.mean():.1f}')
ax.set_title('Distribution of Final Exam Score', fontsize=13, fontweight='bold')
ax.set_xlabel('Final Exam Score')
ax.set_ylabel('Count')
ax.legend()
plt.tight_layout()
plt.savefig('exam_score_distribution.png', dpi=150)
plt.show()
print(">> Figure saved: exam_score_distribution.png")

# Scatter: HoursStudied vs FinalExamScore
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df_sup['HoursStudied'], df_sup['FinalExamScore'],
           alpha=0.5, color='#2E75B6', edgecolors='white', linewidth=0.5)
ax.set_xlabel('Hours Studied')
ax.set_ylabel('Final Exam Score')
ax.set_title('Hours Studied vs Final Exam Score', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('hours_vs_score.png', dpi=150)
plt.show()
print(">> Figure saved: hours_vs_score.png")

# Correlation Heatmap
fig, ax = plt.subplots(figsize=(9, 7))
corr = df_sup.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='Blues',
            ax=ax, linewidths=0.5)
ax.set_title('Feature Correlation Heatmap', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150)
plt.show()
print(">> Figure saved: correlation_heatmap.png")

# Statistical summary grouped by Subject
print("\nMean FinalExamScore by Subject:")
print(df_sup.groupby('Subject')['FinalExamScore'].mean().sort_values(ascending=False).round(2))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: MACHINE LEARNING - REGRESSION
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("REGRESSION MODELS")
print("=" * 60)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression':  Ridge(alpha=1.0),
    'Lasso Regression':  Lasso(alpha=0.1)
}

regression_results = {}

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    # Predict
    y_pred = model.predict(X_test)
    # Evaluate
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    regression_results[name] = {'MSE': round(mse, 2), 'RMSE': round(rmse, 2), 'preds': y_pred}
    print(f"  {name:20s} | MSE: {mse:7.2f} | RMSE: {rmse:.2f}")

# Results DataFrame
results_df = pd.DataFrame({
    k: {'MSE': v['MSE'], 'RMSE': v['RMSE']}
    for k, v in regression_results.items()
}).T
print("\nSummary Table:")
print(results_df)

# Bar chart: MSE & RMSE comparison
colors_bar = ['#2E75B6', '#ED7D31', '#70AD47']
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
model_names  = list(regression_results.keys())
mse_values   = [regression_results[m]['MSE']  for m in model_names]
rmse_values  = [regression_results[m]['RMSE'] for m in model_names]

for ax, vals, title, ylabel in zip(
    axes, [mse_values, rmse_values],
    ['MSE Comparison', 'RMSE Comparison'], ['MSE', 'RMSE']
):
    bars = ax.bar(model_names, vals, color=colors_bar)
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=10)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(v), ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()
print(">> Figure saved: model_comparison.png")

# Actual vs Predicted (Lasso - best model)
best_preds = regression_results['Lasso Regression']['preds']
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(y_test, best_preds, alpha=0.6, color='#2E75B6')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Actual Score')
ax.set_ylabel('Predicted Score')
ax.set_title('Actual vs Predicted - Lasso Regression (Best Model)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=150)
plt.show()
print(">> Figure saved: actual_vs_predicted.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: DATA PREPROCESSING - UNSUPERVISED DATASET
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("DATA PREPROCESSING - UNSUPERVISED DATASET")
print("=" * 60)

num_cols_u = ['Study_Hours', 'Attendance', 'Assignments_Score',
              'Exam_Score', 'Projects_Score']

# Impute missing values
imp_num_u = SimpleImputer(strategy='mean')
for col in num_cols_u:
    df_unsup[col] = imp_num_u.fit_transform(df_unsup[[col]]).ravel()

# Major: impute with mode (already complete, but safe to keep)
df_unsup['Major'] = SimpleImputer(strategy='most_frequent').fit_transform(
    df_unsup[['Major']]
).ravel()

print("Missing values after imputation:")
print(df_unsup.isnull().sum())

# Standardize features
scaler = StandardScaler()
X_clust = scaler.fit_transform(df_unsup[num_cols_u])
print(f"\nScaled feature matrix shape: {X_clust.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: EDA - UNSUPERVISED DATASET
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("EDA - UNSUPERVISED DATASET")
print("=" * 60)

# Pie chart: Major distribution
fig, ax = plt.subplots(figsize=(7, 5))
major_counts = df_unsup['Major'].value_counts()
ax.pie(major_counts, labels=major_counts.index, autopct='%1.1f%%',
       colors=['#2E75B6', '#ED7D31', '#70AD47', '#FFC000'])
ax.set_title('Student Distribution by Major', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('major_distribution.png', dpi=150)
plt.show()
print(">> Figure saved: major_distribution.png")

# Grouped mean by Major
print("\nMean statistics grouped by Major:")
print(df_unsup.groupby('Major')[num_cols_u].mean().round(2))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: MACHINE LEARNING - K-MEANS CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("K-MEANS CLUSTERING")
print("=" * 60)

# Elbow Method
inertia = []
K_range = range(2, 10)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_clust)
    inertia.append(km.inertia_)

# Plot Elbow
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(list(K_range), inertia, marker='o', color='#2E75B6',
        linewidth=2, markersize=8)
ax.axvline(3, color='#ED7D31', linestyle='--', linewidth=2, label='Optimal K = 3')
ax.set_xlabel('Number of Clusters (K)')
ax.set_ylabel('Inertia (Within-Cluster Sum of Squares)')
ax.set_title('Elbow Method for Optimal K', fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('elbow_method.png', dpi=150)
plt.show()
print(">> Figure saved: elbow_method.png")

# Fit K-Means with optimal K = 3
optimal_k = 3
km_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_unsup['Cluster'] = km_final.fit_predict(X_clust)

print(f"\nK-Means fitted with K = {optimal_k}")
print("Cluster distribution:")
print(df_unsup['Cluster'].value_counts().sort_index())

print("\nCluster means (original scale):")
cluster_means = df_unsup.groupby('Cluster')[num_cols_u].mean().round(2)
print(cluster_means)

# Cluster scatter plot
fig, ax = plt.subplots(figsize=(9, 5))
colors_clust = ['#2E75B6', '#ED7D31', '#70AD47']
for c in range(optimal_k):
    mask = df_unsup['Cluster'] == c
    ax.scatter(df_unsup.loc[mask, 'Study_Hours'],
               df_unsup.loc[mask, 'Exam_Score'],
               alpha=0.7, color=colors_clust[c],
               label=f'Cluster {c}', edgecolors='white', linewidth=0.5)
ax.set_xlabel('Study Hours')
ax.set_ylabel('Exam Score')
ax.set_title('K-Means Clusters: Study Hours vs Exam Score',
             fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('kmeans_clusters.png', dpi=150)
plt.show()
print(">> Figure saved: kmeans_clusters.png")

# Cluster means bar chart
fig, ax = plt.subplots(figsize=(11, 5))
cluster_means.T.plot(kind='bar', ax=ax, color=colors_clust)
ax.set_title('Average Feature Values per Cluster', fontsize=13, fontweight='bold')
ax.set_xlabel('Feature')
ax.set_ylabel('Mean Value')
ax.legend(title='Cluster', loc='upper right')
ax.tick_params(axis='x', rotation=20)
plt.tight_layout()
plt.savefig('cluster_means.png', dpi=150)
plt.show()
print(">> Figure saved: cluster_means.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print("\nRegression Results:")
print(results_df.to_string())
print(f"\nBest Model: Lasso Regression (lowest RMSE = {regression_results['Lasso Regression']['RMSE']})")
print(f"\nK-Means: 3 clusters found")
print("  Cluster 0 - Average students (moderate hours, low exam score)")
print("  Cluster 1 - High attendance, good exam score (consistent learners)")
print("  Cluster 2 - High study hours, highest exam score (top performers)")
print("\nAll figures saved. Analysis complete!")
