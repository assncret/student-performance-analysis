[README (1).md](https://github.com/user-attachments/files/27977747/README.1.md)
# 📊 Student Performance Analysis & Prediction

> End-to-end Data Science project applying supervised and unsupervised machine learning on student performance data.

---

## 🧠 Project Overview

This project was completed as a **Final Project for Fundamentals of Data Science** course at the Faculty of Artificial Intelligence, Badr University – Assiut (BUA).

It covers the full data science pipeline:
- Data loading & exploration
- Missing value handling & preprocessing
- Exploratory Data Analysis (EDA)
- Supervised Learning (Regression)
- Unsupervised Learning (Clustering)
- Visualization & interpretation

---

## 📁 Project Structure

```
student-performance-analysis/
│
├── analysis.py                              # Main analysis script (full pipeline)
├── synthetic_student_performance_hidden.csv # Supervised learning dataset
├── student_performance_dirty.csv            # Unsupervised learning dataset (dirty)
└── README.md
```

---

## 🔬 Methodology

### Part 1 – Supervised Learning (Regression)

**Goal:** Predict a student's `FinalExamScore` based on study habits and demographics.

| Step | Details |
|------|---------|
| Dataset | `synthetic_student_performance_hidden.csv` |
| Missing Values | Imputed using `SimpleImputer` (mean for numerical, mode for categorical) |
| Encoding | Label Encoding for categorical features |
| Split | 80% Train / 20% Test |
| Models | Linear Regression, Ridge, Lasso |
| Metric | MSE & RMSE |

**Best Model:** Lasso Regression (lowest RMSE)

---

### Part 2 – Unsupervised Learning (Clustering)

**Goal:** Segment students into groups based on performance patterns.

| Step | Details |
|------|---------|
| Dataset | `student_performance_dirty.csv` |
| Missing Values | Imputed using `SimpleImputer` |
| Scaling | `StandardScaler` |
| Algorithm | KMeans Clustering |
| Optimal K | 3 (determined via Elbow Method) |

**Cluster Interpretation:**
- 🔵 **Cluster 0** – Average students (moderate study hours, lower scores)
- 🟠 **Cluster 1** – Consistent learners (high attendance, good scores)
- 🟢 **Cluster 2** – Top performers (high study hours, highest scores)

---

## 📈 Visualizations Generated

- `missing_values.png` – Missing value bar charts (before imputation)
- `exam_score_distribution.png` – Distribution of Final Exam Score
- `hours_vs_score.png` – Study Hours vs Exam Score scatter plot
- `correlation_heatmap.png` – Feature correlation heatmap
- `model_comparison.png` – MSE & RMSE comparison across models
- `actual_vs_predicted.png` – Actual vs Predicted (Lasso)
- `major_distribution.png` – Student distribution by major (pie chart)
- `elbow_method.png` – Elbow curve for optimal K selection
- `kmeans_clusters.png` – KMeans cluster scatter plot
- `cluster_means.png` – Average feature values per cluster

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat)
![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat)

**Libraries used:**
`pandas` · `numpy` · `scikit-learn` · `matplotlib` · `seaborn`

**Models:**
`LinearRegression` · `Ridge` · `Lasso` · `KMeans`

**Preprocessing:**
`SimpleImputer` · `LabelEncoder` · `StandardScaler` · `train_test_split`

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/assncret/student-performance-analysis.git
cd student-performance-analysis

# 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# 3. Run the analysis
python analysis.py
```

---

## 👤 Author

**Abdelrahman Nasser Mahmoud Mosbah**
Faculty of Artificial Intelligence – Badr University, Assiut (BUA)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/musbh)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/assncret)
