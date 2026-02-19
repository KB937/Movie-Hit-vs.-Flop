# Movie-Hit-vs.-Flop-Classifier
The film industry is a high-risk business where investments often exceed $100 million. Predicting financial success solely based on "gut feeling" is insufficient. 
This project builds a **Risk Assessment Model** to predict whether a movie will be a **Hit** or a **Flop** *before* it enters production, using data available at the script/planning stage.

### The Problem
*   **High Sunk Costs:** Movies are "single-shot" products. A Flop results in massive financial loss.
*   **Goal:** Identify high-risk projects (Flops) to save investors money.

### Definition of Success
*   **🟢 Hit:** Return on Investment (ROI) $\ge$ 2.0 (Double the budget).
*   **🔴 Flop:** ROI < 1.0 (Lost money).
*   *(Movies with ROI between 1.0 and 2.0 were excluded to sharpen the classification boundary.)*

---

## 📊 Dataset & Features
**Source:** [TMDB 5000 Movie Dataset](https://www.kaggle.com/tmdb/tmdb-movie-metadata)
**Size:** ~5,000 Movies

To prevent **Data Leakage**, we strictly used **Pre-release Features** only:
| Feature | Description | Preprocessing |
| :--- | :--- | :--- |
| **Budget** | Production cost in USD | Log-transformed (`log1p`) to handle skewness. |
| **Genres** | Action, Comedy, Drama, etc. | Multi-hot encoded (a movie can have multiple). |
| **Production Company** | Warner Bros, Universal, etc. | Top 20 frequent companies encoded; others grouped. |
| **Release Month** | Month of release | Extracted to capture seasonality (e.g., Summer vs. January). |
| **Runtime** | Duration in minutes | Kept as numeric. |

---

## ⚙️ Methodology
### 1. Data Cleaning
*   Removed movies with $0 budget/revenue (missing data).
*   Filtered for valid feature films (Run time 40-240 mins).
*   Dropped duplicates.

### 2. Risk-Averse Modeling Strategy
Our primary goal is **Risk Reduction**. Missing a Flop (False Negative) is worse than missing a Hit.
*   **Aggressive Class Weights:** We penalized the model **5x more** for missing a Flop than for missing a Hit.
    *   `class_weight = {0: 5, 1: 1}` (where 0 is Flop).

### 3. Models Compared
*   **Logistic Regression:** Linear, interpretable baseline.
*   **Random Forest:** Non-linear ensemble, robust to outliers.
*   **XGBoost:** Gradient boosting, state-of-the-art for tabular data.

---

## 🏆 Key Results
We prioritized **Recall for Flops** (finding the disasters).

| Model | Hit F1-Score | Flop F1-Score | **Flop Recall** | Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **Logistic Regression** | 0.72 | **0.52** | **63%** | **✅ Best for Risk Assessment** |
| Random Forest | **0.82** | 0.20 | 12% | Too optimistic favoring Hits |
| XGBoost | **0.82** | 0.31 | 22% | Balanced but misses risks |

> **Insight:** Logistic Regression correctly flags **63%** of financial disasters, making it the safest tool for investors, even if it generates some false alarms.
