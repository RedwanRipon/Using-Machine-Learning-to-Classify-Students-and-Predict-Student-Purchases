# Using-Machine-Learning-to-Classify-Students-and-Predict-Student-Purchases
## 📌 Project Overview
This project applies machine learning classification techniques to predict whether students on an online learning platform will upgrade from a **free plan(0)** to a **paid plan(1)**.  

The dataset (an anonymized excerpt of real engagement data) contains features such as:
- `days_on_platform` – number of days active on the platform  
- `minutes_watched` – total video content consumed  
- `courses_started` – number of courses started  
- `practice_exams_started` – number of exams started (dropped due to multicollinearity)  
- `practice_exams_passed` – number of exams passed  
- `minutes_spent_on_exams` – time spent in exams  
- `student_country` – encoded user country  

The **target variable** is:
- `purchased` → `1` if the student upgraded to paid, `0` otherwise.

---

## 🎯 Business Objective
Accurately predicting which students are likely to upgrade enables the company to:
- Target students with promotions or special offers  
- Allocate marketing budgets more effectively  
- Increase conversion rates and revenue  

---

## 🛠️ Methods & Models
1. **Data Cleaning & Preprocessing**
   - Outlier detection and removal  
   - Variance Inflation Factor (VIF) to drop highly collinear features  
   - Ordinal encoding for categorical feature `student_country`  
   - Scaling (StandardScaler / MinMaxScaler) depending on the model  

2. **Models Trained**
   - Logistic Regression (statsmodels + sklearn)  
   - k-Nearest Neighbors (KNN)  
   - Support Vector Machine (SVM, RBF kernel)  
   - Decision Tree (with pruning via `ccp_alpha`)  
   - Random Forest  
   - Voting Classifier (ensemble of all models)  

3. **Evaluation Metrics**
   - Confusion Matrices  
   - Precision, Recall, F1-score  
   - ROC-AUC  

---

## 📊 Key Results
- Logistic Regression: Pseudo R² ≈ 0.526, significant predictors identified  
- KNN: Best params → `n_neighbors=8`, `weights='distance'`  
- SVM: Best params → `kernel='rbf'`, `C=10`, `gamma='scale'`  
- Decision Tree: Best `ccp_alpha = 0.003`  
- Voting Classifier: Most robust predictions, balancing recall & precision  

---

## 🚀 How to Run
1. Clone this repository and install requirements:
   ```bash
   git clone <repo_url>
   cd <repo_folder>
   pip install -r requirements.txt
   ```

2. Open the notebook:
   ```bash
   jupyter notebook "Machine Learning Project.ipynb"
   ```

3. Run cells in sequence:
   - Preprocess data  
   - Train models  
   - Evaluate predictions  

---

## 🔮 Predicting New Students
```python
pred = voting_clf.predict(new_data)
prob = voting_clf.predict_proba(new_data)[:,1]
```

- `pred = 1` → likely to **buy**  
- `pred = 0` → likely to stay **free**  

---

## 📂 Repository Structure
```
├── Machine Learning Project.ipynb         # Main notebook
├── ml_datasource.csv                      # Dataset (anonymized)
├── README.md                              # Documentation
└── requirements.txt                       # Dependencies
```

---

## 📈 Future Work
- Use SMOTE/class weighting to fix class imbalance  
- Collect additional engagement features  
- Hyperparameter tuning with wider search spaces  
- Deploy model via API or web app  

---

## 📜 License
For educational purposes only.  
