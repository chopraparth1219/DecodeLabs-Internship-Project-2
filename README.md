# DecodeLabs-Internship-Project-2
# Project 2: Fraud Detection Pipeline


## Objective

Build and tune a classification model to identify fraudulent transactions in a highly imbalanced dataset.  
This project demonstrates:

- Handling extreme class imbalance using SMOTE (Synthetic Minority Over‑sampling).
- Training two classification algorithms: Logistic Regression and Random Forest.
- Evaluation using **Precision, Recall, and ROC‑AUC** – accuracy is discarded.
- Hyperparameter tuning for Random Forest using GridSearchCV.
- A leak‑free pipeline that prevents data leakage during resampling and scaling.

## Dataset

- **File:** `data1.csv` (transaction data with columns: `OrderID`, `Date`, `customerID`, `Product`, `Quantity`, `UnitPrice`, `TotalPrice`).
- Because the original file does not contain a fraud label, a synthetic binary column `Class` was added (1 = fraud, 0 = legitimate) with a controlled imbalance (e.g., 80% legitimate, 20% fraud). In a real‑world project, a genuine labelled fraud dataset would be used.

## Key Challenges and Solutions

| Challenge | Solution |
|-----------|----------|
| Extreme class imbalance (fraud << legitimate) | SMOTE creates synthetic fraud samples via interpolation: `x_new = x_i + λ·(x_nn – x_i)`. |
| Misleading accuracy (e.g., 99% accuracy by predicting everything as legitimate) | Discard accuracy; use Precision, Recall, and ROC‑AUC. |
| Data leakage (applying SMOTE or scaling before train/test split) | Use `imblearn.pipeline.Pipeline` to apply SMOTE and scaler **only** on training folds. |
| Different scaling requirements | StandardScaler for Logistic Regression; no scaling for Random Forest (tree‑based models are scale‑invariant). |
| Hyperparameter tuning without leakage | Wrap the pipeline inside `GridSearchCV` with `roc_auc` scoring. |

## Pipeline Architecture (Leak‑Free)

### Logistic Regression Pipeline
- Scaling is necessary because logistic regression uses distance‑based regularisation.
- Both scaler and SMOTE are applied **inside** the pipeline – never on the full dataset before splitting.

### Random Forest Pipeline

- Random Forest is invariant to feature scale; no scaler is required.

### Hyperparameter Tuning
- `GridSearchCV` with `roc_auc` scoring and cross‑validation (`cv=2` for small dataset; higher values in production).
- The entire pipeline is passed to GridSearchCV, ensuring SMOTE is reapplied correctly on each validation fold.

## Technologies Used

| Library | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data loading and manipulation |
| `scikit-learn` | Train/test split, scalers, classifiers, metrics, GridSearchCV |
| `imbalanced-learn` | SMOTE and `imblearn.pipeline.Pipeline` |

## How to Run

1. Ensure `data1.csv` is in the same folder as the script `project2.py`.
2. Install required packages:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn
