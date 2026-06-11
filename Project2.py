import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

df = pd.read_csv('data1.csv', parse_dates=['Date'])

np.random.seed(42)
df['Class'] = np.random.choice([0, 1], size=len(df), p=[0.8, 0.2])

X = df.select_dtypes(include=[np.number]).drop('Class', axis=1)
y = df['Class']

print("Original class distribution:")
print(y.value_counts())
print(f"Fraud rate: {y.mean():.4%}")

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

logpipe = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42, k_neighbors=1)),
    ('clf', LogisticRegression(random_state=42, max_iter=1000))
])

rfpipe = ImbPipeline([
    ('smote', SMOTE(random_state=42, k_neighbors=1)),
    ('clf', RandomForestClassifier(random_state=42, n_estimators=100))
])

print("\nTraining Logistic Regression...")
logpipe.fit(Xtrain, ytrain)

print("Training Random Forest...")
rfpipe.fit(Xtrain, ytrain)

models = {
    'Logistic Regression': logpipe,
    'Random Forest': rfpipe
}

for name, model in models.items():
    ypred = model.predict(Xtest)
    yproba = model.predict_proba(Xtest)[:, 1]
    
    prec = precision_score(ytest, ypred)
    rec = recall_score(ytest, ypred)
    roc = roc_auc_score(ytest, yproba)
    
    print(f"\n{name} Results (Test Set):")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  ROC-AUC:   {roc:.4f}")
    print("\nClassification Report:")
    print(classification_report(ytest, ypred))
    print("Confusion Matrix:")
    print(confusion_matrix(ytest, ypred))

print("\n" + "="*60)
print("Hyperparameter Tuning for Random Forest (using ROC-AUC)")
print("="*60)

paramgrid = {
    'clf__n_estimators': [50, 100],
    'clf__max_depth': [None, 10],
    'clf__min_samples_split': [2, 5]
}

grid = GridSearchCV(
    rfpipe, paramgrid, cv=2, scoring='roc_auc', n_jobs=1, verbose=1
)

grid.fit(Xtrain, ytrain)

print(f"\nBest parameters: {grid.best_params_}")
print(f"Best CV ROC-AUC: {grid.best_score_:.4f}")

bestrf = grid.best_estimator_
ypredbest = bestrf.predict(Xtest)
yprobabest = bestrf.predict_proba(Xtest)[:, 1]

print("\nTuned Random Forest – Final Test Set Performance:")
print(f"  Precision: {precision_score(ytest, ypredbest):.4f}")
print(f"  Recall:    {recall_score(ytest, ypredbest):.4f}")
print(f"  ROC-AUC:   {roc_auc_score(ytest, yprobabest):.4f}")
