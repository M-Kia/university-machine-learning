```python
import os
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier,
    BaggingClassifier,
    StackingClassifier
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
```


```python
# Load data
data = pd.read_csv(os.path.join(".", 'alzheimers_disease_data.csv'))

data = data.drop(columns=['DoctorInCharge'])
```


```python

# Normalize features
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.drop(columns=['Diagnosis']))

# Separate features and targets
X = data.drop(columns=['PatientID', 'Diagnosis'])
y = data['Diagnosis']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42)
```


```python
# Define individual base models
cart = DecisionTreeClassifier(random_state=42)
c45 = DecisionTreeClassifier(criterion='entropy', random_state=42)
rf = RandomForestClassifier(random_state=42)
gbm = GradientBoostingClassifier(random_state=42)
ada = AdaBoostClassifier(random_state=42)
xgb = XGBClassifier(random_state=42, use_label_encoder=True, eval_metric='logloss')
lgbm = LGBMClassifier(random_state=42)

# Define additional models for stacking
svc = SVC(probability=True, random_state=42)
knn = KNeighborsClassifier()
logreg = LogisticRegression()

# Create ensemble methods
voting_clf = VotingClassifier(
    estimators=[
        ('rf', rf), ('gbm', gbm), ('ada', ada), ('xgb', xgb), ('lgbm', lgbm)
    ],
    voting='soft'
)

bagging_clf = BaggingClassifier(estimator=rf, n_estimators=10, random_state=42)

stacking_clf = StackingClassifier(
    estimators=[
        ('cart', cart),
        ('svc', svc),
        ('knn', knn),
        ('rf', rf)
    ],
    final_estimator=logreg
)

```


```python

# Dictionary of all models to evaluate
models = {
    'CART': cart,
    'C4.5': c45,
    'Random Forest': rf,
    'Gradient Boosting': gbm,
    'AdaBoost': ada,
    'XGBoost': xgb,
    'LightGBM': lgbm,
    'Voting Ensemble': voting_clf,
    'Bagging Ensemble': bagging_clf,
    'Stacking Ensemble': stacking_clf
}

results = []
```


```python
# Evaluate models
for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred, output_dict=True)

        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Recall': recall,
            'Precision': precision,
            'F1 Score': f1,
            'Classification Report': classification_rep
        })


```

    c:\Users\mohammad-hossein\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\ensemble\_weight_boosting.py:527: FutureWarning: The SAMME.R algorithm (the default) is deprecated and will be removed in 1.6. Use the SAMME algorithm to circumvent this warning.
      warnings.warn(
    c:\Users\mohammad-hossein\AppData\Local\Programs\Python\Python313\Lib\site-packages\xgboost\core.py:158: UserWarning: [22:41:55] WARNING: C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0c55ff5f71b100e98-1\xgboost\xgboost-ci-windows\src\learner.cc:740: 
    Parameters: { "use_label_encoder" } are not used.
    
      warnings.warn(smsg, UserWarning)
    

    [LightGBM] [Info] Number of positive: 532, number of negative: 972
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000220 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 3282
    [LightGBM] [Info] Number of data points in the train set: 1504, number of used features: 32
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.353723 -> initscore=-0.602712
    [LightGBM] [Info] Start training from score -0.602712
    

    c:\Users\mohammad-hossein\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\ensemble\_weight_boosting.py:527: FutureWarning: The SAMME.R algorithm (the default) is deprecated and will be removed in 1.6. Use the SAMME algorithm to circumvent this warning.
      warnings.warn(
    c:\Users\mohammad-hossein\AppData\Local\Programs\Python\Python313\Lib\site-packages\xgboost\core.py:158: UserWarning: [22:41:56] WARNING: C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0c55ff5f71b100e98-1\xgboost\xgboost-ci-windows\src\learner.cc:740: 
    Parameters: { "use_label_encoder" } are not used.
    
      warnings.warn(smsg, UserWarning)
    

    [LightGBM] [Info] Number of positive: 532, number of negative: 972
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000229 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 3282
    [LightGBM] [Info] Number of data points in the train set: 1504, number of used features: 32
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.353723 -> initscore=-0.602712
    [LightGBM] [Info] Start training from score -0.602712
    


```python

# Save results to JSON and Excel files
results_df = pd.DataFrame(results)
results_json_path = os.path.join(
    ".", "ensemble_model_results.json")
results_excel_path = os.path.join(
    ".", "ensemble_comparison_results.xlsx")

os.makedirs(os.path.dirname(results_json_path), exist_ok=True)
results_df.to_json(results_json_path, orient='records', indent=4)
results_df.to_excel(results_excel_path, index=False)

print("Evaluation completed. Results saved.")
```

    Evaluation completed. Results saved.
    
