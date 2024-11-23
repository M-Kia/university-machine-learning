import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
import xgboost as xgb
# from lightgbm
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, recall_score, precision_score, f1_score
from imblearn.over_sampling import SMOTE
from openpyxl import Workbook

def main():
  # Load data
  data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'alzheimers_disease_data.csv'))
  
  data = data.drop(columns=['DoctorInCharge'])
  # Replace Ethnicity data with meaningful data
  
  # Translating from numbers to words
  # 0 => Caucasian
  # 1 => African American
  # 2 => Asian
  # 3 => Other
  
  data['Ethnicity'] = data['Ethnicity'].replace([0,1,2,3], ['Caucasian', 'African_American', 'Asian', 'Other'])

  # Normalize features
  scaler = MinMaxScaler()
  data_scaled = scaler.fit_transform(data.drop(columns=['Diagnosis']))

  # Separate features and targets
  X = data.drop(columns=['Diagnosis'])
  Y = data['Diagnosis']

  label_encoder = LabelEncoder()
  y_encoded = label_encoder.fit_transform()
  
  # Stratified train-test split
  X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42)
  
  

main()