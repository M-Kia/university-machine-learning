import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def split_camel_case(text):
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', text)


def main():
    # Load data
    data = pd.read_csv(os.path.join(os.path.dirname(
        __file__), 'alzheimers_disease_data.csv'))

    data.head().T

    data.info()

    data.describe().T

    sum(data.duplicated())

    data = data.drop(columns=['PatientID', 'DoctorInCharge'])

    # Identify numerical columns: columns with more than 10 unique values are considered numerical
    numerical_columns = [
        col for col in data.columns if data[col].nunique() > 10]

    # Identify categorical columns: columns that are not numerical and not 'Diagnosis'
    categorical_columns = data.columns.difference(
        numerical_columns).difference(['Diagnosis']).to_list()

    custom_labels = {
        'Gender': ['Male', 'Female'],
        'Ethnicity': ['Caucasian', 'African American', 'Asian', 'Other'],
        'EducationLevel': ['None', 'High School', 'Bachelor\'s', 'Higher'],
        'Smoking': ['No', 'Yes'],
        'FamilyHistoryAlzheimers': ['No', 'Yes'],
        'CardiovascularDisease': ['No', 'Yes'],
        'Diabetes': ['No', 'Yes'],
        'Depression': ['No', 'Yes'],
        'HeadInjury': ['No', 'Yes'],
        'Hypertension': ['No', 'Yes'],
        'MemoryComplaints': ['No', 'Yes'],
        'BehavioralProblems': ['No', 'Yes'],
        'Confusion': ['No', 'Yes'],
        'Disorientation': ['No', 'Yes'],
        'PersonalityChanges': ['No', 'Yes'],
        'DifficultyCompletingTasks': ['No', 'Yes'],
        'Forgetfulness': ['No', 'Yes']
    }

    for column in categorical_columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=data, x=column)
        plt.title(f'Countplot of {column}')

        manager = plt.get_current_fig_manager()
        title = split_camel_case(column)
        manager.set_window_title(title)

        # Directly set custom labels
        labels = custom_labels[column]
        ticks = range(len(labels))
        plt.xticks(ticks=ticks, labels=labels)

        plt.show()

    # # Normalize features
    # scaler = MinMaxScaler()
    # data_scaled = scaler.fit_transform(data.drop(columns=['Diagnosis']))

    # # Separate features and targets
    # X = data.drop(columns=['Diagnosis'])
    # y = data['Diagnosis']

    # label_encoder = LabelEncoder()
    # y_encoded = label_encoder.fit_transform(y)

    # # Stratified train-test split
    # X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42)

    # plt.figure(figsize=(8,6))
    # sns.heatmap(data, annot=True, cmap="YlGnBu", linewidths=0.5)

    # plt.title("Heatmap Chart")

    # plt.show()


main()
