import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(0)
num_students = 1000
# sample dataset
data = pd.DataFrame({
    'GPA': np.round(np.random.uniform(2.0, 4.0, num_students), 2),
    'AttendanceRate': np.random.uniform(50, 100, num_students),
    'PastAchievements': np.random.randint(0, 5, num_students),  # 0-4 scale for achievements
    'Age': np.random.randint(18, 30, num_students),
    'SocioEconomicStatus': np.random.choice(['Low', 'Middle', 'High'], num_students),
    'ParentalEducation': np.random.choice(['HighSchool', 'Bachelor', 'Master', 'Doctorate'], num_students),
    'PriorEnrollmentAttempts': np.random.randint(0, 3, num_students),
    'CreditLoad': np.random.randint(12, 21, num_students),
    'Enroll': np.random.choice([0, 1], num_students, p=[0.3, 0.7]),  # 0 = No, 1 = Yes
    'SupportNeeded': np.random.choice([0, 1], num_students, p=[0.6, 0.4])  # 0 = No, 1 = Yes
})

data = pd.get_dummies(data, columns=['SocioEconomicStatus', 'ParentalEducation'], drop_first=True)

X = data.drop(['Enroll', 'SupportNeeded'], axis=1)
y_enroll = data['Enroll']
y_support = data['SupportNeeded']

# Spliting the dataset into training and test sets for each target
X_train_enroll, X_test_enroll, y_train_enroll, y_test_enroll = train_test_split(X, y_enroll, test_size=0.2, random_state=0)
X_train_support, X_test_support, y_train_support, y_test_support = train_test_split(X, y_support, test_size=0.2, random_state=0)

#  Training a Decision Tree Classifier for each target
model_enroll = DecisionTreeClassifier(max_depth=5, random_state=0)
model_enroll.fit(X_train_enroll, y_train_enroll)

model_support = DecisionTreeClassifier(max_depth=5, random_state=0)
model_support.fit(X_train_support, y_train_support)

# Predicting  and evaluating each model
y_pred_enroll = model_enroll.predict(X_test_enroll)
y_pred_support = model_support.predict(X_test_support)

print("Enrollment Prediction Report:")
print(classification_report(y_test_enroll, y_pred_enroll))

print("Support Needed Prediction Report:")
print(classification_report(y_test_support, y_pred_support))
