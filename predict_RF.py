import csv
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

dir = os.path.dirname(os.path.abspath(__file__))
origin_path = os.path.join(dir, 'dataset', 'origin','train.csv')
generate_path = os.path.join(dir, 'dataset', 'generate','train.csv')
test_path = os.path.join(dir, 'dataset', 'origin','test.csv')

origin_train = pd.read_csv(origin_path)
generate_train = pd.read_csv(generate_path)
test_data = pd.read_csv(test_path)

features = origin_train.columns[1:-1]
X_origin_train = origin_train[features]
y_origin_train = origin_train['loan_status']
X_generate_train = generate_train[features]
y_generate_train = generate_train['loan_status']
X_test = test_data[features]
y_test = test_data['loan_status']

rf = RandomForestClassifier()
rf.fit(X_origin_train, y_origin_train)
pred_origin = rf.predict(X_test)
print('Accuracy with origin_train:', accuracy_score(y_test, pred_origin))

rf = RandomForestClassifier()
rf.fit(X_generate_train, y_generate_train)
pred_generate = rf.predict(X_test)
print('Accuracy with mix_train:', accuracy_score(y_test, pred_generate))

rf = RandomForestClassifier()
rf.fit(X_generate_train[2900:], y_generate_train[2900:])
pred_generate = rf.predict(X_test)
print('Accuracy with generate_train:', accuracy_score(y_test, pred_generate))
