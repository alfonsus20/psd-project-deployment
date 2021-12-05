import pandas as pd
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("Website Phishing.csv")

target = df['Result']
df = df.drop('Result', axis=1)

enc = OneHotEncoder()
enc.fit(df)

x = enc.transform(df)
y = target.values


sm = SMOTE(random_state=2)
x_res, y_res = sm.fit_resample(x, y.ravel())

x_train, x_test, y_train, y_test = train_test_split(x_res, y_res, test_size= 0.2, random_state= 0)


# Random Forest Classifier
model = RandomForestClassifier()
grid = {'criterion': ["gini", "entropy"],
        'min_samples_split': [2, 3, 4],
        'min_samples_leaf': [1, 2, 3, 4]}
clf = GridSearchCV(model, grid, scoring='accuracy')
clf.fit(x_train, y_train)
model = RandomForestClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=4)
model.fit(x_train, y_train)

# Decision Tree Classifier
model_DT = DecisionTreeClassifier()
grid_DT = {'criterion': ["gini", "entropy"],
        'splitter' : ["best", "random"],
        'min_samples_split': [2, 3, 4, 5, 6],
        'min_samples_leaf': [1, 2, 3, 4]}
clf_DT = GridSearchCV(model_DT, grid_DT, scoring='accuracy')
clf_DT.fit(x_train, y_train)
model_DT = DecisionTreeClassifier(criterion='gini', splitter='best', min_samples_leaf=1, min_samples_split=5)
model_DT.fit(x_train, y_train)

# Support Vector Machine
model_SVC = SVC()
grid_SVC = {'C': [0.1, 1, 10, 100],
        'kernel' : ["rbf", "poly", "sigmoid"],
        'gamma' : [1, 0.1, 0.01, 0.001]}
clf_SVC = GridSearchCV(model_SVC, grid_SVC, scoring='accuracy')
clf_SVC.fit(x_train, y_train)
model_SVC = SVC(gamma=0.1, C=10, kernel='rbf')
model_SVC.fit(x_train, y_train)

pickle.dump(model, open('model_random_forest.pkl', 'wb'))
pickle.dump(model_DT, open('model_decision_tree.pkl', 'wb'))
pickle.dump(model_SVC, open('model_svm.pkl', 'wb'))
pickle.dump(enc, open('enc.pkl', 'wb'))