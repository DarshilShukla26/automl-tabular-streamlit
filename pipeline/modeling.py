from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def get_models():
    return {
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(probability=True),
        'Decision Tree': DecisionTreeClassifier()
    }

def tune_model(model, X, y, param_grid):
    search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    search.fit(X, y)
    return search.best_estimator_
