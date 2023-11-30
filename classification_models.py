from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

def model_selection_options(model_type):
    print("Selecting Model....\n")
    if model_type == 'RandomForest':
        model = RandomForestClassifier()
    elif model_type == 'SVM':
        model = SVC()
    elif model_type == 'LogisticRegression':
        model = LogisticRegression()
    elif model_type == 'GradientBoosting':
        model = GradientBoostingClassifier()
    elif model_type == 'NaiveBayes':
        model = GaussianNB()
    print(f"{model} selected.")

    return model