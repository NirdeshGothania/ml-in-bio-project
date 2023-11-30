from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LogisticRegression

def feature_selection_options(data, target, method, k=5):
    print("Applying Feature Selection Technique....\n")
    if method == 'SelectKBest':
        selector = SelectKBest(mutual_info_classif, k=k)
    elif method == 'RFE':
        selector = RFE(RandomForestClassifier(), n_features_to_select = k)
    elif method == 'VarianceThreshold':
        selector = VarianceThreshold(threshold=0.0)
    elif method == 'PCA':
        selector = PCA(n_components=k)
    elif method == 'SFS':
        selector = SFS(LogisticRegression(), k_features=5, forward=True, floating=False, scoring="accuracy", cv=5)
    
    selected_data = selector.fit_transform(data, target)
    print(f"Applied {selector} feature selection technique.")
    return selected_data
