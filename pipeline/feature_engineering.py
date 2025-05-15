from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

def apply_rfe(X, y, n_features=10):
    model = RandomForestClassifier()
    selector = RFE(model, n_features_to_select=n_features)
    X_new = selector.fit_transform(X, y)
    return X_new

def apply_pca(X, n_components=5):
    pca = PCA(n_components=n_components)
    X_new = pca.fit_transform(X)
    return X_new
