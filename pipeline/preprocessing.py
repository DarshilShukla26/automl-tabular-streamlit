from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(df, target_column):
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop(target_column, errors='ignore')
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.drop(target_column, errors='ignore')

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    X_processed = preprocessor.fit_transform(X)
    return X_processed, y, preprocessor
