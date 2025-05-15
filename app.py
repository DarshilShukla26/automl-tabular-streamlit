import streamlit as st
import pandas as pd
from pipeline import preprocessing, feature_engineering, modeling, evaluation
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

st.title("AutoML for Tabular Data")

uploaded_file = st.file_uploader("Upload your CSV file")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())

    target_column = st.selectbox("Select the target column", df.columns)

    X, y, preprocessor = preprocessing.preprocess_data(df, target_column)

    if st.checkbox("Apply SMOTE for class balancing"):
        smote = SMOTE()
        X, y = smote.fit_resample(X, y)

    feat_eng = st.selectbox("Feature Engineering", ["None", "RFE", "PCA"])
    if feat_eng == "RFE":
        X = feature_engineering.apply_rfe(X, y)
    elif feat_eng == "PCA":
        X = feature_engineering.apply_pca(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model_choice = st.selectbox("Choose model", list(modeling.get_models().keys()))
    model = modeling.get_models()[model_choice]

    st.write("Training the model...")
    tuned_model = modeling.tune_model(model, X_train, y_train, param_grid={})

    st.write("Evaluation results:")
    evaluation.evaluate_model(tuned_model, X_test, y_test)
