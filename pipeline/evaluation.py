from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=False)
    matrix = confusion_matrix(y_test, preds)

    st.text(report)
    ConfusionMatrixDisplay(matrix).plot()
    st.pyplot(plt.gcf())
