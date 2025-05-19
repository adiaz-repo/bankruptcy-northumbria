import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    accuracy_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Company_House_Info.csv")

# Data Pre-processing,  which include rename features, remove duplicates rows to avoid bias,
# fill missing numeric values, then separate features and targets and finally split data for training
# and testing in 70% and 30%
def load_and_prepare_data(data_frame):
    if "Bankrupt?" in data_frame.columns:
        data_frame.rename(columns={"Bankrupt?": "Target"}, inplace=True)
    data_frame.drop_duplicates(inplace=True)
    data_frame.fillna(data_frame.median(numeric_only=True), inplace=True)
    no_target = data_frame.drop(columns=["Target"])
    with_target = data_frame["Target"]
    standard_scaler = StandardScaler()
    no_target_scaled = standard_scaler.fit_transform(no_target)
    return (train_test_split(no_target_scaled,
                             with_target,
                            test_size=0.3,
                            random_state=42),
            scaler,
            no_target.columns.tolist())


(X_train, X_test, y_train, y_test), scaler, feature_names = load_and_prepare_data(df)

# Random Forest Model, Firstly addressing class imbalance using SMOTEto balance it,
# secondly train random forest model, thirdly predicted class labels and probabilities for
# the positive class and Lastly, evaluation through confusion matrix and accuracy
# and different curve graphs
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
y_probs = rf_model.predict_proba(X_test)[:, 1]
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probs)
fpr, tpr, _ = roc_curve(y_test, y_probs)
precision, recall, _ = precision_recall_curve(y_test, y_probs)

# Visualisation of Confusion Matrix to evaluate the performance of a classification model.
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Healthy", "Bankrupt"], yticklabels=["Healthy", "Bankrupt"])
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Visualisation of ROC Curve Plot to evaluate true positives rate and false positive rate.
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualisation of Precision-Recall Curve for focusing on model performance for positive class
# in imbalance data.
plt.figure(figsize=(6, 4))
plt.plot(recall, precision, label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Random Forest")
plt.grid(True)
plt.tight_layout()
plt.show()

# Print
print("Classification Report:\n", class_report)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")
