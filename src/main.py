import os
import cv2
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


DATA_DIR = "../data/fruits/train"
CATEGORIES = ["fresh", "rotten"]
IMG_SIZE = 100

X = []
y = []

for label, category in enumerate(CATEGORIES):
    folder_path = os.path.join(DATA_DIR, category)

    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)

        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Feature extraction: mean RGB
        mean_color = img.mean(axis=(0, 1))
        X.append(mean_color)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("Feature matrix shape:", X.shape)
print("Labels shape:", y.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=CATEGORIES))

# -------- SVM MODEL --------

svm_model = SVC(kernel='rbf', gamma='scale')
svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)

print("\nSVM Model Accuracy:", svm_accuracy)

# Confusion Matrix for SVM
svm_cm = confusion_matrix(y_test, svm_pred)
print("SVM Confusion Matrix:")
print(svm_cm)

print("\nSVM Classification Report:")
print(classification_report(y_test, svm_pred, target_names=CATEGORIES))

# -------- RANDOM FOREST MODEL --------

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy:", rf_accuracy)

rf_cm = confusion_matrix(y_test, rf_pred)
print("Random Forest Confusion Matrix:")
print(rf_cm)

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred, target_names=CATEGORIES))

# Save the trained model
joblib.dump(rf_model, "fruit_spoilage_rf_model.pkl")
print("Random Forest model saved as fruit_spoilage_rf_model.pkl")

# -------- CROSS-VALIDATION --------

cv_scores = cross_val_score(
    rf_model,
    X,
    y,
    cv=5,
    scoring='accuracy'
)

print("\nCross-Validation Accuracy Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())
print("Standard Deviation:", cv_scores.std())


# -------- FEATURE IMPORTANCE ANALYSIS --------

import matplotlib.pyplot as plt

feature_names = ["Red", "Green", "Blue"]

importances = rf_model.feature_importances_

for name, importance in zip(feature_names, importances):
    print(f"{name} importance: {importance:.4f}")

# Plot feature importance
plt.figure()
plt.bar(feature_names, importances)
plt.xlabel("Color Features")
plt.ylabel("Importance Score")
plt.title("Feature Importance using Random Forest")
plt.show()



