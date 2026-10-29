
import os
import numpy as np

from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Random Forest Orange Classification")

dataset_path = "Orange Dataset"

def load_images(folder_path):
    images = []
    labels = []

    for class_name in os.listdir(folder_path):
        class_folder = os.path.join(folder_path, class_name)

        if not os.path.isdir(class_folder):
            continue

        for file_name in os.listdir(class_folder):
            image_path = os.path.join(class_folder, file_name)

            try:
                img = Image.open(image_path).convert("RGB")
                img = img.resize((32, 32))

                images.append(np.array(img).flatten())
                labels.append(class_name)

            except Exception:
                print("Skipping file:", image_path)

    return np.array(images), np.array(labels)


X_train, y_train = load_images(os.path.join(dataset_path, "train"))
X_test, y_test = load_images(os.path.join(dataset_path, "test"))

print("Training images:", len(X_train))
print("Testing images:", len(X_test))
print("Classes:", np.unique(y_train))

print("Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Random Forest training completed!")

print("Testing Random Forest model...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nRandom Forest Test Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nRandom Forest evaluation completed!")

import joblib

joblib.dump(model, "orange_random_forest.pkl")

print("Random Forest model saved successfully!")
