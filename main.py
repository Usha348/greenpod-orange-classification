
import tensorflow as tf

dataset_path = "Orange Dataset"

train_data = tf.keras.utils.image_dataset_from_directory(
    dataset_path + "/train",
    image_size=(128, 128),
    batch_size=32
)

# Build our CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128, 128, 3)),
    tf.keras.layers.Rescaling(1.0 / 255),
    tf.keras.layers.Conv2D(16, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(4, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
print("CNN model created successfully!")
print("Starting model training...")

history = model.fit(
    train_data,
    epochs=5
)

print("CNN training completed!")

print("Loading test images...")

test_data = tf.keras.utils.image_dataset_from_directory(
    dataset_path + "/test",
    image_size=(128, 128),
    batch_size=32,
    shuffle=False
)

print("Test images loaded successfully!")

print("Evaluating CNN model...")

test_loss, test_accuracy = model.evaluate(test_data)

print("Test Loss:", test_loss)
print("Test Accuracy:", round(test_accuracy * 100, 2), "%")

model.save("orange_cnn_model.keras")

print("CNN model saved successfully!")

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

print("Calculating detailed evaluation metrics...")

y_true = np.concatenate([
    labels.numpy() for images, labels in test_data
])

y_prob = model.predict(test_data)
y_pred = np.argmax(y_prob, axis=1)

class_names = test_data.class_names

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("Detailed evaluation completed!")

from PIL import Image
import numpy as np

print("Orange classification model is ready!")

# Load the saved model
model = tf.keras.models.load_model("orange_cnn_model.keras")

# Enter the image path
image_path = input("Enter the orange image path: ")

# Open and resize the image
img = Image.open(image_path).convert("RGB")
img = img.resize((128, 128))

# Convert image into an array
img_array = np.array(img)
img_array = np.expand_dims(img_array, axis=0)

# Predict the class
prediction = model.predict(img_array)
class_names = ["blackspot", "canker", "fresh", "grenning"]

predicted_class = class_names[np.argmax(prediction)]

print("Predicted orange class:", predicted_class)









