# Project 3: Deep Learning - Fashion MNIST (Simple Version)

import pandas as pd
import numpy as np
import tensorflow as tf
import os

# --------------------------------------------------
# 1. Load datasets
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

train_path = os.path.join(BASE_DIR, "fashion-mnist_train.csv")
test_path = os.path.join(BASE_DIR, "fashion-mnist_test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# --------------------------------------------------
# 2. Split features and labels
# --------------------------------------------------
X_train = train_df.iloc[:, 1:].values
y_train = train_df.iloc[:, 0].values

X_test = test_df.iloc[:, 1:].values
y_test = test_df.iloc[:, 0].values

# --------------------------------------------------
# 3. Normalize pixel values
# --------------------------------------------------
X_train = X_train / 255.0
X_test = X_test / 255.0

# --------------------------------------------------
# 4. Reshape data for neural network
# --------------------------------------------------
X_train = X_train.reshape(-1, 28, 28)
X_test = X_test.reshape(-1, 28, 28)

# --------------------------------------------------
# 5. Build the Neural Network
# --------------------------------------------------
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# --------------------------------------------------
# 6. Compile the model
# --------------------------------------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# --------------------------------------------------
# 7. Train the model
# --------------------------------------------------
model.fit(X_train, y_train, epochs=5)

# --------------------------------------------------
# 8. Evaluate the model
# --------------------------------------------------
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", test_accuracy)

# --------------------------------------------------
# 9. Save the model
# --------------------------------------------------
model.save("fashion_mnist_model.keras")

print("\nModel saved successfully as: fashion_mnist_model.keras")

