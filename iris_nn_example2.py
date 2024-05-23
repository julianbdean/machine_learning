### load packages
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Preprocess the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the neural network model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 classes for the Iris dataset
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)


# Get the model's predictions for the test data
predictions = model.predict(X_test)

# Plot the first sample's predictions
plt.figure(figsize=(8, 6))
plt.bar(range(3), predictions[0], tick_label=iris.target_names)
plt.title('Model Predictions for the First Sample')
plt.xlabel('Iris Species')
plt.ylabel('Probability')
plt.show()
