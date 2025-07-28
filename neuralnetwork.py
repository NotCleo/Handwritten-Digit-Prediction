import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

# === Load training data ===
train_df = pd.read_csv('train.csv')
train_data = np.array(train_df)
np.random.shuffle(train_data)

# === Split training/dev ===
m, n = train_data.shape
data_dev = train_data[:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n] / 255.

data_train = train_data[1000:].T
Y_train = data_train[0]
X_train = data_train[1:n] / 255.
_, m_train = X_train.shape

# === Init Weights and Biases for 4-Layer NN ===
def init_params():
    W1 = np.random.randn(128, 784) * np.sqrt(2./784)
    b1 = np.zeros((128, 1))
    W2 = np.random.randn(64, 128) * np.sqrt(2./128)
    b2 = np.zeros((64, 1))
    W3 = np.random.randn(32, 64) * np.sqrt(2./64)
    b3 = np.zeros((32, 1))
    W4 = np.random.randn(10, 32) * np.sqrt(2./32)
    b4 = np.zeros((10, 1))
    return W1, b1, W2, b2, W3, b3, W4, b4


# === Activations ===
def ReLU(Z): return np.maximum(0, Z)
def ReLU_deriv(Z): return Z > 0
def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

# === Forward Propagation ===
def forward_prop(W1, b1, W2, b2, W3, b3, W4, b4, X):
    Z1 = W1 @ X + b1
    A1 = ReLU(Z1)
    Z2 = W2 @ A1 + b2
    A2 = ReLU(Z2)
    Z3 = W3 @ A2 + b3
    A3 = ReLU(Z3)
    Z4 = W4 @ A3 + b4
    A4 = softmax(Z4)
    return Z1, A1, Z2, A2, Z3, A3, Z4, A4

# we do this to encode our digits as a column vector with that specific row as 1 and rest as 0
def one_hot(Y):
    one_hot_Y = np.zeros((10, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

# === Backpropagation ===
def backward_prop(Z1, A1, Z2, A2, Z3, A3, Z4, A4, W1, W2, W3, W4, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ4 = A4 - one_hot_Y
    dW4 = (1 / m) * dZ4 @ A3.T
    db4 = (1 / m) * np.sum(dZ4, axis=1, keepdims=True)

    dZ3 = W4.T @ dZ4 * ReLU_deriv(Z3)
    dW3 = (1 / m) * dZ3 @ A2.T
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)

    dZ2 = W3.T @ dZ3 * ReLU_deriv(Z2)
    dW2 = (1 / m) * dZ2 @ A1.T
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T @ dZ2 * ReLU_deriv(Z1)
    dW1 = (1 / m) * dZ1 @ X.T
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3, dW4, db4

# === Update Parameters ===
def update_params(W1, b1, W2, b2, W3, b3, W4, b4,
                  dW1, db1, dW2, db2, dW3, db3, dW4, db4, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    W3 -= alpha * dW3
    b3 -= alpha * db3
    W4 -= alpha * dW4
    b4 -= alpha * db4
    return W1, b1, W2, b2, W3, b3, W4, b4

# === Predictions ===
def get_predictions(A4): return np.argmax(A4, axis=0)
def get_accuracy(preds, Y): return np.mean(preds == Y)

# === Training Loop ===
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2, W3, b3, W4, b4 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3, Z4, A4 = forward_prop(W1, b1, W2, b2, W3, b3, W4, b4, X)
        dW1, db1, dW2, db2, dW3, db3, dW4, db4 = backward_prop(
            Z1, A1, Z2, A2, Z3, A3, Z4, A4, W1, W2, W3, W4, X, Y)
        W1, b1, W2, b2, W3, b3, W4, b4 = update_params(
            W1, b1, W2, b2, W3, b3, W4, b4,
            dW1, db1, dW2, db2, dW3, db3, dW4, db4, alpha)
        if i % 75 == 0:
            preds = get_predictions(A4)
            acc = get_accuracy(preds, Y)
            print(f"Iteration {i}: Accuracy = {acc:.4f}")
    return W1, b1, W2, b2, W3, b3, W4, b4

# === Predict Function ===
def make_predictions(X, W1, b1, W2, b2, W3, b3, W4, b4):
    _, _, _, _, _, _, _, A4 = forward_prop(W1, b1, W2, b2, W3, b3, W4, b4, X)
    return get_predictions(A4)

# === Predict Image Function ===
def predict_from_image(image_path, W1, b1, W2, b2, W3, b3, W4, b4):
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed to read image.")
        return

    img_resized = cv2.resize(img, (28, 28))
    if np.mean(img_resized) > 127:
        img_resized = 255 - img_resized

    img_flat = img_resized.flatten().astype(np.float32).reshape(-1, 1) / 255.0
    prediction = make_predictions(img_flat, W1, b1, W2, b2, W3, b3, W4, b4)[0]

    print(f"Predicted digit: {prediction}")
    plt.imshow(img_resized, cmap='gray')
    plt.title(f"Predicted: {prediction}")
    plt.axis('off')
    plt.show()

# === Train the model ===
W1, b1, W2, b2, W3, b3, W4, b4 = gradient_descent(X_train, Y_train, alpha=0.2, iterations=600)

# === Evaluate on dev set ===
dev_preds = make_predictions(X_dev, W1, b1, W2, b2, W3, b3, W4, b4)
print("Dev Set Accuracy:", get_accuracy(dev_preds, Y_dev))

# === Test on an image ===
image_path = 'test_digit.jpeg'
predict_from_image(image_path, W1, b1, W2, b2, W3, b3, W4, b4)
