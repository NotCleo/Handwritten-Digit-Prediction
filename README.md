# Handwritten-Digit-Prediction

This project is a from-scratch implementation of a fully connected feedforward neural network for handwritten digit recognition using the MNIST dataset — without using any deep learning libraries like TensorFlow or PyTorch. 

All neural network mechanics (forward propagation, backpropagation, activation functions, training loop) are manually implemented using NumPy.

# Dataset

We use the MNIST handwritten digit dataset (provided in csv format):

    Each row represents a 28x28 grayscale image, flattened into 784 pixel values.

    The first column is the label (i.e., the digit 0–9).

    The remaining 784 columns are the pixel intensity values (0–255).

# Network Architecture

This is a 4-layer fully connected feedforward neural network;

    Input Layer  ->  Hidden Layer 1 -> Hidden Layer 2 -> Hidden Layer 3 -> Output Layer
        784             128               64               32               10

# Breakdown of nodes:

    Input Layer (784 neurons): Each input image is flattened from 28x28 to a 784-dimensional vector.

    Hidden Layer 1 (128 neurons): First transformation with ReLU activation.

    Hidden Layer 2 (64 neurons): Second transformation with ReLU.

    Hidden Layer 3 (32 neurons): Third transformation with ReLU.
    
# Image Prediction

You can upload a new digit image (preferably white digit on black background) and the trained model will:

    Preprocess the image (resize to 28x28, grayscale, invert if needed)

    Flatten and normalize it

    Run it through the neural network to predict the digit

    Output Layer (10 neurons): Outputs class probabilities for digits 0–9 using softmax.


This project intentionally does NOT use any machine learning frameworks or high-level libraries. The goal is to show how neural networks work under the hood by implementing:

    Forward & Backward propagation

    Weight initialization

    Activation functions

    Loss gradients

    Manual parameter updates
