# Handwritten Digit Recognition – From Scratch (NumPy)

A fully custom implementation of a deep neural network for MNIST handwritten digit classification using **only NumPy**.

No TensorFlow.
No PyTorch.
No high-level ML libraries.

This project was built to understand neural networks at a fundamental level by implementing every component manually — from forward propagation to backpropagation.

---

## Project Overview

The model classifies grayscale handwritten digits (0–9) from the MNIST dataset.

It includes:

* Manual forward propagation
* Manual backpropagation
* Gradient descent optimization
* Multi-layer neural network architecture
* He weight initialization
* Cross-entropy loss
* Softmax output layer
* Data shuffling per epoch
* Live digit prediction GUI (Tkinter)

---

## Network Architecture

Input Layer: **784 neurons** (28×28 flattened image)
Hidden Layer 1: **128 neurons** (ReLU)
Hidden Layer 2: **64 neurons** (ReLU)
Hidden Layer 3: **32 neurons** (ReLU)
Output Layer: **10 neurons** (Softmax)

The network learns hierarchical feature representations:

* Early layers capture stroke-level patterns
* Middle layers capture digit shape structures
* Final layer maps features to digit classes

---

## Mathematical Components

### Activation Functions

* **ReLU** for hidden layers
* **Softmax** for output probabilities

### Loss Function

* Categorical Cross-Entropy

### Optimization

* Gradient Descent
* Manual derivative computation for each layer

### Initialization

* He initialization for stable training with ReLU

---

## Training Pipeline

1. Load MNIST dataset
2. Normalize pixel values (0–255 → 0–1)
3. One-hot encode labels
4. Shuffle dataset every epoch
5. Forward propagation
6. Backpropagation
7. Parameter updates

The model learns by minimizing cross-entropy loss over multiple epochs.

---

## Live Drawing Interface

The project includes a Tkinter GUI that allows real-time digit classification.

### Features:

* Draw digit using mouse
* Automatic preprocessing to match MNIST format
* Instant prediction using trained weights

### Preprocessing Steps:

1. Crop empty space
2. Resize digit to 20×20
3. Center inside 28×28 frame
4. Apply slight Gaussian blur (to mimic MNIST softness)
5. Normalize pixel values
6. Flatten to 784-dimensional vector

This ensures consistency between training data and live input.

---

## Repository Structure
/data                ← (Create this folder and extract dataset here)
archive.zip          ← MNIST dataset (compressed)

NN.py                ← Training script
predict.py           ← Tkinter drawing + prediction GUI

W1.npy
b1.npy
W2.npy
b2.npy
W3.npy
b3.npy
W4.npy
b4.npy

Dataset Setup

The dataset is compressed as archive.zip because it exceeds GitHub's file size limits.

Setup Instructions:

Extract archive.zip

Create a folder named:

data/

Place the extracted dataset contents inside the data folder

Final structure should look like:

/data
    train-images-idx3-ubyte
    train-labels-idx1-ubyte
    ...

## How To Run

Step 1 — Train the Model
python NN.py

This will:
Train the neural network
Save all weight files

Step 2 — Run the Drawing Interface
python predict.py

Draw a digit → Click Predict → View classification result.

---

## Key Learnings

* Neural networks are highly sensitive to input distribution
* Proper preprocessing is critical for generalization
* Depth increases representational capacity
* Backpropagation is fundamentally matrix calculus
* Debugging ML systems requires structured reasoning

---

Built for understanding neural networks from first principles.

Pure NumPy. No abstraction layers.
