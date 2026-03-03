import numpy as np


# ============================
# MNIST LOADING (IDX FORMAT)
# ============================

# images file contains:
# 16 byte header (magic number, count, rows, cols)
# followed by raw pixel bytes

def load_mnist_images(filename):
    
    with open(filename, 'rb') as f:
        f.read(16)  # skip header
        data = np.frombuffer(f.read(), dtype=np.uint8)
    
    # reshape into (number_of_images, 784)
    return data.reshape(-1, 28*28)


# labels file contains:
# 8 byte header (magic number, count)
# followed by label bytes

def load_mnist_labels(filename):
    
    with open(filename, 'rb') as f:
        f.read(8)  # skip header
        data = np.frombuffer(f.read(), dtype=np.uint8)
    
    return data


# ============================
# PARAMETER INITIALISATION
# ============================

# Using He initialization for ReLU
# prevents vanishing / exploding gradients

def initialize_parameters():
    
    np.random.seed(42)  # reproducibility
    
    n_input = 784
    n_h1 = 128
    n_h2 = 64
    n_h3 = 32
    n_output = 10
    
    # Layer 1
    W1 = np.random.randn(n_input, n_h1) * np.sqrt(2. / n_input)
    b1 = np.zeros((1, n_h1))
    
    # Layer 2
    W2 = np.random.randn(n_h1, n_h2) * np.sqrt(2. / n_h1)
    b2 = np.zeros((1, n_h2))
    
    # Layer 3
    W3 = np.random.randn(n_h2, n_h3) * np.sqrt(2. / n_h2)
    b3 = np.zeros((1, n_h3))
    
    # Output Layer
    W4 = np.random.randn(n_h3, n_output) * np.sqrt(2. / n_h3)
    b4 = np.zeros((1, n_output))
    
    return W1, b1, W2, b2, W3, b3, W4, b4


# ============================
# ACTIVATION FUNCTIONS
# ============================

# ReLU removes negative values

def relu(Z):
    return np.maximum(0, Z)


# Softmax converts raw scores into probabilities
# subtract max for numerical stability

def softmax(Z):
    
    Z_stable = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_stable)
    
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


# ============================
# FORWARD PROPAGATION
# ============================

# data flows layer by layer
# Linear -> ReLU -> Linear -> ReLU -> Linear -> ReLU -> Linear -> Softmax

def forward_propagation(X, W1, b1, W2, b2, W3, b3, W4, b4):
    
    # Layer 1
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    
    # Layer 2
    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)
    
    # Layer 3
    Z3 = np.dot(A2, W3) + b3
    A3 = relu(Z3)
    
    # Output Layer
    Z4 = np.dot(A3, W4) + b4
    A4 = softmax(Z4)
    
    return Z1, A1, Z2, A2, Z3, A3, Z4, A4


# ============================
# LOSS FUNCTION
# ============================

# categorical cross entropy
# comparing true labels with predicted probabilities

def compute_loss(Y_true, Y_pred):
    
    m = Y_true.shape[0]
    
    # add small value to avoid log(0)
    loss = -np.sum(Y_true * np.log(Y_pred + 1e-8)) / m
    
    return loss


# ============================
# BACKPROPAGATION
# ============================

# chain rule applied from output to input

def backward_propagation(X, Y,
                         Z1, A1,
                         Z2, A2,
                         Z3, A3,
                         Z4, A4,
                         W2, W3, W4):
    
    m = X.shape[0]
    
    # =====================
    # Output Layer
    # =====================
    
    # derivative of (softmax + cross entropy)
    dZ4 = A4 - Y
    
    dW4 = (1/m) * np.dot(A3.T, dZ4)
    db4 = (1/m) * np.sum(dZ4, axis=0, keepdims=True)
    
    
    # =====================
    # Hidden Layer 3
    # =====================
    
    dA3 = np.dot(dZ4, W4.T)
    dZ3 = dA3 * (Z3 > 0)  # derivative of ReLU
    
    dW3 = (1/m) * np.dot(A2.T, dZ3)
    db3 = (1/m) * np.sum(dZ3, axis=0, keepdims=True)
    
    
    # =====================
    # Hidden Layer 2
    # =====================
    
    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * (Z2 > 0)
    
    dW2 = (1/m) * np.dot(A1.T, dZ2)
    db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
    
    
    # =====================
    # Hidden Layer 1
    # =====================
    
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * (Z1 > 0)
    
    dW1 = (1/m) * np.dot(X.T, dZ1)
    db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
    
    
    return dW1, db1, dW2, db2, dW3, db3, dW4, db4


# ============================
# PARAMETER UPDATE
# ============================

# gradient descent
# move opposite to gradient

def update_parameters(W1, b1, W2, b2, W3, b3, W4, b4,
                      dW1, db1, dW2, db2, dW3, db3, dW4, db4, lr):
    
    W1 -= lr * dW1
    b1 -= lr * db1
    
    W2 -= lr * dW2
    b2 -= lr * db2
    
    W3 -= lr * dW3
    b3 -= lr * db3
    
    W4 -= lr * dW4
    b4 -= lr * db4
    
    return W1, b1, W2, b2, W3, b3, W4, b4


# ============================
# TRAINING LOOP
# ============================

# forward -> loss -> backward -> update
# repeated many times

def train(X, Y,
        W1, b1, W2, b2, W3, b3, W4, b4,
        epochs=40, lr=0.01, batch_size=64):
    
    m = X.shape[0]  # total training samples
    
    for epoch in range(epochs):
        
        # shuffle data every epoch
        # prevents model from learning order patterns
        
        permutation = np.random.permutation(m)
        X_shuffled = X[permutation]
        Y_shuffled = Y[permutation]
        
        # iterate over mini batches
        
        for i in range(0, m, batch_size):
            
            # create mini batch
            
            X_batch = X_shuffled[i:i+batch_size]
            Y_batch = Y_shuffled[i:i+batch_size]
            
            # forward pass on mini batch
            
            Z1, A1, Z2, A2, Z3, A3, Z4, A4 = forward_propagation(
                X_batch, W1, b1, W2, b2, W3, b3, W4, b4
            )
            
            # compute gradients
            
            dW1, db1, dW2, db2, dW3, db3, dW4, db4 = backward_propagation(
                X_batch, Y_batch,
                Z1, A1,
                Z2, A2,
                Z3, A3,
                Z4, A4,
                W2, W3, W4
            )
            
            # update parameters
            
            W1, b1, W2, b2, W3, b3, W4, b4 = update_parameters(
                W1, b1, W2, b2, W3, b3, W4, b4,
                dW1, db1, dW2, db2, dW3, db3, dW4, db4,
                lr
            )
        
        # compute loss on full dataset after epoch
        
        _, _, _, _, _, _, _, A4_full = forward_propagation(
            X, W1, b1, W2, b2, W3, b3, W4, b4
        )
        
        loss = compute_loss(Y, A4_full)
        
        print(f"Epoch {epoch}, Loss: {loss}")
    
    return W1, b1, W2, b2, W3, b3, W4, b4


# ============================
# PREDICTION FUNCTION
# ============================

# convert probabilities to digit (0–9)

def predict(X, W1, b1, W2, b2, W3, b3, W4, b4):
    
    _, _, _, _, _, _, _, A4 = forward_propagation(
        X, W1, b1, W2, b2, W3, b3, W4, b4
    )
    
    return np.argmax(A4, axis=1)


# ============================
# ONE HOT ENCODING
# ============================

def one_hot_encode(Y, num_classes=10):
    
    m = Y.shape[0]
    
    one_hot = np.zeros((m, num_classes))
    one_hot[np.arange(m), Y] = 1
    
    return one_hot


# ============================
# ACCURACY FUNCTION
# ============================

def compute_accuracy(Y_true, Y_pred):
    return np.mean(Y_true == Y_pred)


# ============================
# MAIN EXECUTION FLOW
# ============================

# load dataset

X_train = load_mnist_images(r"data\train-images.idx3-ubyte")
Y_train = load_mnist_labels(r"data\train-labels.idx1-ubyte")

X_test = load_mnist_images(r"data\t10k-images.idx3-ubyte")
Y_test = load_mnist_labels(r"data\t10k-labels.idx1-ubyte")


# normalize input (0–255 -> 0–1)

X_train = X_train / 255.0
X_test = X_test / 255.0


# one hot encoding

Y_train_encoded = one_hot_encode(Y_train)


# initialize parameters

W1, b1, W2, b2, W3, b3, W4, b4 = initialize_parameters()


# train model

W1, b1, W2, b2, W3, b3, W4, b4 = train(
    X_train,
    Y_train_encoded,
    W1, b1, W2, b2, W3, b3, W4, b4
)


# evaluate on test set

predictions = predict(X_test, W1, b1, W2, b2, W3, b3, W4, b4)

accuracy = compute_accuracy(Y_test, predictions)

print("Test Accuracy:", accuracy)


# save trained weights

np.save("W1.npy", W1)
np.save("b1.npy", b1)
np.save("W2.npy", W2)
np.save("b2.npy", b2)
np.save("W3.npy", W3)
np.save("b3.npy", b3)
np.save("W4.npy", W4)
np.save("b4.npy", b4)
