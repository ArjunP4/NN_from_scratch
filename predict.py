import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np


# ============================
# LOAD TRAINED MODEL
# ============================

W1 = np.load("W1.npy")
b1 = np.load("b1.npy")
W2 = np.load("W2.npy")
b2 = np.load("b2.npy")
W3 = np.load("W3.npy")
b3 = np.load("b3.npy")
W4 = np.load("W4.npy")
b4 = np.load("b4.npy")


# ============================
# ACTIVATION FUNCTIONS
# ============================

def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    Z_stable = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_stable)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


# ============================
# FORWARD PROPAGATION
# ============================

def forward(X):
    
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    
    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)
    
    Z3 = np.dot(A2, W3) + b3
    A3 = relu(Z3)
    
    Z4 = np.dot(A3, W4) + b4
    A4 = softmax(Z4)
    
    return A4


def predict(X):
    probs = forward(X)
    return np.argmax(probs, axis=1)


# ============================
# TKINTER DRAWING APP
# ============================

canvas_size = 280  # 10x MNIST size for smooth drawing

root = tk.Tk()
root.title("Draw a Digit")

canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg="black")
canvas.pack()

# PIL image to mirror drawing
image = Image.new("L", (canvas_size, canvas_size), "black")
draw = ImageDraw.Draw(image)


# ============================
# DRAWING FUNCTION
# ============================

def draw_lines(event):
    
    x1, y1 = (event.x - 8), (event.y - 8)
    x2, y2 = (event.x + 8), (event.y + 8)
    
    canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
    draw.ellipse([x1, y1, x2, y2], fill="white")


canvas.bind("<B1-Motion>", draw_lines)


# ============================
# PREPROCESSING FUNCTION
# ============================

def preprocess(img):
    
    # canvas is already white digit on black background
    # which matches MNIST format — NO invert needed

    # crop empty space around digit
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    
    # resize digit area to 20x20 (like MNIST digit area)
    img = img.resize((20, 20), Image.Resampling.LANCZOS)
    
    # create 28x28 blank image and paste digit in center
    new_img = Image.new("L", (28, 28), 0)
    new_img.paste(img, (4, 4))
    
    # convert to numpy
    img_array = np.array(new_img)
    
    # normalize (0-255 → 0-1)
    img_array = img_array / 255.0
    
    # flatten to (1, 784)
    img_array = img_array.reshape(1, 784)
    
    return img_array


# ============================
# PREDICT BUTTON
# ============================

def classify():
    
    processed = preprocess(image)
    
    prediction = predict(processed)
    
    result_label.config(text="Prediction: " + str(prediction[0]))


def clear_canvas():
    
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_size, canvas_size], fill="black")
    result_label.config(text="Draw a digit")


predict_button = tk.Button(root, text="Predict", command=classify)
predict_button.pack()

clear_button = tk.Button(root, text="Clear", command=clear_canvas)
clear_button.pack()

result_label = tk.Label(root, text="Draw a digit", font=("Helvetica", 20))
result_label.pack()

root.mainloop()