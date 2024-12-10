import os
import tkinter as tk
from tkinter import filedialog
from tkinter import Label, Button
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the model
model_path = os.path.join(current_dir, 'food_classification.h5')

# Load the trained model
model = load_model(model_path)

class_names = ['Ayam Goreng', 'Nasi Goreng', 'Nasi', 'Tempe']

# Define image size based on your model
IMAGE_SIZE = (640, 640)

# Function to preprocess the image
def preprocess_image(img_path):
    # Load and preprocess the image
    img = Image.open(img_path).resize(IMAGE_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to predict the uploaded image
def predict_image(img_path):
    preprocessed_img = preprocess_image(img_path)
    prediction = model.predict(preprocessed_img)
    # For multi-class classification, use argmax
    predicted_class = np.argmax(prediction[0])
    # Convert the prediction into a readable format
    result = class_names[predicted_class]
    return result

# Function to handle file upload and display image
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        # Display the image in the GUI
        img = Image.open(file_path).resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk
        
        # Predict and display the result
        result = predict_image(file_path)
        result_label.config(text=f"Prediction: {result}")

# Create the GUI window
root = tk.Tk()
root.title("Food Classification")

# Image display label
image_label = Label(root)
image_label.pack()

# Upload button
upload_button = Button(root, text="Upload Image", command=upload_image)
upload_button.pack()

# Prediction result label
result_label = Label(root, text="Prediction: ", font=("Arial", 16))
result_label.pack()

# Run the GUI
root.mainloop()
