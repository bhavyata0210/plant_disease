import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import json
import numpy as np
import io # Import io module

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the same transformations used for validation/testing during training
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Use Streamlit's caching to load the model and class names efficiently
@st.cache_resource
def load_resources():
    # Load class names from a JSON file
    try:
        with open('class_names.json', 'r') as f:
            class_names = json.load(f)
    except FileNotFoundError:
        st.error("Error: 'class_names.json' not found. Please ensure the class names file from training is in the same directory.")
        st.stop() # Stop the app if file is not found

    num_classes = len(class_names)

    # Load the pre-trained EfficientNet-B3 model structure
    model = models.efficientnet_b3(pretrained=False) # No pretrained weights initially

    # Modify the final layer to match the trained model
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Sequential(
        torch.nn.Linear(num_ftrs, 512),
        torch.nn.BatchNorm1d(512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(512, 128),
        torch.nn.BatchNorm1d(128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(128, num_classes)
    )

    # Load the trained model weights
    try:
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
        model = model.to(device)
        model.eval()
    except FileNotFoundError:
        st.error("Error: 'best_model.pth' not found. Please ensure the trained model weights file is in the same directory.")
        st.stop() # Stop the app if file is not found
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        st.stop()

    return model, class_names

# Function to make a prediction on an uploaded image file
def predict_image(image_file, model, transform, class_names, device):
    # Read the image file from the uploader
    image = Image.open(io.BytesIO(image_file.getvalue())).convert('RGB')

    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0) # Add batch dimension
    image_tensor = image_tensor.to(device)

    # Perform prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted_class_idx = torch.max(outputs, 1)

    predicted_class_name = class_names[predicted_class_idx.item()]
    confidence = probabilities[0][predicted_class_idx.item()].item()

    return predicted_class_name, confidence, image # Return the image for display

# --- Streamlit UI --- (removed the old example usage block)
st.title("Plant Leaf Disease Detection")
st.write("Upload an image of a plant leaf to detect diseases.")

# Load the model and class names
model, class_names = load_resources()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Make prediction when a file is uploaded
    predicted_class, confidence, image_for_display = predict_image(uploaded_file, model, transform, class_names, device)

    # Display the uploaded image
    st.image(image_for_display, caption='Uploaded Image', use_column_width=True)

    # Display the prediction results
    st.subheader("Prediction:")
    st.write(f"Predicted Disease: **{predicted_class}**")
    st.write(f"Confidence: **{confidence:.4f}**")

    if "healthy" in predicted_class.lower(): # Assuming 'healthy' is part of your healthy class name
        st.success("The leaf appears healthy!")
    else:
        st.warning("The leaf may have a disease.")

else:
    st.info("Please upload an image file.") 