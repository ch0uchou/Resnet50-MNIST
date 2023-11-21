import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
# Load the pretrained model
model = models.resnet50(pretrained=True)
model.eval()
# Define the image transformation
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
def predict(image):
    """Predict the class of an image using ResNet-50."""
    # Preprocess the image
    image_tensor = preprocess(image).unsqueeze(0)
    # Forward pass
    with torch.no_grad():
        outputs = model(image_tensor)
    # Get the predicted class index
    _, predicted_idx = outputs.max(1)
    return predicted_idx.item()
def main():
    st.title("Image Classification with ResNet-50")
    # Image uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Convert the file to an image
        image = Image.open(uploaded_file).convert("RGB")
        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        # Make a prediction button
        if st.button('Predict'):
            # Predict the class of the image
            pred_idx = predict(image)
            # Display the prediction
            st.write(f"Predicted Class Index: {pred_idx}")
if __name__ == "__main__":
    main()
