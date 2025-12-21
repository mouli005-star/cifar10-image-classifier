import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
# ---------------- Page configuration ----------------
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🧠",
    layout="centered"
)
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}
.pred-box {
    background-color: #0f172a;
    padding: 1.2rem;
    border-radius: 12px;
    border: 1px solid #1e293b;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; margin-bottom:2rem;">
    <h1>🧠 CIFAR-10 Image Classifier</h1>
    <p style="color:#94a3b8;">
        Upload an image and see how a deep learning model classifies it
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("""
### 📌 What is this app?

This is a **deep learning demo** built using **PyTorch** and **ResNet-18**.
The model is trained on the **CIFAR-10 dataset**, which contains images of
common objects.

The model can classify images into **10 categories**:
""")

st.markdown("""
✈️ Airplane · 🚗 Car · 🐦 Bird · 🐱 Cat · 🦌 Deer  
🐶 Dog · 🐸 Frog · 🐴 Horse · 🚢 Ship · 🚚 Truck
""")
st.markdown("""
### 🧪 How to use

1. Upload an image (JPG / PNG)
2. The model analyzes the image
3. You will see:
   - Predicted class
   - Confidence score

💡 Tip: Clear images with one main object work best.
""")
classes = [
    'airplane','car','bird','cat','deer',
    'dog','frog','horse','ship','truck'
]

@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(
        torch.load("resnet18_cifar10.pth", map_location="cpu")
    )
    model.eval()
    return model

model = load_model()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])
st.markdown("### 📤 Try it yourself")

uploaded_file = st.file_uploader(
    "Upload an image (JPG / PNG)",
    type=["jpg", "png", "jpeg"]
)
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=300)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    st.markdown(
        f"""
        <div class="pred-box">
            <h3>🧠 Prediction Result</h3>
            <p><b>Class:</b> <span style="color:#38bdf8">{classes[pred.item()]}</span></p>
            <p><b>Confidence:</b> {confidence.item()*100:.2f}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )

else:
    st.info("👆 Upload an image to get started")
st.markdown("""
---
<p style="text-align:center; color:#64748b;">
End-to-end ML project • Training → Evaluation → Deployment
</p>
""", unsafe_allow_html=True)
