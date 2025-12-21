import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🧠",
    layout="wide"
)

# ---------------- Sidebar ----------------
st.sidebar.title("🧠 CIFAR-10 Classifier")
st.sidebar.markdown("""
**End-to-End Deep Learning Project**

This app demonstrates:
- CNN + Transfer Learning
- PyTorch model training
- Real-world deployment

**Model**
- ResNet-18
- Fine-tuned on CIFAR-10

**Dataset**
- 60,000 images
- 10 classes
""")

st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 Built for learning & interviews")

# ---------------- Main Title ----------------
st.markdown(
    "<h1 style='text-align:center;'>🖼️ CIFAR-10 Image Classifier</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;color:#94a3b8;'>Upload an image and see how a deep learning model classifies it</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# ---------------- Class Labels ----------------
classes = [
    'airplane','car','bird','cat','deer',
    'dog','frog','horse','ship','truck'
]

# ---------------- Load Model ----------------
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

# ---------------- Image Transform ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

# ---------------- Layout ----------------
left_col, right_col = st.columns([1, 1])

# ---------------- Left Column (Explanation) ----------------
with left_col:
    st.subheader("📌 What does this model do?")
    st.markdown("""
    This deep learning model classifies images into **10 everyday objects**
    using a **Convolutional Neural Network (CNN)**.
    """)

    st.markdown("### 🏷️ Supported Classes")
    st.markdown("""
    ✈️ Airplane  
    🚗 Car  
    🐦 Bird  
    🐱 Cat  
    🦌 Deer  
    🐶 Dog  
    🐸 Frog  
    🐴 Horse  
    🚢 Ship  
    🚚 Truck  
    """)

    st.markdown("### 💡 Tips")
    st.info("Use clear images with a single main object for best results.")

# ---------------- Right Column (Interaction) ----------------
with right_col:
    st.subheader("📤 Try it yourself")

    uploaded_file = st.file_uploader(
        "Upload an image (JPG / PNG)",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, dim=1)

        st.markdown("### 🧠 Prediction Result")
        st.success(f"**Class:** {classes[pred.item()]}")
        st.progress(confidence.item())
        st.caption(f"Confidence: {confidence.item()*100:.2f}%")

    else:
        st.info("👆 Upload an image to see prediction")

# ---------------- Footer ----------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#64748b;'>End-to-End ML Project • Training → Evaluation → Deployment</p>",
    unsafe_allow_html=True
)
