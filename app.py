import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

# ---------------- Page config ----------------
st.set_page_config(
    page_title="CIFAR-10 Classifier",
    page_icon="🧠",
    layout="centered"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #0f172a;
    color: white;
}
.block-container {
    padding-top: 2rem;
}
.pred-box {
    background-color: #020617;
    padding: 1.2rem;
    border-radius: 12px;
    border: 1px solid #1e293b;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Title ----------------
st.markdown("<h1 style='text-align:center;'>🖼️ CIFAR-10 Image Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#94a3b8;'>Upload an image and let the model predict its class</p>", unsafe_allow_html=True)
st.divider()

# ---------------- Class labels ----------------
classes = [
    'airplane','car','bird','cat','deer',
    'dog','frog','horse','ship','truck'
]

# ---------------- Load model ----------------
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load("resnet18_cifar10.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ---------------- Image preprocessing ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

# ---------------- File uploader ----------------
uploaded_file = st.file_uploader(
    "📤 Upload an image (JPG / PNG)",
    type=["jpg", "png", "jpeg"]
)

# ---------------- Prediction ----------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=320
    )

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    st.markdown(
        f"""
        <div class="pred-box">
            <h3>Prediction: <span style="color:#38bdf8">{classes[pred.item()]}</span></h3>
            <p>Confidence: <b>{confidence.item()*100:.2f}%</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )

else:
    st.info("👆 Upload an image to get started")
