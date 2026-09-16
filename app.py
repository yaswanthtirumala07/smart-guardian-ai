import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

st.set_page_config(
    page_title="Smart Guardian AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ SMART GUARDIAN AI")
st.subheader("AI-Powered Road Safety Assistant")
st.write("Upload a road image and let AI detect potholes automatically.")

@st.cache_resource
def load_model():
    return YOLO(
        "https://huggingface.co/peterhdd/pothole-detection-yolov8/resolve/main/best.pt"
    )

model = load_model()

uploaded_file = st.file_uploader(
    "📷 Upload Road Image",
    type=["jpg", "jpeg", "png"]
)

confidence = st.slider(
    "Detection Confidence",
    0.1,
    0.9,
    0.25,
    0.05
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Road Image",
        use_container_width=True
    )

    if st.button("🔍 Submit"):

        with st.spinner("🤖 AI is analyzing the road..."):

            results = model(
                np.array(image),
                conf=confidence
            )

            annotated = results[0].plot()

            annotated = cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB
            )

            count = len(results[0].boxes)

        st.subheader("🔍 AI Detection Result")

        st.image(
            annotated,
            use_container_width=True
        )

        st.subheader("🛡️ Smart Guardian Safety Alert")

        if count > 0:

            st.error(
                f"🚨 POTHOLE ALERT!\n\n"
                f"🔴 Potholes detected: {count}\n\n"
                f"⚠️ Slow down and ride carefully.\n\n"
                f"🔵 Smart Guardian recommends maintaining a safe distance."
            )

        else:

            st.success(
                "✅ No potholes detected.\n\n"
                "🟢 Ride safely."
            )
