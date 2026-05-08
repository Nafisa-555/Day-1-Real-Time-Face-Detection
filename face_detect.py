import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Live Face Detection", layout="centered")
st.title("Face Detection App")
st.write("Upload an image or take a photo to detect faces.")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(image, "Face", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return image, len(faces)

tab1, tab2 = st.tabs(["Upload Image", "📷 Use Camera"])

with tab1:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        img_array = np.array(pil_img)
        bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        result, count = detect_faces(bgr)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        st.image(result_rgb, caption=f"{count} face(s) detected", use_container_width=True)

with tab2:
    st.info("Click **Take Photo** below — face detection will run on the captured image.")
    camera_img = st.camera_input("Take a photo")
    if camera_img:
        pil_img = Image.open(camera_img).convert("RGB")
        img_array = np.array(pil_img)
        bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        result, count = detect_faces(bgr)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        st.image(result_rgb, caption=f"{count} face(s) detected", use_container_width=True)

st.markdown("---")
st.caption("Built with OpenCV + Streamlit · Haar Cascade face detection")
