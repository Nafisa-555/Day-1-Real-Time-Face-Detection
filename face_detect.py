import cv2
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.title("Live Face Detection")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

class FaceDetector(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="face-detection",
    video_processor_factory=FaceDetector,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
)
