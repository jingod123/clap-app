import streamlit as st
import cv2
import numpy as np
import av
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


@st.cache_resource
def load_yolo_model():
    return YOLO("yolo11n-pose.pt")  # 배포용은 n 추천


model = load_yolo_model()

st.title("🤸 실시간 포즈 인식 웹캠 (YOLOv11)")
st.write("START 버튼을 누르고 브라우저 카메라 권한을 허용하세요.")

if "clap_count" not in st.session_state:
    st.session_state.clap_count = 0


class PoseClapProcessor(VideoProcessorBase):
    def __init__(self):
        self.clap_flag = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = model(img, verbose=False)
        annotated_frame = results[0].plot()

        keypoints = results[0].keypoints

        if keypoints is not None:
            xy = keypoints.xy.cpu().numpy()

            if len(xy) > 0:
                person = xy[0]

                left_wrist = person[9]
                right_wrist = person[10]

                if left_wrist.sum() > 0 and right_wrist.sum() > 0:
                    dist = np.linalg.norm(left_wrist - right_wrist)

                    if dist < 50:
                        if not self.clap_flag:
                            st.session_state.clap_count += 1
                            self.clap_flag = True
                    else:
                        self.clap_flag = False

                cv2.putText(
                    annotated_frame,
                    f"Clap Count: {st.session_state.clap_count}",
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3,
                )

        return av.VideoFrame.from_ndarray(
            annotated_frame,
            format="bgr24"
        )


webrtc_streamer(
    key="pose-clap",
    video_processor_factory=PoseClapProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
)