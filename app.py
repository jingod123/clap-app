import streamlit as st
import cv2
import numpy as np
import av
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


@st.cache_resource
def load_yolo_model():
    return YOLO("yolo11n-pose.pt")


model = load_yolo_model()

st.title("🤸 실시간 포즈 인식 웹캠 (YOLOv11)")
st.write("START 버튼을 누르고 브라우저 카메라 권한을 허용하세요.")


class PoseClapProcessor(VideoProcessorBase):
    def __init__(self):
        self.clap_flag = False
        self.clap_count = 0
        self.frame_count = 0
        self.last_frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # 화면 크기 줄이기
        small = cv2.resize(img, (320, 240))

        # 5프레임마다 한 번만 YOLO 실행
        if self.frame_count % 5 == 0:
            results = model(small, verbose=False)
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

                        if dist < 35:
                            if not self.clap_flag:
                                self.clap_count += 1
                                self.clap_flag = True
                        else:
                            self.clap_flag = False

            cv2.putText(
                annotated_frame,
                f"Clap Count: {self.clap_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

            self.last_frame = annotated_frame

        # YOLO를 실행하지 않는 프레임에는 이전 결과 화면 재사용
        if self.last_frame is None:
            self.last_frame = small

        return av.VideoFrame.from_ndarray(
            self.last_frame,
            format="bgr24"
        )


webrtc_streamer(
    key="pose-clap",
    video_processor_factory=PoseClapProcessor,
    media_stream_constraints={
        "video": {
            "width": 320,
            "height": 240,
            "frameRate": 10,
        },
        "audio": False,
    },
    async_processing=True,
)