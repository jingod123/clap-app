import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

#Streamlit은 버튼 클리이나 화면 변경이 있을 때 마다 코드가 다시 실행됨
#모델을 매번 다시 불러오면 속도가 느려지기 때문에
#최초 1회만 모델을 로드하고 캐시에 저장해둠
@st.cache_resource
def load_yolo_model():
    return YOLO('yolo11m-pose.pt')

st.title("🤸 실시간 포즈 인식 웹캠 (YOLOv11)")
st.write("아래 '웹캠 실행' 버튼을 클릭하여 실시간으로 포즈 인식을 시작하세요.")

# 모델 로드
model = load_yolo_model()

#session_state를 활용하여 박수 친 횟수를 저장하기 위함
# clap 카운트 초기화
if "clap_count" not in st.session_state:
    st.session_state.clap_count = 0
if "clap_flag" not in st.session_state:  # 손 모으고 있는 동안 연속 카운트 방지용
    st.session_state.clap_flag = False
#clap_flag가 false 일때만 박수 인식
#true이면 이미 박수친 상태를 인식한거라 카운트 하지 않음

#key : session_state와 자동 연동 시 사용 키
run_webcam = st.toggle('웹캠 실행', key='run_webcam')
#비디오 프레임 빈공간 미리 할당
frame_placeholder = st.empty()

#토글 스위치가 켜져 있을 때
if run_webcam:
    cap = cv2.VideoCapture(0) #컴퓨터 기본 웹캠 켜기

    if not cap.isOpened():
        st.error("웹캠을 열 수 없습니다.")
    else:
        #웹캠이 켜져 있고 토글 스위치가 켜져 있는 동안 무한 루프 돌기
        while cap.isOpened() and st.session_state.run_webcam:
            success, frame = cap.read() #웹캠에서 현재 이미지(프레임) 읽어오기
            if not success:
                break

            results = model(frame) #현재 프레임 포즈 추론
            annotated_frame = results[0].plot() #추론 결과 시각화
            # keypoint(관절), skeleton(뼈대)

            # keypoints 좌표 가져오기
            keypoints = results[0].keypoints
            if keypoints is not None:
                xy = keypoints.xy.cpu().numpy() # 감지된 모든 사람의 17개 주요 관졀 좌표(x,y) 를 배열로 가져옴 
                if len(xy) > 0:
                    person = xy[0]  # 첫 번째 사람
                    left_wrist = person[9]   # 왼쪽손목
                    right_wrist = person[10] # 오른쪽손목

                    # 두 손목 거리 계산 (유클리드 거리)
                    dist = np.linalg.norm(left_wrist - right_wrist)

                    if dist < 50:  # 가까우면 박수! (50픽셀 보다 작으면)
                        # clap_flag가 false(박수를 안친 상태였다면)
                        # => True로 바꿔서 1번만 카운트 될 수 있도록
                        if not st.session_state.clap_flag:
                            st.session_state.clap_count += 1
                            st.session_state.clap_flag = True

                            # 5번마다 풍선효과 실행
                            if st.session_state.clap_count % 5 == 0:
                                st.balloons()
                    else: #멀어지면 다시 clap_flag를 False로 바꾸기
                        st.session_state.clap_flag = False

                    # 화면에 카운트 표시
                    cv2.putText(
                        annotated_frame,
                        f"Clap Count: {st.session_state.clap_count}",
                        (30, 60), #글자 위치 좌표 (왼쪽 상단 (0,0))
                        cv2.FONT_HERSHEY_SIMPLEX, #폰트
                        1.2, #폰트크기
                        (0, 0, 255), #폰트색상 (B,G,R)
                        3 #폰트두께
                    )

            #OpenCV (B,G,R) ~> Streamlit (R,G,B)
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            #처음에 만들어준 빈공간에 표현하기
            frame_placeholder.image(annotated_frame_rgb, channels="RGB", use_container_width=True)

        cap.release() #웹캠 자원해제
        cv2.destroyAllWindows() 
else: #토글 스위치 꺼져있을 때
    st.write("버튼을 눌러 웹캠을 시작하세요.")
