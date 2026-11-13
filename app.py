from flask import Flask, send_from_directory, render_template, Response, redirect, url_for
import cv2
import mediapipe as mp
import numpy as np
import os
import winsound
from PIL import ImageFont, ImageDraw, Image
import threading
from datetime import datetime
import socket
from flask_cors import CORS
import webbrowser

current_video_filename = ""

app = Flask(__name__)
CORS(app)  # CORS 설정 추가

# 전역 변수: 타이머 페이지 표시 여부
show_timer = False

def set_show_timer(value):
    global show_timer
    show_timer = value

def start_squat_detection():
    global current_video_filename
    
    # 현재 날짜와 시간을 기반으로 파일 이름 생성
    current_time = datetime.now().strftime("%m%d_%H%M")
    current_video_filename = f"squat_{current_time}.mp4"

    # MediaPipe Pose 초기화
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    
    video_dir = 'static/squat_videos'  # 비디오를 저장할 디렉토리
    count_dir = 'static/squat_count'  # 카운트를 저장할 디렉토리

    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    if not os.path.exists(count_dir):
        os.makedirs(count_dir)

    filename = os.path.join(video_dir, current_video_filename)
    count_file = os.path.join(count_dir, f"squat_{current_time}.txt")

    # 비디오 녹화 설정
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'H264')  # 'H264' 코덱 사용
    out = cv2.VideoWriter(filename, fourcc, 8.0, (height, width))  # 파일명, 코덱, FPS, 프레임 크기

    # 완료 사운드 설정
    frequency = 1000
    duration = 500

    def play_beep(frequency, duration):
        winsound.Beep(frequency, duration)

    # 폰트 설정 (한글 폰트를 다운로드하여 경로 설정)
    font_path = "NanumGothic.ttf"  # 한글을 지원하는 폰트 파일 경로
    font_size = 24

    # 파일이 존재하면 카운트 불러오기, 존재하지 않으면 0으로 초기화
    if os.path.exists(count_file):
        with open(count_file, "r") as file:
            squat_count = int(file.read())
    else:
        squat_count = 0

    in_squat = False
    in_knee = False
    in_knee2 = False
    in_ankle = False
    in_ready = False

    def calculate_angle(a, b, c):
        """삼각형의 각도를 계산하는 함수"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radian = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radian * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def get_body_angles(landmarks):
        """무릎 각도와 상체 각도를 계산하는 함수"""
        
        # 왼쪽 무릎 각도 계산 (발목, 무릎, 엉덩이)
        knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

        # 좌표 추출 (x, y)
        knee_coord = (knee.x, knee.y)
        hip_coord = (hip.x, hip.y)
        ankle_coord = (ankle.x, ankle.y)
        
        # 무릎 각도 계산
        knee_angle = calculate_angle(ankle_coord, knee_coord, hip_coord)

        # 상체 각도 계산 (어깨와 엉덩이 사이의 각도)
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        torso = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        
        # 어깨와 엉덩이 좌표 추출 (x, y)
        shoulder_coord = (shoulder.x, shoulder.y)
        torso_coord = (torso.x, torso.y)
        
        # 수직선 (y축)과 상체 벡터 사이의 각도 계산
        vertical_point = (torso.x, torso.y - 0.1)  # 상체 방향을 기준으로 수직선을 생성
        body_angle = calculate_angle(shoulder_coord, torso_coord, vertical_point)

        return knee_angle, body_angle
    
    def draw_text_korean(img, text, position, font_path, font_size, color=(0, 0, 0), bg_color=(255, 255, 255, 128)):
        """한글과 텍스트 박스를 만드는 함수"""
        # OpenCV 이미지를 PIL 이미지로 변환
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype(font_path, font_size)
        
        # 텍스트 크기 계산 (textbbox를 사용하여 크기를 계산)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # 흰색 불투명 배경 그리기
        x, y = position
        background = Image.new('RGBA', (text_width, text_height), bg_color)
        img_pil.paste(background, position, background)

        # 텍스트 그리기 (흰 배경일 때 텍스트 색상을 검정으로 설정)
        draw.text(position, text, font=font, fill=color)
        
        # 다시 OpenCV 이미지로 변환
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            knee_angle, body_angle = get_body_angles(landmarks)
            
            knee_coord = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y)
            ankle_coord = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y)
            
            # 무릎이 너무 펴져있을 때    
            in_knee2 = knee_angle > 175

            # 1. 스쿼트 시작 여부 체크 (무릎 각도 170도 이상)
            if knee_angle >= 150 and not in_squat:
                in_squat = True            

            # 2. 스쿼트 동작 중
            elif in_squat:
                # 무릎이 펴져있고 발보다 너무 앞에 나가있을 경우
                in_knee = 95 < knee_angle < 170
                in_ankle = abs(knee_coord[1] - ankle_coord[1]) > 0.2

                # 아예 자세가 준비되지 않았을 경우
                in_ready = knee_angle >= 170

                # 바른 자세가 되었을 때
                if knee_angle <= 95 and abs(knee_coord[1] - ankle_coord[1]) <= 0.2:
                    squat_count += 1
                    beep_thread = threading.Thread(target=play_beep, args=(frequency, duration))
                    beep_thread.start()
                    in_squat = False
                    in_knee = False
                    in_knee2 = False
                    in_ankle = False
                    in_ready = False

            # 랜드마크를 그리기
            if knee_angle <= 95 and abs(knee_coord[1] - ankle_coord[1]) <= 0.2:           
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 225, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 225, 0), thickness=2, circle_radius=2)
                )
            else:                   
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )
            
            # 화면 90도 회전
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # 카메라 프레임 좌우 반전
            frame = cv2.flip(frame, 1)

            # 각도 및 카운트, 안내 문장 표시 (한글 텍스트 추가)
            frame = draw_text_korean(frame, f'무릎 각도: {int(knee_angle)}', (10, 50), font_path, font_size, (0, 0, 0), bg_color=(255, 255, 255, 128))
            frame = draw_text_korean(frame, f'상체 각도: {int(body_angle)}', (10, 100), font_path, font_size, (0, 0, 0), bg_color=(255, 255, 255, 128))
            frame = draw_text_korean(frame, f'스쿼트 횟수: {squat_count}', (10, 150), font_path, font_size, (0, 0, 0), bg_color=(255, 255, 255, 128))

        if in_knee:
            frame = draw_text_korean(frame, f'무릎 굽히기', (10, 200), font_path, font_size, (0, 0, 0), bg_color=(255, 255, 255, 128))
        if in_ankle:
            frame = draw_text_korean(frame, f'엉덩이를 뒤로 더 빼세요', (10, 250), font_path, font_size, (0, 0, 0), bg_color=(255, 255, 255, 128))
        if in_ready:
            frame = draw_text_korean(frame, f'스쿼트 자세를 잡으세요', (10, 300), font_path, font_size, (0, 0, 0), bg_color=(255, 255, 255, 128))
        if in_knee2:
            frame = draw_text_korean(frame, f'무릎을 너무 펴면 안됩니다.', (10, 200), font_path, font_size, (0, 0, 0), bg_color=(255, 255, 255, 128))

        # 녹화 중인 프레임 저장
        out.write(frame)

        cv2.imshow('MediaPipe Pose', frame)

        if cv2.waitKey(40) & 0xFF == ord('q'):  # FPS에 맞는 `waitKey` 사용 (1000 / fps = 40)
            break

        if squat_count >= 10:
            print("스쿼트 10회 달성, 종료합니다.")
            break

        ref, buffer = cv2.imencode('.jpg', frame)  # JPEG 형식으로 이미지를 인코딩
        frame = buffer.tobytes()  # 인코딩된 이미지를 바이트 스트림으로 변환
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    out.release()  # 녹화 종료
    cv2.destroyAllWindows()

    # 프로그램 종료 시 카운트를 파일에 저장
    with open(count_file, "w") as file:
        file.write(str(squat_count))
        webbrowser.open("http://127.0.0.1:5000/bye_squat/")

        #주의주의주의주의주의주의주의 라즈베리파이랑 노트북이랑 와이파이 같은걸로 해야됨
def udp_server(host='0.0.0.0', port=65432):
    global show_timer
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind((host, port))
        print(f"Server listening on {host}:{port}")
        while True:
            data, addr = s.recvfrom(1024)
            message = data.decode()
            print(f"Received from {addr}: {message}")
            if message == "1":
                set_show_timer(True)  # 타이머 페이지를 표시할 플래그 설정
                webbrowser.open("http://127.0.0.1:5000/timer_squat/")  # 해당 URL을 브라우저에서 엶
            if message == "2":
                set_show_timer(True)  # 타이머 페이지를 표시할 플래그 설정
                webbrowser.open("http://127.0.0.1:5000/timer_bench")  # 해당 URL을 브라우저에서 엶

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/bye_squat/')
def bye_squat():
    return render_template('bye_squat.html')

@app.route('/bye_bench/')
def bye_bench():
    return render_template('bye_bench.html')

@app.route('/pullup/')
def pullup():
    return render_template('pullup.html')

@app.route('/feedback_squat/')
def feedback_squat():
    return render_template('feedback_squat.html', video_filename=current_video_filename)


@app.route('/feedback_bench/')
def feedback_bench():
    return render_template('feedback_bench.html')

@app.route('/timer_squat/')
def timer_squat():
    return render_template('timer_squat.html')

@app.route('/timer_bench/')
def timer_bench():
    return render_template('timer_bench.html')

@app.route('/squat/')
def squat():
    return render_template('squat.html')

@app.route('/bench/')
def bench():
    return render_template('bench.html')

@app.route('/check_redirect')
def check_redirect():
    return {'redirect': show_timer}

@app.route('/my_model/<path:filename>')
def serve_my_model(filename):
    return send_from_directory(os.path.join(app.root_path, 'my_model'), filename)

@app.route('/squat_videos/<path:filename>')
def serve_video(filename):
    return send_from_directory('static/squat_videos', filename, mimetype='video/mp4')

@app.route('/Stream')
def Stream():
    # GenerateFrames 함수를 통해 비디오 프레임을 클라이언트에게 실시간으로 반환합니다.
    return Response(start_squat_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/redirect')
def redirect_to_page():
    if show_timer:
        return redirect(url_for('timer_squat'))
    return redirect(url_for('index'))

if __name__ == '__main__':
    # UDP 서버를 별도의 스레드에서 실행
    app.run(debug=True)
    
    # Flask 애플리케이션 실행
    app.run(host='0.0.0.0', port=5000)