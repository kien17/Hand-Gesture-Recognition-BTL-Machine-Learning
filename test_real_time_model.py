import cv2
import mediapipe as mp
import numpy as np
import pickle
import joblib
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# MODEL_KNN_PATH = 'knn_model.pkl' 
# MODEL_KNN_PATH = 'svm_hand_gesture.pkl' 
# MODEL_KNN_PATH = 'adaboost_GA_best.pkl' 
MODEL_KNN_PATH = 'bagging_GA_best.pkl'

TASK_FILE_PATH = 'hand_landmarker.task' 

# Map nhãn (Khớp với lúc train)
LABEL_MAP = {
    0: 'Call',
    1: 'Dislike',
    2: 'Fist',
    3: 'Four',
    4: 'Like',
    5: 'Mute',
    6: 'OK'
}

def clean_axis(raw_landmarks):
    """Chuyển đổi tọa độ để KNN hiểu"""
    data = np.array(raw_landmarks)
    base_x, base_y = data[0][0], data[0][1]
    data[:, 0] = data[:, 0] - base_x
    data[:, 1] = data[:, 1] - base_y
    return data.flatten().tolist()

with open(MODEL_KNN_PATH, 'rb') as f:
    try:
        knn_model = pickle.load(f)
    except Exception as e:
        knn_model = joblib.load(f)

base_options = python.BaseOptions(model_asset_path=TASK_FILE_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO, 
    num_hands=1,                           
    min_hand_detection_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

print("Bấm 'q' để thoát.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    timestamp_ms = int(time.time() * 1000)
    detection_result = detector.detect_for_video(mp_image, timestamp_ms)

    display_text = "..."
    box_color = (100, 100, 100)

    if detection_result.hand_landmarks:
        for landmarks in detection_result.hand_landmarks:
            h, w, _ = frame.shape
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            
            try:
                lm_list = []
                for lm in landmarks:
                    lm_list.append([lm.x, lm.y])
                
                cleaned_data = clean_axis(lm_list)

                prediction = knn_model.predict([cleaned_data])
                class_id = prediction[0]
                
                display_text = LABEL_MAP.get(class_id, "Unknown")
                box_color = (0, 200, 255) 
            except Exception as e:
                print(f"Lỗi KNN: {e}")

    cv2.rectangle(frame, (0, 0), (300, 80), box_color, -1)
    cv2.putText(frame, f"Type: {display_text}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    cv2.imshow('Test KNN', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()