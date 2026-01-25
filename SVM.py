import numpy as np

def clean_axis(raw_landmarks):
    data = np.array(raw_landmarks)

    x_root = data[0][0]
    y_root = data[0][1]

    data[:, 0] -= x_root
    data[:, 1] -= y_root

    return data.flatten().tolist()

import json 
import os

NUM_CHANGE = {
    'call': 0,
    'dislike': 1,
    'fist': 2,
    'four': 3,
    'like': 4,
    'mute': 5,
    'ok': 6
}

def get_data(JSON_FOLDER):
    X_temp = []
    y_temp = []

    for file_name in os.listdir(JSON_FOLDER):
        class_name = file_name.replace('.json', '')

        if class_name not in NUM_CHANGE:
            continue
        
        path = os.path.join(JSON_FOLDER, file_name)
        with open(path, 'r') as file:
            data = json.load(file)

        for id_photo, infomation in data.items():
            labels = infomation['labels']
            landmarks = infomation['landmarks']

            for i, cur_labels in enumerate(labels):
                if labels[i] == class_name:
                    if i < len(landmarks):
                        raw = landmarks[i]

                        try:
                            clean = clean_axis(raw)
                            X_temp.append(clean)
                            y_temp.append(NUM_CHANGE[class_name])
                        except:
                            continue
    return X_temp, y_temp

JSON_FOLDER_TRAIN = 'Dataset\\ann_train_val'
JSON_FOLDER_TEST = 'Dataset\\ann_test'

X_train, y_train = get_data(JSON_FOLDER_TRAIN)
X_test, y_test = get_data(JSON_FOLDER_TEST)

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
# ================== SVM MODEL ==================
# Pipeline = Scaling + SVM (bắt buộc với SVM)
svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",     # hoặc "linear"
        C=10,
        gamma="scale",
        decision_function_shape="ovr"
    ))
])

# Train
svm_model.fit(X_train, y_train)

# Predict & evaluate
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# ================== SAVE MODEL ==================
joblib.dump(svm_model, "svm_hand_gesture.pkl")
print("Model saved!")