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

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time

# ================== NAIVE BAYES MODEL ==================
nb_model = GaussianNB()

# Train
nb_model.fit(X_train, y_train)

# Predict & evaluate
nb_start = time.time()
y_pred_nb = nb_model.predict(X_test)
nb_time = time.time() - nb_start

nb_acc = accuracy_score(y_test, y_pred_nb)

print(f"Accuracy     : {nb_acc:.4f}")

# ================== SAVE MODEL ==================
joblib.dump(nb_model, "naive_bayes_hand_gesture.pkl")
print("Naive Bayes model saved!")