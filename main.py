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




import pickle
import joblib
import time

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
# Load KNN
with open("knn_model.pkl", "rb") as f:
    knn = pickle.load(f)
print("Models loaded successfully!")
knn_start = time.time()
y_pred_knn = knn.predict(X_test)
knn_time = time.time() - knn_start

knn_acc = accuracy_score(y_test, y_pred_knn)
knn_precision = precision_score(y_test, y_pred_knn, average='macro')
knn_recall = recall_score(y_test, y_pred_knn, average='macro')
knn_f1 = f1_score(y_test, y_pred_knn, average='macro')

print("===== KNN RESULTS =====")
print(f"Accuracy  : {knn_acc:.4f}")
print(f"Precision : {knn_precision:.4f}")
print(f"Recall    : {knn_recall:.4f}")
print(f"F1-score  : {knn_f1:.4f}")
print(f"Predict time: {knn_time:.4f} seconds\n")

print("KNN Classification Report:")
print(classification_report(y_test, y_pred_knn, digits=4))

# ada_start = time.time()
# y_pred_ada = ada.predict(X_test)
# ada_time = time.time() - ada_start

# ada_acc = accuracy_score(y_test, y_pred_ada)
# ada_precision = precision_score(y_test, y_pred_ada, average='macro')
# ada_recall = recall_score(y_test, y_pred_ada, average='macro')
# ada_f1 = f1_score(y_test, y_pred_ada, average='macro')

# print("===== ADABOOST RESULTS =====")
# print(f"Accuracy  : {ada_acc:.4f}")
# print(f"Precision : {ada_precision:.4f}")
# print(f"Recall    : {ada_recall:.4f}")
# print(f"F1-score  : {ada_f1:.4f}")
# print(f"Predict time: {ada_time:.4f} seconds\n")

# print("AdaBoost Classification Report:")
# print(classification_report(y_test, y_pred_ada, digits=4))

# ===== GA + AdaBoost Evaluation =====
# Load GA-AdaBoost
GA_ada = joblib.load("adaboost_GA_best.pkl")
ga_ada_start = time.time()
y_pred_ga_ada = GA_ada.predict(X_test)
ga_ada_time = time.time() - ga_ada_start

ga_ada_acc = accuracy_score(y_test, y_pred_ga_ada)
ga_ada_precision = precision_score(y_test, y_pred_ga_ada, average='macro')
ga_ada_recall = recall_score(y_test, y_pred_ga_ada, average='macro')
ga_ada_f1 = f1_score(y_test, y_pred_ga_ada, average='macro')

print("===== GA-OPTIMIZED ADABOOST RESULTS =====")
print(f"Accuracy     : {ga_ada_acc:.4f}")
print(f"Precision    : {ga_ada_precision:.4f}")
print(f"Recall       : {ga_ada_recall:.4f}")
print(f"F1-score     : {ga_ada_f1:.4f}")
print(f"Predict time : {ga_ada_time:.4f} seconds\n")

print("GA-AdaBoost Classification Report:")
print(classification_report(y_test, y_pred_ga_ada, digits=4))

# Load GA-Bagging model
GA_bagging = joblib.load("bagging_GA_best.pkl")

# ===== GA + Bagging Evaluation =====
ga_bag_start = time.time()
y_pred_ga_bag = GA_bagging.predict(X_test)
ga_bag_time = time.time() - ga_bag_start

ga_bag_acc = accuracy_score(y_test, y_pred_ga_bag)
ga_bag_precision = precision_score(y_test, y_pred_ga_bag, average='macro')
ga_bag_recall = recall_score(y_test, y_pred_ga_bag, average='macro')
ga_bag_f1 = f1_score(y_test, y_pred_ga_bag, average='macro')

print("===== GA-OPTIMIZED BAGGING RESULTS =====")
print(f"Accuracy     : {ga_bag_acc:.4f}")
print(f"Precision    : {ga_bag_precision:.4f}")
print(f"Recall       : {ga_bag_recall:.4f}")
print(f"F1-score     : {ga_bag_f1:.4f}")
print(f"Predict time : {ga_bag_time:.4f} seconds\n")

print("GA-Bagging Classification Report:")
print(classification_report(y_test, y_pred_ga_bag, digits=4))

# Load SVM model
svm_model = joblib.load("svm_hand_gesture.pkl") 

# ===== SVM (RBF) Evaluation =====
svm_start = time.time()
y_pred_svm = svm_model.predict(X_test)
svm_time = time.time() - svm_start

svm_acc = accuracy_score(y_test, y_pred_svm)
svm_precision = precision_score(y_test, y_pred_svm, average='macro')
svm_recall = recall_score(y_test, y_pred_svm, average='macro')
svm_f1 = f1_score(y_test, y_pred_svm, average='macro')

print("===== SVM (RBF) RESULTS =====")
print(f"Accuracy     : {svm_acc:.4f}")
print(f"Precision    : {svm_precision:.4f}")
print(f"Recall       : {svm_recall:.4f}")
print(f"F1-score     : {svm_f1:.4f}")
print(f"Predict time : {svm_time:.4f} seconds\n")

print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm, digits=4))

# Load Bayes model
bayes_model = joblib.load("naive_bayes_hand_gesture.pkl")

# ===== BAYES Evaluation =====
bayes_start = time.time()
y_pred_bayes = bayes_model.predict(X_test)
bayes_time = time.time() - bayes_start

bayes_acc = accuracy_score(y_test, y_pred_bayes)
bayes_precision = precision_score(y_test, y_pred_bayes, average='macro')
bayes_recall = recall_score(y_test, y_pred_bayes, average='macro')
bayes_f1 = f1_score(y_test, y_pred_bayes, average='macro')

print("===== NAIVE BAYES RESULTS =====")
print(f"Accuracy     : {bayes_acc:.4f}")
print(f"Precision    : {bayes_precision:.4f}")
print(f"Recall       : {bayes_recall:.4f}")
print(f"F1-score     : {bayes_f1:.4f}")
print(f"Predict time : {bayes_time:.4f} seconds\n")

print("Bayes Classification Report:")
print(classification_report(y_test, y_pred_bayes, digits=4))