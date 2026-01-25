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

from sklearn.model_selection import train_test_split
# Lấy 30% tập train, giữ đều class
from sklearn.model_selection import train_test_split
X_GA, _, y_GA, _ = train_test_split(
    X_train,
    y_train,
    test_size=0.7,        # bỏ 70%, giữ 30%
    stratify=y_train,     # giữ phân bố lớp
    random_state=42
)

X_train_ga, X_val_ga, y_train_ga, y_val_ga = train_test_split(
    X_GA,
    y_GA,
    test_size=0.3,        # 30% cho validation
    stratify=y_GA,
    random_state=42
)

from collections import Counter

print("Full train:", Counter(y_train))
print("Part train:", Counter(y_train_ga))
print("Part test:", Counter(y_val_ga))

import random

def random_individual():
    return {
        "max_depth": random.randint(5, 60),
        "min_samples_leaf": random.randint(5, 100),
        "n_estimators": random.randint(10, 100),
        "learning_rate": random.uniform(0.01, 0.5)
    }

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def fitness(individual, X_train, y_train, X_val, y_val):
    base_tree = DecisionTreeClassifier(
        max_depth=individual["max_depth"],
        min_samples_leaf=individual["min_samples_leaf"],
        random_state=42
    )

    model = AdaBoostClassifier(
        estimator=base_tree,
        n_estimators=individual["n_estimators"],
        learning_rate=individual["learning_rate"],
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return accuracy_score(y_val, y_pred)

def tournament_selection(population, scores, k=3):
    selected = random.sample(list(zip(population, scores)), k)
    selected.sort(key=lambda x: x[1], reverse=True)
    return selected[0][0]

def crossover(parent1, parent2):
    child = {}
    for key in parent1:
        child[key] = parent1[key] if random.random() < 0.5 else parent2[key]
    return child

def mutate(individual, mutation_rate=0.2):
    if random.random() < mutation_rate:
        individual["max_depth"] = random.randint(5, 60)
    if random.random() < mutation_rate:
        individual["min_samples_leaf"] = random.randint(5, 100)
    if random.random() < mutation_rate:
        individual["n_estimators"] = random.randint(10, 100)
    if random.random() < mutation_rate:
        individual["learning_rate"] = random.uniform(0.01, 0.5)
    return individual

def genetic_algorithm(
    X_train, y_train, X_val, y_val,
    population_size=10,
    generations=15,
    log_file="ga_adaboost_log.txt"
):
    population = [random_individual() for _ in range(population_size)]

    best_individual = None
    best_score = -1

    # mở file log
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("===== GENETIC ALGORITHM LOG =====\n")
        f.write(f"Population size: {population_size}\n")
        f.write(f"Generations: {generations}\n\n")

        for gen in range(generations):
            scores = [fitness(ind, X_train, y_train, X_val, y_val) for ind in population]

            gen_best_score = max(scores)
            gen_best_ind = population[scores.index(gen_best_score)]

            if gen_best_score > best_score:
                best_score = gen_best_score
                best_individual = gen_best_ind
                status = "NEW BEST"
            else:
                status = "OK"

            # log text
            log_line = (
                f"Gen {gen+1:02d} | "
                f"Acc = {gen_best_score:.5f} | "
                f"Params = {gen_best_ind} | "
                f"{status}\n"
            )

            # in ra màn hình
            print(log_line.strip())

            # ghi file
            f.write(log_line)

            # GA steps
            new_population = []
            new_population.append(gen_best_ind)  # elitism

            while len(new_population) < population_size:
                p1 = tournament_selection(population, scores)
                p2 = tournament_selection(population, scores)

                child = crossover(p1, p2)
                child = mutate(child)

                new_population.append(child)

            population = new_population

        f.write("\n===== FINAL RESULT =====\n")
        f.write(f"Best Acc: {best_score:.5f}\n")
        f.write(f"Best Params: {best_individual}\n")

    return best_individual, best_score


best_params, best_acc = genetic_algorithm(
    X_train_ga, y_train_ga, X_val_ga, y_val_ga,
    population_size=12,
    generations=20
)

print("Best hyperparameters:", best_params)
print("Validation accuracy:", best_acc)

# Train lại trên full train
base_tree = DecisionTreeClassifier(
    max_depth=best_params["max_depth"],
    min_samples_leaf=best_params["min_samples_leaf"],
    random_state=42
)

final_model = AdaBoostClassifier(
    estimator=base_tree,
    n_estimators=best_params["n_estimators"],
    learning_rate=best_params["learning_rate"],
    random_state=42
)

final_model.fit(X_train, y_train)

import joblib
joblib.dump(final_model, "adaboost_GA_best.pkl")
print("Final model saved!")
