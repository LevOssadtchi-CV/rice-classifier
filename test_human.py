import os
import json
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

RESULTS_DIR = "results"
CATEGORIES = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]

all_true = []
all_pred = []

# Проходим по всем папкам пользователей
for user_id in os.listdir(RESULTS_DIR):
    user_folder = os.path.join(RESULTS_DIR, user_id)
    if not os.path.isdir(user_folder):
        continue

    # Проходим по всем JSON файлам для этого пользователя
    for fname in os.listdir(user_folder):
        if not fname.endswith(".json"):
            continue
        file_path = os.path.join(user_folder, fname)
        with open(file_path, "r") as f:
            rounds = json.load(f)
            for r in rounds:
                # Извлекаем "правильный" класс из имени файла
                true_label = r["shown_image"].split("(")[0].strip().capitalize()
                pred_label = r["chosen"].capitalize()

                # Проверка, что метка входит в категории
                if true_label not in CATEGORIES:
                    print(f"Предупреждение: неизвестный правильный класс {true_label} в файле {file_path}")
                    continue
                if pred_label not in CATEGORIES:
                    print(f"Предупреждение: неизвестный выбранный класс {pred_label} в файле {file_path}")
                    continue

                all_true.append(true_label)
                all_pred.append(pred_label)

# --- Метрики ---
accuracy = accuracy_score(all_true, all_pred)
print(f"Общая точность: {accuracy*100:.2f}%")

# Точность по классам
print("\nТочность по классам:")
for cat in CATEGORIES:
    indices = [i for i, t in enumerate(all_true) if t == cat]
    if len(indices) == 0:
        acc = 0.0
    else:
        acc = sum(all_true[i] == all_pred[i] for i in indices) / len(indices)
    print(f"{cat}: {acc*100:.2f}%")

# Матрица ошибок
cm = confusion_matrix(all_true, all_pred, labels=CATEGORIES)
print("\nМатрица ошибок:")
print(cm)
