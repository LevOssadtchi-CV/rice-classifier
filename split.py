import os
import shutil
import random
from pathlib import Path

# === Параметры ===
DATASET_DIR = "Rice_Image_Dataset"   # исходная папка
OUTPUT_DIR = "Rice_Split"            # папка для train/val/test
SPLIT_RATIOS = [0.7, 0.15, 0.15]     # train/val/test

random.seed(42)  # для воспроизводимости

# === Создаём структуру папок ===
for split in ["train", "val", "test"]:
    for class_name in os.listdir(DATASET_DIR):
        Path(os.path.join(OUTPUT_DIR, split, class_name)).mkdir(parents=True, exist_ok=True)

# === Разделение по классам ===
for class_name in os.listdir(DATASET_DIR):
    class_dir = os.path.join(DATASET_DIR, class_name)
    images = os.listdir(class_dir)
    random.shuffle(images)

    n_total = len(images)
    n_train = int(SPLIT_RATIOS[0] * n_total)
    n_val   = int(SPLIT_RATIOS[1] * n_total)
    # test = всё, что осталось

    train_files = images[:n_train]
    val_files   = images[n_train:n_train+n_val]
    test_files  = images[n_train+n_val:]

    # Копируем файлы
    for fname in train_files:
        shutil.copy(os.path.join(class_dir, fname),
                    os.path.join(OUTPUT_DIR, "train", class_name, fname))
    for fname in val_files:
        shutil.copy(os.path.join(class_dir, fname),
                    os.path.join(OUTPUT_DIR, "val", class_name, fname))
    for fname in test_files:
        shutil.copy(os.path.join(class_dir, fname),
                    os.path.join(OUTPUT_DIR, "test", class_name, fname))

print("✅ Датасет успешно разделён на train/val/test в папке:", OUTPUT_DIR)

