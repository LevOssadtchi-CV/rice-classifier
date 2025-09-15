import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.nn.functional as F
from model import RiceClassifier

# === 1. Подготовка датасета ===
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])

# Загружаем весь датасет
full_dataset = datasets.ImageFolder(root="Rice_Image_Dataset", transform=transform)

# Отбираем первые 1000 изображений из каждой папки
indices = []
seen = {cls: 0 for cls in range(len(full_dataset.classes))}
for idx, (_, label) in enumerate(full_dataset.samples):
    if seen[label] < 10000:
        indices.append(idx)
        seen[label] += 1

subset_dataset = Subset(full_dataset, indices)
loader = DataLoader(subset_dataset, batch_size=64, shuffle=False)

# === 2. Загружаем обученную модель ===
model = RiceClassifier()
model.load_state_dict(torch.load("checkpoints/classifier_epoch_3.pth", map_location="cpu"))
model.eval()

# === 3. Предсказания модели ===
all_preds = []
all_labels = []
all_confidences = []

with torch.no_grad():
    for imgs, lbls in loader:
        out = model(imgs)
        probs = F.softmax(out, dim=1)
        preds = probs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(lbls.cpu().numpy())
        all_confidences.extend(probs.max(dim=1).values.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_confidences = np.array(all_confidences)

# === 4. Визуализации ===

# (1) Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=full_dataset.classes)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()

# (2) Accuracy per class
acc_per_class = []
for i, cls in enumerate(full_dataset.classes):
    cls_idx = all_labels == i
    acc = (all_preds[cls_idx] == all_labels[cls_idx]).mean()
    acc_per_class.append(acc)

plt.figure(figsize=(10,6))
plt.bar(full_dataset.classes, acc_per_class, color="skyblue")
plt.xticks(rotation=45)
plt.ylabel("Accuracy")
plt.title("Accuracy per class")
plt.show()

# (3) Distribution of prediction confidences
plt.figure(figsize=(8,6))
plt.hist(all_confidences, bins=20, color="orange", alpha=0.7)
plt.xlabel("Prediction confidence (softmax max)")
plt.ylabel("Frequency")
plt.title("Distribution of prediction confidences")
plt.show()
