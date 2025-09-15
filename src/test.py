import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import RiceDataset
from model import RiceClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from tqdm import tqdm

# === 1. Подготовка данных ===
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

test_dataset = RiceDataset("Rice_Split/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# === 2. Модель ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RiceClassifier().to(device)

# Загружаем веса последней эпохи (или любую другую)
model.load_state_dict(torch.load("checkpoints/classifier_epoch_3.pth", map_location=device))
model.eval()

# === 3. Тестирование ===
all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in tqdm(test_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# === 4. Результаты ===
acc = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {acc:.4f}")

# Дополнительно: матрица ошибок
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)

