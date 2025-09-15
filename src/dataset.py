import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class RiceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(root=root_dir, transform=transform)
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return img, label  # img = тензор, label = int

    def __len__(self):
        return len(self.dataset)
