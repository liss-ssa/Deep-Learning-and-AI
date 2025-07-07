import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from datasets import CustomImageDataset

# Путь к данным
DATA_ROOT = "5/data/train"
RESULTS_DIR = "5/results/task1"  

# Создадим папку для сохранения результатов (рекурсивно)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Загрузка датасета без аугментаций (только ресайз и ToTensor)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
dataset = CustomImageDataset(DATA_ROOT, transform=transform)

# Выбираем по одному изображению из 5 разных классов
sample_indices = []
class_indices = set()
idx = 0
while len(sample_indices) < 5 and idx < len(dataset):
    _, label = dataset[idx]
    if label not in class_indices:
        sample_indices.append(idx)
        class_indices.add(label)
    idx += 1

# Определяем стандартные аугментации
standard_augmentations = {
    "RandomHorizontalFlip": transforms.RandomHorizontalFlip(p=1.0),
    "RandomCrop": transforms.RandomCrop(size=(180, 180), padding=20),
    "ColorJitter": transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    "RandomRotation": transforms.RandomRotation(degrees=45),
    "RandomGrayscale": transforms.RandomGrayscale(p=1.0)
}

# Визуализация оригиналов и аугментированных изображений
for i, idx in enumerate(sample_indices):
    original_img, label = dataset[idx]
    class_name = dataset.get_class_names()[label]
    
    # Конвертируем tensor в PIL для визуализации
    original_pil = transforms.ToPILImage()(original_img)
    
    # Сохраняем оригинал
    plt.figure(figsize=(5, 5))
    plt.imshow(original_pil)
    plt.title(f"Original: {class_name}")
    plt.axis("off")
    plt.savefig(f"{RESULTS_DIR}/original_{i}_{class_name}.png", bbox_inches="tight")  # Исправлено
    plt.close()
    
    # Применяем каждую аугментацию отдельно
    for aug_name, aug in standard_augmentations.items():
        aug_img = aug(original_pil)
        
        plt.figure(figsize=(5, 5))
        plt.imshow(aug_img)
        plt.title(f"{aug_name}: {class_name}")
        plt.axis("off")
        plt.savefig(f"{RESULTS_DIR}/{aug_name}_{i}_{class_name}.png", bbox_inches="tight")  # Исправлено
        plt.close()
    
    # Применяем все аугментации вместе
    combined_aug = transforms.Compose(list(standard_augmentations.values()))
    combined_img = combined_aug(original_pil)
    
    plt.figure(figsize=(5, 5))
    plt.imshow(combined_img)
    plt.title(f"Combined Augs: {class_name}")
    plt.axis("off")
    plt.savefig(f"{RESULTS_DIR}/combined_{i}_{class_name}.png", bbox_inches="tight")  # Исправлено
    plt.close()

print(f"Задание 1 выполнено! Результаты сохранены в {RESULTS_DIR}/")