import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from datasets import CustomImageDataset

# Пути
DATA_ROOT = "5/data/train"
RESULTS_DIR = "5/results/task4"
os.makedirs(RESULTS_DIR, exist_ok=True)

class AugmentationPipeline:
    """Гибкий пайплайн для применения аугментаций к изображениям."""
    
    def __init__(self):
        self.augmentations = {}
    
    def add_augmentation(self, name, aug):
        """Добавляет аугментацию в пайплайн.
        Args:
            name (str): Уникальное имя аугментации.
            aug: Функция или класс аугментации (например, из torchvision.transforms).
        """
        self.augmentations[name] = aug
    
    def remove_augmentation(self, name):
        """Удаляет аугментацию из пайплайна."""
        if name in self.augmentations:
            del self.augmentations[name]
    
    def apply(self, image):
        """Применяет все аугментации к изображению последовательно.
        Args:
            image (PIL.Image или torch.Tensor): Входное изображение.
        Returns:
            Преобразованное изображение.
        """
        for aug in self.augmentations.values():
            image = aug(image)
        return image
    
    def get_augmentations(self):
        """Возвращает словарь всех аугментаций."""
        return self.augmentations

# Примеры конфигураций аугментаций
def get_light_augmentations():
    """Легкие аугментации: минимальные изменения."""
    return {
        "RandomHorizontalFlip": transforms.RandomHorizontalFlip(p=0.3),
        "ColorJitter": transforms.ColorJitter(brightness=0.1, contrast=0.1)
    }

def get_medium_augmentations():
    """Средние аугментации: умеренные изменения."""
    return {
        "RandomHorizontalFlip": transforms.RandomHorizontalFlip(p=0.5),
        "ColorJitter": transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        "RandomRotation": transforms.RandomRotation(degrees=15)
    }

def get_heavy_augmentations():
    """Сильные аугментации: значительные искажения."""
    return {
        "RandomHorizontalFlip": transforms.RandomHorizontalFlip(p=0.7),
        "ColorJitter": transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        "RandomRotation": transforms.RandomRotation(degrees=30),
        "RandomGrayscale": transforms.RandomGrayscale(p=0.2),
        "GaussianBlur": transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
    }

# Загрузка датасета (без аугментаций)
dataset = CustomImageDataset(DATA_ROOT, transform=None)

# Выбираем 3 изображения для демонстрации
sample_indices = [0, 10, 20]  # Примерные индексы

# Создаем и тестируем конфигурации
configs = {
    "light": get_light_augmentations(),
    "medium": get_medium_augmentations(),
    "heavy": get_heavy_augmentations()
}

for config_name, augs in configs.items():
    pipeline = AugmentationPipeline()
    for name, aug in augs.items():
        pipeline.add_augmentation(name, aug)
    
    # Применяем к выбранным изображениям
    for i, idx in enumerate(sample_indices):
        original_img, label = dataset[idx]
        class_name = dataset.get_class_names()[label]
        
        # Применяем пайплайн
        augmented_img = pipeline.apply(original_img)
        
        # Сохраняем результаты
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original_img)
        plt.title(f"Original: {class_name}")
        plt.axis("off")
        
        plt.subplot(1, 2, 2)
        plt.imshow(augmented_img)
        plt.title(f"{config_name.capitalize()}: {class_name}")
        plt.axis("off")
        
        plt.savefig(f"{RESULTS_DIR}/{config_name}_{i}.png", bbox_inches="tight")
        plt.close()

print(f"Задание 4 выполнено! Результаты сохранены в {RESULTS_DIR}/")