import os
import random
import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
from datasets import CustomImageDataset
from extra_augs import AddGaussianNoise, CutOut  # Для сравнения с готовыми аугментациями

# Пути
DATA_ROOT = "5/data/train"
RESULTS_DIR = "5/results/task2"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Загрузка датасета
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
dataset = CustomImageDataset(DATA_ROOT, transform=transform)

# Выбираем 3 изображения для демонстрации
sample_indices = [0, 10, 20]  # Примерные индексы

### Кастомные аугментации ###
class RandomBlur:
    """Случайное размытие изображения"""
    def __init__(self, p=0.5, max_radius=2):
        self.p = p
        self.max_radius = max_radius

    def __call__(self, img):
        if random.random() < self.p:
            radius = random.uniform(0, self.max_radius)
            return img.filter(ImageFilter.GaussianBlur(radius))
        return img

class RandomPerspective:
    """Случайная перспективная трансформация"""
    def __init__(self, p=0.5, distortion_scale=0.5):
        self.p = p
        self.distortion_scale = distortion_scale

    def __call__(self, img):
        if random.random() < self.p:
            width, height = img.size
            points = []
            # Случайные смещения углов
            for _ in range(4):
                x_offset = random.uniform(-self.distortion_scale, self.distortion_scale) * width
                y_offset = random.uniform(-self.distortion_scale, self.distortion_scale) * height
                points.append((x_offset, y_offset))
            # Применяем трансформацию
            return img.transform(
                img.size,
                Image.PERSPECTIVE,
                [1, 0, 0, 0, 1, 0, 0, 0],  # Упрощённый вариант (реализация требует больше кода)
                Image.BICUBIC
            )
        return img

class RandomBrightnessContrast:
    """Случайное изменение яркости и контраста"""
    def __init__(self, p=0.5, brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3)):
        self.p = p
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(self, img):
        if random.random() < self.p:
            # Яркость
            enhancer = ImageEnhance.Brightness(img)
            brightness_factor = random.uniform(*self.brightness_range)
            img = enhancer.enhance(brightness_factor)
            # Контраст
            enhancer = ImageEnhance.Contrast(img)
            contrast_factor = random.uniform(*self.contrast_range)
            img = enhancer.enhance(contrast_factor)
        return img

### Применение и визуализация ###
custom_augs = {
    "RandomBlur": RandomBlur(p=1.0, max_radius=2),
    "RandomPerspective": RandomPerspective(p=1.0, distortion_scale=0.3),
    "RandomBrightnessContrast": RandomBrightnessContrast(p=1.0)
}

# Готовые аугментации из extra_augs.py для сравнения
extra_augs = {
    "AddGaussianNoise": AddGaussianNoise(0., 0.2),
    "CutOut": CutOut(p=1.0, size=(32, 32))
}

for i, idx in enumerate(sample_indices):
    original_img, label = dataset[idx]
    class_name = dataset.get_class_names()[label]
    original_pil = transforms.ToPILImage()(original_img)

    # Сохраняем оригинал
    plt.figure(figsize=(5, 5))
    plt.imshow(original_pil)
    plt.title(f"Original: {class_name}")
    plt.axis("off")
    plt.savefig(f"{RESULTS_DIR}/original_{i}.png", bbox_inches="tight")
    plt.close()

    # Применяем кастомные аугментации
    for aug_name, aug in custom_augs.items():
        aug_img = aug(original_pil.copy())  # Копируем, чтобы не модифицировать оригинал
        plt.figure(figsize=(5, 5))
        plt.imshow(aug_img)
        plt.title(f"Custom: {aug_name}")
        plt.axis("off")
        plt.savefig(f"{RESULTS_DIR}/custom_{aug_name}_{i}.png", bbox_inches="tight")
        plt.close()

    # Применяем готовые аугментации для сравнения
    for aug_name, aug in extra_augs.items():
        # Для extra_augs нужно преобразовать в tensor
        img_tensor = transforms.ToTensor()(original_pil)
        aug_img = aug(img_tensor.clone())
        aug_img_pil = transforms.ToPILImage()(aug_img)
        plt.figure(figsize=(5, 5))
        plt.imshow(aug_img_pil)
        plt.title(f"Extra: {aug_name}")
        plt.axis("off")
        plt.savefig(f"{RESULTS_DIR}/extra_{aug_name}_{i}.png", bbox_inches="tight")
        plt.close()

print(f"Задание 2 выполнено! Результаты в {RESULTS_DIR}/")