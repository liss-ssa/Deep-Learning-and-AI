import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
from datasets import CustomImageDataset

# Пути
DATA_ROOT = "5/data/train"
RESULTS_DIR = "5/results/task3"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Загрузка датасета БЕЗ преобразований
dataset = CustomImageDataset(DATA_ROOT, transform=None)

# 1. Подсчет количества изображений по классам
class_counts = defaultdict(int)
for _, label in dataset:
    class_name = dataset.get_class_names()[label]
    class_counts[class_name] += 1

# 2. Анализ размеров изображений
widths, heights = [], []
for idx in range(len(dataset)):
    img_path = dataset.images[idx]  # Получаем путь к изображению
    with Image.open(img_path) as img:
        widths.append(img.width)
        heights.append(img.height)

# Статистика
min_size = (min(widths), min(heights))
max_size = (max(widths), max(heights))
avg_size = (np.mean(widths), np.mean(heights))

# 3. Визуализация
plt.figure(figsize=(15, 5))

# Гистограмма по классам
plt.subplot(1, 2, 1)
plt.bar(class_counts.keys(), class_counts.values())
plt.title("Количество изображений по классам")
plt.xlabel("Класс")
plt.ylabel("Количество")
plt.xticks(rotation=45, ha='right')

# Распределение размеров
plt.subplot(1, 2, 2)
plt.scatter(widths, heights, alpha=0.5)
plt.title("Распределение размеров изображений")
plt.xlabel("Ширина (px)")
plt.ylabel("Высота (px)")
plt.grid(True)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/dataset_stats.png", bbox_inches="tight")
plt.close()

# Сохранение статистики
with open(f"{RESULTS_DIR}/stats.txt", "w") as f:
    f.write("=== Количество изображений по классам ===\n")
    for class_name, count in class_counts.items():
        f.write(f"{class_name}: {count}\n")
    
    f.write("\n=== Размеры изображений ===\n")
    f.write(f"Минимальный: {min_size} (ширина x высота)\n")
    f.write(f"Максимальный: {max_size}\n")
    f.write(f"Средний: {avg_size}\n")
    f.write(f"Всего изображений: {len(dataset)}\n")

print(f"Задание 3 выполнено! Результаты в {RESULTS_DIR}/")