import os
import time
import torch
import psutil
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from datasets import CustomImageDataset

# Пути
DATA_ROOT = "5/data/train"
RESULTS_DIR = "5/results/task5"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Конфигурация эксперимента
SIZES = [64, 128, 224, 512]  # Тестируемые размеры
NUM_IMAGES = 100              # Количество изображений для теста
AUGMENTATIONS = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(degrees=15)
])

# Результаты
results = {
    "size": [],
    "load_time": [],
    "augment_time": [],
    "memory_usage": []
}

# Процесс тестирования для каждого размера
for size in SIZES:
    print(f"Тестируем размер {size}x{size}...")
    
    # Создаем трансформации
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    
    # Загрузка датасета
    dataset = CustomImageDataset(DATA_ROOT, transform=transform)
    
    # Измеряем время загрузки
    start_time = time.time()
    for i in range(NUM_IMAGES):
        _ = dataset[i % len(dataset)]  # Зацикливаемся, если изображений <100
    load_time = time.time() - start_time
    
    # Измеряем время аугментаций и память
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / 1024 ** 2  # В МБ
    start_time = time.time()
    for i in range(NUM_IMAGES):
        img, _ = dataset[i % len(dataset)]
        _ = AUGMENTATIONS(img)
    augment_time = time.time() - start_time
    end_mem = process.memory_info().rss / 1024 ** 2
    
    # Сохраняем результаты
    results["size"].append(size)
    results["load_time"].append(load_time)
    results["augment_time"].append(augment_time)
    results["memory_usage"].append(end_mem - start_mem)
    print(f"Размер {size}x{size}: Загрузка={load_time:.2f}с, Аугментация={augment_time:.2f}с, Память={end_mem - start_mem:.2f}МБ")

# Визуализация
plt.figure(figsize=(15, 5))

# График времени загрузки
plt.subplot(1, 3, 1)
plt.plot(results["size"], results["load_time"], marker='o')
plt.title("Время загрузки")
plt.xlabel("Размер (px)")
plt.ylabel("Время (с)")
plt.grid(True)

# График времени аугментации
plt.subplot(1, 3, 2)
plt.plot(results["size"], results["augment_time"], marker='o', color='orange')
plt.title("Время аугментации")
plt.xlabel("Размер (px)")
plt.ylabel("Время (с)")
plt.grid(True)

# График использования памяти
plt.subplot(1, 3, 3)
plt.plot(results["size"], results["memory_usage"], marker='o', color='green')
plt.title("Потребление памяти")
plt.xlabel("Размер (px)")
plt.ylabel("Память (МБ)")
plt.grid(True)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/size_experiment.png", bbox_inches="tight")
plt.close()

# Сохранение результатов в текстовый файл
with open(f"{RESULTS_DIR}/results.txt", "w") as f:
    f.write("Размер\tЗагрузка (с)\tАугментация (с)\tПамять (МБ)\n")
    for size, load, aug, mem in zip(results["size"], results["load_time"], results["augment_time"], results["memory_usage"]):
        f.write(f"{size}x{size}\t{load:.4f}\t{aug:.4f}\t{mem:.2f}\n")

print(f"Задание 5 выполнено! Результаты сохранены в {RESULTS_DIR}/")