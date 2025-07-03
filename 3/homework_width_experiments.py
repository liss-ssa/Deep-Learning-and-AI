import torch
import time
from datasets import get_cifar_loaders
from models import FullyConnectedModel
from trainer import train_model
from utils import count_parameters
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Настройки
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
epochs = 10
results_dir = 'results/width_experiments'
os.makedirs(results_dir, exist_ok=True)

# Загрузка данных
train_loader, test_loader = get_cifar_loaders(batch_size)

# Конфигурации ширины слоёв (глубина = 3 слоя)
width_configs = {
    'narrow': [64, 32, 16],     # Узкая
    'medium': [256, 128, 64],   # Средняя
    'wide': [1024, 512, 256],   # Широкая
    'xwide': [2048, 1024, 512]  # Очень широкая
}

def run_width_experiment(name, width_config):
    """Обучает модель с заданной шириной слоёв."""
    layers = []
    prev_size = 3072  # CIFAR: 32x32x3
    
    for size in width_config:
        layers.extend([
            {'type': 'linear', 'size': size},
            {'type': 'relu'}
        ])
        prev_size = size
    
    model = FullyConnectedModel(
        input_size=3072,
        num_classes=10,
        layers=layers
    ).to(device)
    
    params = count_parameters(model)
    print(f"\n=== Конфигурация: {name} {width_config} ===")
    print(f"Параметров: {params:,}")
    
    start_time = time.time()
    history = train_model(model, train_loader, test_loader, epochs=epochs, device=device)
    train_time = time.time() - start_time
    
    return {
        'config_name': name,  # Используем имя конфигурации вместо списка
        'config': str(width_config),  # Преобразуем список в строку для отображения
        'params': params,
        'train_time': train_time,
        'train_acc': history['train_accs'][-1],
        'test_acc': history['test_accs'][-1],
        'history': history
    }

# Запуск экспериментов
results = []
for name, config in width_configs.items():
    results.append(run_width_experiment(name, config))

# Сохранение результатов
df_results = pd.DataFrame(results)
df_results.to_csv(f"{results_dir}/width_results.csv", index=False)

# Визуализация
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.barplot(x='config', y='test_acc', data=df_results)
plt.title('Accuracy vs Ширина слоёв')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
sns.barplot(x='config', y='train_time', data=df_results)
plt.title('Время обучения vs Ширина слоёв')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{results_dir}/width_comparison.png")
plt.show()

# 2.2 Grid Search для оптимизации архитектуры
def grid_search_width_patterns():
    """Поиск лучшей схемы изменения ширины."""
    patterns = {
        'narrowing': [1024, 512, 256],  # Сужение
        'expanding': [256, 512, 1024],  # Расширение
        'constant': [512, 512, 512],    # Постоянная
        'bottleneck': [1024, 256, 1024] # "Бутылочное горлышко"
    }
    
    grid_results = []
    for name, config in patterns.items():
        grid_results.append(run_width_experiment(name, config))
    
    # Визуализация heatmap
    df_grid = pd.DataFrame(grid_results)
    
    # Создаем DataFrame для heatmap
    heatmap_data = df_grid[['config_name', 'test_acc']].set_index('config_name')
    
    plt.figure(figsize=(8, 4))
    sns.heatmap(heatmap_data.T, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title('Accuracy для разных схем ширины')
    plt.savefig(f"{results_dir}/width_patterns_heatmap.png")
    plt.show()

grid_search_width_patterns()