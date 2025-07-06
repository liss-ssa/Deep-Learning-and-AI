import os
import time
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils.datasets import get_cifar_loaders
from utils.training_utils import train_model
from utils.visualization_utils import plot_feature_maps, count_parameters

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('4/results/custom_layers_experiment.log'),
        logging.StreamHandler()
    ]
)

# --- 3.1 Кастомные слои ---
class CustomConv2d(nn.Module):
    """Кастомный сверточный слой с энергетической регуляризацией"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.energy_scale = nn.Parameter(torch.ones(1))
        self.energy_bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        x = self.conv(x)
        if self.training:
            energy = x.pow(2).mean(dim=[1,2,3], keepdim=True)
            x = x * (self.energy_scale / (energy.sqrt() + 1e-6) + self.energy_bias)
        return x

class ChannelAttention(nn.Module):
    """Attention механизм по каналам (аналог SE-block)"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))
        return x * (avg_out + max_out).view(b, c, 1, 1)

class Swish(nn.Module):
    """Кастомная функция активации Swish"""
    def forward(self, x):
        return x * torch.sigmoid(x)

# --- 3.2 Residual блоки ---
class BasicResidualBlock(nn.Module):
    """Базовый Residual блок (2 слоя 3x3)"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

# --- Модели для экспериментов ---
class CustomLayerModel(nn.Module):
    """Модель для тестирования кастомных слоев (без PositionAwarePooling)"""
    def __init__(self, layer_type='standard'):
        super().__init__()
        
        # Выбор типа слоя
        if layer_type == 'conv':
            self.conv = CustomConv2d(3, 32, 3, padding=1)
        else:
            self.conv = nn.Conv2d(3, 32, 3, padding=1)
            
        self.attn = ChannelAttention(32) if layer_type == 'attention' else nn.Identity()
        self.pool = nn.MaxPool2d(2)  # Всегда используем стандартный pooling
        self.fc = nn.Linear(32*16*16, 10)
        self.act = Swish() if layer_type == 'activation' else nn.ReLU()

    def forward(self, x):
        x = self.act(self.pool(self.attn(self.conv(x))))
        return self.fc(x.view(x.size(0), -1))

class ResidualTestModel(nn.Module):
    """Модель для тестирования Residual блоков"""
    def __init__(self, block_type='basic', num_blocks=2):
        super().__init__()
        self.in_channels = 64
        self.block = BasicResidualBlock  # Используем только базовый блок
        
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Создание слоев
        self.layer1 = self._make_layer(64, num_blocks, 1)
        self.layer2 = self._make_layer(128, num_blocks, 2)
        self.layer3 = self._make_layer(256, num_blocks, 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, 10)
    
    def _make_layer(self, out_channels, blocks, stride):
        layers = [self.block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        layers.extend([self.block(out_channels, out_channels, 1) for _ in range(1, blocks)])
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.fc(self.avgpool(x).view(x.size(0), -1))

# --- Эксперименты ---
def run_custom_layers_experiment():
    """Эксперимент с кастомными слоями (без PositionAwarePooling)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_cifar_loaders()
    
    # Убрали 'pool' из тестируемых вариантов
    layer_types = ['standard', 'conv', 'attention', 'activation']
    results = {}
    
    for layer_type in tqdm(layer_types, desc="Testing custom layers"):
        try:
            model = CustomLayerModel(layer_type).to(device)
            params = count_parameters(model)
            
            logging.info(f"\nTesting {layer_type} layer...")
            logging.info(f"Parameters: {params:,}")
            
            start_time = time.time()
            history = train_model(model, train_loader, test_loader, epochs=10, device=device)
            train_time = time.time() - start_time
            
            # Визуализация feature maps
            sample, _ = next(iter(test_loader))
            features = model.conv(sample.to(device))
            plot_feature_maps(features, title=f"{layer_type} features", save_path=f"{layer_type}_features.png")
            
            results[layer_type] = {
                'test_acc': history['test_accs'][-1],
                'train_time': train_time,
                'params': params
            }
            
        except Exception as e:
            logging.error(f"Error testing {layer_type}: {e}")
            continue
    
    save_results(results, 'custom_layers_results.json')
    plot_comparison(results, 'custom_layers')
    return results

def save_results(results, filename):
    """Сохраняет результаты в JSON файл"""
    os.makedirs('results', exist_ok=True)
    with open(f'results/{filename}', 'w') as f:
        json.dump(results, f, indent=2)

def plot_comparison(results, experiment_name):
    """Визуализирует результаты экспериментов"""
    os.makedirs('plots', exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Точность
    axs[0].bar(results.keys(), [res['test_acc'] for res in results.values()])
    axs[0].set_title('Test Accuracy')
    axs[0].set_ylim(0, 1)
    
    # Время обучения
    axs[1].bar(results.keys(), [res['train_time'] for res in results.values()])
    axs[1].set_title('Training Time (s)')
    
    # Параметры
    axs[2].bar(results.keys(), [res['params'] for res in results.values()])
    axs[2].set_title('Parameters Count')
    
    plt.tight_layout()
    plt.savefig(f'plots/{experiment_name}_comparison.png')
    plt.close()

if __name__ == "__main__":
    logging.info("Starting custom layers experiment (without PositionAwarePooling)...")
    custom_results = run_custom_layers_experiment()
    
    logging.info("\n=== Custom Layers Results ===")
    for name, res in custom_results.items():
        logging.info(f"{name}: Acc={res['test_acc']:.4f}, Time={res['train_time']:.1f}s, Params={res['params']:,}")
