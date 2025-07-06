import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.training_utils import train_model
from utils.datasets import get_cifar_loaders

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('4/results/cnn_architecture_analysis.log'),
        logging.StreamHandler()
    ]
)

# 1. Модели для исследования размера ядра
class KernelSizeCNN(nn.Module):
    def __init__(self, kernel_sizes=[(3,3)], in_channels=3, num_classes=10):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        filters = [64, 32, 16]  # Количество фильтров для каждого слоя
        
        for i, kernel_size in enumerate(kernel_sizes):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels if i == 0 else filters[i-1], 
                             filters[i], 
                             kernel_size=kernel_size,
                             padding=kernel_size[0]//2),
                    nn.BatchNorm2d(filters[i]),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )
            )
        
        # Автоматический расчет размера входа для FC слоя
        self.fc_input_size = self._get_fc_input_size(in_channels)
        self.fc = nn.Linear(self.fc_input_size, num_classes)
        
    def _get_fc_input_size(self, in_channels):
        # Пробный проход для определения размера
        x = torch.zeros(1, in_channels, 32, 32)  # CIFAR-10 размер 32x32
        for layer in self.conv_layers:
            x = layer(x)
        return x.view(1, -1).size(1)
        
    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 2. Модели для исследования глубины
class DepthCNN(nn.Module):
    def __init__(self, depth=2, in_channels=3, num_classes=10):
        super().__init__()
        self.layers = nn.ModuleList()
        filters = [64, 128, 256, 512, 512, 512]
        
        for i in range(depth):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels if i == 0 else filters[i-1], 
                             filters[i], 
                             kernel_size=3, 
                             padding=1),
                    nn.BatchNorm2d(filters[i]),
                    nn.ReLU(),
                    nn.MaxPool2d(2) if i % 2 == 0 else nn.Identity()
                )
            )
        
        pool_size = 32 // (2 ** (depth // 2))
        self.fc = nn.Linear(filters[depth-1] * pool_size * pool_size, num_classes)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ResidualBlock(nn.Module):
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
        out = F.relu(out)
        return out

class ResidualDepthCNN(nn.Module):
    def __init__(self, depth=6, in_channels=3, num_classes=10):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(ResidualBlock(64, 64))
        
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.initial(x)
        for block in self.blocks:
            x = block(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Вспомогательные функции
def calculate_receptive_field(kernel_sizes):
    rf = 1
    for size in kernel_sizes:
        rf += (size[0] - 1)
    return rf

def analyze_gradients(model):
    gradients = []
    for param in model.parameters():
        if param.grad is not None:
            gradients.append(param.grad.abs().mean().item())
    return {
        'mean': np.mean(gradients),
        'std': np.std(gradients)
    }

def plot_activations(activations, title):
    plt.figure(figsize=(12, 6))
    for i in range(min(16, activations.size(1))):
        plt.subplot(4, 4, i+1)
        plt.imshow(activations[0, i].detach().cpu().numpy(), cmap='viridis')
        plt.axis('off')
    plt.suptitle(title)
    plt.savefig(f'4/plots/activations_{title}.png')
    plt.close()

def plot_feature_maps(features, title):
    plt.figure(figsize=(12, 6))
    for i in range(min(16, features.size(1))):
        plt.subplot(4, 4, i+1)
        plt.imshow(features[0, i].detach().cpu().numpy(), cmap='viridis')
        plt.axis('off')
    plt.suptitle(title)
    plt.savefig(f'4/plots/features_{title}.png')
    plt.close()

def save_results(results, filename):
    os.makedirs('results', exist_ok=True)
    with open(f'results/{filename}', 'w') as f:
        json.dump(results, f, indent=2)

def run_kernel_size_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_cifar_loaders(batch_size=64)
    
    configs = [
        {'name': '3x3', 'kernels': [(3,3), (3,3), (3,3)]},
        {'name': '5x5', 'kernels': [(5,5), (5,5)]},
        {'name': '7x7', 'kernels': [(7,7)]},
        {'name': '1x1+3x3', 'kernels': [(1,1), (3,3), (3,3)]}
    ]
    
    results = {}
    for config in configs:
        model = KernelSizeCNN(config['kernels']).to(device)
        params = sum(p.numel() for p in model.parameters())
        
        logging.info(f"Training {config['name']} with {params:,} parameters")
        logging.info(f"FC input size: {model.fc_input_size}")
        
        history = train_model(model, train_loader, test_loader, epochs=10, device=device)
        
        # Визуализация и сохранение результатов
        sample, _ = next(iter(test_loader))
        activations = model.conv_layers[0](sample.to(device))
        plot_activations(activations, f"kernel_{config['name']}")
        
        results[config['name']] = {
            'test_acc': history['test_accs'][-1],
            'params': params,
            'fc_input': model.fc_input_size,
            'receptive_field': calculate_receptive_field(config['kernels'])
        }
    
    save_results(results, 'kernel_size_results.json')
    return results

def run_depth_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_cifar_loaders(batch_size=64)
    
    configs = [
        {'name': 'Shallow (2)', 'depth': 2},
        {'name': 'Medium (4)', 'depth': 4},
        {'name': 'Deep (6)', 'depth': 6},
        {'name': 'ResNet (6)', 'cls': ResidualDepthCNN}
    ]
    
    results = {}
    for config in configs:
        if 'cls' in config:
            model = config['cls'](depth=6).to(device)
        else:
            model = DepthCNN(config['depth']).to(device)
        
        params = sum(p.numel() for p in model.parameters())
        logging.info(f"Training {config['name']} with {params:,} parameters")
        history = train_model(model, train_loader, test_loader, epochs=15, device=device)
        
        # Анализ градиентов
        grad_stats = analyze_gradients(model)
        
        # Визуализация feature maps
        sample, _ = next(iter(test_loader))
        if hasattr(model, 'initial'):
            features = model.initial(sample.to(device))
            features = model.blocks[0](features)
        else:
            features = model.layers[0](sample.to(device))
        plot_feature_maps(features, f"depth_{config['name']}")
        
        results[config['name']] = {
            'test_acc': history['test_accs'][-1],
            'params': params,
            'grad_mean': grad_stats['mean'],
            'grad_std': grad_stats['std']
        }
    
    save_results(results, 'depth_results.json')
    return results

def plot_results(kernel_results, depth_results):
    # Графики для размеров ядер
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(kernel_results.keys(), [x['test_acc'] for x in kernel_results.values()])
    plt.title('Test Accuracy by Kernel Size')
    plt.ylim(0.5, 0.8)
    
    plt.subplot(1, 2, 2)
    plt.bar(kernel_results.keys(), [x['receptive_field'] for x in kernel_results.values()])
    plt.title('Receptive Field Size')
    plt.savefig('4/plots/kernel_size_results.png')
    plt.close()
    
    # Графики для глубины сети
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.bar(depth_results.keys(), [x['test_acc'] for x in depth_results.values()])
    plt.title('Test Accuracy by Depth')
    plt.ylim(0.5, 0.85)
    
    plt.subplot(1, 3, 2)
    plt.bar(depth_results.keys(), [x['params'] for x in depth_results.values()])
    plt.title('Number of Parameters')
    
    plt.subplot(1, 3, 3)
    plt.errorbar(depth_results.keys(), 
                [x['grad_mean'] for x in depth_results.values()],
                yerr=[x['grad_std'] for x in depth_results.values()],
                fmt='o')
    plt.title('Gradient Statistics')
    plt.savefig('4/plots/depth_results.png')
    plt.close()

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    logging.info("Starting kernel size experiments...")
    kernel_results = run_kernel_size_experiment()
    
    logging.info("\nStarting depth experiments...")
    depth_results = run_depth_experiment()
    
    plot_results(kernel_results, depth_results)
    
    logging.info("\nExperiments completed!")
    logging.info("\nKernel Size Results:")
    for name, res in kernel_results.items():
        logging.info(f"{name}: Accuracy={res['test_acc']:.4f}, RF={res['receptive_field']}")
    
    logging.info("\nDepth Results:")
    for name, res in depth_results.items():
        logging.info(f"{name}: Accuracy={res['test_acc']:.4f}, Gradients={res['grad_mean']:.2e}±{res['grad_std']:.2e}")
