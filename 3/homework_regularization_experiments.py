import torch
import torch.nn as nn
import torch.optim as optim
from datasets import get_cifar_loaders
from models import FullyConnectedModel
from trainer import train_model
from utils import count_parameters, plot_training_history
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import logging
from tqdm import tqdm

# Настройки
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
epochs = 15
results_dir = '3/results/regularization_experiments'
os.makedirs(results_dir, exist_ok=True)

# Настройка логирования
logging.basicConfig(
    filename=os.path.join(results_dir, 'experiments.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Загрузка данных
train_loader, test_loader = get_cifar_loaders(batch_size)

# Базовые параметры модели
base_config = {
    'input_size': 3072,  # 32x32x3 для CIFAR-10
    'num_classes': 10,
    'layers': [
        {'type': 'linear', 'size': 512},
        {'type': 'relu'},
        {'type': 'linear', 'size': 256},
        {'type': 'relu'},
        {'type': 'linear', 'size': 128},
        {'type': 'relu'}
    ]
}

def run_regularization_experiments():
    """Основная функция для экспериментов с регуляризацией."""
    # 3.1 Сравнение техник регуляризации
    logging.info("Starting regularization techniques comparison")
    basic_results = compare_regularization_techniques()
    
    # 3.2 Адаптивная регуляризация
    logging.info("Starting adaptive regularization experiments")
    adaptive_results = run_adaptive_regularization()
    
    return basic_results, adaptive_results

def compare_regularization_techniques():
    """Сравнение различных техник регуляризации."""
    techniques = [
        {'name': 'no_reg', 'layers': base_config['layers'], 'weight_decay': 0.0},
        {'name': 'dropout_0.1', 'layers': add_dropout(base_config['layers'], 0.1), 'weight_decay': 0.0},
        {'name': 'dropout_0.3', 'layers': add_dropout(base_config['layers'], 0.3), 'weight_decay': 0.0},
        {'name': 'dropout_0.5', 'layers': add_dropout(base_config['layers'], 0.5), 'weight_decay': 0.0},
        {'name': 'batchnorm', 'layers': add_batchnorm(base_config['layers']), 'weight_decay': 0.0},
        {'name': 'dropout_0.3_batchnorm', 'layers': add_batchnorm(add_dropout(base_config['layers'], 0.3)), 'weight_decay': 0.0},
        {'name': 'l2_reg', 'layers': base_config['layers'], 'weight_decay': 0.01},
        {'name': 'combined', 'layers': add_batchnorm(add_dropout(base_config['layers'], 0.3)), 'weight_decay': 0.001}
    ]
    
    results = []
    for tech in techniques:
        logging.info(f"Running experiment: {tech['name']}")
        model = FullyConnectedModel(
            input_size=base_config['input_size'],
            num_classes=base_config['num_classes'],
            layers=tech['layers']
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=tech['weight_decay'])
        
        history = train_model_with_details(model, train_loader, test_loader, criterion, optimizer, epochs, device)
        
        # Визуализация кривых обучения
        plot_training_history(history)
        plt.title(f"Learning curves - {tech['name']}")
        plt.savefig(os.path.join(results_dir, f"learning_curves_{tech['name']}.png"))
        plt.close()
        
        # Анализ весов
        analyze_weights(model, tech['name'])
        
        results.append({
            'technique': tech['name'],
            'train_acc': history['train_accs'][-1],
            'test_acc': history['test_accs'][-1],
            'overfitting_gap': history['train_accs'][-1] - history['test_accs'][-1],
            'min_train_loss': min(history['train_losses']),
            'min_test_loss': min(history['test_losses'])
        })
    
    # Сохранение и визуализация результатов
    plot_techniques_comparison(results)
    return results

def run_adaptive_regularization():
    """Эксперименты с адаптивной регуляризацией."""
    adaptive_configs = [
        {
            'name': 'progressive_dropout',
            'layers': create_progressive_dropout_layers(base_config['layers']), 
            'weight_decay': 0.0
        },
        {
            'name': 'batchnorm_variable_momentum',
            'layers': create_batchnorm_variable_momentum(base_config['layers']),
            'weight_decay': 0.0
        },
        {
            'name': 'layerwise_combined',
            'layers': create_layerwise_combined_reg(base_config['layers']),
            'weight_decay': 0.001
        }
    ]
    
    results = []
    for config in adaptive_configs:
        logging.info(f"Running adaptive experiment: {config['name']}")
        model = FullyConnectedModel(
            input_size=base_config['input_size'],
            num_classes=base_config['num_classes'],
            layers=config['layers']
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=config['weight_decay'])
        
        history = train_model_with_details(model, train_loader, test_loader, criterion, optimizer, epochs, device)
        
        # Анализ стабильности обучения
        analyze_training_stability(history, config['name'])
        
        results.append({
            'technique': config['name'],
            'train_acc': history['train_accs'][-1],
            'test_acc': history['test_accs'][-1],
            'loss_std': np.std(history['test_losses'][-5:]),  # Стандартное отклонение последних 5 эпох
            'acc_std': np.std(history['test_accs'][-5:])
        })
    
    # Визуализация сравнения адаптивных методов
    plot_adaptive_results(results)
    return results

# Вспомогательные функции для модификации слоев
def add_dropout(layers, rate):
    """Добавляет Dropout после каждого ReLU."""
    new_layers = []
    for layer in layers:
        new_layers.append(layer)
        if layer['type'] == 'relu':
            new_layers.append({'type': 'dropout', 'rate': rate})
    return new_layers

def add_batchnorm(layers):
    """Добавляет BatchNorm после каждого Linear слоя."""
    new_layers = []
    for layer in layers:
        new_layers.append(layer)
        if layer['type'] == 'linear':
            new_layers.append({'type': 'batch_norm'})
    return new_layers

def create_progressive_dropout_layers(layers):
    """Создает слои с прогрессивно увеличивающимся Dropout."""
    new_layers = []
    relu_count = sum(1 for layer in layers if layer['type'] == 'relu')
    current_dropout = 0.1
    
    for layer in layers:
        new_layers.append(layer)
        if layer['type'] == 'relu':
            new_layers.append({'type': 'dropout', 'rate': current_dropout})
            current_dropout += 0.1  # Увеличиваем dropout для следующих слоев
    return new_layers

def create_batchnorm_variable_momentum(layers):
    """Создает слои с BatchNorm и разным momentum."""
    new_layers = []
    bn_count = 0
    
    for layer in layers:
        new_layers.append(layer)
        if layer['type'] == 'linear':
            # Уменьшаем momentum для более глубоких слоев
            momentum = max(0.1, 0.9 - bn_count * 0.2)
            new_layers.append({
                'type': 'batch_norm',
                'momentum': momentum
            })
            bn_count += 1
    return new_layers

def create_layerwise_combined_reg(layers):
    """Комбинированная регуляризация с разными методами для разных слоев."""
    new_layers = []
    layer_depth = 0
    
    for layer in layers:
        new_layers.append(layer)
        
        if layer['type'] == 'linear':
            # Для первых слоев используем BatchNorm
            if layer_depth < 2:
                new_layers.append({'type': 'batch_norm'})
            # Для средних слоев используем Dropout
            elif 2 <= layer_depth < 4:
                new_layers.append({'type': 'dropout', 'rate': 0.3})
            # Для последних слоев используем LayerNorm
            else:
                new_layers.append({'type': 'layer_norm'})
            
            layer_depth += 1
    return new_layers

# Функции для анализа и визуализации
def train_model_with_details(model, train_loader, test_loader, criterion, optimizer, epochs, device):
    """Расширенная версия train_model с дополнительной логикой."""
    history = {
        'train_losses': [],
        'train_accs': [],
        'test_losses': [],
        'test_accs': []
    }
    
    for epoch in range(epochs):
        model.train()
        train_loss, train_acc = 0, 0
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            train_acc += pred.eq(target.view_as(pred)).sum().item()
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)
        
        # Оценка на тестовом наборе
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        
        history['train_losses'].append(train_loss)
        history['train_accs'].append(train_acc)
        history['test_losses'].append(test_loss)
        history['test_accs'].append(test_acc)
        
        logging.info(
            f"Epoch {epoch+1}: "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
        )
    
    return history

def evaluate_model(model, data_loader, criterion, device):
    """Оценка модели на данных."""
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / len(data_loader.dataset)
    return avg_loss, accuracy

def analyze_weights(model, technique_name):
    """Анализ распределения весов модели."""
    weights = []
    for name, param in model.named_parameters():
        if 'weight' in name and 'bn' not in name and 'ln' not in name:
            weights.extend(param.data.cpu().numpy().flatten())
    
    plt.figure(figsize=(10, 5))
    plt.hist(weights, bins=50, alpha=0.7)
    plt.title(f"Weight distribution - {technique_name}")
    plt.xlabel("Weight value")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(results_dir, f"weight_dist_{technique_name}.png"))
    plt.close()

def analyze_training_stability(history, technique_name):
    """Анализ стабильности обучения."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Train')
    plt.plot(history['test_losses'], label='Test')
    plt.title(f"Loss - {technique_name}")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accs'], label='Train')
    plt.plot(history['test_accs'], label='Test')
    plt.title(f"Accuracy - {technique_name}")
    plt.legend()
    
    plt.savefig(os.path.join(results_dir, f"stability_{technique_name}.png"))
    plt.close()

def plot_techniques_comparison(results):
    """Визуализация сравнения техник регуляризации."""
    techniques = [r['technique'] for r in results]
    test_accs = [r['test_acc'] for r in results]
    overfitting_gaps = [r['overfitting_gap'] for r in results]
    
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(techniques, test_accs)
    plt.title('Test Accuracy by Regularization Technique')
    plt.xticks(rotation=45)
    plt.ylabel('Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.bar(techniques, overfitting_gaps)
    plt.title('Overfitting Gap (Train Acc - Test Acc)')
    plt.xticks(rotation=45)
    plt.ylabel('Gap')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'techniques_comparison.png'))
    plt.close()

def plot_adaptive_results(results):
    """Визуализация результатов адаптивной регуляризации."""
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(df['technique'], df['test_acc'])
    plt.title('Test Accuracy')
    plt.ylabel('Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.bar(df['technique'], df['loss_std'])
    plt.title('Loss Stability (std of last 5 epochs)')
    plt.ylabel('Standard Deviation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'adaptive_results.png'))
    plt.close()

if __name__ == "__main__":
    basic_results, adaptive_results = run_regularization_experiments()