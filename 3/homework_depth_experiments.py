import torch
import time
from datasets import get_mnist_loaders, get_cifar_loaders
from models import FullyConnectedModel
from trainer import train_model
from utils import count_parameters
import matplotlib.pyplot as plt
plt.ion()
import os

# Настройки
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
epochs = 10
results_dir = '3/results/depth_experiments'
os.makedirs(results_dir, exist_ok=True)

# Загрузка данных
mnist_train, mnist_test = get_mnist_loaders(batch_size)
cifar_train, cifar_test = get_cifar_loaders(batch_size)

# Конфигурации моделей разной глубины
depth_configs = [
    {'name': '1_layer', 'layers': []},  # Только выходной слой
    {'name': '2_layers', 'layers': [{'type': 'linear', 'size': 256}, {'type': 'relu'}]},
    {'name': '3_layers', 'layers': [{'type': 'linear', 'size': 256}, {'type': 'relu'}, 
                                    {'type': 'linear', 'size': 128}, {'type': 'relu'}]},
    {'name': '5_layers', 'layers': [{'type': 'linear', 'size': 512}, {'type': 'relu'}, 
                                    {'type': 'linear', 'size': 256}, {'type': 'relu'},
                                    {'type': 'linear', 'size': 128}, {'type': 'relu'},
                                    {'type': 'linear', 'size': 64}, {'type': 'relu'}]},
    {'name': '7_layers', 'layers': [{'type': 'linear', 'size': 512}, {'type': 'relu'},
                                    {'type': 'linear', 'size': 512}, {'type': 'relu'},
                                    {'type': 'linear', 'size': 256}, {'type': 'relu'},
                                    {'type': 'linear', 'size': 256}, {'type': 'relu'},
                                    {'type': 'linear', 'size': 128}, {'type': 'relu'},
                                    {'type': 'linear', 'size': 64}, {'type': 'relu'}]}
]

def run_depth_experiment(dataset_name, train_loader, test_loader, input_size):
    """Запускает эксперименты с разной глубиной сети."""
    results = []
    
    for config in depth_configs:
        print(f"\n=== {dataset_name.upper()}: {config['name']} ===")
        
        # Создание модели
        model = FullyConnectedModel(
            input_size=input_size,
            num_classes=10,
            layers=config['layers']
        ).to(device)
        
        print(f"Параметров: {count_parameters(model):,}")
        
        # Обучение
        start_time = time.time()
        history = train_model(model, train_loader, test_loader, epochs=epochs, device=device)
        train_time = time.time() - start_time
        
        # Сохранение результатов
        results.append({
            'name': config['name'],
            'train_acc': history['train_accs'][-1],
            'test_acc': history['test_accs'][-1],
            'train_time': train_time,
            'history': history
        })
        
        # Визуализация
        plt.figure(figsize=(10, 4))
        plt.plot(history['train_accs'], label='Train Accuracy')
        plt.plot(history['test_accs'], label='Test Accuracy')
        plt.title(f"{dataset_name} - {config['name']} (Accuracy)")
        plt.legend()
        plt.savefig(f"{results_dir}/{dataset_name}_{config['name']}_acc.png")
        plt.close()
    
    return results

# Запуск экспериментов для MNIST
mnist_results = run_depth_experiment(
    dataset_name='mnist',
    train_loader=mnist_train,
    test_loader=mnist_test,
    input_size=784
)

# Запуск экспериментов для CIFAR
cifar_results = run_depth_experiment(
    dataset_name='cifar',
    train_loader=cifar_train,
    test_loader=cifar_test,
    input_size=3072
)

def print_training_summary(history, model_name):
    """Выводит итоговые метрики обучения без графиков"""
    print(f"\nРезультаты для {model_name}:")
    print(f"Финальная Train Accuracy: {history['train_accs'][-1]:.4f}")
    print(f"Финальная Test Accuracy: {history['test_accs'][-1]:.4f}")
    print(f"Min Train Loss: {min(history['train_losses']):.4f}")
    print(f"Min Test Loss: {min(history['test_losses']):.4f}")
    
    # Анализ переобучения
    overfit_gap = history['train_accs'][-1] - history['test_accs'][-1]
    print(f"Разрыв Train/Test Accuracy: {overfit_gap:.4f} (чем больше, тем хуже)")

# Анализ переобучения (добавление Dropout/BatchNorm)
def run_regularization_experiment():
    """Сравнение моделей с регуляризацией."""
    config = depth_configs[3]  # 5 слоёв (выбрана как склонная к переобучению)
    
    # Варианты регуляризации
    variants = [
        {'name': 'no_reg', 'layers': config['layers']},
        {'name': 'dropout', 'layers': config['layers'] + [{'type': 'dropout', 'rate': 0.5}]},
        {'name': 'batchnorm', 'layers': [{'type': 'batch_norm'} if i % 2 == 1 else layer 
                                        for i, layer in enumerate(config['layers'])]},
        {'name': 'both', 'layers': [{'type': 'batch_norm'} if i % 2 == 1 else layer 
                                   for i, layer in enumerate(config['layers'])] + [{'type': 'dropout', 'rate': 0.3}]}
    ]
    
    for var in variants:
        model = FullyConnectedModel(
            input_size=3072,
            num_classes=10,
            layers=var['layers']
        ).to(device)
        
        history = train_model(model, cifar_train, cifar_test, epochs=epochs, device=device)
        print_training_summary(history, "5 слоёв + Dropout")
        plt.show()
        plt.title(f"CIFAR-10 5 слоёв + {var['name']}")
        plt.show()

run_regularization_experiment()
