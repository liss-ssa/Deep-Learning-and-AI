import os
import time
import json
import logging
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from models.fc_models import FCNetwork
from models.cnn_models import SimpleCNN, CNNWithResidual, CIFARCNN
from utils.training_utils import train_model
from utils.visualization_utils import plot_training_history as original_plot_history
from utils.datasets import get_mnist_loaders, get_cifar_loaders

# Модифицированная функция для визуализации
def plot_training_history(history, title=None, save_path=None):
    """Обертка для визуализации истории обучения с поддержкой заголовка и сохранения"""
    original_plot_history(history)
    fig = plt.gcf()
    if title:
        fig.suptitle(title)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()

RESULTS_ROOT = "4/results"
PLOTS_ROOT = "4/plots"
os.makedirs(RESULTS_ROOT, exist_ok=True)
os.makedirs(PLOTS_ROOT, exist_ok=True)

def setup_logger(dataset_name, model_name):
    """Настраивает логгер для конкретной модели и датасета"""
    logger = logging.getLogger(f"{dataset_name}_{model_name}")
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    log_dir = os.path.join(RESULTS_ROOT, f"{dataset_name}_comparison", "logs")
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{model_name}.log"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def save_metrics(metrics, dataset_name, model_name):
    """Сохраняет метрики в JSON файл"""
    metrics_dir = os.path.join(RESULTS_ROOT, f"{dataset_name}_comparison", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    with open(os.path.join(metrics_dir, f"{model_name}.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

def save_model(model, dataset_name, model_name):
    """Сохраняет веса модели"""
    models_dir = os.path.join(RESULTS_ROOT, f"{dataset_name}_comparison", "models")
    os.makedirs(models_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(models_dir, f"{model_name}.pth"))

def plot_confusion_matrix(model, loader, device, dataset_name, model_name):
    """Генерирует и сохраняет confusion matrix"""
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name} on {dataset_name}')
    
    plot_dir = os.path.join(PLOTS_ROOT, dataset_name)
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f'cm_{model_name}.png'))
    plt.close()

def run_experiment(models_dict, train_loader, test_loader, device, dataset_name):
    """Запускает эксперимент для набора моделей"""
    results = {}
    
    for model_name, model in models_dict.items():
        logger = setup_logger(dataset_name, model_name)
        logger.info(f"\n{'='*50}\nTraining {model_name} on {dataset_name}\n{'='*50}")
        
        start_time = time.time()
        history = train_model(
            model, train_loader, test_loader,
            epochs=10, lr=0.001, device=device
        )
        train_time = time.time() - start_time
        
        with torch.no_grad():
            start_inference = time.time()
            for data, _ in test_loader:
                _ = model(data.to(device))
            inference_time = (time.time() - start_inference) / len(test_loader)
        
        metrics = {
            'train_history': {
                'loss': history['train_losses'],
                'accuracy': history['train_accs']
            },
            'test_history': {
                'loss': history['test_losses'],
                'accuracy': history['test_accs']
            },
            'performance': {
                'train_time': train_time,
                'inference_time': inference_time,
                'params': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'final_test_acc': history['test_accs'][-1]
            }
        }
        
        save_metrics(metrics, dataset_name, model_name)
        save_model(model, dataset_name, model_name)
        
        plot_training_history(history,
                            title=f"{model_name} on {dataset_name}",
                            save_path=os.path.join(PLOTS_ROOT, dataset_name, f"training_{model_name}.png"))
        
        if 'CNN' in model_name:
            plot_confusion_matrix(model, test_loader, device, dataset_name, model_name)
        
        logger.info(f"Training completed in {train_time:.2f} seconds")
        logger.info(f"Final test accuracy: {metrics['performance']['final_test_acc']:.4f}")
        logger.info(f"Model parameters: {metrics['performance']['params']:,}")
        
        results[model_name] = metrics
    
    return results

def generate_summary_report(all_results):
    """Генерирует сводный отчет по всем экспериментам"""
    summary = {}
    
    for dataset_name, results in all_results.items():
        summary[dataset_name] = {}
        for model_name, metrics in results.items():
            summary[dataset_name][model_name] = {
                'final_test_acc': metrics['performance']['final_test_acc'],
                'train_time': metrics['performance']['train_time'],
                'params': metrics['performance']['params']
            }
    
    summary_path = os.path.join(RESULTS_ROOT, "comparison_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    report_path = os.path.join(RESULTS_ROOT, "report.txt")
    with open(report_path, 'w') as f:
        f.write("Model Comparison Summary\n")
        f.write("="*50 + "\n\n")
        
        for dataset_name, models in summary.items():
            f.write(f"{dataset_name} Results:\n")
            f.write("-"*50 + "\n")
            for model_name, metrics in models.items():
                f.write(f"{model_name}:\n")
                f.write(f"  Test Accuracy: {metrics['final_test_acc']:.4f}\n")
                f.write(f"  Train Time: {metrics['train_time']:.2f}s\n")
                f.write(f"  Parameters: {metrics['params']:,}\n\n")
    
    return summary

def get_mnist_models(device):
    return {
        'FC': FCNetwork(input_size=28*28).to(device),
        'SimpleCNN': SimpleCNN(input_channels=1).to(device),
        'ResidualCNN': CNNWithResidual(input_channels=1).to(device)
    }

def get_cifar_models(device):
    return {
        'FC': FCNetwork(input_size=3*32*32).to(device),
        'ResidualCNN': CNNWithResidual(input_channels=3).to(device),
        'ResCNN_Regularized': CIFARCNN().to(device)
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    all_results = {}
    
    # MNIST эксперимент
    mnist_train, mnist_test = get_mnist_loaders(batch_size=64)
    mnist_models = get_mnist_models(device)
    all_results['MNIST'] = run_experiment(
        mnist_models, mnist_train, mnist_test, device, 'MNIST'
    )
    
    # CIFAR-10 эксперимент
    cifar_train, cifar_test = get_cifar_loaders(batch_size=64)
    cifar_models = get_cifar_models(device)
    all_results['CIFAR-10'] = run_experiment(
        cifar_models, cifar_train, cifar_test, device, 'CIFAR-10'
    )
    
    summary = generate_summary_report(all_results)
    
    print("\nFinal Summary:")
    for dataset, models in summary.items():
        print(f"\n{dataset}:")
        for model, metrics in models.items():
            print(f"  {model}:")
            print(f"    Test Acc: {metrics['final_test_acc']:.4f}")
            print(f"    Train Time: {metrics['train_time']:.2f}s")
            print(f"    Params: {metrics['params']:,}")

if __name__ == "__main__":
    main()