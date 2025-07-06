import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history):
    """Визуализирует историю обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['test_losses'], label='Test Loss')
    ax1.set_title('Loss')
    ax1.legend()
    
    ax2.plot(history['train_accs'], label='Train Acc')
    ax2.plot(history['test_accs'], label='Test Acc')
    ax2.set_title('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def plot_feature_maps(features, title=None, max_maps=16, save_path=None):
    """
    Визуализирует карты признаков (feature maps)
    
    Args:
        features (torch.Tensor): Тензор с картами признаков [batch, channels, height, width]
        title (str): Заголовок для графика
        max_maps (int): Максимальное количество карт для отображения
        save_path (str): Путь для сохранения изображения (None - не сохранять)
    """
    # Конвертируем тензор в numpy и выбираем первый образец в батче
    features = features.detach().cpu().numpy()
    if len(features.shape) == 4:
        features = features[0]  # Берем первый образец в батче
    
    # Ограничиваем количество карт для отображения
    num_channels = min(features.shape[0], max_maps)
    
    # Создаем сетку для отображения
    rows = int(np.ceil(np.sqrt(num_channels)))
    cols = int(np.ceil(num_channels / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Отображаем каждую карту признаков
    for i, ax in enumerate(axes.flat):
        if i < num_channels:
            channel = features[i]
            im = ax.imshow(channel, cmap='viridis')
            ax.axis('off')
            ax.set_title(f'Channel {i}')
        else:
            ax.axis('off')
    
    if title:
        plt.suptitle(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def count_parameters(model):
    """Подсчитывает количество параметров модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model, path):
    """Сохраняет модель"""
    torch.save(model.state_dict(), path)


def load_model(model, path):
    """Загружает модель"""
    model.load_state_dict(torch.load(path))
    return model


def compare_models(fc_history, cnn_history):
    """Сравнивает результаты полносвязной и сверточной сетей"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(fc_history['test_accs'], label='FC Network', marker='o')
    ax1.plot(cnn_history['test_accs'], label='CNN', marker='s')
    ax1.set_title('Test Accuracy Comparison')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(fc_history['test_losses'], label='FC Network', marker='o')
    ax2.plot(cnn_history['test_losses'], label='CNN', marker='s')
    ax2.set_title('Test Loss Comparison')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show() 