import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix)
import logging
import os
import unittest

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("2/training.log"),
        logging.StreamHandler()
    ]
)

class LinearRegression(nn.Module):
    def __init__(self, in_features, l1_lambda=0.01, l2_lambda=0.01):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.best_loss = float('inf')
        self.best_state = None

    # Прямой проход модели
    def forward(self, x) -> torch.Tensor:
        return self.linear(x)
    
    # Вычисление регуляризационных потерь
    def regularization_loss(self) -> torch.Tensor:
        l1_loss = sum(p.abs().sum() for p in self.parameters())
        l2_loss = sum(p.pow(2).sum() for p in self.parameters())
        return self.l1_lambda * l1_loss + self.l2_lambda * l2_loss
    
    # Сохранение лучших весов модели
    def save_best(self, current_loss):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_state = {k: v.clone() for k, v in self.state_dict().items()}
            logging.info(f"New best loss: {current_loss:}")
    
    # Восстановление лучших весов модели
    def restore_best(self):
        if self.best_state is not None:
            self.load_state_dict(self.best_state)
            logging.info("Best weights restored")

class MulticlassLogisticRegression(nn.Module):
    """
    Многоклассовая логистическая регрессия с метриками качества
    
    Args:
        in_features (int): Количество входных признаков
        num_classes (int): Количество классов
    """
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    # Прямой проход модели    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.linear(x), dim=1)
    
    # Сохранение модели
    def save(self, path: str = "2/models/logreg_model.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        logging.info(f"Model saved to {path}")    

    # Загрузка модели
    @classmethod
    def load(cls, in_features: int, num_classes: int, path: str = "2/models/logreg_model.pth"):
        model = cls(in_features, num_classes)
        model.load_state_dict(torch.load(path))
        model.eval()
        logging.info(f"Model loaded from {path}")
        return model
    
def train_linear_regression(model: nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader,
    epochs: int = 100,
    patience: int = 5
) -> None:
    """
    Обучение модели линейной регрессии с early stopping
    
    Args:
        model: Модель для обучения
        train_loader: DataLoader для обучающих данных
        val_loader: DataLoader для валидационных данных
        epochs: Максимальное количество эпох
        patience: Количество эпох без улучшения для early stopping
    """
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    no_improve = 0
    
    logging.info("Starting linear regression training")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # Обучение на батчах
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch) + model.regularization_loss()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Валидация
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                y_pred = model(X_val)
                val_loss += criterion(y_pred, y_val).item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        model.save_best(avg_val_loss)
        
        # Логирование
        logging.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {avg_train_loss:.4f} - "
            f"Val Loss: {avg_val_loss:.4f}"
        )
        
        # Early stopping
        if avg_val_loss >= model.best_loss:
            no_improve += 1
            if no_improve >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break
        else:
            no_improve = 0
    
    model.restore_best()

def evaluate_classification(model: nn.Module, test_loader: torch.utils.data.DataLoader, num_classes) -> dict:
    """
    Оценка модели классификации с различными метриками
    
    Args:
        model: Модель для оценки
        test_loader: DataLoader для тестовых данных
        num_classes: Количество классов
    
    Returns:
        Словарь с метриками качества
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            probs, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            if num_classes == 2:
                all_probs.extend(outputs[:, 1].cpu().numpy())
    
    metrics = {
        'precision': precision_score(all_targets, all_preds, average='weighted'),
        'recall': recall_score(all_targets, all_preds, average='weighted'),
        'f1': f1_score(all_targets, all_preds, average='weighted'),
    }
    
    if num_classes == 2:
        metrics['roc_auc'] = roc_auc_score(all_targets, all_probs)
    
    # Визуализация матрицы ошибок
    plot_confusion_matrix(all_targets, all_preds)
    
    return metrics

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Confusion Matrix") -> None:
    """
    Построение и сохранение матрицы ошибок
    
    Args:
        y_true: Истинные метки
        y_pred: Предсказанные метки
        title: Заголовок графика
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    os.makedirs("plots", exist_ok=True)
    plt.savefig("2/plots/confusion_matrix.png")
    plt.close()
    logging.info("Confusion matrix saved to 2/plots/confusion_matrix.png")

# Юнит-тесты для критических функций
class TestModels(unittest.TestCase):
    
    def test_linear_regression_forward(self):
        model = LinearRegression(in_features=5)
        x = torch.randn(10, 5)
        y = model(x)
        self.assertEqual(y.shape, (10, 1))
    
    def test_logistic_regression_forward(self):
        model = MulticlassLogisticRegression(in_features=4, num_classes=3)
        x = torch.randn(10, 4)
        y = model(x)
        self.assertEqual(y.shape, (10, 3))
        self.assertTrue(torch.allclose(y.sum(dim=1), torch.ones(10)))
    
    def test_regularization_loss(self):
        model = LinearRegression(in_features=3, l1_lambda=0.1, l2_lambda=0.1)
        loss = model.regularization_loss()
        self.assertGreaterEqual(loss.item(), 0)

if __name__ == '__main__':
    # Запуск тестов
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    logging.info("Example usage with synthetic data")
    
    # Пример для линейной регрессии
    X = torch.randn(100, 3)
    y = X @ torch.tensor([1.5, -2.0, 0.5]) + 0.1 * torch.randn(100)
    
    dataset = torch.utils.data.TensorDataset(X, y)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
    
    lin_model = LinearRegression(in_features=3)
    train_linear_regression(lin_model, train_loader, val_loader, epochs=50)
    torch.save(lin_model.state_dict(), "2/models/linear_model.pth")
    
    # Пример для логистической регрессии
    X_clf = torch.randn(100, 4)
    y_clf = (X_clf[:, 0] > 0).long()
    
    clf_dataset = torch.utils.data.TensorDataset(X_clf, y_clf)
    test_loader = torch.utils.data.DataLoader(clf_dataset, batch_size=16)
    
    logreg_model = MulticlassLogisticRegression(in_features=4, num_classes=2)
    metrics = evaluate_classification(logreg_model, test_loader, num_classes=2)
    logging.info(f"Classification metrics: {metrics}")
    logreg_model.save("2/models/logreg_model.pth")

