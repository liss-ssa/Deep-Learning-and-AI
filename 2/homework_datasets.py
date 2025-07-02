import sklearn
from sklearn.metrics import roc_curve, auc
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, accuracy_score, 
                           precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
from homework_model_modification import LinearRegression, MulticlassLogisticRegression

class CSVDataset(Dataset):
    """
    Кастомный Dataset класс для работы с CSV файлами
    
    Args:
        csv_file (str): Путь к CSV файлу
        target_col (str): Название целевой колонки
        normalize (bool): Нормализовать ли числовые признаки
        task_type (str): Тип задачи ('regression' или 'classification')
    """
    def __init__(self, csv_file, target_col, normalize=True, task_type='regression'):
        self.data = pd.read_csv(csv_file)
        self.target_col = target_col
        self.task_type = task_type
        self.normalize = normalize
        
        # Предобработка данных
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Предобработка данных: кодирование категорий и нормализация"""
        # Отделяем целевую переменную
        self.y = self.data[self.target_col].values
        self.X = self.data.drop(columns=[self.target_col])
        
        # Кодируем категориальные признаки
        self.categorical_cols = self.X.select_dtypes(include=['object', 'category']).columns
        self.encoders = {}
        
        for col in self.categorical_cols:
            le = LabelEncoder()
            self.X[col] = le.fit_transform(self.X[col])
            self.encoders[col] = le
        
        # Нормализуем числовые признаки
        self.numeric_cols = self.X.select_dtypes(include=['int64', 'float64']).columns
        if self.normalize and len(self.numeric_cols) > 0:
            self.scaler = StandardScaler()
            self.X[self.numeric_cols] = self.scaler.fit_transform(self.X[self.numeric_cols])
        
        # Преобразуем в тензоры
        self.X = torch.FloatTensor(self.X.values)
        
        if self.task_type == 'regression':
            self.y = torch.FloatTensor(self.y).view(-1, 1)
        else:
            self.label_encoder = LabelEncoder()
            self.y = self.label_encoder.fit_transform(self.y)
            self.y = torch.LongTensor(self.y)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def get_feature_dim(self):
        """Возвращает размерность признаков"""
        return self.X.shape[1]
    
    def get_num_classes(self):
        """Возвращает количество классов (для классификации)"""
        if hasattr(self, 'label_encoder'):
            return len(self.label_encoder.classes_)
        return 1

def train_and_evaluate(model, train_loader, val_loader, epochs=100, task_type='regression'):
    """Обучение и оценка модели"""
    criterion = torch.nn.MSELoss() if task_type == 'regression' else torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    
    # Оценка на валидационном наборе
    model.eval()
    val_preds, val_true = [], []
    with torch.no_grad():
        for X_val, y_val in val_loader:
            outputs = model(X_val)
            val_preds.extend(outputs.numpy())
            val_true.extend(y_val.numpy())
    
    if task_type == 'regression':
        mse = mean_squared_error(val_true, val_preds)
        r2 = sklearn.metrics.r2_score(val_true, val_preds)
        print(f"Validation MSE: {mse:.4f}, R2 Score: {r2:.4f}")
        plt.scatter(val_true, val_preds)
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.title("Regression Results")
        plt.savefig("2/plots/regression_results.png")
        plt.close()
    else:
        val_preds = np.argmax(val_preds, axis=1) if len(np.array(val_preds).shape) > 1 else (np.array(val_preds) > 0.5).astype(int)
        acc = accuracy_score(val_true, val_preds)
        precision = precision_score(val_true, val_preds, average='weighted')
        recall = recall_score(val_true, val_preds, average='weighted')
        f1 = f1_score(val_true, val_preds, average='weighted')
        
        print(f"Validation Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Confusion matrix
        plot_confusion_matrix(val_true, val_preds, class_names=model.label_encoder.classes_ if hasattr(model, 'label_encoder') else None)

        # ROC-кривая для бинарной классификации
        if len(np.unique(val_true)) == 2: 
            fpr, tpr, thresholds = roc_curve(val_true, val_preds)
            roc_auc = auc(fpr, tpr)
            
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig("2/plots/roc_curve.png")
            plt.close()
            print(f"AUC-ROC: {roc_auc:.4f}")

def plot_confusion_matrix(y_true, y_pred, class_names=None, title="Confusion Matrix"):
    """Визуализация матрицы ошибок"""
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig("2/plots/confusion_matrix_dataset.png")
    plt.close()

def experiment_with_real_datasets():
    """Эксперименты с реальными датасетами"""
    
    # 1. Регрессия: Акции Tesla (предсказание цены закрытия)
    print("\n=== Регрессия: Акции Tesla ===")
    tesla_dataset = CSVDataset("2/data/Tesla.csv", "Close", task_type='regression')
    X_train, X_val = train_test_split(tesla_dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(X_train, batch_size=32, shuffle=True)
    val_loader = DataLoader(X_val, batch_size=32)
    
    lin_model = LinearRegression(in_features=tesla_dataset.get_feature_dim())
    train_and_evaluate(lin_model, train_loader, val_loader, task_type='regression')
    torch.save(lin_model.state_dict(), "2/models/tesla_model.pth")
    
    # 2. Регрессия: Автомобили (предсказание цены)
    print("\n=== Регрессия: Автомобили в Молдове ===")
    cars_dataset = CSVDataset("2/data/cars_moldova_cat_num.csv", "Price(euro)", task_type='regression')
    X_train, X_val = train_test_split(cars_dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(X_train, batch_size=32, shuffle=True)
    val_loader = DataLoader(X_val, batch_size=32)
    
    car_model = LinearRegression(in_features=cars_dataset.get_feature_dim())
    train_and_evaluate(car_model, train_loader, val_loader, task_type='regression')
    torch.save(car_model.state_dict(), "2/models/cars_model.pth")
    
    # 3. Классификация: Изюм (предсказание сорта)
    print("\n=== Классификация: Изюм ===")
    raisin_dataset = CSVDataset("2/data/Raisin_Dataset.csv", "Class", task_type='classification')
    X_train, X_val = train_test_split(raisin_dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(X_train, batch_size=32, shuffle=True)
    val_loader = DataLoader(X_val, batch_size=32)
    
    raisin_model = MulticlassLogisticRegression(
        in_features=raisin_dataset.get_feature_dim(),
        num_classes=raisin_dataset.get_num_classes()
    )
    
    raisin_model.label_encoder = raisin_dataset.label_encoder
    
    train_and_evaluate(raisin_model, train_loader, val_loader, task_type='classification')
    torch.save({
        'model_state_dict': raisin_model.state_dict(),
        'label_encoder': raisin_dataset.label_encoder
    }, "2/models/raisin_model.pth")

if __name__ == '__main__':
    experiment_with_real_datasets()

    