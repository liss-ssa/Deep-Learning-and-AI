import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from homework_datasets import CSVDataset
from homework_model_modification import LinearRegression, MulticlassLogisticRegression
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import os

# Настройка директорий
os.makedirs("2/plots/experiments", exist_ok=True)
os.makedirs("2/models/experiments", exist_ok=True)

def run_complete_experiments():
    """Полные эксперименты для всех задач"""
    # Регрессионные датасеты
    regression_data = [
        ("cars_moldova_cat_num.csv", "Price(euro)", "regression"),
        ("Tesla.csv", "Close", "regression")
    ]
    
    # Классификационные датасеты
    classification_data = [
        ("Raisin_Dataset.csv", "Class", "classification")
    ]
    
    # Запуск экспериментов
    for dataset, target, task_type in regression_data + classification_data:
        print(f"\n=== Эксперименты для {dataset} ===")
        if task_type == "regression":
            run_full_regression_experiments(dataset, target)
        else:
            run_full_classification_experiments(dataset, target)

def run_full_regression_experiments(dataset_name, target_col):
    """Полные эксперименты для регрессии"""
    # Загрузка данных
    dataset = CSVDataset(f"2/data/{dataset_name}", target_col, task_type="regression")
    X, y = dataset.X.numpy(), dataset.y.numpy()
    
    # Разделение данных
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. Эксперименты с гиперпараметрами
    hp_results = []
    
    # Конфигурации для тестирования
    configs = [
        {"lr": 0.001, "bs": 32, "opt": "SGD"},
        {"lr": 0.01, "bs": 32, "opt": "SGD"},
        {"lr": 0.1, "bs": 32, "opt": "SGD"},
        {"lr": 0.01, "bs": 16, "opt": "SGD"},
        {"lr": 0.01, "bs": 64, "opt": "SGD"},
        {"lr": 0.01, "bs": 32, "opt": "Adam"},
        {"lr": 0.01, "bs": 32, "opt": "RMSprop"}
    ]
    
    for config in configs:
        # Подготовка DataLoader
        train_data = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        train_loader = DataLoader(train_data, batch_size=config["bs"], shuffle=True)
        
        val_data = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        val_loader = DataLoader(val_data, batch_size=config["bs"])
        
        # Инициализация модели
        model = LinearRegression(in_features=X.shape[1])
        
        # Настройка оптимизатора
        if config["opt"] == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
        elif config["opt"] == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        else:
            optimizer = torch.optim.RMSprop(model.parameters(), lr=config["lr"])
        
        # Обучение
        mse = train_regression_model(model, train_loader, val_loader, optimizer, epochs=50)
        
        # Сохранение результатов
        hp_results.append({
            "Dataset": dataset_name,
            "Learning Rate": config["lr"],
            "Batch Size": config["bs"],
            "Optimizer": config["opt"],
            "MSE": mse
        })
    
    # 2. Feature Engineering эксперименты
    fe_results = []
    
    # Базовые признаки
    base_mse = hp_results[1]["MSE"]  # Используем лучшую конфигурацию
    fe_results.append({"Feature Type": "Base", "MSE": base_mse})
    
    # Полиномиальные признаки (степень 2)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Разделение полиномиальных признаков
    X_train_poly, X_val_poly, y_train_poly, y_val_poly = train_test_split(
        X_poly, y, test_size=0.2, random_state=42)
    
    # Обучение на полиномиальных признаках
    train_data = TensorDataset(torch.FloatTensor(X_train_poly), torch.FloatTensor(y_train_poly))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    
    val_data = TensorDataset(torch.FloatTensor(X_val_poly), torch.FloatTensor(y_val_poly))
    val_loader = DataLoader(val_data, batch_size=32)
    
    model = LinearRegression(in_features=X_poly.shape[1])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    poly_mse = train_regression_model(model, train_loader, val_loader, optimizer)
    fe_results.append({"Feature Type": "Polynomial", "MSE": poly_mse})
    
    # Сохранение результатов
    save_results(hp_results, fe_results, dataset_name, "regression")

def run_full_classification_experiments(dataset_name, target_col):
    """Полные эксперименты для классификации"""
    # Загрузка данных
    dataset = CSVDataset(f"2/data/{dataset_name}", target_col, task_type="classification")
    X, y = dataset.X.numpy(), dataset.y.numpy()
    
    # Разделение данных
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. Эксперименты с гиперпараметрами
    hp_results = []
    
    # Конфигурации для тестирования
    configs = [
        {"lr": 0.001, "bs": 32, "opt": "SGD"},
        {"lr": 0.01, "bs": 32, "opt": "SGD"},
        {"lr": 0.01, "bs": 64, "opt": "Adam"}
    ]
    
    for config in configs:
        # Подготовка DataLoader
        train_data = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        train_loader = DataLoader(train_data, batch_size=config["bs"], shuffle=True)
        
        val_data = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        val_loader = DataLoader(val_data, batch_size=config["bs"])
        
        # Инициализация модели
        model = MulticlassLogisticRegression(in_features=X.shape[1], 
                                          num_classes=len(np.unique(y)))
        
        # Настройка оптимизатора
        if config["opt"] == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
        elif config["opt"] == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        else:
            optimizer = torch.optim.RMSprop(model.parameters(), lr=config["lr"])
        
        # Обучение
        accuracy = train_classification_model(model, train_loader, val_loader, optimizer)
        
        # Сохранение результатов
        hp_results.append({
            "Dataset": dataset_name,
            "Learning Rate": config["lr"],
            "Batch Size": config["bs"],
            "Optimizer": config["opt"],
            "Accuracy": accuracy
        })
    
    # 2. Feature Engineering эксперименты
    fe_results = []
    
    # Базовые признаки
    base_acc = hp_results[1]["Accuracy"]  # Используем лучшую конфигурацию
    fe_results.append({"Feature Type": "Base", "Accuracy": base_acc})
    
    # Сохранение результатов
    save_results(hp_results, fe_results, dataset_name, "classification")

def train_regression_model(model, train_loader, val_loader, optimizer, epochs=50):
    """Обучение регрессионной модели"""
    criterion = torch.nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch) + model.regularization_loss()
            loss.backward()
            optimizer.step()
    
    # Оценка
    model.eval()
    val_preds = []
    val_true = []
    with torch.no_grad():
        for X_val, y_val in val_loader:
            val_preds.append(model(X_val))
            val_true.append(y_val)
    
    return criterion(torch.cat(val_preds), torch.cat(val_true)).item()

def train_classification_model(model, train_loader, val_loader, optimizer, epochs=50):
    """Обучение классификационной модели"""
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
    
    # Оценка
    model.eval()
    val_preds = []
    val_true = []
    with torch.no_grad():
        for X_val, y_val in val_loader:
            outputs = model(X_val)
            _, predicted = torch.max(outputs.data, 1)
            val_preds.extend(predicted.numpy())
            val_true.extend(y_val.numpy())
    
    return accuracy_score(val_true, val_preds)

def save_results(hp_results, fe_results, dataset_name, task_type):
    """Сохранение результатов и визуализация"""
    # Создание DataFrame
    hp_df = pd.DataFrame(hp_results)
    fe_df = pd.DataFrame(fe_results)
    
    # Сохранение в CSV
    hp_df.to_csv(f"2/plots/experiments/{dataset_name[:-4]}_hp_results.csv", index=False)
    fe_df.to_csv(f"2/plots/experiments/{dataset_name[:-4]}_fe_results.csv", index=False)
    
    # Визуализация гиперпараметров
    plt.figure(figsize=(12, 6))
    if task_type == "regression":
        sns.barplot(x="Optimizer", y="MSE", hue="Batch Size", data=hp_df)
        plt.title(f"Hyperparameter Tuning for {dataset_name} (MSE)")
    else:
        sns.barplot(x="Optimizer", y="Accuracy", hue="Batch Size", data=hp_df)
        plt.title(f"Hyperparameter Tuning for {dataset_name} (Accuracy)")
    plt.savefig(f"2/plots/experiments/{dataset_name[:-4]}_hyperparams.png")
    plt.close()
    
    # Визуализация feature engineering
    plt.figure(figsize=(10, 6))
    if task_type == "regression":
        sns.barplot(x="Feature Type", y="MSE", data=fe_df)
    else:
        sns.barplot(x="Feature Type", y="Accuracy", data=fe_df)
    plt.title(f"Feature Engineering for {dataset_name}")
    plt.savefig(f"2/plots/experiments/{dataset_name[:-4]}_features.png")
    plt.close()

if __name__ == '__main__':
    run_complete_experiments()