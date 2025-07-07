import os
import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datasets import CustomImageDataset

# Настройки
DATA_ROOT = "5/data"
RESULTS_DIR = "5/results/task6"
os.makedirs(RESULTS_DIR, exist_ok=True)
BATCH_SIZE = 32
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Проверка наличия папок
if not os.path.exists(f"{DATA_ROOT}/train"):
    raise FileNotFoundError(f"Train folder not found at {DATA_ROOT}/train")
if not os.path.exists(f"{DATA_ROOT}/val"):
    print(f"Warning: Validation folder not found at {DATA_ROOT}/val. Using train folder for validation.")
    val_path = f"{DATA_ROOT}/train"  # Используем train как val если val нет
else:
    val_path = f"{DATA_ROOT}/val"

# Подготовка данных
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = CustomImageDataset(f'{DATA_ROOT}/train', transform=train_transform)
val_dataset = CustomImageDataset(val_path, transform=val_transform)  # Исправленный путь

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Загрузка модели
model = models.resnet18(weights='IMAGENET1K_V1')
num_classes = len(train_dataset.get_class_names())
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

# Оптимизатор и функция потерь
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

# Тренировка
train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(EPOCHS):
    model.train()
    epoch_train_loss = 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(x)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    train_losses.append(epoch_train_loss / len(train_loader))

    model.eval()
    epoch_val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            loss = loss_fn(outputs, y)
            epoch_val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    val_losses.append(epoch_val_loss / len(val_loader))
    val_accuracies.append(100 * correct / total)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_losses[-1]:.4f} | "
          f"Val Loss: {val_losses[-1]:.4f} | "
          f"Val Acc: {val_accuracies[-1]:.2f}%")

# Визуализация
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, color='green', label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/metrics.png')
plt.close()

# Сохранение модели
torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': train_dataset.get_class_names()
}, f'{RESULTS_DIR}/model.pth')

print(f"Обучение завершено! Результаты сохранены в {RESULTS_DIR}/")