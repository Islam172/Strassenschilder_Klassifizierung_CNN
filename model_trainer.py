#Author: Gerhard Horst Reglich

import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision.models as models
from tqdm import tqdm

# === CONFIGURATION ===
data_dir = "/Users/mac/Desktop/Nachher"
model_dir = os.path.join(data_dir, "models")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "model_traffic_signs.pth")


batch_size = 32
num_epochs = 10
learning_rate = 0.0001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD DATA ===
X_train = np.load(os.path.join(data_dir, "X_train.npy"))
y_train = np.load(os.path.join(data_dir, "y_train.npy"))
X_val = np.load(os.path.join(data_dir, "X_val.npy"))
y_val = np.load(os.path.join(data_dir, "y_val.npy"))

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# === LOAD LABEL ENCODER ===
with open(os.path.join(data_dir, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

num_classes = len(label_encoder.classes_)

# === MODEL SETUP ===
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# === TRAIN LOOP ===
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{num_epochs}")

    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"[{epoch+1}/{num_epochs}] Loss: {running_loss:.4f}")

    # Validation Accuracy
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")

# === SAVE MODEL ===
torch.save(model.state_dict(), model_path)
print(f"\nModell gespeichert unter: {model_path}")
