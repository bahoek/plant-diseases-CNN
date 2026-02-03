import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
import matplotlib.pyplot as plt
import time

scaler = GradScaler()
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


data_dir = r"C:\data\archive (2)"

train_dataset = datasets.ImageFolder(
    root=os.path.join(data_dir, 'new plant diseases dataset(augmented)', 'New Plant Diseases Dataset(Augmented)', 'train'),
    transform=train_transform)
valid_dataset = datasets.ImageFolder(
    root=os.path.join(data_dir, 'new plant diseases dataset(augmented)', 'New Plant Diseases Dataset(Augmented)','valid'),
    transform=val_transform)

class CNN(nn.Module):
    def __init__(self,num_classes):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3,padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(256, num_classes) 
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x) 
        x = torch.flatten(x,1)       
        x = self.fc_layers(x)
        return x

# --- [4] 학습 실행 ---
if __name__ == '__main__':
    
    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, pin_memory=True, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, num_workers=4, pin_memory=True, shuffle=False)

    num_classes = len(train_dataset.classes)
    print(f"Classes: {num_classes}")

    history = {'loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience = 5
    counter = 0
    epochs = 30
    LR = 0.001
    start_time = time.time()
    
    model = CNN(num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    MODEL_DIR = './model'
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
        
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * images.size(0)
            
            # 진행상황 출력
            if (i + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Step [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
                
        train_loss /= len(train_loader.dataset)
        history['loss'].append(train_loss)

        # 검증
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()

        val_loss /= len(valid_loader.dataset)
        val_acc = correct / len(valid_loader.dataset)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # 저장 및 Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # ★ 경로 안전하게 수정됨
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pth"))
            print(f"✅ Best Model Saved! (Acc: {val_acc:.4f})")
            best_model_time = time.time() - start_time
            best_epoch_idx = epoch + 1
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print("Early Stopping!")
            break

    # 그래프 그리기
    total_time = time.time() - start_time # 전체 걸린 시간

    print("-" * 50)
    print("🏁 Training Finished.")
    print(f"1. Total Training Time : {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"2. Time to Best Model  : {best_model_time // 60:.0f}m {best_model_time % 60:.0f}s (at Epoch {best_epoch_idx})")
    print("-" * 50)
    plt.figure(figsize=(10, 5))
    plt.plot(history['val_loss'], marker='.', c="red", label='Validation Loss')
    plt.plot(history['loss'], marker='.', c="blue", label='Train Loss')
    plt.title('EfficientNet-B0 Training Result')
    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()