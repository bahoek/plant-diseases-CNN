import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

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
        x = self.adaptive_pool(x) # 여기서 강제로 (Batch, 128, 1, 1)로 만듦
        x = torch.flatten(x,1)       # (Batch, 128)
        x = self.fc_layers(x)
        return x
    
if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = r"C:\data\archive (2)"
    model_path = r"C:\Users\infs1\OneDrive\Desktop\.대학\code\.주요한 코드들\plant_diseases_CNN\best_model.pth"
    
    # 2. 테스트 데이터셋 준비
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("📁 테스트 데이터를 불러오는 중...")
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'new plant diseases dataset(augmented)', 'New Plant Diseases Dataset(Augmented)',  'valid'), transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4,pin_memory=True)
    
    
    class_names = test_dataset.classes
    num_classes = len(class_names)
    print(f"📊 감지된 클래스 개수: {num_classes}개")

    
    print("🧠 모델을 로드하는 중...")
    model = CNN(num_classes=num_classes).to(device)    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval() # ★ 평가 모드 (필수!)
        print("✅ 학습된 모델 로드 완료!")
    else:
        print(f"❌ 모델 파일이 없습니다: {model_path}")
        exit()

    # 4. 전체 정확도 평가
    print("🚀 정확도 측정 시작...")
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\n🏆 최종 테스트 정확도: {accuracy:.2f}%")
    
# 결과 눈으로 확인
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images = images.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # 시각화 함수
    def imshow(img, title):
        img = img.cpu().numpy().transpose((1, 2, 0))
        img = np.array([0.229, 0.224, 0.225]) * img + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(title, fontsize=10)
        plt.axis('off')

    plt.figure(figsize=(12, 8))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        color = 'blue' if predicted[i] == labels[i] else 'red'
        title = f"Pred: {class_names[predicted[i]]}\nActual: {class_names[labels[i]]}"
        imshow(images[i], title)
    plt.show()