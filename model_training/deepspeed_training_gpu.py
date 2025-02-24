import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import deepspeed
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载数据集
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 创建模型
    model = CNNModel()

    # DeepSpeed 配置
    ds_config = {
        "train_batch_size": 64,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.001
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0.0,
                "warmup_max_lr": 0.001,
                "warmup_num_steps": 1000
            }
        },
        "fp16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 2
        },
        "gradient_accumulation_steps": 1
    }

    # 初始化 DeepSpeed
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )

    # 训练模型
    epochs = 10
    for epoch in range(epochs):
        model_engine.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            outputs = model_engine(data)
            loss = F.cross_entropy(outputs, target)
            
            model_engine.backward(loss)
            model_engine.step()
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        # 计算平均训练损失
        train_loss /= len(train_loader)
        print(f'Epoch: {epoch}, Average Training Loss: {train_loss:.4f}')

        # 评估模型
        model_engine.eval()
        test_loss = 0
        correct = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model_engine(data)
                test_loss += F.cross_entropy(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                
                predictions.extend(pred.cpu().numpy())
                true_labels.extend(target.cpu().numpy())

        test_loss /= len(test_loader)
        accuracy = correct / len(test_loader.dataset)

        print(f'\nEpoch: {epoch}')
        print(f'测试集平均损失: {test_loss:.4f}')
        print(f'测试集准确率: {accuracy:.4f}')

    # 最终模型评估
    print('\n=== 最终模型评估 ===')
    
    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, predictions)
    print('\n混淆矩阵:')
    print(cm)
    
    # 输出详细分类报告
    print('\n分类报告:')
    print(classification_report(true_labels, predictions))

    # 保存模型
    os.makedirs('d:/Workspace/ModelTrainingDemo/models', exist_ok=True)
    
    # 保存 DeepSpeed 检查点
    model_engine.save_checkpoint('d:/Workspace/ModelTrainingDemo/models/deepspeed_gpu_checkpoint')
    
    # 保存为普通 PyTorch 模型
    torch.save(model.state_dict(), 'd:/Workspace/ModelTrainingDemo/models/mnist_deepspeed_gpu.pt')
    
    # 保存为 TorchScript 格式
    scripted_model = torch.jit.script(model)
    scripted_model.save('d:/Workspace/ModelTrainingDemo/models/mnist_deepspeed_gpu_scripted.pt')
    
    print('\n=== 模型已保存 ===')
    print('DeepSpeed检查点保存路径:d:/Workspace/ModelTrainingDemo/models/deepspeed_gpu_checkpoint')
    print('PyTorch模型保存路径:d:/Workspace/ModelTrainingDemo/models/mnist_deepspeed_gpu.pt')
    print('TorchScript模型保存路径:d:/Workspace/ModelTrainingDemo/models/mnist_deepspeed_gpu_scripted.pt')

if __name__ == '__main__':
    train_model()