import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import optuna
from optuna.trial import Trial
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

class CNNModel(nn.Module):
    def __init__(self, trial: Trial):
        super(CNNModel, self).__init__()
        # 搜索网络架构超参数
        self.n_conv_layers = trial.suggest_int('n_conv_layers', 1, 4)
        self.n_dense_layers = trial.suggest_int('n_dense_layers', 1, 3)
        
        layers = []
        in_channels = 1
        
        # 动态构建卷积层
        for i in range(self.n_conv_layers):
            out_channels = trial.suggest_int(f'conv_{i}_filters', 16, 128)
            kernel_size = trial.suggest_int(f'conv_{i}_kernel', 3, 5)
            
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size))
            layers.append(nn.ReLU())
            
            # 可选的批归一化
            if trial.suggest_categorical(f'batch_norm_{i}', [True, False]):
                layers.append(nn.BatchNormalization2d(out_channels))
            
            # 可选的池化层
            if trial.suggest_categorical(f'pooling_{i}', [True, False]):
                pool_size = trial.suggest_int(f'pool_{i}_size', 2, 3)
                layers.append(nn.MaxPool2d(pool_size))
            
            # 可选的 Dropout
            if trial.suggest_categorical(f'dropout_{i}', [True, False]):
                dropout_rate = trial.suggest_float(f'dropout_{i}_rate', 0.1, 0.5)
                layers.append(nn.Dropout(dropout_rate))
            
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # 动态计算展平后的特征数量
        with torch.no_grad():
            x = torch.randn(1, 1, 28, 28)
            x = self.conv_layers(x)
            n_features = x.view(1, -1).size(1)
        
        # 动态构建全连接层
        dense_layers = []
        in_features = n_features
        
        for i in range(self.n_dense_layers):
            out_features = trial.suggest_int(f'dense_{i}_units', 32, 512)
            
            dense_layers.append(nn.Linear(in_features, out_features))
            dense_layers.append(nn.ReLU())
            
            if trial.suggest_categorical(f'dense_dropout_{i}', [True, False]):
                dropout_rate = trial.suggest_float(f'dense_dropout_{i}_rate', 0.1, 0.5)
                dense_layers.append(nn.Dropout(dropout_rate))
            
            in_features = out_features
        
        dense_layers.append(nn.Linear(in_features, 10))
        self.dense_layers = nn.Sequential(*dense_layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.dense_layers(x)
        return x

def objective(trial: Trial):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 超参数搜索空间
    batch_size = trial.suggest_int('batch_size', 16, 128)
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    
    # GPU 版本优化：增大批量大小搜索范围
    batch_size = trial.suggest_int('batch_size', 32, 256)
    
    # GPU 版本优化：使用更激进的学习率范围
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    
    # GPU 版本优化：启用 CUDA 优化
    torch.backends.cudnn.benchmark = True
    
    # 数据加载和预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    model = CNNModel(trial).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 训练模型
    for epoch in range(10):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # 使用混合精度训练
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = F.cross_entropy(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 定期报告中间结果
            if batch_idx % 100 == 0:
                trial.report(loss.item(), epoch * len(train_loader) + batch_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
    
    # 评估模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = correct / total
    return accuracy

def main():
    # 创建保存目录
    os.makedirs('d:/Workspace/ModelTrainingDemo/automl_results', exist_ok=True)
    
    # 创建学习器
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(),
        study_name='mnist_automl'
    )
    
    # 开始超参数搜索
    print("开始搜索最佳模型架构...")
    study.optimize(objective, n_trials=100, timeout=3600)
    
    # 输出最佳结果
    print("\n=== 最佳超参数 ===")
    print("最佳准确率:", study.best_value)
    print("最佳超参数:", study.best_params)
    
    # 保存最佳模型
    best_trial = study.best_trial
    best_model = CNNModel(best_trial)
    
    # 保存模型结构和超参数
    save_path = 'd:/Workspace/ModelTrainingDemo/automl_results'
    torch.save(best_model.state_dict(), f'{save_path}/best_model.pt')
    
    # 保存搜索历史
    study.trials_dataframe().to_csv(f'{save_path}/search_history.csv')
    
    print(f"\n结果已保存至: {save_path}")
    # GPU 版本优化：增加搜索次数，启用并行
    study.optimize(
        objective, 
        n_trials=200,  # 增加到200次
        timeout=7200,  # 增加到7200秒
        n_jobs=-1  # 使用所有可用 GPU
    )
    
    # ... 后面的代码保持不变 ...