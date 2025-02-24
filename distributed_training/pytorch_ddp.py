import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

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
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 64 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    # 初始化分布式环境
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset)
    train_loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # 创建模型
    model = CNNModel().cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    for epoch in range(10):
        model.train()
        sampler.set_epoch(epoch)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(local_rank), target.cuda(local_rank)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if local_rank == 0 and batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

    # 保存模型（仅在主进程）
    if local_rank == 0:
        torch.save(model.state_dict(), 
                  'd:/Workspace/ModelTrainingDemo/distributed_training/models/ddp_model.pt')

if __name__ == '__main__':
    main()