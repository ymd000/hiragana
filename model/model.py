import torch
import torch.nn as nn
import torch.nn.functional as F

class HiraganaNet(nn.Module):
    def __init__(self, num_classes=46):
        super(HiraganaNet, self).__init__()
        
        # 単純なCNNモデル
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全結合層
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 入力サイズ: [batch_size, 1, 64, 64]
        x = self.pool(F.relu(self.conv1(x)))  # -> [batch_size, 32, 32, 32]
        x = self.pool(F.relu(self.conv2(x)))  # -> [batch_size, 64, 16, 16]
        x = self.pool(F.relu(self.conv3(x)))  # -> [batch_size, 128, 8, 8]
        
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x