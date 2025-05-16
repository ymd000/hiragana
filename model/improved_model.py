import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # スキップ接続用のダウンサンプリング層
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out

class ImprovedHiraganaNet(nn.Module):
    def __init__(self, num_classes=46):
        super(ImprovedHiraganaNet, self).__init__()
        
        # 初期層
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 出力: 32x32
        
        # 残差ブロック
        self.res1 = ResidualBlock(32, 64, stride=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 出力: 16x16
        
        self.res2 = ResidualBlock(64, 128, stride=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 出力: 8x8
        
        self.res3 = ResidualBlock(128, 256, stride=1)
        self.pool4 = nn.MaxPool2d(2, 2)  # 出力: 4x4
        
        # グローバル平均プーリング
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # 出力: 256x1x1
        
        # 全結合層
        self.fc1 = nn.Linear(256, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # 入力サイズ: [batch_size, 1, 64, 64]
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # -> [batch_size, 32, 32, 32]
        
        x = self.pool2(self.res1(x))  # -> [batch_size, 64, 16, 16]
        x = self.pool3(self.res2(x))  # -> [batch_size, 128, 8, 8]
        x = self.pool4(self.res3(x))  # -> [batch_size, 256, 4, 4]
        
        x = self.global_avg_pool(x)  # -> [batch_size, 256, 1, 1]
        x = x.view(-1, 256)
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return x 