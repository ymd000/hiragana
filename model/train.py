import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import datetime
from tqdm import tqdm
from model.model import HiraganaNet

# データセットクラス
class HiraganaDataset(Dataset):
    def __init__(self, data_dir, transform=None, classes_file=None):
        self.data_dir = data_dir
        self.transform = transform    
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):  # ディレクトリが存在することを確認
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # グレースケールに変換
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 学習用の前処理
transform = transforms.Compose([
    transforms.CenterCrop(50),  # 画像の中心を64x64にトリミング
    transforms.RandomRotation(degrees=10),  # 画像の回転
    transforms.Resize((64, 64)),  # 画像のサイズを64x64に変更
    transforms.ToTensor(),  # 画像をテンソルに変換
    transforms.Normalize((0.5,), (0.5,))  # テンソルを正規化
])

# データセットとローダーの準備
train_dataset = HiraganaDataset(data_dir='./data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = HiraganaDataset(data_dir='./data/val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32)

# クラス情報をデバッグ表示
print(f"Number of classes: {len(train_dataset.classes)}")
print(f"Classes: {train_dataset.classes}")

# 日本語フォントの設定
# plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 
                            #    'Meiryo', 'Hiragino Sans', 'MS Gothic'
                                   ]

'''
'''
# データセットの最初の数個のデータを表示（デバッグ用）
def show_samples(save_path=None):    
    plt.figure(figsize=(10, 4))
    for i in range(10):
        if i >= len(val_dataset):
            break
        image, label = val_dataset[i]
        plt.subplot(2, 5, i + 1)
        plt.imshow(image.squeeze().numpy(), cmap='gray')
        plt.title(f'Label: {train_dataset.classes[label]}')
        plt.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# サンプル画像を表示（デバッグ用）
show_samples()

# モデル、損失関数、オプティマイザの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = HiraganaNet(num_classes=len(train_dataset.classes))
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 結果を格納するリスト
train_losses = []
val_accuracies = []

# 学習ループ
num_epochs = 20  # エポック数
for epoch in tqdm(range(num_epochs),
                  desc='Training',
                  leave=False
                  ):
    # 訓練モード
    model.train()
    running_loss = 0.0
    
    for inputs, labels in tqdm(train_loader,
                               desc=f'Training Epoch {epoch+1}/{num_epochs}',
                               leave=False
                               ):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    
    # 評価モード
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader,
                                   desc=f'Validation Epoch {epoch+1}/{num_epochs}',
                                   leave=False
                                   ):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    val_accuracies.append(accuracy)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, '
          f'Accuracy: {accuracy:.2f}%')

# クラス名のリストを保存
classes_file_path = './model/classes.txt'
if not os.path.exists(classes_file_path):
    os.makedirs(os.path.dirname(classes_file_path), exist_ok=True)
    with open(classes_file_path, 'w', encoding='utf-8') as f:
        for class_name in train_dataset.classes:
            f.write(f"{class_name}\n")
    print(f"Classes written to {classes_file_path}")

# 保存用のディレクトリ
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = os.path.join('./model/saved_model', timestamp)
os.makedirs(save_dir, exist_ok=True)

# 学習の推移をグラフ化
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(val_accuracies)
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')

plt.tight_layout()
plt.savefig(save_dir + '/training_progress.png')

# モデルの保存
model_path = f'{save_dir}/hiragana_model.pth'
torch.save(model.state_dict(), model_path)
print(f'Training complete. Model saved to {model_path}')

# 混同行列の計算と表示
def compute_confusion_matrix():
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # 混同行列の計算
    cm = confusion_matrix(all_labels, all_preds)
    
    # 可視化
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=train_dataset.classes,
                yticklabels=train_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_dir + '/confusion_matrix.png')

# 混同行列の計算と表示（オプション）
try:
    compute_confusion_matrix()
except ImportError:
    print("sklearn, seaborn not available for confusion matrix")