import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import datetime
from improved_model import ImprovedHiraganaNet

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

# 早期停止用のクラス
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = 0
        self.delta = delta
        self.path = path

    def __call__(self, val_acc, model):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        if self.verbose:
            print(f'Validation accuracy increased ({self.val_acc_max:.6f} --> {val_acc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_acc_max = val_acc

# データ拡張を強化したtransforms
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomRotation(10),  # ±10度のランダム回転
    transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),  # アフィン変換
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

val_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# データセットとローダーの準備
train_dataset = HiraganaDataset(data_dir='./data/train', transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = HiraganaDataset(data_dir='./data/val', transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=32)

# クラス情報を表示
print(f"Number of classes: {len(train_dataset.classes)}")
print(f"Classes: {train_dataset.classes}")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# フォント設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic']

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 保存用のディレクトリ
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = os.path.join('./model/saved_model', timestamp)
os.makedirs(save_dir, exist_ok=True)

# モデル、損失関数、オプティマイザの設定
model = ImprovedHiraganaNet(num_classes=len(train_dataset.classes))
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # 正則化を追加

# 学習率スケジューラー
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# 早期停止
early_stopping = EarlyStopping(patience=7, verbose=True, path=f'{save_dir}/best_model.pth')

# 結果を格納するリスト
train_losses = []
val_accuracies = []

# 学習ループ
num_epochs = 50  # エポック数を増やす
for epoch in range(num_epochs):
    # 訓練モード
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
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
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    val_accuracies.append(accuracy)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, '
          f'Accuracy: {accuracy:.2f}%')
    
    # 学習率の調整
    scheduler.step(accuracy)
    
    # 早期停止のチェック
    early_stopping(accuracy, model)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

# クラス名のリストを保存
classes_file_path = './model/classes.txt'
if os.path.exists(classes_file_path):
    print(f"Updating existing classes file: {classes_file_path}")
else:
    print(f"Creating new classes file: {classes_file_path}")
with open(classes_file_path, 'w', encoding='utf-8') as f:
    for class_name in train_dataset.classes:
        f.write(f"{class_name}\n")
print(f"Classes written to {classes_file_path}")

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
plt.savefig(f'{save_dir}/training_progress.png')

# 最終モデルの保存
model_path = f'{save_dir}/hiragana_model.pth'
torch.save(model.state_dict(), model_path)
print(f'Training complete. Final model saved to {model_path}')
print(f'Best model saved to {save_dir}/best_model.pth')

# クラスごとの精度を計算
def compute_class_accuracy():
    model.eval()
    class_correct = [0] * len(train_dataset.classes)
    class_total = [0] * len(train_dataset.classes)
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # クラスごとの精度を計算して表示
    plt.figure(figsize=(12, 8))
    accuracies = []
    for i in range(len(train_dataset.classes)):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            accuracies.append((train_dataset.classes[i], accuracy))
            print(f'Accuracy of {train_dataset.classes[i]}: {accuracy:.2f}%')
    
    # 精度の低いクラスを特定
    accuracies.sort(key=lambda x: x[1])
    print("\nClasses with lowest accuracy:")
    for char, acc in accuracies[:5]:
        print(f"{char}: {acc:.2f}%")
    
    # 棒グラフで表示
    chars, accs = zip(*accuracies)
    plt.figure(figsize=(15, 6))
    plt.bar(chars, accs)
    plt.axhline(y=sum(accs)/len(accs), color='r', linestyle='-', label='平均精度')
    plt.xlabel('ひらがな')
    plt.ylabel('精度 (%)')
    plt.title('クラスごとの精度')
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/class_accuracy.png')

# クラスごとの精度を計算して表示
compute_class_accuracy()

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
    
    try:
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        # 混同行列の計算
        cm = confusion_matrix(all_labels, all_preds)
        
        # 可視化
        plt.figure(figsize=(15, 15))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                    xticklabels=train_dataset.classes,
                    yticklabels=train_dataset.classes)
        plt.xlabel('予測')
        plt.ylabel('実際')
        plt.title('混同行列')
        plt.savefig(f'{save_dir}/confusion_matrix.png')
    except ImportError:
        print("sklearn or seaborn not available for confusion matrix")

# 混同行列の計算と表示
compute_confusion_matrix() 