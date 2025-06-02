import os
import torch
import torch.nn as nn
import json
from improved_model import ImprovedHiraganaNet

class ActivationMapModel(nn.Module):
    def __init__(self, base_model):
        super(ActivationMapModel, self).__init__()
        self.base_model = base_model
        
        # 元のモデルからコンポーネントを取得
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.pool1 = base_model.pool1
        
        self.res1 = base_model.res1
        self.pool2 = base_model.pool2
        
        self.res2 = base_model.res2
        self.pool3 = base_model.pool3
        
        self.res3 = base_model.res3
        self.pool4 = base_model.pool4
        
        self.global_avg_pool = base_model.global_avg_pool
        
        self.fc1 = base_model.fc1
        self.dropout1 = base_model.dropout1
        self.fc2 = base_model.fc2
    
    def forward(self, x):
        # 元のモデルと同じ順序で計算
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.res1(x))
        x = self.pool3(self.res2(x))
        
        # 活性化マップを抽出するレイヤー
        res3_out = self.res3(x)  # [batch_size, 256, 8, 8]
        activation_map = self.pool4(res3_out)  # [batch_size, 256, 4, 4]
        
        x = self.global_avg_pool(activation_map)
        x = x.view(-1, 256)
        
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        output = self.fc2(x)
        
        # 通常の出力と共に活性化マップも返す
        return output, activation_map

def export_model():
    # クラス名の読み込み
    classes = []
    with open('./model/classes.txt', 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f.readlines()]
    
    print(f"Loaded {len(classes)} classes")
    
    # 保存済みモデルのディレクトリを取得
    saved_model_dir = './model/saved_model'
    model_dirs = [os.path.join(saved_model_dir, d) for d in os.listdir(saved_model_dir) 
                  if os.path.isdir(os.path.join(saved_model_dir, d))]
    
    if not model_dirs:
        print("No saved model directories found.")
        return
    
    # 最新のモデルディレクトリを取得
    latest_model_dir = max(model_dirs, key=os.path.getctime)
    print(f"Using latest model directory: {latest_model_dir}")
    
    # ベストモデルのパスを取得
    best_model_path = os.path.join(latest_model_dir, 'best_model.pth')
    if not os.path.exists(best_model_path):
        print(f"Best model not found at {best_model_path}")
        print("Falling back to final model")
        best_model_path = os.path.join(latest_model_dir, 'hiragana_model.pth')
    
    print(f"Loading model from: {best_model_path}")
    
    # ベースモデルのロード
    base_model = ImprovedHiraganaNet(num_classes=len(classes))
    base_model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
    base_model.eval()
    
    # 活性化マップ出力を含むモデルを作成
    model = ActivationMapModel(base_model)
    model.eval()
    
    # Webディレクトリの作成
    web_model_dir = './web/model'
    os.makedirs(web_model_dir, exist_ok=True)
    
    # クラス名をJSONとして保存
    with open(f'{web_model_dir}/classes.json', 'w', encoding='utf-8') as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)
    
    # サンプル入力
    example = torch.rand(1, 1, 64, 64)
    
    # ONNX形式へ変換（opset_versionを9に設定）
    try:
        # 活性化マップを含むONNXモデル
        onnx_path = f"{web_model_dir}/hiragana_model.onnx"
        
        # 2つの出力名を定義
        output_names = ['output', 'activation_map']
        
        torch.onnx.export(
            model,
            example,
            onnx_path,
            export_params=True,
            opset_version=9,  # ブラウザでの互換性のために9を使用
            input_names=['input'],
            output_names=output_names,
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'},
                'activation_map': {0: 'batch_size'}
            },
            do_constant_folding=True,
            verbose=False
        )
        print(f"Model with activation map exported to ONNX format: {onnx_path}")
        
        # ONNXモデルの検証（オプション）
        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model check passed!")
        except ImportError:
            print("ONNX validation skipped: onnx package not installed")
            
    except Exception as e:
        print(f"Error exporting to ONNX: {e}")
    
    print("Export completed!")

if __name__ == "__main__":
    export_model() 