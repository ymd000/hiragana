import os
import torch
import json
from model.improved_model import ImprovedHiraganaNet

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
        regular_model_path = os.path.join(latest_model_dir, 'hiragana_model.pth')
        if os.path.exists(regular_model_path):
            print(f"Best model not found at {best_model_path}")
            user_input = input("Use hiragana_model.pth instead? (y/n): ")
            if user_input.lower() == 'y':
                best_model_path = regular_model_path
                print(f"Using regular model: {regular_model_path}")
            else:
                print("Export cancelled by user")
                return
        else:
            print(f"Neither best_model.pth nor hiragana_model.pth found in {latest_model_dir}")
            print("Export cancelled")
            return
    
    print(f"Loading model from: {best_model_path}")
    
    # モデルのロード
    model = ImprovedHiraganaNet(num_classes=len(classes))
    model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
    model.eval()
    
    # docsディレクトリの作成
    docs_model_dir = './docs/model'
    os.makedirs(docs_model_dir, exist_ok=True)
    
    # クラス名をJSONとして保存
    with open(f'{docs_model_dir}/classes.json', 'w', encoding='utf-8') as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)
    
    # サンプル入力
    example = torch.rand(1, 1, 64, 64)
    
    # ONNX形式へ変換（opset_versionを9に指定）
    try:
        onnx_path = f"{docs_model_dir}/hiragana_model.onnx"
        torch.onnx.export(
            model,
            example,
            onnx_path,
            export_params=True,
            opset_version=9,  # ブラウザでの互換性のために9を使用
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            do_constant_folding=True,
            verbose=False
        )
        print(f"Model exported to ONNX format: {onnx_path}")
        
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
    
    # TorchScript形式へ変換（バックアップ用）
    try:
        traced_script_module = torch.jit.trace(model, example)
        torchscript_path = f"{docs_model_dir}/hiragana_model.pt"
        traced_script_module.save(torchscript_path)
        print(f"Model exported to TorchScript format: {torchscript_path}")
    except Exception as e:
        print(f"Error exporting to TorchScript: {e}")
    
    print("Export completed!")

if __name__ == "__main__":
    export_model() 