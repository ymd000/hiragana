import os
import torch
import json
from model import HiraganaNet

def export_model():
    # クラス名の読み込み
    classes = []
    with open('./model/classes.txt', 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f.readlines()]
    
    print(f"Loaded {len(classes)} classes")
    
    # モデルのロード
    model = HiraganaNet(num_classes=len(classes))
    model.load_state_dict(torch.load('./model/saved_model/20250516_003536/hiragana_model.pth'))
    model.eval()
    
    # Webディレクトリの作成
    web_model_dir = './web/model'
    os.makedirs(web_model_dir, exist_ok=True)
    
    # クラス名をJSONとして保存
    with open(f'{web_model_dir}/classes.json', 'w', encoding='utf-8') as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)
    
    # サンプル入力
    example = torch.rand(1, 1, 64, 64)
    
    # ONNX形式へ変換（opset_versionを9に変更）
    onnx_path = f"{web_model_dir}/hiragana_model.onnx"
    torch.onnx.export(
        model,
        example,
        onnx_path,
        export_params=True,
        opset_version=9,  # opset_versionを9に変更（ブラウザでの互換性向上）
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        do_constant_folding=True,  # 定数畳み込みを有効化
        verbose=False
    )
    print(f"Model exported to ONNX format: {onnx_path}")
    
    # TorchScript形式へ変換
    traced_script_module = torch.jit.trace(model, example)
    torchscript_path = f"{web_model_dir}/hiragana_model.pt"
    traced_script_module.save(torchscript_path)
    print(f"Model exported to TorchScript format: {torchscript_path}")

if __name__ == "__main__":
    export_model()