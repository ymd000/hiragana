# ひらがな手書き認識アプリケーション

PyTorchで学習した手書きひらがな認識モデルをONNX形式に変換し、ブラウザで実行するWebアプリケーションです。GitHub Pagesでホストされています。

## デモページ

以下のURLでアプリケーションを試すことができます：

[ひらがな手書き認識アプリケーション](https://ymd000.github.io/hiragana/)

## プロジェクトの構成

```
hiragana/
├── data/               # 学習データ（.gitignoreで除外）
│   ├── train/          # トレーニングデータ
│   └── val/            # 検証データ
├── model/              # モデル関連
│   ├── model.py        # モデル定義
│   ├── train.py        # 学習スクリプト
│   ├── export.py       # モデルエクスポート
│   └── saved_model/    # 保存済みモデル
├── web/                # Webアプリケーション（GitHub Pagesでデプロイ）
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   └── model/          # ブラウザで実行するモデル
│       ├── hiragana_model.onnx
│       ├── hiragana_model.pt
│       └── classes.json
├── .github/            # GitHub Actions設定
├── .gitignore
└── README.md
```

## 機能

- 手書きのひらがなをリアルタイムで認識
- ブラウザ上で動作（サーバーレス）
- レスポンシブデザイン対応
- タッチデバイス対応

## 使用技術

- **モデル学習**: PyTorch
- **モデル変換**: ONNX
- **フロントエンド**: HTML, CSS, JavaScript
- **モデル実行**: ONNX Runtime Web
- **ホスティング**: GitHub Pages

## ローカルでの実行

1. リポジトリをクローン
   ```
   git clone https://github.com/あなたのGitHubユーザー名/hiragana.git
   cd hiragana
   ```

2. ウェブサーバーを起動
   ```
   cd web
   python -m http.server
   ```

3. ブラウザで `http://localhost:8000` にアクセス

## モデルの再学習

1. 必要なパッケージをインストール
   ```
   pip install torch torchvision numpy matplotlib pillow
   ```

2. 学習データを `/data/train` と `/data/val` に配置

3. モデルの学習
   ```
   python -m model.train
   ```

4. モデルをエクスポート
   ```
   python -m model.export
   ```

## 貢献方法

1. このリポジトリをフォーク
2. 新しいブランチを作成
3. 変更をコミット
4. プルリクエストを送信

## ライセンス

MITライセンス

## 謝辞

- ETL Character Databaseの手書き文字データを使用しています。
- ONNX Runtime Web を使用してブラウザ上で推論を実行しています。 