// HTML要素の取得
const canvas = document.getElementById('drawingCanvas');
const heatmapCanvas = document.getElementById('heatmapCanvas');
const ctx = canvas.getContext('2d');
const heatmapCtx = heatmapCanvas.getContext('2d');
const clearButton = document.getElementById('clearButton');
const recognizeButton = document.getElementById('recognizeButton');
const showHeatmapCheckbox = document.getElementById('showHeatmap');
const resultDisplay = document.getElementById('result');
const topResultsContainer = document.getElementById('topResults');

// ONNX Runtime関連の変数
let session;
let classes = [];
let lastActivations = null; // 最終層の活性化マップを保存

// 描画関連の変数
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// カラーマップの定義 (Viridis風)
const colorMap = [
    [68, 1, 84],       // 濃い紫
    [70, 50, 126],     // 紫
    [54, 92, 141],     // 青紫
    [39, 127, 142],    // ターコイズ
    [31, 161, 135],    // 青緑
    [74, 194, 109],    // 緑
    [160, 217, 58],    // 黄緑
    [253, 231, 37]     // 黄色
];

// モデルとクラスのロード
async function loadModelAndClasses() {
    try {
        // クラス情報の読み込み
        const classesResponse = await fetch('model/classes.json');
        if (!classesResponse.ok) {
            throw new Error(`HTTP error! status: ${classesResponse.status}`);
        }
        classes = await classesResponse.json();
        console.log(`Loaded ${classes.length} classes`);
        
        // ONNX Runtime の初期化
        resultDisplay.textContent = "モデルを読み込み中...";
        
        try {
            // モデルファイルの読み込み
            const modelResponse = await fetch('model/hiragana_model.onnx');
            const modelArrayBuffer = await modelResponse.arrayBuffer();
            
            // ONNX Runtimeの初期化と実行オプション
            const options = {
                executionProviders: ['wasm'], 
                graphOptimizationLevel: 'all'
            };
            
            // セッションの作成
            session = await ort.InferenceSession.create(modelArrayBuffer, options);
            console.log('Model loaded successfully');
            resultDisplay.textContent = "ここに結果が表示されます";
        } catch (onnxError) {
            console.error('ONNX Runtime error:', onnxError);
            throw onnxError;
        }
    } catch (error) {
        console.error('Error loading model or classes:', error);
        resultDisplay.textContent = `モデル読み込みエラー: ${error.message || ''}`;
    }
}

// キャンバスをクリア
function clearCanvas() {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    clearHeatmap();
    topResultsContainer.innerHTML = '';
    resultDisplay.textContent = "ここに結果が表示されます";
    lastActivations = null;
}

// ヒートマップをクリア
function clearHeatmap() {
    heatmapCtx.clearRect(0, 0, heatmapCanvas.width, heatmapCanvas.height);
}

// 画像の前処理
function preprocessImage() {
    // キャンバスを64x64にリサイズ
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    tempCanvas.width = 64;
    tempCanvas.height = 64;
    
    // リサイズして描画
    tempCtx.fillStyle = 'white';
    tempCtx.fillRect(0, 0, 64, 64);
    tempCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, 64, 64);
    
    // グレースケールデータを取得し、正規化
    const imageData = tempCtx.getImageData(0, 0, 64, 64);
    const data = imageData.data;
    const input = new Float32Array(64 * 64);
    
    for (let i = 0; i < 64 * 64; i++) {
        // RGBからグレースケールに変換
        const r = data[i * 4];
        const g = data[i * 4 + 1];
        const b = data[i * 4 + 2];
        const gray = (r + g + b) / 3;
        
        // PyTorchの入力形式に合わせて[0,1]から[-1,1]に正規化
        // 白黒反転（白背景に黒文字⇒黒背景に白文字）
        input[i] = (255 - gray) / 127.5 - 1;
    }
    
    return input;
}

// 認識処理
async function recognizeHiragana() {
    if (!session) {
        resultDisplay.textContent = "モデルが読み込まれていません";
        return;
    }
    
    try {
        resultDisplay.textContent = "認識中...";
        resultDisplay.classList.add('loading');
        
        // 画像の前処理
        const input = preprocessImage();
        
        // テンソルの作成
        const tensor = new ort.Tensor('float32', input, [1, 1, 64, 64]);
        
        // 入力名を取得
        const inputName = session.inputNames[0];
        const outputNames = session.outputNames;
        
        // ONNXモデルの出力には、活性化マップの中間出力が含まれている必要があります
        // これはモデルのエクスポート時に設定する必要があります
        
        // 推論の実行
        const feeds = {};
        feeds[inputName] = tensor;
        
        const outputData = await session.run(feeds);
        
        // 最後の全結合層の前の活性化マップを取得（存在する場合）
        // この部分はモデルの構造に依存します
        if (outputData['activation_map']) {
            lastActivations = outputData['activation_map'].data;
            // 活性化マップの形状を取得
            const activationShape = outputData['activation_map'].dims;
            console.log('Activation map shape:', activationShape);
            
            // ヒートマップの表示
            if (showHeatmapCheckbox.checked) {
                displayHeatmap(lastActivations, activationShape);
            }
        } else {
            console.log('Activation map not available in model output');
            
            // モデルが活性化マップを返さない場合は、疑似的なヒートマップを生成
            // これは視覚的な効果のみで、実際のモデルの注目領域とは異なります
            generatePseudoHeatmap();
        }
        
        const output = outputData[outputNames[0]].data;
        
        // 結果表示
        displayResults(output);
    } catch (error) {
        console.error('Recognition error:', error);
        resultDisplay.textContent = "エラーが発生しました";
    } finally {
        resultDisplay.classList.remove('loading');
    }
}

// ヒートマップの表示
function displayHeatmap(activations, shape) {
    if (!showHeatmapCheckbox.checked) {
        clearHeatmap();
        return;
    }
    
    // 活性化マップのサイズ
    const [batch, channels, height, width] = shape;
    
    // チャネルごとの活性化を平均化
    const heatmap = new Float32Array(height * width);
    for (let i = 0; i < height * width; i++) {
        let sum = 0;
        for (let c = 0; c < channels; c++) {
            sum += activations[c * height * width + i];
        }
        heatmap[i] = sum / channels;
    }
    
    // 正規化
    const min = Math.min(...heatmap);
    const max = Math.max(...heatmap);
    const normalizedHeatmap = heatmap.map(val => (val - min) / (max - min || 1));
    
    // ヒートマップの描画
    drawHeatmap(normalizedHeatmap, height, width);
}

// 疑似ヒートマップの生成と表示
function generatePseudoHeatmap() {
    if (!showHeatmapCheckbox.checked) {
        clearHeatmap();
        return;
    }
    
    // キャンバスから現在の描画領域を取得
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    // グレースケールに変換して白黒を反転（描画部分をハイライト）
    const heatmap = new Float32Array(canvas.width * canvas.height);
    for (let i = 0; i < canvas.width * canvas.height; i++) {
        const r = data[i * 4];
        const g = data[i * 4 + 1];
        const b = data[i * 4 + 2];
        const gray = (r + g + b) / 3;
        
        // 白黒反転（白=0, 黒=1）
        heatmap[i] = 1 - (gray / 255);
    }
    
    // ガウシアンブラーを適用してスムージング
    const smoothedHeatmap = applyGaussianBlur(heatmap, canvas.width, canvas.height);
    
    // ヒートマップの描画
    drawHeatmap(smoothedHeatmap, canvas.height, canvas.width);
}

// ガウシアンブラーの適用
function applyGaussianBlur(data, width, height) {
    const result = new Float32Array(width * height);
    const kernel = [
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ];
    const kernelSum = 16;
    
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            let sum = 0;
            for (let ky = -1; ky <= 1; ky++) {
                for (let kx = -1; kx <= 1; kx++) {
                    const idx = (y + ky) * width + (x + kx);
                    sum += data[idx] * kernel[ky + 1][kx + 1];
                }
            }
            result[y * width + x] = sum / kernelSum;
        }
    }
    
    return result;
}

// ヒートマップの描画
function drawHeatmap(heatmapData, height, width) {
    clearHeatmap();
    
    // リサイズファクターの計算
    const scaleX = heatmapCanvas.width / width;
    const scaleY = heatmapCanvas.height / height;
    
    // ヒートマップの描画
    const imageData = heatmapCtx.createImageData(heatmapCanvas.width, heatmapCanvas.height);
    const data = imageData.data;
    
    for (let y = 0; y < heatmapCanvas.height; y++) {
        for (let x = 0; x < heatmapCanvas.width; x++) {
            // 元のヒートマップの座標に変換
            const srcX = Math.min(Math.floor(x / scaleX), width - 1);
            const srcY = Math.min(Math.floor(y / scaleY), height - 1);
            
            const value = heatmapData[srcY * width + srcX];
            const idx = (y * heatmapCanvas.width + x) * 4;
            
            // カラーマップの適用
            const colorIdx = Math.min(Math.floor(value * colorMap.length), colorMap.length - 1);
            const [r, g, b] = colorMap[colorIdx];
            
            data[idx] = r;     // Red
            data[idx + 1] = g; // Green
            data[idx + 2] = b; // Blue
            data[idx + 3] = value * 200; // Alpha (透明度)
        }
    }
    
    heatmapCtx.putImageData(imageData, 0, 0);
}

// 結果表示
function displayResults(outputData) {
    // softmax関数で確率に変換
    const softmax = (arr) => {
        const max = Math.max(...arr);
        const expValues = arr.map(value => Math.exp(value - max));
        const sumExp = expValues.reduce((sum, value) => sum + value, 0);
        return expValues.map(value => value / sumExp);
    };
    
    const probabilities = softmax(Array.from(outputData));
    
    // 最も確率の高いクラスを表示
    const maxIndex = probabilities.indexOf(Math.max(...probabilities));
    resultDisplay.textContent = classes[maxIndex];
    
    // 上位5つの候補を表示
    const topIndices = Array.from(Array(probabilities.length).keys())
        .sort((a, b) => probabilities[b] - probabilities[a])
        .slice(0, 5);
    
    topResultsContainer.innerHTML = '';
    topIndices.forEach(index => {
        const probability = probabilities[index] * 100;
        const item = document.createElement('div');
        item.className = 'result-item';
        item.textContent = `${classes[index]}: ${probability.toFixed(1)}%`;
        topResultsContainer.appendChild(item);
    });
}

// 描画関連の関数
function startDrawing(e) {
    isDrawing = true;
    [lastX, lastY] = getCoordinates(e);
}

function draw(e) {
    if (!isDrawing) return;
    
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    
    const [currentX, currentY] = getCoordinates(e);
    ctx.lineTo(currentX, currentY);
    ctx.stroke();
    
    [lastX, lastY] = [currentX, currentY];
}

function stopDrawing() {
    isDrawing = false;
}

// マウス/タッチ座標の取得
function getCoordinates(e) {
    const rect = canvas.getBoundingClientRect();
    let x, y;
    
    if (e.type.includes('touch')) {
        x = e.touches[0].clientX - rect.left;
        y = e.touches[0].clientY - rect.top;
    } else {
        x = e.offsetX;
        y = e.offsetY;
    }
    
    return [x, y];
}

// ヒートマップ表示切り替え
function toggleHeatmap() {
    if (showHeatmapCheckbox.checked) {
        if (lastActivations) {
            // 既存の活性化マップがある場合は再表示
            displayHeatmap(lastActivations);
        } else {
            // なければ疑似ヒートマップを生成
            generatePseudoHeatmap();
        }
    } else {
        clearHeatmap();
    }
}

// 初期化処理
function init() {
    // キャンバスの初期化
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    // マウスイベントの設定
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
    
    // タッチイベントの設定
    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        startDrawing(e);
    });
    canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        draw(e);
    });
    canvas.addEventListener('touchend', (e) => {
        e.preventDefault();
        stopDrawing();
    });
    
    // ボタンイベントの設定
    clearButton.addEventListener('click', clearCanvas);
    recognizeButton.addEventListener('click', recognizeHiragana);
    showHeatmapCheckbox.addEventListener('change', toggleHeatmap);
    
    // モデル読み込み
    loadModelAndClasses();
}

// ページロード時に初期化を実行
window.onload = init; 