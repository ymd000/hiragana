// HTML要素の取得
const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
const clearButton = document.getElementById('clearButton');
const recognizeButton = document.getElementById('recognizeButton');
const resultDisplay = document.getElementById('result');
const topResultsContainer = document.getElementById('topResults');

// ONNX Runtime関連の変数
let session;
let classes = [];

// 描画関連の変数
let isDrawing = false;
let lastX = 0;
let lastY = 0;

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
    topResultsContainer.innerHTML = '';
    resultDisplay.textContent = "ここに結果が表示されます";
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
        const outputName = session.outputNames[0];
        
        // 推論の実行
        const feeds = {};
        feeds[inputName] = tensor;
        
        const outputData = await session.run(feeds);
        const output = outputData[outputName].data;
        
        // 結果表示
        displayResults(output);
    } catch (error) {
        console.error('Recognition error:', error);
        resultDisplay.textContent = "エラーが発生しました";
    } finally {
        resultDisplay.classList.remove('loading');
    }
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
    
    // モデル読み込み
    loadModelAndClasses();
}

// ページロード時に初期化を実行
window.onload = init;