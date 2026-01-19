# 訓練工具 - 快速參考

## 安裝
```python
# 無需安裝 - 純 NumPy
from training_utils import *
```

## 常用工作流程

### 基本分類訓練
```python
from lstm_baseline import LSTM
from training_utils import train_model, evaluate

model = LSTM(input_size=10, hidden_size=32, output_size=3)

history = train_model(
    model,
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    learning_rate=0.01,
    task='classification'
)

test_loss, test_acc = evaluate(model, X_test, y_test)
```

### 迴歸訓練
```python
history = train_model(
    model,
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    task='regression',  # 使用 MSE 損失
    epochs=100
)
```

### 帶所有功能
```python
history = train_model(
    model,
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    learning_rate=0.01,
    lr_decay=0.95,           # LR 衰減 5%
    lr_decay_every=10,       # 每 10 epochs
    clip_norm=5.0,           # 梯度裁剪到範數 5
    patience=10,             # 提前停止 patience
    task='classification',
    verbose=True
)
```

## 函數參考

### 損失函數
```python
# 分類
loss = cross_entropy_loss(predictions, targets)  # targets: (batch,) 或 (batch, n_classes)

# 迴歸
loss = mse_loss(predictions, targets)  # 連續值的 MSE

# 準確度
acc = accuracy(predictions, targets)  # 分類準確度 [0, 1]
```

### 單一訓練步驟
```python
loss, metric, grad_norm = train_step(
    model, X_batch, y_batch,
    learning_rate=0.01,
    clip_norm=5.0,
    task='classification'
)
```

### 評估
```python
loss, metric = evaluate(
    model, X_test, y_test,
    task='classification',
    batch_size=32
)
```

### 梯度裁剪
```python
clipped_grads, global_norm = clip_gradients(grads, max_norm=5.0)
```

### 學習率排程
```python
lr = learning_rate_schedule(
    epoch,
    initial_lr=0.001,
    decay=0.95,
    decay_every=10
)
```

### 提前停止
```python
early_stop = EarlyStopping(patience=10, min_delta=1e-4)

for epoch in range(epochs):
    # ... 訓練 ...
    if early_stop(val_loss, model.get_params()):
        print("提前停止！")
        best_params = early_stop.get_best_params()
        model.set_params(best_params)
        break
```

### 視覺化
```python
plot_training_curves(history, save_path='training.png')
```

## 歷史字典

```python
history = {
    'train_loss': [1.2, 1.1, 1.0, ...],      # 每 epoch 訓練損失
    'train_metric': [0.3, 0.4, 0.5, ...],    # 每 epoch 訓練指標
    'val_loss': [1.3, 1.2, 1.1, ...],        # 每 epoch 驗證損失
    'val_metric': [0.25, 0.35, 0.45, ...],   # 每 epoch 驗證指標
    'learning_rates': [0.01, 0.01, ...],     # 每 epoch 使用的 LR
    'grad_norms': [0.5, 0.4, 0.3, ...]       # 每 epoch 梯度範數
}
```

## 資料格式

### 輸入資料
```python
X_train: (num_samples, seq_len, input_size)  # 序列
y_train: (num_samples,)                       # 類別標籤（分類）
         或 (num_samples, output_size)        # 目標（迴歸）
```

### 模型介面
```python
class YourModel:
    def forward(self, X, return_sequences=False):
        # X: (batch, seq_len, input_size)
        # return: (batch, output_size) if return_sequences=False
        pass

    def get_params(self):
        return {'W': self.W, 'b': self.b}

    def set_params(self, params):
        self.W = params['W']
        self.b = params['b']
```

## 超參數建議

### 小型資料集（< 1000 樣本）
```python
epochs=100
batch_size=16
learning_rate=0.01
lr_decay=0.95
lr_decay_every=10
clip_norm=5.0
patience=10
```

### 中型資料集（1000-10000 樣本）
```python
epochs=50
batch_size=32
learning_rate=0.01
lr_decay=0.95
lr_decay_every=5
clip_norm=5.0
patience=10
```

### 大型資料集（> 10000 樣本）
```python
epochs=30
batch_size=64
learning_rate=0.01
lr_decay=0.95
lr_decay_every=5
clip_norm=5.0
patience=5
```

### 過擬合徵兆
```python
# 檢查訓練-驗證差距
train_acc = history['train_metric'][-1]
val_acc = history['val_metric'][-1]
gap = train_acc - val_acc

if gap > 0.1:  # 過擬合
    # 解決方案：
    # - 增加 patience（更多 epochs）
    # - 使用更小的學習率
    # - 添加正則化（未實作）
    # - 取得更多資料
```

### 欠擬合徵兆
```python
# 訓練和驗證準確度都低
if train_acc < 0.6 and val_acc < 0.6:
    # 解決方案：
    # - 增加模型大小（hidden_size）
    # - 訓練更長（更多 epochs）
    # - 增加學習率
    # - 檢查資料品質
```

## 常見問題

### 損失中的 NaN
```python
# 可能原因：
# 1. 學習率太高 → 降低 LR
# 2. 梯度爆炸 → 檢查 clip_norm
# 3. 數值不穩定 → 損失使用穩定實作

# 解決方案：
learning_rate=0.001  # 降低
clip_norm=1.0        # 較低的裁剪閾值
```

### 損失不下降
```python
# 可能原因：
# 1. 學習率太低
# 2. 錯誤的任務類型
# 3. 資料/標籤不匹配

# 檢查：
print(f"損失：{loss}，指標：{metric}")
print(f"預測：{model.forward(X_batch[:1])}")
print(f"目標：{y_batch[:1]}")
```

### 訓練太慢
```python
# 數值梯度很慢
# 為了更快訓練：
# 1. 使用更小的批次
# 2. 減小模型大小
# 3. 使用更少的 epochs
# 4. 實作解析梯度（BPTT）
```

## 測試

### 快速測試
```bash
python3 test_training_utils_quick.py
```

### 完整測試套件
```bash
python3 training_utils.py
```

### 示範
```bash
python3 training_demo.py
```

## 檔案

- `training_utils.py` - 主要實作（37KB）
- `training_demo.py` - 示範（11KB）
- `test_training_utils_quick.py` - 快速測試（5KB）
- `TRAINING_UTILS_README.md` - 完整文件（10KB）
- `TRAINING_QUICK_REFERENCE.md` - 本檔案（8KB）
- `TASK_P2_T3_SUMMARY.md` - 任務摘要（9KB）

## 後續步驟

1. 實作具有相同介面的 Relational RNN
2. 使用這些工具訓練 LSTM 和 Relational RNN
3. 比較推理任務上的效能
4. （可選）實作解析梯度以獲得更快的訓練
