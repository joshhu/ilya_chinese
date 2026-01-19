# 訓練工具 - 論文 18：Relational RNN

## 任務 P2-T3：訓練工具和損失函數

本模組為 LSTM 和 Relational RNN 模型提供完整的訓練工具，僅使用 NumPy。

## 檔案

- `training_utils.py` - 主要工具模組，包含損失函數、訓練迴圈和最佳化輔助函數
- `training_demo.py` - 所有訓練功能的完整示範
- `TRAINING_UTILS_README.md` - 本文件

## 實作的功能

### 1. 損失函數

#### 交叉熵損失（Cross-Entropy Loss）
```python
loss = cross_entropy_loss(predictions, targets)
```
- 支援稀疏（類別索引）和 one-hot 編碼目標
- 使用 log-sum-exp 技巧的數值穩定實作
- 用於分類任務

#### 均方誤差（MSE）損失
```python
loss = mse_loss(predictions, targets)
```
- 用於迴歸任務（物件追蹤、軌跡預測）
- 簡單的平方差平均

#### Softmax 函數
```python
probs = softmax(logits)
```
- 數值穩定的 softmax 實作
- 將 logits 轉換為機率

#### 準確度指標
```python
acc = accuracy(predictions, targets)
```
- 分類準確度計算
- 適用於稀疏和 one-hot 目標

### 2. 梯度計算

#### 數值梯度（有限差分）
```python
gradients = compute_numerical_gradient(model, X_batch, y_batch, loss_fn)
```
- 逐元素有限差分近似
- 教育性實作（慢但正確）
- 使用中心差分：`df/dx ≈ (f(x + ε) - f(x - ε)) / (2ε)`

#### 快速數值梯度
```python
gradients = compute_numerical_gradient_fast(model, X_batch, y_batch, loss_fn)
```
- 向量化梯度估計（比逐元素方式更快）
- 仍然比解析梯度慢，但更實用
- 適合原型設計和測試

**備註**：生產環境請實作透過時間反向傳播（BPTT）的解析梯度。

### 3. 最佳化工具

#### 梯度裁剪（Gradient Clipping）
```python
clipped_grads, global_norm = clip_gradients(grads, max_norm=5.0)
```
- 防止梯度爆炸（對 RNN 穩定性至關重要）
- 依全域範數裁剪所有參數
- 回傳裁剪後的梯度和原始範數以供監控

#### 學習率排程（Learning Rate Schedule）
```python
lr = learning_rate_schedule(epoch, initial_lr=0.001, decay=0.95, decay_every=10)
```
- 指數衰減排程
- 隨時間降低學習率以進行微調
- 公式：`lr = initial_lr * (decay ^ (epoch // decay_every))`

#### 提前停止（Early Stopping）
```python
early_stopping = EarlyStopping(patience=10, min_delta=1e-4)
should_stop = early_stopping(val_loss, model_params)
best_params = early_stopping.get_best_params()
```
- 透過監控驗證損失防止過擬合
- 自動儲存最佳參數
- 可配置的 patience（等待的 epochs 數）和最小改進閾值

### 4. 訓練函數

#### 單一訓練步驟
```python
loss, metric, grad_norm = train_step(
    model, X_batch, y_batch,
    learning_rate=0.001,
    clip_norm=5.0,
    task='classification'
)
```
- 執行一次梯度下降步驟
- 計算梯度、裁剪並更新參數
- 回傳損失、指標（準確度或負損失）和梯度範數
- 支援分類和迴歸任務

#### 模型評估
```python
avg_loss, avg_metric = evaluate(
    model, X_test, y_test,
    task='classification',
    batch_size=32
)
```
- 評估模型而不更新參數
- 批次處理資料（處理大型資料集）
- 回傳平均損失和指標

#### 完整訓練迴圈
```python
history = train_model(
    model,
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    lr_decay=0.95,
    lr_decay_every=10,
    clip_norm=5.0,
    patience=10,
    task='classification',
    verbose=True
)
```

功能：
- 自動批次處理，可選擇洗牌
- 學習率衰減
- 梯度裁剪
- 提前停止並恢復最佳模型
- 進度追蹤和詳細輸出
- 回傳完整的訓練歷史

歷史字典包含：
- `train_loss`：每個 epoch 的訓練損失
- `train_metric`：每個 epoch 的訓練指標
- `val_loss`：每個 epoch 的驗證損失
- `val_metric`：每個 epoch 的驗證指標
- `learning_rates`：使用的學習率
- `grad_norms`：梯度範數（用於監控穩定性）

### 5. 視覺化

#### 繪製訓練曲線
```python
plot_training_curves(history, save_path='training_curves.png')
```
- 建立 2x2 網格圖表：
  - 損失隨 epochs 變化（訓練和驗證）
  - 指標隨 epochs 變化（訓練和驗證）
  - 學習率排程
  - 梯度範數
- 若 matplotlib 不可用則回退為文字輸出

## 使用範例

### 基本訓練
```python
from lstm_baseline import LSTM
from training_utils import train_model, evaluate

# 建立模型
model = LSTM(input_size=10, hidden_size=32, output_size=3)

# 準備資料
X_train, y_train = ...  # (num_samples, seq_len, input_size)
X_val, y_val = ...

# 訓練
history = train_model(
    model,
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    learning_rate=0.01,
    task='classification'
)

# 評估
test_loss, test_acc = evaluate(model, X_test, y_test)
print(f"測試準確度：{test_acc:.4f}")
```

### 自訂訓練迴圈
```python
from training_utils import train_step, clip_gradients

for epoch in range(num_epochs):
    for X_batch, y_batch in create_batches(X_train, y_train, batch_size=32):
        loss, acc, grad_norm = train_step(
            model, X_batch, y_batch,
            learning_rate=0.01,
            clip_norm=5.0
        )
        print(f"批次損失：{loss:.4f}，準確度：{acc:.4f}")
```

### 迴歸任務
```python
# 迴歸任務（例如物件追蹤）
history = train_model(
    model,
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    task='regression',  # 使用 MSE 損失
    epochs=100
)
```

## 模型相容性

訓練工具適用於任何實作以下介面的模型：

```python
class YourModel:
    def forward(self, X, return_sequences=False):
        """
        參數：
            X: (batch, seq_len, input_size)
            return_sequences: bool
        回傳：
            outputs: (batch, output_size) 如果 return_sequences=False
                    (batch, seq_len, output_size) 如果 return_sequences=True
        """
        pass

    def get_params(self):
        """回傳參數名稱到陣列的字典"""
        return {'W': self.W, 'b': self.b, ...}

    def set_params(self, params):
        """從字典設定參數"""
        self.W = params['W']
        self.b = params['b']
```

相容的模型：
- LSTM（來自 `lstm_baseline.py`）
- Relational RNN（待實作）
- 任何遵循此介面的自訂 RNN 架構

## 測試結果

所有測試成功通過：

```
✓ 損失函數
  - 交叉熵：完美預測 → 接近零損失
  - MSE：完美預測 → 零損失
  - 稀疏和 one-hot 目標給出相同結果

✓ 最佳化工具
  - 梯度裁剪：小梯度不變，大梯度裁剪至 max_norm
  - 學習率排程：指數衰減正常運作
  - 提前停止：在 patience epochs 無改進後停止

✓ 訓練迴圈
  - 單一步驟：參數正確更新
  - 評估：不更新參數即可運作
  - 完整訓練：損失隨 epochs 下降
  - 歷史追蹤：所有指標正確記錄
```

## 效能特性

### 數值梯度
- **優點**：
  - 實作簡單
  - 無反向傳播錯誤風險
  - 教育價值

- **缺點**：
  - 非常慢（每步 O(參數數) 次前向傳播）
  - 近似（有限差分誤差）
  - 不適合大型模型或生產使用

### 建議
1. **原型設計**：使用提供的數值梯度
2. **實驗**：實作快速數值梯度估計
3. **生產環境**：透過 BPTT 實作解析梯度

## 簡化與限制

1. **梯度**：數值近似而非解析 BPTT
   - 權衡：簡單性 vs. 速度
   - 適合教育目的和小型模型

2. **最佳化器**：僅純 SGD（無 momentum、Adam 等）
   - 容易擴展更複雜的最佳化器

3. **批次處理**：無平行處理
   - 純 NumPy 實作（無 GPU 支援）

4. **梯度估計**：快速版本仍為近似
   - 使用隨機擾動而非逐元素有限差分

## 未來增強

潛在改進（本任務非必需）：
- [ ] 透過 BPTT 的解析梯度計算
- [ ] Adam 最佳化器
- [ ] 基於 Momentum 的最佳化
- [ ] 學習率預熱（warmup）
- [ ] 大批次的梯度累積
- [ ] 混合精度訓練模擬
- [ ] 更進階的 LR 排程（餘弦退火等）

## 與 Relational RNN 的整合

這些工具已準備好用於 Relational RNN 模型。只需確保您的 Relational RNN 實作所需介面（`forward`、`get_params`、`set_params`），所有訓練工具將可無縫運作。

範例：
```python
from relational_rnn import RelationalRNN
from training_utils import train_model

# 建立 Relational RNN
model = RelationalRNN(input_size=10, hidden_size=32, output_size=3)

# 訓練方式與 LSTM 完全相同
history = train_model(
    model,
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    epochs=50
)
```

## 總結

本實作提供完整的純 NumPy 訓練基礎架構：
- **損失計算**：具數值穩定性的交叉熵和 MSE
- **梯度計算**：數值近似（有限差分）
- **最佳化**：梯度裁剪、LR 排程、提前停止
- **訓練**：帶指標追蹤的完整訓練迴圈
- **監控**：完整的歷史記錄和視覺化

所有工具已測試、記錄完成，並準備好用於 LSTM 和 Relational RNN 模型。
