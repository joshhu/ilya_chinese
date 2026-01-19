# 訓練工具 - 論文 18：Relational RNN

## 任務 P2-T3：訓練工具和損失函數

本模組為 LSTM 和 Relational RNN 模型提供完整的訓練工具，僅使用 NumPy。

## 檔案

- `training_utils.py` - 主要工具模組，包含損失函數、訓練迴圈和最佳化輔助
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
- 向量