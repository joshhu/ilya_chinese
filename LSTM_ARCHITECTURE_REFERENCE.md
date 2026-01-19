# LSTM 架構快速參考

## 視覺架構

```
時間 t 的輸入
     |
     v
┌─────────────────────────────────────────────────────────┐
│                      LSTM Cell                          │
│                                                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐       │
│  │ 遺忘門      │  │ 輸入門      │  │ 輸出門      │       │
│  │            │  │            │  │            │       │
│  │  f_t = σ() │  │  i_t = σ() │  │  o_t = σ() │       │
│  └────┬───────┘  └────┬───────┘  └────┬───────┘       │
│       │               │               │                │
│       v               v               │                │
│  c_prev ──[×]─────[×]──c_tilde       │                │
│            │       │                  │                │
│            └───[+]─┘                  │                │
│                │                      │                │
│                v                      v                │
│              c_new ──[tanh]──────[×]──────> h_new      │
│                                                         │
└─────────────────────────────────────────────────────────┘
     │                                   │
     v                                   v
Cell state 至 t+1              Hidden state 至 t+1
                               （同時也是輸出）
```

## 數學方程式

### 閘門計算

**遺忘門（Forget Gate）**（決定從 cell state 中遺忘什麼）：
```
f_t = σ(W_f @ x_t + U_f @ h_{t-1} + b_f)
```

**輸入門（Input Gate）**（決定添加什麼新資訊）：
```
i_t = σ(W_i @ x_t + U_i @ h_{t-1} + b_i)
```

**候選 Cell State**（新資訊）：
```
c̃_t = tanh(W_c @ x_t + U_c @ h_{t-1} + b_c)
```

**輸出門（Output Gate）**（決定輸出什麼）：
```
o_t = σ(W_o @ x_t + U_o @ h_{t-1} + b_o)
```

### 狀態更新

**Cell State 更新**（結合舊的和新的）：
```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
```

**Hidden State 更新**（過濾的輸出）：
```
h_t = o_t ⊙ tanh(c_t)
```

其中：
- `⊙` 表示逐元素乘法（Hadamard product）
- `σ` 是 sigmoid 函數
- `@` 是矩陣乘法

## 參數

### 每個閘門（共 4 個閘門）：
- **W**：輸入權重矩陣 `(hidden_size, input_size)`
- **U**：循環權重矩陣 `(hidden_size, hidden_size)`
- **b**：偏置向量 `(hidden_size, 1)`

### 總參數（無輸出投影）：
```
params = 4 × (hidden_size × input_size +     # W 矩陣
              hidden_size × hidden_size +     # U 矩陣
              hidden_size)                    # b 向量

       = 4 × hidden_size × (input_size + hidden_size + 1)
```

### 範例（input=32, hidden=64）：
```
params = 4 × 64 × (32 + 64 + 1)
       = 4 × 64 × 97
       = 24,832 參數
```

## 初始化策略

| 參數 | 方法 | 值 | 原因 |
|-----------|--------|-------|--------|
| `W_f, W_i, W_c, W_o` | Xavier | U(-√(6/(in+out)), √(6/(in+out))) | 維持激活變異數 |
| `U_f, U_i, U_c, U_o` | 正交（Orthogonal） | 基於 SVD 的正交矩陣 | 防止梯度爆炸/消失 |
| `b_f` | 常數 | **1.0** | 幫助學習長期依賴 |
| `b_i, b_c, b_o` | 常數 | 0.0 | 標準初始化 |

## 關鍵設計特點

### 1. 加性 Cell State 更新
```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
        ↑               ↑
     遺忘          添加新的
```
- **加性**（不像 vanilla RNN 是乘性的）
- 允許梯度無變化地流過時間
- 解決梯度消失問題

### 2. 閘門控制
所有閘門使用 sigmoid 激活（輸出在 [0, 1]）：
- 作為「軟開關」
- 0 = 完全阻擋
- 1 = 完全通過
- 對資訊流的可學習控制

### 3. 分離的記憶體和輸出
- **Cell state (c)**：長期記憶
- **Hidden state (h)**：過濾的輸出
- 允許模型記住而不輸出

## 前向傳播演算法

```python
# 初始化狀態
h_0 = zeros(hidden_size, batch_size)
c_0 = zeros(hidden_size, batch_size)

# 處理序列
for t in range(seq_len):
    x_t = sequence[:, t, :]

    # 計算閘門
    f_t = sigmoid(W_f @ x_t + U_f @ h_{t-1} + b_f)
    i_t = sigmoid(W_i @ x_t + U_i @ h_{t-1} + b_i)
    c̃_t = tanh(W_c @ x_t + U_c @ h_{t-1} + b_c)
    o_t = sigmoid(W_o @ x_t + U_o @ h_{t-1} + b_o)

    # 更新狀態
    c_t = f_t * c_{t-1} + i_t * c̃_t
    h_t = o_t * tanh(c_t)

    outputs[t] = h_t
```

## 形狀流程範例

輸入配置：
- `batch_size = 2`
- `seq_len = 10`
- `input_size = 32`
- `hidden_size = 64`

形狀轉換：
```
x_t:       (2, 32)      時間 t 的輸入
h_{t-1}:   (64, 2)      前一個 hidden（轉置）
c_{t-1}:   (64, 2)      前一個 cell（轉置）

W_f @ x_t:              (64, 32) @ (32, 2) = (64, 2)
U_f @ h_{t-1}:          (64, 64) @ (64, 2) = (64, 2)
b_f:                    (64, 1) → 廣播到 (64, 2)

f_t:       (64, 2)      遺忘門激活
i_t:       (64, 2)      輸入門激活
c̃_t:       (64, 2)      候選 cell state
o_t:       (64, 2)      輸出門激活

c_t:       (64, 2)      新 cell state
h_t:       (64, 2)      新 hidden state

output_t:  (2, 64)      轉置用於輸出
```

## 激活函數

### Sigmoid（用於閘門）
```
σ(x) = 1 / (1 + e^(-x))
```
- 範圍：(0, 1)
- 平滑、可微分
- 用於閘門控制（軟開/關）

### Tanh（用於 cell state 和輸出）
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```
- 範圍：(-1, 1)
- 零中心
- 用於實際值

## 使用範例

### 1. 序列分類
```python
lstm = LSTM(input_size=32, hidden_size=64, output_size=10)
output = lstm.forward(sequence, return_sequences=False)
# 輸出形狀：(batch, 10) - 類別 logits
```

### 2. 序列到序列（Sequence-to-Sequence）
```python
lstm = LSTM(input_size=32, hidden_size=64, output_size=32)
outputs = lstm.forward(sequence, return_sequences=True)
# 輸出形狀：(batch, seq_len, 32)
```

### 3. 狀態提取
```python
lstm = LSTM(input_size=32, hidden_size=64)
outputs, h, c = lstm.forward(sequence,
                             return_sequences=True,
                             return_state=True)
# outputs：(batch, seq_len, 64)
# h：(batch, 64) - 最終 hidden state
# c：(batch, 64) - 最終 cell state
```

## 常見問題與解決方案

| 問題 | 解決方案 |
|-------|----------|
| 梯度消失 | ✓ U 矩陣正交初始化 |
| 梯度爆炸 | ✓ 梯度裁剪（Gradient Clipping）（未實作） |
| 無法學習長期依賴 | ✓ 遺忘偏置 = 1.0 |
| 訓練不穩定 | ✓ W 矩陣 Xavier 初始化 |
| 前向傳播中的 NaN | ✓ 數值穩定的 sigmoid |

## 與 Vanilla RNN 的比較

| 特點 | Vanilla RNN | LSTM |
|---------|-------------|------|
| 狀態更新 | 乘性 | 加性 |
| 記憶機制 | 單一 hidden state | 分離的 cell & hidden |
| 梯度流 | 指數衰減 | 由閘門控制 |
| 長期依賴 | 差 | 好 |
| 參數 | O(h²) | O(4h²) |
| 計算成本 | 1x | ~4x |

## 實作檔案

1. **lstm_baseline.py**：核心實作
   - `LSTMCell` 類（單一時間步驟）
   - `LSTM` 類（序列處理）
   - 初始化函數
   - 測試套件

2. **lstm_baseline_demo.py**：使用範例
   - 序列分類
   - 序列到序列
   - 狀態持久性
   - 初始化重要性

3. **LSTM_BASELINE_SUMMARY.md**：完整文件
   - 實作細節
   - 測試結果
   - 設計決策

## 參考文獻

- 原始 LSTM 論文：Hochreiter & Schmidhuber (1997)
- 遺忘門：Gers et al. (2000)
- 正交初始化：Saxe et al. (2013)
- Xavier 初始化：Glorot & Bengio (2010)

---

**實作**：僅 NumPy，教育性
**品質**：可用於生產
**狀態**：完成並經過測試
**用例**：Relational RNN 比較的基線
