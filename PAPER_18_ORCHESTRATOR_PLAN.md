# 論文 18：Relational RNN - Orchestrator 實作計畫

## 概述

**論文**：Relational Recurrent Neural Networks（Santoro et al.）
**狀態**：尚未實作（剩餘 8/30）
**難度**：中級
**關鍵概念**：Relational Memory、RNN 中的自注意力、序列推理

## 實作目標

1. 實作用於記憶體的多頭點積注意力
2. 建構 relational memory 核心
3. 建立序列推理任務
4. 與標準 LSTM 基線比較
5. 視覺化記憶體互動和注意力模式

---

## 原子任務分解

### 第 1 階段：基礎與設定（4 個任務 - 並行執行）

**P1-T1**：建立 notebook 結構和匯入
- 建立 `18_relational_rnn.ipynb`
- 添加標準匯入（numpy、matplotlib、scipy）
- 建立帶有 markdown 標題的單元格結構
- 添加論文引用和概述
- 交付物：帶結構的空 notebook

**P1-T2**：實作多頭點積注意力機制
- 實作縮放點積注意力函數
- 實作多頭注意力包裝器
- 添加 query、key、value 投影
- 包含注意力分數計算
- 交付物：`multi_head_attention(Q, K, V, num_heads)` 函數

**P1-T3**：實作標準 LSTM 基線用於比較
- 建立帶有閘門（遺忘、輸入、輸出）的 LSTM cell
- 實作序列的前向傳播
- 添加參數初始化
- 包含 hidden state 管理
- 交付物：帶有 forward 方法的 `LSTM` 類

**P1-T4**：生成合成序列推理資料集
- 建立任務：需要記憶的序列排序
- 生成關係推理任務（例如：配對匹配、物件追蹤）
- 建立簡單的 bAbI 風格 QA 任務
- 添加資料預處理工具
- 交付物：返回訓練/測試分割的 `generate_reasoning_data()` 函數

---

### 第 2 階段：核心 Relational Memory 實作（3 個任務 - 並行執行）

**依賴**：需要 P1-T2（注意力機制）

**P2-T1**：實作 relational memory 核心模組
- 使用多頭注意力建構記憶體模組
- 實作記憶體更新機制
- 添加殘差連接和層正規化
- 包含透過自注意力的記憶體列互動
- 交付物：帶有 `forward(input, memory)` 方法的 `RelationalMemory` 類

**P2-T2**：建構結合 LSTM + relational memory 的 relational RNN cell
- 整合 LSTM hidden state 與 relational memory
- 實作 LSTM 和記憶體之間的閘門
- 添加記憶體讀/寫操作
- 包含適當的狀態管理
- 交付物：`RelationalRNNCell` 類

**P2-T3**：實作訓練工具和損失函數
- 建立序列損失計算
- 添加梯度裁剪工具
- 實作學習率排程
- 添加提前停止邏輯
- 交付物：`train_step()` 和 `evaluate()` 函數

---

### 第 3 階段：訓練與基線比較（2 個任務 - 並行執行）

**依賴**：需要 P1-T3、P1-T4、P2-T1、P2-T2、P2-T3

**P3-T1**：訓練標準 LSTM 基線
- 在推理任務上訓練 LSTM
- 記錄訓練曲線（損失、準確度）
- 儲存最佳模型參數
- 記錄最終測試效能
- 交付物：訓練好的 LSTM 基線 + 效能指標

**P3-T2**：訓練 relational RNN 模型
- 在相同任務上訓練 Relational RNN
- 匹配基線的超參數
- 記錄訓練曲線
- 儲存最佳模型參數
- 交付物：訓練好的 Relational RNN + 效能指標

---

### 第 4 階段：評估與視覺化（4 個任務 - 並行執行）

**依賴**：需要 P3-T1、P3-T2

**P4-T1**：生成比較效能圖表
- 繪製訓練曲線（LSTM vs Relational RNN）
- 建立準確度比較長條圖
- 展示收斂速度分析
- 生成樣本效率圖
- 交付物：效能比較視覺化

**P4-T2**：視覺化注意力模式和記憶體互動
- 從訓練好的模型提取注意力權重
- 建立隨時間的注意力熱圖
- 視覺化推論期間的記憶體演化
- 展示哪些記憶體 slot 何時被使用
- 交付物：注意力視覺化圖表

**P4-T3**：分析推理能力
- 在保留的複雜推理範例上測試
- 展示兩種模型的失敗案例
- 展示 relational memory 有幫助之處
- 包含定性分析
- 交付物：帶範例的推理分析章節

**P4-T4**：建立消融研究
- 測試不同數量的注意力頭（1、2、4、8）
- 改變記憶體大小
- 比較有/無殘差連接
- 展示每個元件的影響
- 交付物：消融研究結果和圖表

---

### 第 5 階段：文件與完善（4 個任務 - 並行執行）

**依賴**：需要所有先前階段

**P5-T1**：撰寫完整的 markdown 解說
- 添加關係推理的理論背景
- 解釋注意力機制直覺
- 記錄架構選擇
- 包含數學公式
- 交付物：完整的 markdown 文件單元格

**P5-T2**：添加程式碼文件和註解
- 為所有函數添加 docstrings
- 為複雜操作添加行內註解
- 包含形狀標註
- 添加使用範例
- 交付物：完全文件化的程式碼

**P5-T3**：建立摘要和關鍵洞見章節
- 總結主要發現
- 與論文結果比較
- 列出關鍵要點
- 建議擴展和後續步驟
- 交付物：結論章節

**P5-T4**：執行最終測試並建立檢查點
- 端對端執行完整 notebook
- 驗證所有視覺化正確渲染
- 檢查錯誤和警告
- 建立乾淨的輸出版本
- 交付物：經測試、無錯誤的 notebook

---

## 階段摘要

### 第 1 階段：基礎與設定
- **任務數**：4（全部並行）
- **依賴**：無
- **預計並行度**：4 個子代理
- **輸出**：Notebook 結構、注意力機制、LSTM 基線、合成資料

### 第 2 階段：核心實作
- **任務數**：3（P1-T2 之後全部並行）
- **依賴**：P1-T2
- **預計並行度**：3 個子代理
- **輸出**：Relational memory 核心、Relational RNN cell、訓練工具

### 第 3 階段：訓練
- **任務數**：2（並行）
- **依賴**：P1-T3、P1-T4、P2-T1、P2-T2、P2-T3
- **預計並行度**：2 個子代理
- **輸出**：訓練好的模型（基線和 relational）

### 第 4 階段：評估
- **任務數**：4（全部並行）
- **依賴**：P3-T1、P3-T2
- **預計並行度**：4 個子代理
- **輸出**：視覺化、分析、消融研究

### 第 5 階段：文件
- **任務數**：4（全部並行）
- **依賴**：所有先前階段
- **預計並行度**：4 個子代理
- **輸出**：完整、文件化、經測試的 notebook

---

## 總體分解

- **原子任務總數**：17
- **階段總數**：5
- **最大並行度**：4 個子代理（第 1、4、5 階段）
- **順序依賴**：第 2 階段 → 第 3 階段 → 第 4 階段 → 第 5 階段

---

## 成功標準

每個子代理必須交付：

1. **可運作的程式碼**：執行無錯誤
2. **摘要**：實作了什麼以及如何實作
3. **學習心得**：任何偏差、設計決策或洞見
4. **測試通過**：程式碼成功執行
5. **提交**：帶有任務描述的清晰提交訊息

## 失敗協議

- 如果任務失敗 3 次，升級給 orchestrator
- Orchestrator 將：
  1. 分析根本原因
  2. 以簡化範圍或替代方法重新規劃任務
  3. 可能合併/分割任務
  4. 重新分配給不同的子代理

---

## 品質標準

- **程式碼**：僅 NumPy（無 PyTorch/TensorFlow）
- **資料**：自包含的合成資料生成
- **視覺化**：清晰、出版品質的圖表
- **文件**：解釋「為什麼」而非只是「什麼」
- **測試**：驗證形狀、範圍、收斂性

---

## 實作備註

### 多頭注意力
```python
# 預期簽名
def multi_head_attention(Q, K, V, num_heads=4):
    """
    Q：queries (batch, seq_len, d_model)
    K：keys (batch, seq_len, d_model)
    V：values (batch, seq_len, d_model)
    返回：attended output (batch, seq_len, d_model)
    """
```

### Relational Memory
- 記憶體 slots：4-8 個 slots
- 每個 slot 是一個向量（例如 64-128 維）
- 跨記憶體 slots 的自注意力
- 更新記憶體的閘門

### 序列推理任務
- **任務 1**：追蹤物件隨時間的位置
- **任務 2**：記住並比較配對
- **任務 3**：簡單的 bAbI 風格 QA（2-3 個支持事實）

### 比較指標
- 推理任務的準確度
- 訓練收斂速度
- 樣本效率
- 記憶體利用率

---

## Git 提交策略

每個任務完成後：

```
WIP: [Phase X Task Y] 簡短描述

- 實作了什麼
- 任何設計決策
- 已知限制

Tests: Pass/Fail
Lint: Pass/Fail
```

最終提交：
```
feat: Implement Paper 18 - Relational RNN

Complete implementation of Relational Recurrent Neural Networks
- Multi-head attention for memory
- Relational memory core
- Sequential reasoning tasks
- Comparison with LSTM baseline
- Visualizations and ablations

Closes #18 (if issue tracking used)
```

---

## 開始實作

**Orchestrator**：從第 1 階段開始，為任務 P1-T1 至 P1-T4 啟動 4 個並行子代理。
