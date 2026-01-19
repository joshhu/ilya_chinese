# Sutskever 30 - 完整實作套件

**Ilya Sutskever 推薦的 30 篇基礎論文的完整玩具實作**

[![實作進度](https://img.shields.io/badge/實作-30%2F30-brightgreen)](https://github.com/pageman/sutskever-30-implementations)
[![覆蓋率](https://img.shields.io/badge/覆蓋率-100%25-blue)](https://github.com/pageman/sutskever-30-implementations)
[![Python](https://img.shields.io/badge/Python-僅使用NumPy-yellow)](https://numpy.org/)

## 概述

本儲存庫包含來自 Ilya Sutskever 著名閱讀清單的詳細教育性實作——這是他告訴 John Carmack 可以教會你深度學習中「90% 重要內容」的論文集合。

**進度：30/30 篇論文（100%）- 完成！🎉**

每個實作：
- ✅ 僅使用 NumPy（不使用深度學習框架）以確保教育清晰度
- ✅ 包含合成/自助抽樣資料以便立即執行
- ✅ 提供豐富的視覺化和說明
- ✅ 展示每篇論文的核心概念
- ✅ 在 Jupyter notebooks 中運行以進行互動式學習

## 快速開始

```bash
# 進入目錄
cd sutskever-30-implementations

# 安裝依賴
pip install numpy matplotlib scipy

# 運行任意 notebook
jupyter notebook 02_char_rnn_karpathy.ipynb
```

## Sutskever 30 論文

### 基礎概念（論文 1-5）

| # | 論文 | Notebook | 關鍵概念 |
|---|------|----------|----------|
| 1 | 複雜動力學第一定律（The First Law of Complexodynamics） | ✅ `01_complexity_dynamics.ipynb` | 熵、複雜度增長、細胞自動機 |
| 2 | RNN 的不合理有效性（The Unreasonable Effectiveness of RNNs） | ✅ `02_char_rnn_karpathy.ipynb` | 字元級模型、RNN 基礎、文本生成 |
| 3 | 理解 LSTM 網路（Understanding LSTM Networks） | ✅ `03_lstm_understanding.ipynb` | 閘門、長期記憶、梯度流 |
| 4 | RNN 正則化（RNN Regularization） | ✅ `04_rnn_regularization.ipynb` | 序列的 Dropout、變分 Dropout |
| 5 | 保持神經網路簡單（Keeping Neural Networks Simple） | ✅ `05_neural_network_pruning.ipynb` | MDL 原理、權重剪枝、90%+ 稀疏性 |

### 架構與機制（論文 6-15）

| # | 論文 | Notebook | 關鍵概念 |
|---|------|----------|----------|
| 6 | 指標網路（Pointer Networks） | ✅ `06_pointer_networks.ipynb` | 注意力作為指標、組合問題 |
| 7 | ImageNet/AlexNet | ✅ `07_alexnet_cnn.ipynb` | CNN、卷積、資料增強 |
| 8 | 順序很重要：集合的 Seq2Seq（Order Matters: Seq2Seq for Sets） | ✅ `08_seq2seq_for_sets.ipynb` | 集合編碼、排列不變性、注意力池化 |
| 9 | GPipe | ✅ `09_gpipe.ipynb` | 管線並行、微批次、重新材料化 |
| 10 | 深度殘差學習（Deep Residual Learning，ResNet） | ✅ `10_resnet_deep_residual.ipynb` | 跳躍連接、梯度高速公路 |
| 11 | 擴張卷積（Dilated Convolutions） | ✅ `11_dilated_convolutions.ipynb` | 感受野、多尺度 |
| 12 | 神經訊息傳遞（Neural Message Passing，GNN） | ✅ `12_graph_neural_networks.ipynb` | 圖網路、訊息傳遞 |
| 13 | **注意力就是你所需要的（Attention Is All You Need）** | ✅ `13_attention_is_all_you_need.ipynb` | Transformer、自注意力、多頭注意力 |
| 14 | 神經機器翻譯（Neural Machine Translation） | ✅ `14_bahdanau_attention.ipynb` | Seq2seq、Bahdanau 注意力 |
| 15 | ResNet 中的恆等映射（Identity Mappings in ResNet） | ✅ `15_identity_mappings_resnet.ipynb` | 預激活、梯度流 |

### 進階主題（論文 16-22）

| # | 論文 | Notebook | 關鍵概念 |
|---|------|----------|----------|
| 16 | 關係推理（Relational Reasoning） | ✅ `16_relational_reasoning.ipynb` | 關係網路、成對函數 |
| 17 | **變分有損自編碼器（Variational Lossy Autoencoder）** | ✅ `17_variational_autoencoder.ipynb` | VAE、ELBO、重參數化技巧 |
| 18 | **關係型 RNN（Relational RNNs）** | ✅ `18_relational_rnn.ipynb` | 關係記憶、多頭自注意力、手動反向傳播（約 1100 行） |
| 19 | 咖啡自動機（The Coffee Automaton） | ✅ `19_coffee_automaton.ipynb` | 不可逆性、熵、時間箭頭、Landauer 原理 |
| 20 | **神經圖靈機（Neural Turing Machines）** | ✅ `20_neural_turing_machine.ipynb` | 外部記憶、可微分定址 |
| 21 | Deep Speech 2（CTC） | ✅ `21_ctc_speech.ipynb` | CTC 損失、語音辨識 |
| 22 | **縮放定律（Scaling Laws）** | ✅ `22_scaling_laws.ipynb` | 冪律、計算最優訓練 |

### 理論與元學習（論文 23-30）

| # | 論文 | Notebook | 關鍵概念 |
|---|------|----------|----------|
| 23 | MDL 原理（MDL Principle） | ✅ `23_mdl_principle.ipynb` | 資訊理論、模型選擇、壓縮 |
| 24 | **機器超級智慧（Machine Super Intelligence）** | ✅ `24_machine_super_intelligence.ipynb` | 通用 AI、AIXI、Solomonoff 歸納、智慧測量、自我改進 |
| 25 | Kolmogorov 複雜度（Kolmogorov Complexity） | ✅ `25_kolmogorov_complexity.ipynb` | 壓縮、演算法隨機性、通用先驗 |
| 26 | **CS231n：視覺辨識的 CNN** | ✅ `26_cs231n_cnn_fundamentals.ipynb` | 影像分類管線、kNN/線性/神經網路/CNN、反向傳播、最佳化、照顧神經網路 |
| 27 | 多詞元預測（Multi-token Prediction） | ✅ `27_multi_token_prediction.ipynb` | 多個未來詞元、樣本效率、快 2-3 倍 |
| 28 | 密集段落檢索（Dense Passage Retrieval） | ✅ `28_dense_passage_retrieval.ipynb` | 雙編碼器、MIPS、批次內負樣本 |
| 29 | 檢索增強生成（Retrieval-Augmented Generation） | ✅ `29_rag.ipynb` | RAG-Sequence、RAG-Token、知識檢索 |
| 30 | 迷失在中間（Lost in the Middle） | ✅ `30_lost_in_middle.ipynb` | 位置偏差、長上下文、U 形曲線 |

## 精選實作

### 🌟 必讀 Notebook

這些實作涵蓋了最具影響力的論文，並展示核心深度學習概念：

#### 基礎篇
1. **`02_char_rnn_karpathy.ipynb`** - 字元級 RNN
   - 從頭建立 RNN
   - 理解時間反向傳播
   - 生成文本

2. **`03_lstm_understanding.ipynb`** - LSTM 網路
   - 實作遺忘閘/輸入閘/輸出閘
   - 視覺化閘門激活
   - 與原始 RNN 比較

3. **`04_rnn_regularization.ipynb`** - RNN 正則化
   - RNN 的變分 Dropout
   - 正確的 Dropout 放置位置
   - 訓練改進

4. **`05_neural_network_pruning.ipynb`** - 網路剪枝與 MDL
   - 基於幅度的剪枝
   - 帶微調的迭代剪枝
   - 90%+ 稀疏性且損失最小
   - 最小描述長度原理

#### 電腦視覺
5. **`07_alexnet_cnn.ipynb`** - CNN 與 AlexNet
   - 從頭建立卷積層
   - 最大池化和 ReLU
   - 資料增強技術

6. **`10_resnet_deep_residual.ipynb`** - ResNet
   - 跳躍連接解決退化問題
   - 梯度流視覺化
   - 恆等映射直覺

7. **`15_identity_mappings_resnet.ipynb`** - 預激活 ResNet
   - 預激活 vs 後激活
   - 更好的梯度流
   - 訓練 1000+ 層網路

8. **`11_dilated_convolutions.ipynb`** - 擴張卷積
   - 多尺度感受野
   - 不需要池化
   - 語義分割

#### 注意力與 Transformer
9. **`14_bahdanau_attention.ipynb`** - 神經機器翻譯
   - 原始注意力機制
   - 帶對齊的 Seq2seq
   - 注意力視覺化

10. **`13_attention_is_all_you_need.ipynb`** - Transformer
    - 縮放點積注意力
    - 多頭注意力
    - 位置編碼
    - 現代 LLM 的基礎

11. **`06_pointer_networks.ipynb`** - 指標網路
    - 注意力作為選擇
    - 組合最佳化
    - 可變輸出大小

12. **`08_seq2seq_for_sets.ipynb`** - 集合的 Seq2Seq
    - 排列不變的集合編碼器
    - 讀取-處理-寫入架構
    - 對無序元素的注意力
    - 排序和集合運算
    - 比較：順序敏感 vs 順序不變

13. **`09_gpipe.ipynb`** - GPipe 管線並行
    - 跨設備的模型分割
    - 微批次以提高管線利用率
    - F-then-B 排程（先全部前向，再全部後向）
    - 重新材料化（梯度檢查點）
    - 氣泡時間分析
    - 訓練超過單一設備記憶體的模型

#### 進階主題
14. **`12_graph_neural_networks.ipynb`** - 圖神經網路
    - 訊息傳遞框架
    - 圖卷積
    - 分子性質預測

15. **`16_relational_reasoning.ipynb`** - 關係網路
    - 成對關係推理
    - 視覺問答
    - 排列不變性

16. **`18_relational_rnn.ipynb`** - 關係型 RNN
    - 帶關係記憶的 LSTM
    - 跨記憶槽的多頭自注意力
    - 架構展示（前向傳遞）
    - 序列推理任務
    - **第 11 節：手動反向傳播實作（約 1100 行）**
    - 所有組件的完整梯度計算
    - 帶數值驗證的梯度檢查

17. **`20_neural_turing_machine.ipynb`** - 記憶增強網路
    - 內容與位置定址
    - 可微分讀/寫
    - 外部記憶

18. **`21_ctc_speech.ipynb`** - CTC 損失與語音辨識
    - 連接主義時序分類
    - 無對齊訓練
    - 前向演算法

#### 生成式模型
19. **`17_variational_autoencoder.ipynb`** - VAE
    - 生成式建模
    - ELBO 損失
    - 潛在空間視覺化

#### 現代應用
20. **`27_multi_token_prediction.ipynb`** - 多詞元預測
    - 預測多個未來詞元
    - 2-3 倍樣本效率
    - 推測性解碼
    - 更快的訓練和推論

21. **`28_dense_passage_retrieval.ipynb`** - 密集檢索
    - 雙編碼器架構
    - 批次內負樣本
    - 語義搜索

22. **`29_rag.ipynb`** - 檢索增強生成
    - RAG-Sequence vs RAG-Token
    - 結合檢索 + 生成
    - 知識基礎輸出

23. **`30_lost_in_middle.ipynb`** - 長上下文分析
    - LLM 中的位置偏差
    - U 形性能曲線
    - 文件排序策略

#### 縮放與理論
24. **`22_scaling_laws.ipynb`** - 縮放定律
    - 冪律關係
    - 計算最優訓練
    - 性能預測

25. **`23_mdl_principle.ipynb`** - 最小描述長度
    - 資訊理論模型選擇
    - 壓縮 = 理解
    - MDL vs AIC/BIC 比較
    - 神經網路架構選擇
    - 基於 MDL 的剪枝（連接到論文 5）
    - Kolmogorov 複雜度預覽

26. **`25_kolmogorov_complexity.ipynb`** - Kolmogorov 複雜度
    - K(x) = 生成 x 的最短程式
    - 隨機性 = 不可壓縮性
    - 演算法機率（Solomonoff）
    - 歸納的通用先驗
    - 與夏農熵的連接
    - 形式化的奧卡姆剃刀
    - 機器學習的理論基礎

27. **`24_machine_super_intelligence.ipynb`** - 通用人工智慧
    - **智慧的形式理論（Legg & Hutter）**
    - 心理測量 g 因素和通用智慧 Υ(π)
    - 用於序列預測的 Solomonoff 歸納
    - AIXI：理論上最優的強化學習代理
    - 蒙特卡羅 AIXI（MC-AIXI）近似
    - Kolmogorov 複雜度估計
    - 跨環境的智慧測量
    - 遞迴自我改進動態
    - 智慧爆炸場景
    - **6 個章節：從心理測量學到超級智慧**
    - 連接論文 #23（MDL）、#25（Kolmogorov）、#8（DQN）

28. **`01_complexity_dynamics.ipynb`** - 複雜度與熵
    - 細胞自動機（規則 30）
    - 熵增長
    - 不可逆性（基本介紹）

28. **`19_coffee_automaton.ipynb`** - 咖啡自動機（深入探討）
    - **全面探索不可逆性**
    - 咖啡混合和擴散過程
    - 熵增長和粗粒化
    - 相空間和 Liouville 定理
    - Poincaré 遞歸定理（將在 e^N 時間後解混合！）
    - Maxwell 妖和 Landauer 原理
    - 計算不可逆性（單向函數、雜湊）
    - 機器學習中的資訊瓶頸
    - 生物不可逆性（生命和熱力學第二定律）
    - 時間箭頭：基本的 vs 湧現的
    - **10 個全面章節探索各尺度的不可逆性**

29. **`26_cs231n_cnn_fundamentals.ipynb`** - CS231n：從第一原理學視覺
    - **純 NumPy 的完整視覺管線**
    - k 近鄰基準
    - 線性分類器（SVM 和 Softmax）
    - 最佳化（SGD、動量、Adam、學習率排程）
    - 帶反向傳播的 2 層神經網路
    - 卷積層（conv、pool、ReLU）
    - 完整 CNN 架構（Mini-AlexNet）
    - 視覺化技術（濾波器、顯著性圖）
    - 遷移學習原理
    - 照顧技巧（健全性檢查、超參數調整、監控）
    - **10 個章節涵蓋整個 CS231n 課程**
    - 連接論文 #7（AlexNet）、#10（ResNet）、#11（擴張卷積）

## 儲存庫結構

```
sutskever-30-implementations/
├── README.md                           # 本文件
├── PROGRESS.md                         # 實作進度追蹤
├── IMPLEMENTATION_TRACKS.md            # 所有 30 篇論文的詳細軌跡
│
├── 01_complexity_dynamics.ipynb        # 熵與複雜度
├── 02_char_rnn_karpathy.ipynb         # 原始 RNN
├── 03_lstm_understanding.ipynb         # LSTM 閘門
├── 04_rnn_regularization.ipynb         # RNN 的 Dropout
├── 05_neural_network_pruning.ipynb     # 剪枝與 MDL
├── 06_pointer_networks.ipynb           # 注意力指標
├── 07_alexnet_cnn.ipynb               # CNN 與 AlexNet
├── 08_seq2seq_for_sets.ipynb          # 排列不變的集合
├── 09_gpipe.ipynb                     # 管線並行
├── 10_resnet_deep_residual.ipynb      # 殘差連接
├── 11_dilated_convolutions.ipynb       # 多尺度卷積
├── 12_graph_neural_networks.ipynb      # 訊息傳遞 GNN
├── 13_attention_is_all_you_need.ipynb # Transformer 架構
├── 14_bahdanau_attention.ipynb         # 原始注意力
├── 15_identity_mappings_resnet.ipynb   # 預激活 ResNet
├── 16_relational_reasoning.ipynb       # 關係網路
├── 17_variational_autoencoder.ipynb   # VAE
├── 18_relational_rnn.ipynb             # 關係型 RNN
├── 19_coffee_automaton.ipynb           # 不可逆性深入探討
├── 20_neural_turing_machine.ipynb     # 外部記憶
├── 21_ctc_speech.ipynb                # CTC 損失
├── 22_scaling_laws.ipynb              # 經驗縮放
├── 23_mdl_principle.ipynb             # MDL 與壓縮
├── 24_machine_super_intelligence.ipynb # 通用 AI 與 AIXI
├── 25_kolmogorov_complexity.ipynb     # K(x) 與隨機性
├── 26_cs231n_cnn_fundamentals.ipynb    # 從第一原理學視覺
├── 27_multi_token_prediction.ipynb     # 多詞元預測
├── 28_dense_passage_retrieval.ipynb    # 密集檢索
├── 29_rag.ipynb                       # RAG 架構
└── 30_lost_in_middle.ipynb            # 長上下文分析
```

**所有 30 篇論文已實作！（100% 完成！）🎉**

## 學習路徑

### 初學者軌跡（從這裡開始！）
1. **字元 RNN**（`02_char_rnn_karpathy.ipynb`）- 學習基本 RNN
2. **LSTM**（`03_lstm_understanding.ipynb`）- 理解閘門機制
3. **CNN**（`07_alexnet_cnn.ipynb`）- 電腦視覺基礎
4. **ResNet**（`10_resnet_deep_residual.ipynb`）- 跳躍連接
5. **VAE**（`17_variational_autoencoder.ipynb`）- 生成式模型

### 中級軌跡
6. **RNN 正則化**（`04_rnn_regularization.ipynb`）- 更好的訓練
7. **Bahdanau 注意力**（`14_bahdanau_attention.ipynb`）- 注意力基礎
8. **指標網路**（`06_pointer_networks.ipynb`）- 注意力作為選擇
9. **集合的 Seq2Seq**（`08_seq2seq_for_sets.ipynb`）- 排列不變性
10. **CS231n**（`26_cs231n_cnn_fundamentals.ipynb`）- 完整視覺管線（kNN → CNN）
11. **GPipe**（`09_gpipe.ipynb`）- 大型模型的管線並行
12. **Transformer**（`13_attention_is_all_you_need.ipynb`）- 現代架構
13. **擴張卷積**（`11_dilated_convolutions.ipynb`）- 感受野
14. **縮放定律**（`22_scaling_laws.ipynb`）- 理解規模

### 進階軌跡
15. **預激活 ResNet**（`15_identity_mappings_resnet.ipynb`）- 架構細節
16. **圖神經網路**（`12_graph_neural_networks.ipynb`）- 圖學習
17. **關係網路**（`16_relational_reasoning.ipynb`）- 關係推理
18. **神經圖靈機**（`20_neural_turing_machine.ipynb`）- 外部記憶
19. **CTC 損失**（`21_ctc_speech.ipynb`）- 語音辨識
20. **密集檢索**（`28_dense_passage_retrieval.ipynb`）- 語義搜索
21. **RAG**（`29_rag.ipynb`）- 檢索增強生成
22. **迷失在中間**（`30_lost_in_middle.ipynb`）- 長上下文分析

### 理論與基礎
23. **MDL 原理**（`23_mdl_principle.ipynb`）- 通過壓縮進行模型選擇
24. **Kolmogorov 複雜度**（`25_kolmogorov_complexity.ipynb`）- 隨機性與資訊
25. **複雜度動力學**（`01_complexity_dynamics.ipynb`）- 熵與湧現
26. **咖啡自動機**（`19_coffee_automaton.ipynb`）- 深入探討不可逆性

## Sutskever 30 的關鍵洞察

### 架構演進
- **RNN → LSTM**：閘門解決梯度消失
- **普通網路 → ResNet**：跳躍連接實現深度
- **RNN → Transformer**：注意力實現並行化
- **固定詞彙 → 指標**：輸出可以引用輸入

### 基本機制
- **注意力**：可微分選擇機制
- **殘差連接**：梯度高速公路
- **閘門**：學習的資訊流控制
- **外部記憶**：將儲存與計算分離

### 訓練洞察
- **縮放定律**：性能隨規模可預測地改進
- **正則化**：Dropout、權重衰減、資料增強
- **最佳化**：梯度裁剪、學習率排程
- **計算最優**：平衡模型大小和訓練資料

### 理論基礎
- **資訊理論**：壓縮、熵、MDL
- **複雜度**：Kolmogorov 複雜度、冪律
- **生成式建模**：VAE、ELBO、潛在空間
- **記憶**：可微分資料結構

## 實作哲學

### 為什麼只用 NumPy？

這些實作刻意避免 PyTorch/TensorFlow 以：
- **加深理解**：看到框架抽象掉了什麼
- **教育清晰度**：沒有魔法，每個操作都明確
- **核心概念**：專注於演算法，而非框架 API
- **可遷移知識**：原理適用於任何框架

### 合成資料方法

每個 notebook 生成自己的資料以：
- **立即執行**：不需要下載資料集
- **受控實驗**：在簡單情況下理解行為
- **概念聚焦**：資料不會遮蔽演算法
- **快速迭代**：修改並立即重新運行

## 擴展與下一步

### 基於這些實作繼續構建

理解核心概念後，嘗試：

1. **擴大規模**：在 PyTorch/JAX 中實作用於真實資料集
2. **結合技術**：例如 ResNet + 注意力
3. **現代變體**：
   - RNN → GRU → Transformer
   - VAE → β-VAE → VQ-VAE
   - ResNet → ResNeXt → EfficientNet
4. **應用**：應用於真實問題

### 研究方向

Sutskever 30 指向：
- 縮放（更大的模型、更多資料）
- 效率（稀疏模型、量化）
- 能力（推理、多模態）
- 理解（可解釋性、理論）

## 資源

### 原始論文
參見 `IMPLEMENTATION_TRACKS.md` 獲取完整引用和連結

### 延伸閱讀
- [Ilya Sutskever 的閱讀清單（GitHub）](https://github.com/dzyim/ilya-sutskever-recommended-reading)
- [Aman 的 AI 日誌 - Sutskever 30 入門](https://aman.ai/primers/ai/top-30-papers/)
- [註解版 Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- [Andrej Karpathy 的部落格](http://karpathy.github.io/)

### 課程
- Stanford CS231n：卷積神經網路
- Stanford CS224n：深度學習 NLP
- MIT 6.S191：深度學習入門

## 貢獻

這些實作是教育性的，可以改進！考慮：
- 添加更多視覺化
- 實作遺漏的論文
- 改進說明
- 發現錯誤
- 添加與框架實作的比較

## 引用

如果你在工作或教學中使用這些實作：

```bibtex
@misc{sutskever30implementations,
  title={Sutskever 30：完整實作套件},
  author={Paul "The Pageman" Pajo, pageman@gmail.com},
  year={2025},
  note={Ilya Sutskever 推薦閱讀清單的教育性實作，靈感來自 https://papercode.vercel.app/}
}
```

## 授權

教育用途。原始研究引用請參見各論文。

## 致謝

- **Ilya Sutskever**：策劃這份必要的閱讀清單
- **論文作者**：他們的基礎性貢獻
- **社群**：使這些想法變得易於理解

---

## 最新添加（2025 年 12 月）

### 最近實作（21 篇新論文！）
- ✅ **論文 4**：RNN 正則化（變分 dropout）
- ✅ **論文 5**：神經網路剪枝（MDL、90%+ 稀疏性）
- ✅ **論文 7**：AlexNet（從頭建立 CNN）
- ✅ **論文 8**：集合的 Seq2Seq（排列不變性、注意力池化）
- ✅ **論文 9**：GPipe（管線並行、微批次、重新材料化）
- ✅ **論文 19**：咖啡自動機（深入探討不可逆性、熵、Landauer 原理）
- ✅ **論文 26**：CS231n（完整視覺管線：kNN → CNN，全部用 NumPy）
- ✅ **論文 11**：擴張卷積（多尺度）
- ✅ **論文 12**：圖神經網路（訊息傳遞）
- ✅ **論文 14**：Bahdanau 注意力（原始注意力）
- ✅ **論文 15**：恆等映射 ResNet（預激活）
- ✅ **論文 16**：關係推理（關係網路）
- ✅ **論文 18**：關係型 RNN（關係記憶 + 第 11 節：手動反向傳播約 1100 行）
- ✅ **論文 21**：Deep Speech 2（CTC 損失）
- ✅ **論文 23**：MDL 原理（壓縮、模型選擇，連接論文 5 和 25）
- ✅ **論文 24**：機器超級智慧（通用 AI、AIXI、Solomonoff 歸納、智慧測量、遞迴自我改進）
- ✅ **論文 25**：Kolmogorov 複雜度（隨機性、演算法機率、理論基礎）
- ✅ **論文 27**：多詞元預測（2-3 倍樣本效率）
- ✅ **論文 28**：密集段落檢索（雙編碼器）
- ✅ **論文 29**：RAG（檢索增強生成）
- ✅ **論文 30**：迷失在中間（長上下文）

## 快速參考：實作複雜度

### 一個下午可完成
- ✅ 字元 RNN
- ✅ LSTM
- ✅ ResNet
- ✅ 簡單 VAE
- ✅ 擴張卷積

### 週末專案
- ✅ Transformer
- ✅ 指標網路
- ✅ 圖神經網路
- ✅ 關係網路
- ✅ 神經圖靈機
- ✅ CTC 損失
- ✅ 密集檢索

### 一週深度探索
- ✅ 完整 RAG 系統
- ⚠️ 大規模實驗
- ⚠️ 超參數最佳化

---

**「如果你真的學會所有這些，你將知道當今 90% 重要的內容。」** - Ilya Sutskever

祝學習愉快！🚀
