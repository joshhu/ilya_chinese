# Sutskever 30 - 完整實作軌道

本文件提供 Ilya Sutskever 著名閱讀清單中每篇論文的詳細實作軌道。

**狀態：30/30 篇論文已實作（100% 完成！）** ✅

所有實作使用純 NumPy 程式碼與合成資料，以便即時執行和教學清晰度。

---

## 1. The First Law of Complexodynamics（Scott Aaronson）

**類型**：理論文章
**可實作**：是（概念性）
**Notebook**：`01_complexity_dynamics.ipynb` ✅

**實作軌道**：
- 使用細胞自動機（Cellular Automata）展示熵（Entropy）與複雜度（Complexity）成長
- 實作顯示複雜度動態的簡單物理模擬
- 視覺化封閉系統中的熵變化

**我們建構的內容**：
- Rule 30 細胞自動機模擬
- 隨時間的熵測量
- 複雜度指標與視覺化
- 不可逆性概念介紹

**關鍵概念**：熵（Entropy）、複雜度（Complexity）、熱力學第二定律、細胞自動機（Cellular Automata）

---

## 2. The Unreasonable Effectiveness of RNNs（Andrej Karpathy）

**類型**：字元級語言模型
**可實作**：是
**Notebook**：`02_char_rnn_karpathy.ipynb` ✅

**實作軌道**：
1. 從文本建立字元級詞彙表
2. 實作帶有前向/反向傳播的 vanilla RNN cell
3. 使用 teacher forcing 在文本序列上訓練
4. 實作帶有溫度控制的取樣/生成
5. 視覺化隱藏狀態激活

**我們建構的內容**：
- 從零開始的完整 vanilla RNN
- 字元級文本生成
- 溫度控制取樣
- 隱藏狀態視覺化
- Shakespeare 風格文本生成

**關鍵概念**：RNN、字元建模、文本生成、BPTT（時間反向傳播）

---

## 3. Understanding LSTM Networks（Christopher Olah）

**類型**：LSTM 架構
**可實作**：是
**Notebook**：`03_lstm_understanding.ipynb` ✅

**實作軌道**：
1. 實作 LSTM cell（遺忘門、輸入門、輸出門）
2. 建構帶有閘門計算的前向傳播
3. 實作時間反向傳播（BPTT）
4. 比較 vanilla RNN 與 LSTM 在序列任務上的表現
5. 視覺化閘門激活隨時間變化

**我們建構的內容**：
- 完整 LSTM 實作，包含所有閘門
- 遺忘門、輸入門、輸出門機制
- Cell state 和 hidden state 追蹤
- 與 vanilla RNN 在長序列上的比較
- 閘門激活視覺化

**關鍵概念**：LSTM、閘門（Gates）、長期依賴（Long-term Dependencies）、梯度流（Gradient Flow）

---

## 4. Recurrent Neural Network Regularization（Zaremba et al.）

**類型**：RNN 的 Dropout
**可實作**：是
**Notebook**：`04_rnn_regularization.ipynb` ✅

**實作軌道**：
1. 實作標準 dropout
2. 實作變分 dropout（Variational Dropout，跨時間步驟使用相同 mask）
3. 僅對非循環連接應用 dropout
4. 比較不同的 dropout 策略
5. 在序列建模任務上評估

**關鍵概念**：Dropout、正則化（Regularization）、防止過擬合（Overfitting Prevention）

---

## 5. Keeping Neural Networks Simple（Hinton & van Camp）

**類型**：MDL 原則 / 權重剪枝
**可實作**：是
**Notebook**：`05_neural_network_pruning.ipynb` ✅

**實作軌道**：
1. 實作簡單神經網路
2. 添加 L1/L2 正則化以實現稀疏性
3. 實作基於幅度的剪枝
4. 計算權重的描述長度
5. 比較模型大小與效能的取捨

**關鍵概念**：最小描述長度（Minimum Description Length）、壓縮（Compression）、剪枝（Pruning）

---

## 6. Pointer Networks（Vinyals et al.）

**類型**：基於注意力的架構
**可實作**：是
**Notebook**：`06_pointer_networks.ipynb` ✅

**實作軌道**：
1. 實作注意力機制
2. 建構帶有指標機制的編碼器-解碼器
3. 在凸包問題上訓練（合成幾何）
4. 在旅行推銷員問題（TSP）上訓練
5. 視覺化測試範例上的注意力權重

**關鍵概念**：注意力（Attention）、指標（Pointers）、組合最佳化（Combinatorial Optimization）

---

## 7. ImageNet Classification（AlexNet）（Krizhevsky et al.）

**類型**：卷積神經網路
**可實作**：是（縮小版本）
**Notebook**：`07_alexnet_cnn.ipynb` ✅

**實作軌道**：
1. 實作卷積層
2. 建構 AlexNet 架構（針對小資料集縮放）
3. 實作資料增強
4. 在 CIFAR-10 或小型 ImageNet 子集上訓練
5. 視覺化學習到的濾波器和特徵圖

**關鍵概念**：CNN、卷積（Convolution）、ReLU、Dropout、資料增強（Data Augmentation）

---

## 8. Order Matters: Sequence to Sequence for Sets（Vinyals et al.）

**類型**：Read-Process-Write 架構
**可實作**：是
**Notebook**：`08_seq2seq_for_sets.ipynb` ✅

**實作軌道**：
1. 實作帶有注意力的集合編碼
2. 建構 read-process-write 網路
3. 在排序任務上訓練
4. 在集合問題上測試（集合聯集、尋找最大值）
5. 與順序無關的基線比較

**關鍵概念**：集合（Sets）、排列不變性（Permutation Invariance）、注意力（Attention）

---

## 9. GPipe: Pipeline Parallelism（Huang et al.）

**類型**：模型並行
**可實作**：是（概念性）
**Notebook**：`09_gpipe.ipynb` ✅

**實作軌道**：
1. 實作帶有層分割的簡單神經網路
2. 以順序執行模擬微批次管道
3. 視覺化管道氣泡開銷
4. 比較管道與順序的吞吐量
5. 展示梯度累積

**關鍵概念**：模型並行（Model Parallelism）、管道（Pipeline）、微批次處理（Micro-batching）

---

## 10. Deep Residual Learning（ResNet）（He et al.）

**類型**：殘差神經網路
**可實作**：是
**Notebook**：`10_resnet_deep_residual.ipynb` ✅

**實作軌道**：
1. 實作帶有跳躍連接的殘差區塊
2. 建構 ResNet 架構（18/34 層）
3. 比較有/無殘差的訓練
4. 視覺化梯度流
5. 在影像分類任務上訓練

**關鍵概念**：跳躍連接（Skip Connections）、梯度流（Gradient Flow）、深度網路（Deep Networks）

---

## 11. Multi-Scale Context Aggregation（Dilated Convolutions）（Yu & Koltun）

**類型**：空洞/膨脹卷積
**可實作**：是
**Notebook**：`11_dilated_convolutions.ipynb` ✅

**實作軌道**：
1. 實作空洞卷積運算
2. 建構多尺度感受野網路
3. 應用於語義分割（玩具資料集）
4. 視覺化不同膨脹率的感受野
5. 與標準卷積比較

**關鍵概念**：空洞卷積（Dilated Convolution）、感受野（Receptive Field）、分割（Segmentation）

---

## 12. Neural Message Passing for Quantum Chemistry（Gilmer et al.）

**類型**：圖神經網路
**可實作**：是
**Notebook**：`12_graph_neural_networks.ipynb` ✅

**實作軌道**：
1. 實作圖表示（鄰接矩陣、特徵）
2. 建構訊息傳遞層
3. 實作節點和邊更新
4. 在分子性質預測上訓練（QM9 子集）
5. 視覺化訊息傳播

**關鍵概念**：圖網路（Graph Networks）、訊息傳遞（Message Passing）、分子機器學習（Molecular ML）

---

## 13. Attention Is All You Need（Vaswani et al.）

**類型**：Transformer 架構
**可實作**：是
**Notebook**：`13_attention_is_all_you_need.ipynb` ✅

**實作軌道**：
1. 實作縮放點積注意力
2. 建構多頭注意力
3. 實作位置編碼
4. 建構編碼器-解碼器 transformer
5. 在序列轉換任務上訓練
6. 視覺化注意力模式

**關鍵概念**：自注意力（Self-Attention）、多頭注意力（Multi-Head Attention）、Transformers

---

## 14. Neural Machine Translation（Attention）（Bahdanau et al.）

**類型**：帶注意力的 Seq2Seq
**可實作**：是
**Notebook**：`14_bahdanau_attention.ipynb` ✅

**實作軌道**：
1. 實作編碼器-解碼器 RNN
2. 添加 Bahdanau（加性）注意力
3. 在簡單翻譯任務上訓練（數字、日期）
4. 實作束搜尋（Beam Search）
5. 視覺化注意力對齊

**關鍵概念**：注意力（Attention）、Seq2Seq、對齊（Alignment）

---

## 15. Identity Mappings in ResNet（He et al.）

**類型**：ResNet 變體
**可實作**：是
**Notebook**：`15_identity_mappings_resnet.ipynb` ✅

**實作軌道**：
1. 實作預激活殘差區塊
2. 比較激活順序（pre vs post）
3. 測試不同的跳躍連接變體
4. 視覺化梯度傳播
5. 比較收斂速度

**關鍵概念**：預激活（Pre-activation）、跳躍連接（Skip Connections）、梯度流（Gradient Flow）

---

## 16. Simple Neural Network for Relational Reasoning（Santoro et al.）

**類型**：關係網路
**可實作**：是
**Notebook**：`16_relational_reasoning.ipynb` ✅

**實作軌道**：
1. 實作成對關係函數
2. 建構關係網路架構
3. 生成合成關係推理任務（類 CLEVR）
4. 在「相同-不同」和「計數」任務上訓練
5. 視覺化學習到的關係

**關鍵概念**：關係推理（Relational Reasoning）、成對函數（Pairwise Functions）、組合性（Compositionality）

---

## 17. Variational Lossy Autoencoder（Chen et al.）

**類型**：VAE 變體
**可實作**：是
**Notebook**：`17_variational_autoencoder.ipynb` ✅

**實作軌道**：
1. 實作標準 VAE
2. 添加 bits-back 編碼用於壓縮
3. 實作階層式潛在結構
4. 在影像資料集上訓練（MNIST/Fashion-MNIST）
5. 視覺化潛在空間和重建
6. 測量率-失真權衡

**關鍵概念**：VAE、率-失真（Rate-Distortion）、階層式潛在變數（Hierarchical Latents）

---

## 18. Relational Recurrent Neural Networks（Santoro et al.）

**類型**：Relational RNN
**可實作**：是
**Notebook**：`18_relational_rnn.ipynb` ✅

**實作軌道**：
1. 實作用於記憶體的多頭點積注意力
2. 建構 relational memory 核心
3. 建立序列推理任務
4. 與標準 LSTM 比較
5. 視覺化記憶體互動

**關鍵概念**：Relational Memory、RNN 中的自注意力、推理（Reasoning）

---

## 19. The Coffee Automaton（Aaronson et al.）

**類型**：複雜度理論 / 不可逆性
**可實作**：是（全面性）
**Notebook**：`19_coffee_automaton.ipynb` ✅

**實作軌道**：
1. 實作咖啡混合模擬（擴散）
2. 測量隨時間的熵和複雜度指標
3. 展示混合和複雜度成長
4. 視覺化熵增加和粗粒化
5. 展示不可逆性和龐加萊遞歸
6. 實作 Maxwell's demon 思想實驗
7. 展示 Landauer 原則（計算不可逆性）
8. 探索機器學習中的資訊瓶頸
9. 連結到時間之箭

**我們建構的內容**：
- **10 個關於不可逆性的全面章節**
- 咖啡擴散模擬與粒子追蹤
- 熵成長視覺化（Shannon、粗粒化）
- 相空間演化和 Liouville 定理
- 龐加萊遞歸計算（將在 e^N 時間後解混！）
- Maxwell's demon 模擬
- Landauer 原則：每位元擦除需 kT ln(2) 能量
- 單向函數和計算不可逆性
- 神經網路中的資訊瓶頸
- 生物系統與熱力學第二定律
- 時間之箭：基本 vs 湧現的爭論
- 約 2,500 行，跨 10 個章節

**關鍵概念**：不可逆性（Irreversibility）、熵（Entropy）、混合（Mixing）、粗粒化（Coarse-graining）、Maxwell's Demon、Landauer 原則、時間之箭（Arrow of Time）、熱力學第二定律

---

## 20. Neural Turing Machines（Graves et al.）

**類型**：記憶體增強神經網路
**可實作**：是
**Notebook**：`20_neural_turing_machine.ipynb` ✅

**實作軌道**：
1. 實作外部記憶體矩陣
2. 建構基於內容的定址
3. 實作基於位置的定址
4. 建構帶有注意力的讀/寫頭
5. 在複製和重複複製任務上訓練
6. 視覺化記憶體存取模式

**關鍵概念**：外部記憶體（External Memory）、可微分定址（Differentiable Addressing）、注意力（Attention）

---

## 21. Deep Speech 2（Baidu Research）

**類型**：語音辨識
**可實作**：是（簡化版）
**Notebook**：`21_ctc_speech.ipynb` ✅

**實作軌道**：
1. 生成合成音頻資料或使用小型語音資料集
2. 實作 RNN/CNN 聲學模型
3. 實作 CTC 損失
4. 端對端訓練語音辨識
5. 視覺化頻譜圖和預測

**關鍵概念**：CTC 損失、Sequence-to-Sequence、語音辨識（Speech Recognition）

---

## 22. Scaling Laws for Neural Language Models（Kaplan et al.）

**類型**：實證分析
**可實作**：是
**Notebook**：`22_scaling_laws.ipynb` ✅

**實作軌道**：
1. 實作簡單語言模型（Transformer）
2. 訓練多個不同大小的模型
3. 改變資料集大小和計算預算
4. 繪製損失與參數/資料/計算的關係圖
5. 擬合冪律關係
6. 預測更大模型的效能

**關鍵概念**：縮放定律（Scaling Laws）、冪律（Power Laws）、計算最優訓練（Compute-Optimal Training）

---

## 23. Minimum Description Length Principle（Grunwald）

**類型**：資訊理論
**可實作**：是（概念性）
**Notebook**：`23_mdl_principle.ipynb` ✅

**實作軌道**：
1. 實作各種壓縮方案
2. 計算資料+模型的描述長度
3. 比較不同的模型複雜度
4. 展示 MDL 用於模型選擇
5. 展示過擬合與壓縮的權衡
6. 應用於神經網路架構選擇
7. 連結到 Kolmogorov 複雜度

**我們建構的內容**：
- Huffman 編碼實作
- 不同模型的 MDL 計算
- 透過壓縮進行模型選擇
- 使用 MDL 進行神經網路架構比較
- 基於 MDL 的剪枝
- 與 AIC/BIC 資訊準則的連結
- 論文 25（Kolmogorov 複雜度）的預備

**關鍵概念**：MDL、模型選擇（Model Selection）、壓縮（Compression）、資訊理論（Information Theory）、奧卡姆剃刀（Occam's Razor）

---

## 24. Machine Super Intelligence（Shane Legg）

**類型**：博士論文 - 通用人工智慧
**可實作**：是（理論與實用近似）
**Notebook**：`24_machine_super_intelligence.ipynb` ✅

**實作軌道**：
1. 實作心理測量智力模型（g-factor）
2. 透過程式枚舉建構 Solomonoff 歸納近似
3. 估計序列的 Kolmogorov 複雜度
4. 實作 Monte Carlo AIXI（MC-AIXI）代理
5. 建立具有不同複雜度的玩具環境套件
6. 計算通用智力測度 Υ(π)
7. 探索計算-效能權衡
8. 模擬遞歸自我改進
9. 建模智力爆炸動態

**我們建構的內容**：
- **6 個關於通用 AI 的全面章節**
- **第 1 章**：心理測量智力和 g-factor 提取（對認知測試的 PCA）
- **第 2 章**：透過程式枚舉的 Solomonoff 歸納、序列預測、K(x) 近似
- **第 3 章**：AIXI 代理理論、使用 MCTS 的 MC-AIXI 實作、玩具網格世界環境
- **第 4 章**：通用智力測度 Υ(π) = Σ 2^(-K(μ)) V_μ^π、跨環境的代理比較
- **第 5 章**：有時間限制的 AIXI、計算預算實驗、不可計算性展示
- **第 6 章**：遞歸自我改進模擬、智力爆炸情境（線性/指數/超指數）
- SimpleProgramEnumerator：帶有 Solomonoff 先驗的加權序列預測
- ToyGridWorld 環境，包含 Random、Greedy 和 MC-AIXI 代理
- 使用 UCB1 選擇的基於 MCTS 的規劃
- 跨不同環境的智力測量
- 具有能力增強的自我改進代理
- 成長模型和起飛情境
- **約 2,000 行，跨 6 個章節**
- **15+ 視覺化**：相關矩陣、Solomonoff 先驗、代理比較、智力測度、能力成長曲線

**關鍵概念**：
- 通用智力（Universal Intelligence）Υ(π)
- AIXI：理論上最優的 RL 代理
- Solomonoff 歸納 & 通用先驗
- Kolmogorov 複雜度 K(x)
- Monte Carlo AIXI（MC-AIXI）
- 智力爆炸 & 遞歸自我改進
- 不可計算性 vs 可近似性
- 心理測量 g-factor
- 環境複雜度加權

**連結**：論文 23（MDL）、論文 25（Kolmogorov 複雜度）、論文 8（DQN）

---

## 25. Kolmogorov Complexity（Shen et al.）

**類型**：書籍/理論
**可實作**：是（概念性）
**Notebook**：`25_kolmogorov_complexity.ipynb` ✅

**實作軌道**：
1. 實作簡單壓縮演算法
2. 透過壓縮估計 Kolmogorov 複雜度
3. 展示隨機字串的不可壓縮性
4. 展示結構化與隨機資料的複雜度
5. 與最小描述長度相關聯
6. 連結到 Solomonoff 歸納和通用先驗
7. 形式化奧卡姆剃刀

**我們建構的內容**：
- K(x) = 生成 x 的最短程式長度
- 基於壓縮的 K(x) 估計
- 隨機性 = 不可壓縮性展示
- 演算法機率（Solomonoff 先驗）
- 用於歸納的通用先驗
- 與 Shannon 熵的連結
- 奧卡姆剃刀形式化
- 機器學習的理論基礎

**關鍵概念**：Kolmogorov 複雜度 K(x)、壓縮（Compression）、資訊理論（Information Theory）、隨機性（Randomness）、演算法機率（Algorithmic Probability）、通用先驗（Universal Prior）

---

## 26. Stanford CS231n: CNNs for Visual Recognition

**類型**：課程 - 完整視覺流程
**可實作**：是（全面性）
**Notebook**：`26_cs231n_cnn_fundamentals.ipynb` ✅

**實作軌道**：
1. 生成合成 CIFAR-10 資料（程序模式）
2. 實作 k-近鄰基線（L1/L2 距離）
3. 建構線性分類器（SVM hinge loss、Softmax cross-entropy）
4. 實作最佳化演算法（SGD、Momentum、Adam）
5. 建構帶反向傳播的 2 層神經網路
6. 實作卷積層（conv2d、maxpool、ReLU）
7. 建構完整 CNN 架構（Mini-AlexNet）
8. 實作視覺化技術（顯著性圖、濾波器視覺化）
9. 展示遷移學習原則
10. 應用調參技巧和除錯策略

**我們建構的內容**：
- **10 個全面章節涵蓋整個 CS231n 課程**
- **第 1 章**：合成 CIFAR-10 生成（具有類別特定模式的程序 32×32 影像）
- **第 2 章**：k-NN 分類器（L1/L2 距離、交叉驗證）
- **第 3 章**：線性分類器（SVM hinge loss、Softmax cross-entropy、梯度計算）
- **第 4 章**：最佳化（SGD、Momentum、Adam、學習率排程）
- **第 5 章**：2 層神經網路（前向傳播、ReLU、反向傳播）
- **第 6 章**：CNN 層（conv2d_forward、maxpool2d_forward，帶快取）
- **第 7 章**：完整 CNN（Mini-AlexNet：Conv→ReLU→Pool→FC）
- **第 8 章**：視覺化（顯著性圖、濾波器視覺化）
- **第 9 章**：遷移學習和微調概念
- **第 10 章**：神經網路調參（健全性檢查、損失曲線、超參數調優）
- 完整視覺流程：kNN → Linear → NN → CNN
- 全部使用純 NumPy（約 2,400 行）
- 合成資料（無需下載）
- **優先考慮教育清晰度而非速度**

**關鍵概念**：
- 影像分類流程
- k-近鄰（kNN）
- 線性分類器（SVM、Softmax）
- 最佳化（SGD、Momentum、Adam）
- 神經網路 & 反向傳播
- 卷積層
- 池化 & ReLU 激活
- CNN 架構
- 顯著性圖 & 視覺化
- 遷移學習
- 神經網路調參

**連結**：論文 7（AlexNet）、論文 10（ResNet）、論文 11（空洞卷積）

---

## 27. Multi-token Prediction（Gloeckle et al.）

**類型**：語言模型訓練
**可實作**：是
**Notebook**：`27_multi_token_prediction.ipynb` ✅

**實作軌道**：
1. 實作標準下一個 token 預測
2. 修改為預測多個未來 token
3. 使用多 token 目標訓練語言模型
4. 比較與單 token 的樣本效率
5. 測量困惑度和生成品質

**關鍵概念**：語言建模（Language Modeling）、多任務學習（Multi-task Learning）、預測（Prediction）

---

## 28. Dense Passage Retrieval（Karpukhin et al.）

**類型**：資訊檢索
**可實作**：是
**Notebook**：`28_dense_passage_retrieval.ipynb` ✅

**實作軌道**：
1. 實作雙編碼器（query + passage）
2. 建立小型文件語料庫
3. 使用批次內負樣本訓練
4. 實作近似最近鄰搜尋
5. 評估檢索準確度
6. 建構簡單 QA 系統

**關鍵概念**：稠密檢索（Dense Retrieval）、雙編碼器（Dual Encoders）、語義搜尋（Semantic Search）

---

## 29. Retrieval-Augmented Generation（Lewis et al.）

**類型**：RAG 架構
**可實作**：是
**Notebook**：`29_rag.ipynb` ✅

**實作軌道**：
1. 建構文件編碼器和檢索器
2. 實作簡單 seq2seq 生成器
3. 結合檢索 + 生成
4. 建立知識密集型 QA 任務
5. 比較 RAG 與非檢索基線
6. 視覺化檢索的文件

**關鍵概念**：檢索（Retrieval）、生成（Generation）、知識密集型 NLP（Knowledge-Intensive NLP）

---

## 30. Lost in the Middle（Liu et al.）

**類型**：長上下文分析
**可實作**：是
**Notebook**：`30_lost_in_middle.ipynb` ✅

**實作軌道**：
1. 實作簡單 Transformer 模型
2. 建立具有不同上下文位置的合成任務
3. 測試從上下文開頭/中間/結尾的檢索
4. 繪製準確度 vs 位置曲線
5. 展示「lost in the middle」現象
6. 測試緩解策略

**關鍵概念**：長上下文（Long Context）、注意力（Attention）、位置偏差（Position Bias）

---

## 摘要統計

**總論文數：30/30（100% 完成！）** 🎉

- **完全實作**：30 篇論文
- **純 NumPy**：所有實作
- **合成資料**：所有 notebook 可即時執行
- **總程式碼行數**：約 50,000+ 行教育程式碼

## 實作難度等級

**初學者**（直接的，下午專案）：
- 2（Char RNN）、4（RNN Regularization）、5（Pruning）、7（AlexNet）、10（ResNet）、15（Pre-activation ResNet）、17（VAE）、21（CTC）

**中級**（週末專案）：
- 3（LSTM）、6（Pointer Networks）、8（Seq2Seq for Sets）、11（Dilated Conv）、12（GNNs）、14（Bahdanau Attention）、16（Relation Networks）、18（Relational RNN）、22（Scaling Laws）、27（Multi-token Prediction）、28（Dense Retrieval）

**進階**（一週深入研究）：
- 9（GPipe）、13（Transformer）、20（NTM）、29（RAG）、30（Lost in Middle）

**全面/理論**（多章節探索）：
- 1（Complexity Dynamics）、19（Coffee Automaton - 10 章節）、23（MDL）、24（Machine Super Intelligence - 6 章節）、25（Kolmogorov Complexity）、26（CS231n - 10 章節）

## 精選亮點

**最長實作**：
- 論文 26（CS231n）：約 2,400 行，10 章節
- 論文 19（Coffee Automaton）：約 2,500 行，10 章節
- 論文 24（Machine Super Intelligence）：約 2,000 行，6 章節
- 論文 18（Relational RNN）：約 1,100 行手動反向傳播章節

**最多視覺化**：
- 論文 24（Machine Super Intelligence）：15+ 圖表
- 論文 19（Coffee Automaton）：20+ 視覺化
- 論文 26（CS231n）：15+ 視覺化
- 論文 22（Scaling Laws）：10+ 圖表

**理論基礎**：
- 論文 23、24、25：資訊理論三部曲（MDL、Universal AI、Kolmogorov）
- 論文 1、19：複雜度和不可逆性
- 論文 22：實證縮放定律

---

**「如果你真正學會所有這些，你將了解今天重要內容的 90%。」** - Ilya Sutskever

**所有 30 篇論文現已實作，可供自主學習！** 🚀
