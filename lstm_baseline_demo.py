"""
LSTM 基準模型 - 使用示範

本腳本展示如何將 LSTM 基準模型應用於各種任務。
"""

import numpy as np
from lstm_baseline import LSTM, LSTMCell


def demo_sequence_classification():
    """
    展示 LSTM 用於序列分類。
    任務：根據序列的模式進行分類。
    """
    print("\n" + "="*60)
    print("Demo 1: Sequence Classification")
    print("="*60)

    # 建立合成資料：具有不同模式的序列
    batch_size = 4
    seq_len = 20
    input_size = 8
    hidden_size = 32
    num_classes = 3

    print(f"\nTask: Classify {num_classes} different sequence patterns")
    print(f"Sequence length: {seq_len}, Input features: {input_size}")

    # 生成具有不同模式的序列
    sequences = []
    labels = []

    # 模式 0：遞增趨勢
    seq0 = np.linspace(0, 1, seq_len).reshape(-1, 1) * np.random.randn(seq_len, input_size) * 0.1
    seq0 = seq0 + np.linspace(0, 1, seq_len).reshape(-1, 1)
    sequences.append(seq0)
    labels.append(0)

    # 模式 1：遞減趨勢
    seq1 = np.linspace(1, 0, seq_len).reshape(-1, 1) * np.random.randn(seq_len, input_size) * 0.1
    seq1 = seq1 + np.linspace(1, 0, seq_len).reshape(-1, 1)
    sequences.append(seq1)
    labels.append(1)

    # 模式 2：振盪
    seq2 = np.sin(np.linspace(0, 4*np.pi, seq_len)).reshape(-1, 1) * np.ones((seq_len, input_size))
    seq2 = seq2 + np.random.randn(seq_len, input_size) * 0.1
    sequences.append(seq2)
    labels.append(2)

    # 再次使用模式 0
    seq0_2 = np.linspace(0, 1, seq_len).reshape(-1, 1) * np.random.randn(seq_len, input_size) * 0.1
    seq0_2 = seq0_2 + np.linspace(0, 1, seq_len).reshape(-1, 1)
    sequences.append(seq0_2)
    labels.append(0)

    # 堆疊成批次
    batch = np.stack(sequences, axis=0)  # (batch_size, seq_len, input_size)

    # 建立 LSTM 模型
    lstm = LSTM(input_size, hidden_size, output_size=num_classes)

    # 前向傳播 - 僅獲取最終輸出用於分類
    outputs = lstm.forward(batch, return_sequences=False)

    print(f"\nInput shape: {batch.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Expected shape: ({batch_size}, {num_classes})")

    # 套用 softmax 取得類別機率
    exp_outputs = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
    probabilities = exp_outputs / np.sum(exp_outputs, axis=1, keepdims=True)

    print(f"\nPredicted class probabilities (before training):")
    for i in range(batch_size):
        pred_class = np.argmax(probabilities[i])
        true_class = labels[i]
        print(f"  Sample {i}: pred={pred_class}, true={true_class}, probs={probabilities[i]}")

    print("\nNote: Model is randomly initialized, so predictions are random.")
    print("After training, it would learn to classify these patterns correctly.")


def demo_sequence_to_sequence():
    """
    展示 LSTM 用於序列到序列任務。
    任務：對輸入序列進行轉換後輸出。
    """
    print("\n" + "="*60)
    print("Demo 2: Sequence-to-Sequence Processing")
    print("="*60)

    batch_size = 2
    seq_len = 15
    input_size = 10
    hidden_size = 24
    output_size = 10

    print(f"\nTask: Process sequences and output transformed sequences")
    print(f"Input sequence length: {seq_len}")
    print(f"Output sequence length: {seq_len}")

    # 建立輸入序列
    sequences = np.random.randn(batch_size, seq_len, input_size) * 0.5

    # 建立 LSTM
    lstm = LSTM(input_size, hidden_size, output_size=output_size)

    # 前向傳播 - 獲取所有時間步的輸出
    outputs = lstm.forward(sequences, return_sequences=True)

    print(f"\nInput shape: {sequences.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {output_size})")

    # 顯示輸出統計資訊
    print(f"\nOutput statistics:")
    print(f"  Mean: {np.mean(outputs):.4f}")
    print(f"  Std: {np.std(outputs):.4f}")
    print(f"  Min: {np.min(outputs):.4f}")
    print(f"  Max: {np.max(outputs):.4f}")


def demo_state_persistence():
    """
    展示 LSTM 如何跨時間步維持狀態。
    """
    print("\n" + "="*60)
    print("Demo 3: State Persistence and Memory")
    print("="*60)

    batch_size = 1
    seq_len = 30
    input_size = 5
    hidden_size = 16

    print(f"\nDemonstrating how LSTM maintains memory over {seq_len} time steps")

    # 建立一個在早期具有模式的序列
    sequence = np.zeros((batch_size, seq_len, input_size))
    # 在前 5 個時間步設定一個獨特的模式
    sequence[:, 0:5, :] = 1.0
    # 其餘為零

    # 建立 LSTM
    lstm = LSTM(input_size, hidden_size, output_size=None)

    # 獲取所有輸出和最終狀態
    outputs, final_h, final_c = lstm.forward(sequence, return_sequences=True, return_state=True)

    print(f"\nInput shape: {sequence.shape}")
    print(f"Output shape: {outputs.shape}")

    # 分析隱藏狀態如何演變
    print(f"\nHidden state evolution:")
    print(f"  At t=5 (after pattern):  mean={np.mean(outputs[0, 5, :]):.4f}, std={np.std(outputs[0, 5, :]):.4f}")
    print(f"  At t=15 (middle):         mean={np.mean(outputs[0, 15, :]):.4f}, std={np.std(outputs[0, 15, :]):.4f}")
    print(f"  At t=29 (end):           mean={np.mean(outputs[0, 29, :]):.4f}, std={np.std(outputs[0, 29, :]):.4f}")

    print(f"\nFinal hidden state shape: {final_h.shape}")
    print(f"Final cell state shape: {final_c.shape}")

    print("\nThe LSTM maintains internal state throughout the sequence,")
    print("allowing it to remember patterns from early time steps.")


def demo_initialization_importance():
    """
    展示正確初始化的重要性。
    """
    print("\n" + "="*60)
    print("Demo 4: Importance of Initialization")
    print("="*60)

    input_size = 16
    hidden_size = 32
    seq_len = 100
    batch_size = 1

    # 使用正確初始化建立 LSTM
    lstm = LSTM(input_size, hidden_size, output_size=None)

    # 建立長序列
    sequence = np.random.randn(batch_size, seq_len, input_size) * 0.1

    # 前向傳播
    outputs = lstm.forward(sequence, return_sequences=True)

    print(f"\nProcessing long sequence (length={seq_len})")
    print(f"\nWith proper initialization:")
    print(f"  Orthogonal recurrent weights")
    print(f"  Xavier input weights")
    print(f"  Forget bias = 1.0")
    print(f"\nResults:")
    print(f"  Output mean: {np.mean(outputs):.4f}")
    print(f"  Output std: {np.std(outputs):.4f}")
    print(f"  Contains NaN: {np.isnan(outputs).any()}")
    print(f"  Contains Inf: {np.isinf(outputs).any()}")

    # 檢查梯度流（近似）
    output_start = outputs[:, 0:10, :]
    output_end = outputs[:, -10:, :]

    print(f"\nGradient flow (variance check):")
    print(f"  Early outputs variance: {np.var(output_start):.4f}")
    print(f"  Late outputs variance: {np.var(output_end):.4f}")
    print(f"  Ratio: {np.var(output_end) / (np.var(output_start) + 1e-8):.4f}")

    print("\nProper initialization helps maintain stable gradients")
    print("and prevents vanishing/exploding gradient problems.")


def demo_cell_level_usage():
    """
    展示直接使用 LSTMCell 進行自訂迴圈。
    """
    print("\n" + "="*60)
    print("Demo 5: Using LSTMCell for Custom Processing")
    print("="*60)

    input_size = 8
    hidden_size = 16
    batch_size = 3

    print(f"\nManually stepping through time with LSTMCell")
    print(f"Useful for custom training loops or variable-length sequences")

    # 建立 cell（細胞單元）
    cell = LSTMCell(input_size, hidden_size)

    # 初始化狀態
    h = np.zeros((hidden_size, batch_size))
    c = np.zeros((hidden_size, batch_size))

    print(f"\nInitial states:")
    print(f"  h shape: {h.shape}, all zeros: {np.allclose(h, 0)}")
    print(f"  c shape: {c.shape}, all zeros: {np.allclose(c, 0)}")

    # 處理多個時間步
    print(f"\nProcessing 5 time steps:")
    for t in range(5):
        # 隨機輸入
        x = np.random.randn(batch_size, input_size) * 0.1

        # 前向一步
        h, c = cell.forward(x, h, c)

        print(f"  t={t}: h_mean={np.mean(h):.4f}, c_mean={np.mean(c):.4f}")

    print(f"\nFinal states:")
    print(f"  h shape: {h.shape}")
    print(f"  c shape: {c.shape}")
    print("\nThis gives you full control over the processing loop.")


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*15 + "LSTM Baseline - Usage Demonstrations")
    print("="*70)

    np.random.seed(42)  # 設定隨機種子以確保可重現性

    # 執行所有展示
    demo_sequence_classification()
    demo_sequence_to_sequence()
    demo_state_persistence()
    demo_initialization_importance()
    demo_cell_level_usage()

    print("\n" + "="*70)
    print(" "*20 + "All Demonstrations Complete!")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. LSTM can handle various sequence tasks (classification, seq2seq)")
    print("2. It maintains internal memory across time steps")
    print("3. Proper initialization is critical for stability")
    print("4. Both LSTM and LSTMCell classes provide flexibility")
    print("5. Ready for comparison with Relational RNN")
    print("="*70 + "\n")
