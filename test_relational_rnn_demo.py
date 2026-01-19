"""
關係 RNN 單元（Relational RNN Cell）展示 - 擴展測試

此腳本提供額外的視覺化和測試，用以展示：
1. 記憶如何隨序列演化
2. LSTM 和記憶如何交互
3. 有記憶和無記憶輸出的比較

論文 18：Relational RNN - 任務 P2-T2 展示
"""

import numpy as np
from relational_rnn_cell import RelationalRNN, RelationalRNNCell
from lstm_baseline import LSTM


def analyze_memory_evolution():
    """詳細分析記憶如何隨序列演化。"""
    print("=" * 80)
    print("Analyzing Memory Evolution Over Sequence")
    print("=" * 80)

    np.random.seed(42)

    # 配置
    batch_size = 1  # 單一範例以便清晰展示
    seq_len = 15
    input_size = 32
    hidden_size = 64
    num_slots = 4
    slot_size = 64

    print(f"\nConfiguration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Memory slots: {num_slots}")
    print(f"  Slot size: {slot_size}")

    # 建立單元
    cell = RelationalRNNCell(
        input_size=input_size,
        hidden_size=hidden_size,
        num_slots=num_slots,
        slot_size=slot_size,
        num_heads=2
    )

    # 建立具有模式的序列
    # 前半部分：小數值；後半部分：大數值
    sequence = np.random.randn(batch_size, seq_len, input_size) * 0.1
    sequence[:, seq_len//2:, :] *= 5.0  # 在後半部分增加數值量級

    print(f"\n[Analysis] Processing sequence and tracking memory...")

    # 初始化狀態
    h = np.zeros((hidden_size, batch_size))
    c = np.zeros((hidden_size, batch_size))
    memory = cell.init_memory(batch_size)

    # 追蹤記憶統計資訊
    memory_norms = []
    memory_means = []
    memory_stds = []
    slot_norms = []  # 分別追蹤每個槽位

    # 處理序列
    for t in range(seq_len):
        x_t = sequence[:, t, :]
        output, h, c, memory = cell.forward(x_t, h, c, memory)

        # 計算統計資訊
        memory_norm = np.linalg.norm(memory)
        memory_mean = np.mean(memory)
        memory_std = np.std(memory)

        memory_norms.append(memory_norm)
        memory_means.append(memory_mean)
        memory_stds.append(memory_std)

        # 追蹤個別槽位範數（norm）
        slot_norm = [np.linalg.norm(memory[0, i, :]) for i in range(num_slots)]
        slot_norms.append(slot_norm)

    print(f"\n[Results] Memory Evolution Statistics:")
    print(f"\n  Overall Memory Norm (L2):")
    print(f"    Initial steps (1-5):  {np.mean(memory_norms[:5]):.4f}")
    print(f"    Middle steps (6-10):  {np.mean(memory_norms[5:10]):.4f}")
    print(f"    Final steps (11-15):  {np.mean(memory_norms[10:]):.4f}")

    print(f"\n  Memory Mean:")
    print(f"    Initial steps (1-5):  {np.mean(memory_means[:5]):.4f}")
    print(f"    Middle steps (6-10):  {np.mean(memory_means[5:10]):.4f}")
    print(f"    Final steps (11-15):  {np.mean(memory_means[10:]):.4f}")

    print(f"\n  Memory Standard Deviation:")
    print(f"    Initial steps (1-5):  {np.mean(memory_stds[:5]):.4f}")
    print(f"    Middle steps (6-10):  {np.mean(memory_stds[5:10]):.4f}")
    print(f"    Final steps (11-15):  {np.mean(memory_stds[10:]):.4f}")

    # 分析槽位特化
    print(f"\n  Individual Slot Norms at Final Step:")
    final_slot_norms = slot_norms[-1]
    for i, norm in enumerate(final_slot_norms):
        print(f"    Slot {i}: {norm:.4f}")

    # 檢查槽位是否有不同的量級（特化的指標）
    slot_variance = np.var(final_slot_norms)
    print(f"\n  Slot norm variance: {slot_variance:.4f}")
    if slot_variance > 0.01:
        print(f"    -> Slots show differentiation (potential specialization)")
    else:
        print(f"    -> Slots relatively uniform")

    print("\n" + "=" * 80 + "\n")


def compare_with_without_memory():
    """比較單獨 LSTM 與 LSTM 加關係記憶的效果。"""
    print("=" * 80)
    print("Comparing LSTM vs. LSTM + Relational Memory")
    print("=" * 80)

    np.random.seed(42)

    # 配置
    batch_size = 2
    seq_len = 10
    input_size = 32
    hidden_size = 64
    output_size = 16

    # 建立相同序列以進行公平比較
    sequence = np.random.randn(batch_size, seq_len, input_size)

    print(f"\nConfiguration:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  input_size: {input_size}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  output_size: {output_size}")

    # LSTM 基準線
    print(f"\n[1] LSTM Baseline (no relational memory)")
    lstm = LSTM(input_size, hidden_size, output_size)
    lstm_outputs = lstm.forward(sequence, return_sequences=True)

    # 關係 RNN（Relational RNN）
    print(f"[2] Relational RNN (LSTM + relational memory)")
    rel_rnn = RelationalRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_slots=4,
        slot_size=64,
        num_heads=2
    )
    rel_outputs = rel_rnn.forward(sequence, return_sequences=True)

    # 比較輸出
    print(f"\n[Comparison] Output Statistics:")
    print(f"\n  LSTM Baseline:")
    print(f"    Mean: {lstm_outputs.mean():.4f}")
    print(f"    Std:  {lstm_outputs.std():.4f}")
    print(f"    Min:  {lstm_outputs.min():.4f}")
    print(f"    Max:  {lstm_outputs.max():.4f}")

    print(f"\n  Relational RNN:")
    print(f"    Mean: {rel_outputs.mean():.4f}")
    print(f"    Std:  {rel_outputs.std():.4f}")
    print(f"    Min:  {rel_outputs.min():.4f}")
    print(f"    Max:  {rel_outputs.max():.4f}")

    # 計算差異
    diff = np.abs(lstm_outputs - rel_outputs)
    print(f"\n  Absolute Difference:")
    print(f"    Mean: {diff.mean():.4f}")
    print(f"    Max:  {diff.max():.4f}")

    # 分析
    print(f"\n[Analysis]")
    print(f"  - Both models process the same sequence")
    print(f"  - Different random initializations lead to different outputs")
    print(f"  - Relational RNN has additional memory mechanism")
    print(f"  - Memory allows for more complex representations")

    print("\n" + "=" * 80 + "\n")


def demonstrate_lstm_memory_interaction():
    """逐步展示 LSTM 和記憶如何交互。"""
    print("=" * 80)
    print("Demonstrating LSTM + Memory Interaction")
    print("=" * 80)

    np.random.seed(42)

    # 簡單配置
    batch_size = 1
    input_size = 8
    hidden_size = 16
    num_slots = 3
    slot_size = 16

    print(f"\nConfiguration:")
    print(f"  Input size: {input_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num slots: {num_slots}")
    print(f"  Slot size: {slot_size}")

    # 建立單元
    cell = RelationalRNNCell(
        input_size=input_size,
        hidden_size=hidden_size,
        num_slots=num_slots,
        slot_size=slot_size,
        num_heads=1
    )

    # 初始化狀態
    h = np.zeros((hidden_size, batch_size))
    c = np.zeros((hidden_size, batch_size))
    memory = cell.init_memory(batch_size)

    print(f"\n[Initial State]")
    print(f"  LSTM h: all zeros")
    print(f"  LSTM c: all zeros")
    print(f"  Memory: all zeros")

    # 處理幾個步驟
    num_steps = 3
    for step in range(num_steps):
        print(f"\n[Step {step + 1}]")

        # 建立輸入
        x = np.random.randn(batch_size, input_size) * 0.5
        print(f"  Input: mean={x.mean():.4f}, std={x.std():.4f}")

        # 前向傳播（Forward Pass）
        output, h_new, c_new, memory_new = cell.forward(x, h, c, memory)

        # 顯示變化
        h_change = np.linalg.norm(h_new - h)
        c_change = np.linalg.norm(c_new - c)
        mem_change = np.linalg.norm(memory_new - memory)

        print(f"  LSTM hidden change: {h_change:.4f}")
        print(f"  LSTM cell change:   {c_change:.4f}")
        print(f"  Memory change:      {mem_change:.4f}")
        print(f"  Output: mean={output.mean():.4f}, std={output.std():.4f}")

        # 更新狀態
        h = h_new
        c = c_new
        memory = memory_new

    print(f"\n[Interaction Summary]")
    print(f"  1. Input -> LSTM -> updates hidden state (h)")
    print(f"  2. Hidden state (h) -> updates memory via projection")
    print(f"  3. Memory slots interact via self-attention")
    print(f"  4. Memory readout combined with LSTM hidden")
    print(f"  5. Combined representation -> output")

    print("\n" + "=" * 80 + "\n")


def test_memory_capacity():
    """測試不同記憶槽位數量如何影響行為。"""
    print("=" * 80)
    print("Testing Memory Capacity (Different Number of Slots)")
    print("=" * 80)

    np.random.seed(42)

    # 配置
    batch_size = 2
    seq_len = 10
    input_size = 32
    hidden_size = 64
    output_size = 16

    # 所有測試使用相同序列
    sequence = np.random.randn(batch_size, seq_len, input_size)

    slot_configs = [1, 2, 4, 8]

    print(f"\nTesting different numbers of memory slots:")

    results = []
    for num_slots in slot_configs:
        print(f"\n[Testing] num_slots = {num_slots}")

        model = RelationalRNN(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_slots=num_slots,
            slot_size=64,
            num_heads=2
        )

        outputs = model.forward(sequence, return_sequences=True)

        print(f"  Output shape: {outputs.shape}")
        print(f"  Output mean:  {outputs.mean():.4f}")
        print(f"  Output std:   {outputs.std():.4f}")

        results.append({
            'num_slots': num_slots,
            'mean': outputs.mean(),
            'std': outputs.std()
        })

    print(f"\n[Summary]")
    print(f"  All configurations successfully process the sequence")
    print(f"  More slots = more memory capacity for relational reasoning")
    print(f"  Flexibility in choosing num_slots based on task complexity")

    print("\n" + "=" * 80 + "\n")


def main():
    """執行所有展示。"""
    print("\n" + "=" * 80)
    print(" " * 15 + "RELATIONAL RNN - EXTENDED DEMONSTRATIONS")
    print(" " * 20 + "Paper 18: Relational RNN - Task P2-T2")
    print("=" * 80 + "\n")

    # 執行展示
    analyze_memory_evolution()
    compare_with_without_memory()
    demonstrate_lstm_memory_interaction()
    test_memory_capacity()

    print("=" * 80)
    print(" " * 25 + "ALL DEMONSTRATIONS COMPLETE")
    print("=" * 80)
    print("\nKey Insights:")
    print("  1. Memory evolves dynamically over sequence processing")
    print("  2. Memory slots can specialize to different patterns")
    print("  3. LSTM provides sequential processing foundation")
    print("  4. Memory adds relational reasoning capability")
    print("  5. Combined system benefits from both mechanisms")
    print("\nArchitecture Benefits:")
    print("  - LSTM: Handles temporal dependencies and sequences")
    print("  - Memory: Maintains multiple related representations")
    print("  - Attention: Enables memory slots to interact")
    print("  - Gates: Control information flow and updates")
    print("  - Combination: Both sequential and relational processing")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
