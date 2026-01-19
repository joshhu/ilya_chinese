"""
關係記憶核心 (Relational Memory Core) - 快速展示
論文 18：Relational RNN（關係型循環神經網路）- 實作任務 P2-T1

本腳本提供關係記憶核心功能的簡潔展示，
展示它如何維護和更新透過自注意力機制 (self-attention) 互動的多個記憶槽。
"""

import numpy as np
from relational_memory import RelationalMemory, init_memory

def main():
    """展示關係記憶核心的功能。"""

    print("\n" + "=" * 80)
    print("RELATIONAL MEMORY CORE - QUICK DEMONSTRATION")
    print("=" * 80 + "\n")

    np.random.seed(42)

    # 配置參數（如任務中所指定）
    batch_size = 2
    num_slots = 4
    slot_size = 64
    num_heads = 2

    print("Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of memory slots: {num_slots}")
    print(f"  Slot dimension: {slot_size}")
    print(f"  Number of attention heads: {num_heads}")

    # 建立關係記憶模組
    print("\n1. Creating Relational Memory Core...")
    rm = RelationalMemory(
        num_slots=num_slots,
        slot_size=slot_size,
        num_heads=num_heads,
        use_gate=True,
        use_input_attention=True
    )
    print("   Created successfully!")

    # 初始化記憶
    print("\n2. Initializing memory state...")
    memory = rm.reset_memory(batch_size)
    print(f"   Memory shape: {memory.shape}")
    print(f"   (batch_size={batch_size}, num_slots={num_slots}, slot_size={slot_size})")

    # 無輸入的前向傳播
    print("\n3. Running self-attention across memory slots (no input)...")
    updated_memory, attention_weights = rm.forward(memory)
    print(f"   Updated memory shape: {updated_memory.shape}")
    print(f"   Attention weights shape: {attention_weights.shape}")
    print(f"   Attention weights sum to 1.0: {np.allclose(np.sum(attention_weights, axis=-1), 1.0)}")

    # 顯示第一個範例、第一個頭的注意力模式
    print("\n4. Attention pattern (batch 0, head 0):")
    print("   How much each slot attends to others:")
    attn = attention_weights[0, 0]
    for i in range(num_slots):
        print(f"   Slot {i}: [{', '.join([f'{attn[i,j]:.3f}' for j in range(num_slots)])}]")

    # 有輸入的前向傳播
    print("\n5. Incorporating external input into memory...")
    input_vec = np.random.randn(batch_size, 32)
    updated_memory_with_input, _ = rm.forward(memory, input_vec)
    print(f"   Input shape: {input_vec.shape}")
    print(f"   Updated memory shape: {updated_memory_with_input.shape}")

    # 當提供輸入時記憶會改變
    difference = np.linalg.norm(updated_memory - updated_memory_with_input)
    print(f"   Difference from no-input case: {difference:.4f}")
    print(f"   Input successfully incorporated: {difference > 0.1}")

    # 模擬一個序列
    print("\n6. Processing a sequence of 5 inputs...")
    memory_seq = rm.reset_memory(batch_size)

    for t in range(5):
        input_t = np.random.randn(batch_size, 32)
        memory_seq, attn_t = rm.forward(memory_seq, input_t)
        mean_attn = np.mean(attn_t[0, 0])
        print(f"   Step {t+1}: Memory updated, mean attention = {mean_attn:.4f}")

    print(f"\n   Final memory shape: {memory_seq.shape}")

    # 摘要
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nKey Capabilities Demonstrated:")
    print("  1. Multiple memory slots maintain distinct representations")
    print("  2. Self-attention allows slots to interact and share information")
    print("  3. External inputs can be incorporated into memory")
    print("  4. Memory evolves over time through sequential processing")
    print("  5. Multi-head attention enables different types of relationships")
    print("\nRelational Reasoning Advantages:")
    print("  - Can represent multiple entities/concepts simultaneously")
    print("  - Slots can reason about relationships via attention")
    print("  - More structured than single-vector hidden states")
    print("  - Better suited for tasks requiring relational reasoning")
    print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    main()
