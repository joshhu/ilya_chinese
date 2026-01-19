"""
整合測試：關係記憶核心（Relational Memory Core）
論文 18：Relational RNN - 任務 P2-T1

此測試展示關係記憶核心如何與來自 P1-T2 的
多頭注意力機制（Multi-head Attention）整合。
"""

import numpy as np
from relational_memory import RelationalMemory, layer_norm, gated_update, init_memory
from attention_mechanism import multi_head_attention, init_attention_params


def test_integration():
    """測試注意力機制與關係記憶之間的整合。"""

    print("\n" + "=" * 80)
    print("INTEGRATION TEST: Attention Mechanism + Relational Memory")
    print("=" * 80 + "\n")

    np.random.seed(42)

    # 測試參數
    batch_size = 2
    num_slots = 4
    slot_size = 64
    num_heads = 2

    print("Testing Components Integration:")
    print(f"  batch_size: {batch_size}")
    print(f"  num_slots: {num_slots}")
    print(f"  slot_size: {slot_size}")
    print(f"  num_heads: {num_heads}")

    # 測試 1：獨立的多頭注意力（Multi-head Attention）
    print("\n[1] Testing multi-head attention (from P1-T2)...")
    memory = np.random.randn(batch_size, num_slots, slot_size)
    attn_params = init_attention_params(slot_size, num_heads)

    attn_out, attn_weights = multi_head_attention(
        Q=memory, K=memory, V=memory,
        num_heads=num_heads,
        **attn_params
    )

    print(f"    Input shape: {memory.shape}")
    print(f"    Output shape: {attn_out.shape}")
    print(f"    Attention weights shape: {attn_weights.shape}")
    assert attn_out.shape == memory.shape, "Shape mismatch"
    print("    ✅ Multi-head attention working")

    # 測試 2：層正規化（Layer Normalization）
    print("\n[2] Testing layer normalization...")
    gamma = np.ones(slot_size)
    beta = np.zeros(slot_size)
    normalized = layer_norm(attn_out, gamma, beta)

    mean = np.mean(normalized, axis=-1)
    std = np.std(normalized, axis=-1)
    print(f"    Mean range: [{np.min(mean):.6f}, {np.max(mean):.6f}]")
    print(f"    Std range: [{np.min(std):.4f}, {np.max(std):.4f}]")
    assert np.allclose(mean, 0.0, atol=1e-5), "Mean not close to 0"
    assert np.allclose(std, 1.0, atol=1e-5), "Std not close to 1"
    print("    ✅ Layer normalization working")

    # 測試 3：門控更新（Gated Update）
    print("\n[3] Testing gated update...")
    old_memory = np.random.randn(batch_size, num_slots, slot_size)
    new_memory = normalized

    gate_weights = np.random.randn(slot_size * 2, slot_size) * 0.1
    gated_memory = gated_update(old_memory, new_memory, gate_weights)

    print(f"    Old memory shape: {old_memory.shape}")
    print(f"    New memory shape: {new_memory.shape}")
    print(f"    Gated memory shape: {gated_memory.shape}")
    assert gated_memory.shape == old_memory.shape, "Shape mismatch"
    print("    ✅ Gated update working")

    # 測試 4：完整的關係記憶（結合所有組件）
    print("\n[4] Testing full Relational Memory Core...")
    rm = RelationalMemory(
        num_slots=num_slots,
        slot_size=slot_size,
        num_heads=num_heads,
        use_gate=True,
        use_input_attention=True
    )

    initial_memory = rm.reset_memory(batch_size)
    final_memory, final_attn = rm.forward(initial_memory)

    print(f"    Initial memory shape: {initial_memory.shape}")
    print(f"    Final memory shape: {final_memory.shape}")
    print(f"    Attention shape: {final_attn.shape}")

    # 驗證整合結果
    assert final_memory.shape == initial_memory.shape, "Shape mismatch"
    assert final_attn.shape == (batch_size, num_heads, num_slots, num_slots), "Attention shape mismatch"
    assert np.allclose(np.sum(final_attn, axis=-1), 1.0), "Attention doesn't sum to 1"
    assert not np.any(np.isnan(final_memory)), "NaN in output"
    assert not np.any(np.isinf(final_memory)), "Inf in output"

    print("    ✅ Full Relational Memory working")

    # 測試 5：多步驟處理
    print("\n[5] Testing multi-step sequential processing...")
    memory_t = rm.reset_memory(batch_size)

    for t in range(3):
        input_t = np.random.randn(batch_size, 32)
        memory_t, attn_t = rm.forward(memory_t, input_t)

        # 在每一步驗證
        assert memory_t.shape == (batch_size, num_slots, slot_size), f"Step {t} shape error"
        assert not np.any(np.isnan(memory_t)), f"Step {t} has NaN"

        mean_attn = np.mean(attn_t)
        print(f"    Step {t+1}: Memory updated (mean attn: {mean_attn:.4f})")

    print("    ✅ Multi-step processing working")

    # 最終總結
    print("\n" + "=" * 80)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 80)
    print("\n✅ All components successfully integrated:")
    print("   1. Multi-head attention (P1-T2) - Working")
    print("   2. Layer normalization - Working")
    print("   3. Gated update mechanism - Working")
    print("   4. Relational Memory Core - Working")
    print("   5. Sequential processing - Working")
    print("\n✅ Ready for integration into Relational RNN Cell (P2-T2)")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    test_integration()
