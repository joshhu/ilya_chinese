"""
關係記憶核心模組 (Relational Memory Core Module)
論文 18：Relational RNN - 實作任務 P2-T1

本模組實作 Relational RNN 論文的核心創新：
一組透過多頭自注意力 (multi-head self-attention) 互動的記憶槽 (memory slots)，
用於在儲存的資訊之間進行關係推理 (relational reasoning)。

關係記憶核心維護一組可以互相注意 (attend) 的記憶向量（槽），
允許資訊共享和關係推理。這是與傳統 RNN 使用單一隱藏狀態向量的關鍵差異。

這是 Sutskever 30 篇論文專案的教育性實作。
"""

import numpy as np
from attention_mechanism import multi_head_attention, init_attention_params


def layer_norm(x, gamma=None, beta=None, eps=1e-6):
    """
    層正規化 (Layer Normalization)。

    對每個樣本獨立地在特徵維度上進行正規化。
    這有助於穩定訓練，並使每一層能夠將其輸入調整為零均值和單位方差。

    參數:
        x: 輸入張量，形狀 (..., d_model)
        gamma: 縮放參數 (scale parameter)，形狀 (d_model,)
        beta: 偏移參數 (shift parameter)，形狀 (d_model,)
        eps: 用於數值穩定性的小常數

    回傳:
        正規化後的輸出，形狀 (..., d_model)

    數學公式:
        mean = mean(x, axis=-1, keepdims=True)
        var = var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / sqrt(var + eps)
        output = gamma * x_norm + beta
    """
    # 計算最後一個維度（特徵維度）的均值和方差
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    # 正規化
    x_norm = (x - mean) / np.sqrt(var + eps)

    # 如果提供了參數，則進行縮放和偏移
    if gamma is not None and beta is not None:
        # 確保 gamma 和 beta 具有正確的形狀以進行廣播 (broadcasting)
        assert gamma.shape[-1] == x.shape[-1], \
            f"gamma shape {gamma.shape} incompatible with x shape {x.shape}"
        assert beta.shape[-1] == x.shape[-1], \
            f"beta shape {beta.shape} incompatible with x shape {x.shape}"

        x_norm = gamma * x_norm + beta

    return x_norm


def gated_update(old_value, new_value, gate_weights=None):
    """
    記憶的門控更新機制 (Gated Update Mechanism)。

    使用學習到的門來在舊記憶值和新記憶值之間進行內插。
    這允許模型學習何時保留舊資訊，何時納入新資訊。

    參數:
        old_value: 先前的記憶，形狀 (..., d_model)
        new_value: 候選的新記憶，形狀 (..., d_model)
        gate_weights: 可選的門參數，形狀 (d_model * 2, d_model)
                     如果為 None，則直接回傳 new_value

    回傳:
        更新後的記憶，形狀 (..., d_model)

    數學公式:
        gate_input = concat([old_value, new_value], axis=-1)
        gate = sigmoid(gate_input @ gate_weights)
        output = gate * new_value + (1 - gate) * old_value
    """
    if gate_weights is None:
        # 沒有門控，直接回傳新值
        return new_value

    assert old_value.shape == new_value.shape, \
        f"Shape mismatch: old_value {old_value.shape} vs new_value {new_value.shape}"

    d_model = old_value.shape[-1]

    # 串接舊值和新值
    gate_input = np.concatenate([old_value, new_value], axis=-1)
    # gate_input: (..., d_model * 2)

    # 使用 sigmoid 計算門值
    # gate_weights: (d_model * 2, d_model)
    gate_logits = np.matmul(gate_input, gate_weights)  # (..., d_model)
    gate = 1.0 / (1.0 + np.exp(-gate_logits))  # sigmoid

    # 門控組合
    output = gate * new_value + (1.0 - gate) * old_value

    return output


def init_memory(batch_size, num_slots, slot_size, init_std=0.1):
    """
    初始化記憶槽。

    為關係記憶核心建立初始記憶狀態。
    記憶以小的隨機值初始化以打破對稱性 (break symmetry)。

    參數:
        batch_size: 批次中的樣本數量
        num_slots: 記憶槽的數量
        slot_size: 每個記憶槽的維度
        init_std: 初始化的標準差

    回傳:
        memory: 形狀 (batch_size, num_slots, slot_size)
    """
    memory = np.random.randn(batch_size, num_slots, slot_size) * init_std
    return memory


class RelationalMemory:
    """
    使用多頭自注意力的關係記憶核心。

    這是 Relational RNN 論文的核心創新。不同於維護單一隱藏狀態向量（如傳統 RNN），
    關係記憶維護多個可以透過自注意力互動的記憶槽。這使模型能夠：

    1. 同時儲存多個資訊片段
    2. 透過允許槽之間互相注意來實現關係推理
    3. 根據相關性在槽之間動態路由資訊

    架構:
        1. 跨記憶槽的多頭自注意力 (Multi-head self-attention)
        2. 注意力周圍的殘差連接 (Residual connection)
        3. 用於穩定性的層正規化 (Layer normalization)
        4. 可選的門控更新以控制資訊流
        5. 可選的透過注意力納入輸入

    記憶作為一個關係推理模組，可以維護和操作結構化表示。
    """

    def __init__(self, num_slots=8, slot_size=64, num_heads=4,
                 use_gate=True, use_input_attention=True):
        """
        初始化關係記憶核心。

        參數:
            num_slots: 記憶槽的數量（預設：8）
            slot_size: 每個記憶槽的維度（預設：64）
            num_heads: 注意力頭的數量（預設：4）
            use_gate: 是否使用門控更新（預設：True）
            use_input_attention: 是否透過注意力納入輸入（預設：True）
        """
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.num_heads = num_heads
        self.use_gate = use_gate
        self.use_input_attention = use_input_attention

        # 檢查 slot_size 是否可被 num_heads 整除
        assert slot_size % num_heads == 0, \
            f"slot_size ({slot_size}) must be divisible by num_heads ({num_heads})"

        # 初始化自注意力的參數
        self.attn_params = init_attention_params(slot_size, num_heads)

        # 初始化層正規化參數
        self.ln1_gamma = np.ones(slot_size)
        self.ln1_beta = np.zeros(slot_size)
        self.ln2_gamma = np.ones(slot_size)
        self.ln2_beta = np.zeros(slot_size)

        # 如果使用門控，則初始化門參數
        if use_gate:
            # 門接收串接的 [舊值, 新值] 並輸出門值
            std = np.sqrt(2.0 / (slot_size * 2 + slot_size))
            self.gate_weights = np.random.randn(slot_size * 2, slot_size) * std
        else:
            self.gate_weights = None

        # 如果使用輸入納入，則初始化相關參數
        if use_input_attention:
            self.ln_input_gamma = np.ones(slot_size)
            self.ln_input_beta = np.zeros(slot_size)

    def forward(self, memory, input_vec=None):
        """
        通過關係記憶核心的前向傳播。

        參數:
            memory: 目前的記憶狀態，形狀 (batch, num_slots, slot_size)
            input_vec: 可選的要納入的輸入，形狀 (batch, input_size)
                      如果提供且 use_input_attention=True，則會對輸入進行注意

        回傳:
            updated_memory: 新的記憶狀態，形狀 (batch, num_slots, slot_size)
            attention_weights: 自注意力權重，形狀 (batch, num_heads, num_slots, num_slots)

        演算法:
            1. 跨記憶槽的自注意力
            2. 加入殘差連接
            3. 層正規化
            4. 可選：對輸入進行注意
            5. 可選：門控更新
        """
        # 驗證輸入形狀
        assert memory.ndim == 3, \
            f"memory must be 3D (batch, num_slots, slot_size), got {memory.shape}"
        batch_size, num_slots, slot_size = memory.shape
        assert num_slots == self.num_slots, \
            f"Expected {self.num_slots} slots, got {num_slots}"
        assert slot_size == self.slot_size, \
            f"Expected slot_size {self.slot_size}, got {slot_size}"

        # 儲存原始記憶以用於殘差和門控
        memory_orig = memory

        # 步驟 1：跨記憶槽的多頭自注意力
        # 記憶對自身進行注意：Q=K=V=memory
        attn_output, attn_weights = multi_head_attention(
            Q=memory,
            K=memory,
            V=memory,
            num_heads=self.num_heads,
            W_q=self.attn_params['W_q'],
            W_k=self.attn_params['W_k'],
            W_v=self.attn_params['W_v'],
            W_o=self.attn_params['W_o'],
            mask=None
        )
        # attn_output: (batch, num_slots, slot_size)
        # attn_weights: (batch, num_heads, num_slots, num_slots)

        # 步驟 2：殘差連接
        memory = memory_orig + attn_output

        # 步驟 3：層正規化
        memory = layer_norm(
            memory,
            gamma=self.ln1_gamma,
            beta=self.ln1_beta
        )

        # 步驟 4：可選的輸入注意
        # 如果提供了輸入且我們使用輸入注意，
        # 透過廣播和門控將輸入納入記憶
        if input_vec is not None and self.use_input_attention:
            # Input_vec: (batch, input_size)
            # 需要使其與記憶相容

            # 如果需要，將輸入投影到 slot_size
            if input_vec.shape[-1] != self.slot_size:
                # 簡單的線性投影
                input_size = input_vec.shape[-1]
                if not hasattr(self, 'input_projection'):
                    # 初始化投影矩陣
                    std = np.sqrt(2.0 / (input_size + self.slot_size))
                    self.input_projection = np.random.randn(input_size, self.slot_size) * std

                # 投影輸入
                input_vec_proj = np.matmul(input_vec, self.input_projection)
            else:
                input_vec_proj = input_vec

            # 將輸入廣播到所有記憶槽：(batch, num_slots, slot_size)
            # 每個槽都能看到相同的輸入
            input_broadcast = np.tile(input_vec_proj[:, np.newaxis, :], (1, self.num_slots, 1))

            # 透過簡單的門控機制組合記憶和輸入
            # 這是相對於完整交叉注意力的簡化方法
            # 完整交叉注意力需要處理不同的序列長度

            # 串接記憶和輸入，然後投影
            memory_input_concat = np.concatenate([memory, input_broadcast], axis=-1)
            # 形狀：(batch, num_slots, slot_size * 2)

            # 投影回 slot_size
            if not hasattr(self, 'input_combine_weights'):
                std = np.sqrt(2.0 / (self.slot_size * 2 + self.slot_size))
                self.input_combine_weights = np.random.randn(self.slot_size * 2, self.slot_size) * std

            input_contribution = np.matmul(memory_input_concat, self.input_combine_weights)
            # 形狀：(batch, num_slots, slot_size)

            # 加入殘差並正規化
            memory_before_input = memory
            memory = memory_before_input + input_contribution
            memory = layer_norm(
                memory,
                gamma=self.ln_input_gamma,
                beta=self.ln_input_beta
            )

        # 步驟 5：可選的門控更新
        if self.use_gate and self.gate_weights is not None:
            memory = gated_update(
                old_value=memory_orig,
                new_value=memory,
                gate_weights=self.gate_weights
            )

        return memory, attn_weights

    def reset_memory(self, batch_size, init_std=0.1):
        """
        建立新的記憶狀態。

        參數:
            batch_size: 批次中的樣本數量
            init_std: 初始化的標準差

        回傳:
            memory: 形狀 (batch_size, num_slots, slot_size)
        """
        return init_memory(batch_size, self.num_slots, self.slot_size, init_std)


# ============================================================================
# 測試函式
# ============================================================================

def test_layer_norm():
    """測試層正規化。"""
    print("=" * 80)
    print("Testing Layer Normalization")
    print("=" * 80)

    np.random.seed(42)

    # Test basic layer norm
    batch_size = 2
    seq_len = 5
    d_model = 8

    x = np.random.randn(batch_size, seq_len, d_model) * 2.0 + 3.0

    print(f"\nInput shape: {x.shape}")
    print(f"Input mean (approx): {np.mean(x):.4f}")
    print(f"Input std (approx): {np.std(x):.4f}")

    # Test without gamma/beta
    print("\n[Test 1] Layer norm without scale/shift")
    x_norm = layer_norm(x)

    # Check that each example has been normalized
    for b in range(batch_size):
        for s in range(seq_len):
            vec_mean = np.mean(x_norm[b, s])
            vec_std = np.std(x_norm[b, s])
            assert np.abs(vec_mean) < 1e-6, f"Mean not close to 0: {vec_mean}"
            assert np.abs(vec_std - 1.0) < 1e-6, f"Std not close to 1: {vec_std}"

    print(f"  Output mean (approx): {np.mean(x_norm):.6f}")
    print(f"  Output std (approx): {np.std(x_norm):.4f}")
    print(f"  Each vector normalized: PASS")

    # Test with gamma/beta
    print("\n[Test 2] Layer norm with scale/shift")
    gamma = np.ones(d_model) * 2.0
    beta = np.ones(d_model) * 0.5

    x_norm_scaled = layer_norm(x, gamma=gamma, beta=beta)

    # Check that scaling is applied
    # Each normalized vector should be scaled and shifted
    for b in range(batch_size):
        for s in range(seq_len):
            vec_mean = np.mean(x_norm_scaled[b, s])
            # Mean should be close to mean(beta) = 0.5
            assert np.abs(vec_mean - 0.5) < 0.1, f"Mean not close to 0.5: {vec_mean}"

    print(f"  Output mean (approx): {np.mean(x_norm_scaled):.4f}")
    print(f"  Scaling applied: PASS")

    print("\n" + "=" * 80)
    print("Layer Normalization: ALL TESTS PASSED")
    print("=" * 80 + "\n")


def test_gated_update():
    """測試門控更新機制。"""
    print("=" * 80)
    print("Testing Gated Update")
    print("=" * 80)

    np.random.seed(42)

    batch_size = 2
    num_slots = 4
    d_model = 8

    old_value = np.random.randn(batch_size, num_slots, d_model)
    new_value = np.random.randn(batch_size, num_slots, d_model)

    print(f"\nOld value shape: {old_value.shape}")
    print(f"New value shape: {new_value.shape}")

    # Test without gate (should return new_value)
    print("\n[Test 1] Update without gate")
    result = gated_update(old_value, new_value, gate_weights=None)
    assert np.allclose(result, new_value), "Without gate should return new_value"
    print(f"  Returns new_value: PASS")

    # Test with gate
    print("\n[Test 2] Update with gate")
    std = np.sqrt(2.0 / (d_model * 2 + d_model))
    gate_weights = np.random.randn(d_model * 2, d_model) * std

    result_gated = gated_update(old_value, new_value, gate_weights=gate_weights)

    print(f"  Output shape: {result_gated.shape}")
    assert result_gated.shape == old_value.shape, "Output shape mismatch"
    print(f"  Output shape correct: PASS")

    # Check that output is a combination of old and new
    # It should be different from both old_value and new_value (in general)
    # unless gate is all 0s or all 1s
    assert not np.allclose(result_gated, old_value), "Output should differ from old_value"
    assert not np.allclose(result_gated, new_value), "Output should differ from new_value"
    print(f"  Output is combination of old and new: PASS")

    # Check numerical stability
    assert not np.any(np.isnan(result_gated)), "Output contains NaN"
    assert not np.any(np.isinf(result_gated)), "Output contains Inf"
    print(f"  No NaN/Inf in output: PASS")

    print("\n" + "=" * 80)
    print("Gated Update: ALL TESTS PASSED")
    print("=" * 80 + "\n")


def test_init_memory():
    """測試記憶初始化。"""
    print("=" * 80)
    print("Testing Memory Initialization")
    print("=" * 80)

    np.random.seed(42)

    batch_size = 2
    num_slots = 4
    slot_size = 64

    print(f"\nParameters:")
    print(f"  batch_size: {batch_size}")
    print(f"  num_slots: {num_slots}")
    print(f"  slot_size: {slot_size}")

    memory = init_memory(batch_size, num_slots, slot_size)

    print(f"\nMemory shape: {memory.shape}")
    assert memory.shape == (batch_size, num_slots, slot_size), \
        f"Shape mismatch: expected {(batch_size, num_slots, slot_size)}, got {memory.shape}"
    print(f"  Shape correct: PASS")

    # Check initialization statistics
    print(f"\nMemory statistics:")
    print(f"  Mean: {np.mean(memory):.6f}")
    print(f"  Std: {np.std(memory):.4f}")

    # Should be roughly zero mean, small std
    assert np.abs(np.mean(memory)) < 0.1, "Mean too far from 0"
    assert 0.05 < np.std(memory) < 0.2, "Std outside expected range"
    print(f"  Statistics reasonable: PASS")

    print("\n" + "=" * 80)
    print("Memory Initialization: ALL TESTS PASSED")
    print("=" * 80 + "\n")


def test_relational_memory():
    """測試 RelationalMemory 類別。"""
    print("=" * 80)
    print("Testing Relational Memory Core")
    print("=" * 80)

    np.random.seed(42)

    # Test parameters (as specified in task)
    batch_size = 2
    num_slots = 4
    slot_size = 64
    num_heads = 2

    print(f"\nParameters:")
    print(f"  batch_size: {batch_size}")
    print(f"  num_slots: {num_slots}")
    print(f"  slot_size: {slot_size}")
    print(f"  num_heads: {num_heads}")

    # Initialize relational memory
    print("\n[Test 1] Initialization")
    rm = RelationalMemory(
        num_slots=num_slots,
        slot_size=slot_size,
        num_heads=num_heads,
        use_gate=True,
        use_input_attention=True
    )

    print(f"  RelationalMemory created")
    print(f"  num_slots: {rm.num_slots}")
    print(f"  slot_size: {rm.slot_size}")
    print(f"  num_heads: {rm.num_heads}")
    print(f"  use_gate: {rm.use_gate}")
    print(f"  use_input_attention: {rm.use_input_attention}")

    # Verify parameters initialized
    assert rm.attn_params is not None, "Attention params not initialized"
    assert rm.gate_weights is not None, "Gate weights not initialized"
    print(f"  All parameters initialized: PASS")

    # Test memory reset
    print("\n[Test 2] Memory reset")
    memory = rm.reset_memory(batch_size)
    print(f"  Memory shape: {memory.shape}")
    assert memory.shape == (batch_size, num_slots, slot_size), \
        f"Memory shape mismatch: expected {(batch_size, num_slots, slot_size)}, got {memory.shape}"
    print(f"  Memory shape correct: PASS")

    # Test forward pass without input
    print("\n[Test 3] Forward pass without input")
    updated_memory, attn_weights = rm.forward(memory)

    print(f"  Updated memory shape: {updated_memory.shape}")
    print(f"  Attention weights shape: {attn_weights.shape}")

    assert updated_memory.shape == (batch_size, num_slots, slot_size), \
        f"Updated memory shape mismatch"
    assert attn_weights.shape == (batch_size, num_heads, num_slots, num_slots), \
        f"Attention weights shape mismatch"
    print(f"  Output shapes correct: PASS")

    # Check attention weights sum to 1
    attn_sums = np.sum(attn_weights, axis=-1)
    assert np.allclose(attn_sums, 1.0), "Attention weights don't sum to 1"
    print(f"  Attention weights sum to 1: PASS")

    # Check for NaN/Inf
    assert not np.any(np.isnan(updated_memory)), "Memory contains NaN"
    assert not np.any(np.isinf(updated_memory)), "Memory contains Inf"
    assert not np.any(np.isnan(attn_weights)), "Attention weights contain NaN"
    assert not np.any(np.isinf(attn_weights)), "Attention weights contain Inf"
    print(f"  No NaN/Inf in outputs: PASS")

    # Test forward pass with input
    print("\n[Test 4] Forward pass with input")
    input_size = 32
    input_vec = np.random.randn(batch_size, input_size)

    updated_memory_with_input, attn_weights_with_input = rm.forward(memory, input_vec)

    print(f"  Input shape: {input_vec.shape}")
    print(f"  Updated memory shape: {updated_memory_with_input.shape}")
    print(f"  Attention weights shape: {attn_weights_with_input.shape}")

    assert updated_memory_with_input.shape == (batch_size, num_slots, slot_size), \
        f"Updated memory shape mismatch"
    print(f"  Output shape correct: PASS")

    # Memory should be different when input is provided
    assert not np.allclose(updated_memory, updated_memory_with_input), \
        "Input should affect memory"
    print(f"  Input affects memory: PASS")

    # Test multiple forward passes (simulating sequence)
    print("\n[Test 5] Multiple forward passes")
    memory_seq = rm.reset_memory(batch_size)
    memories = [memory_seq]

    for t in range(5):
        input_t = np.random.randn(batch_size, input_size)
        memory_seq, _ = rm.forward(memory_seq, input_t)
        memories.append(memory_seq)

    print(f"  Processed {len(memories)-1} timesteps")

    # Check that memory evolves over time
    for t in range(1, len(memories)):
        assert not np.allclose(memories[0], memories[t]), \
            f"Memory should change at timestep {t}"
    print(f"  Memory evolves over time: PASS")

    # Test without gating
    print("\n[Test 6] Without gating")
    rm_no_gate = RelationalMemory(
        num_slots=num_slots,
        slot_size=slot_size,
        num_heads=num_heads,
        use_gate=False,
        use_input_attention=False
    )

    memory_no_gate = rm_no_gate.reset_memory(batch_size)
    updated_no_gate, _ = rm_no_gate.forward(memory_no_gate)

    print(f"  Forward pass without gate: PASS")
    assert updated_no_gate.shape == (batch_size, num_slots, slot_size), \
        "Shape mismatch without gate"
    print(f"  Output shape correct: PASS")

    # Test different configurations
    print("\n[Test 7] Different configurations")
    configs = [
        {'num_slots': 8, 'slot_size': 64, 'num_heads': 4},
        {'num_slots': 4, 'slot_size': 128, 'num_heads': 8},
        {'num_slots': 16, 'slot_size': 32, 'num_heads': 2},
    ]

    for i, config in enumerate(configs):
        test_rm = RelationalMemory(**config)
        test_memory = test_rm.reset_memory(batch_size)
        test_updated, test_attn = test_rm.forward(test_memory)

        assert test_updated.shape == (batch_size, config['num_slots'], config['slot_size']), \
            f"Config {i} failed"
        assert test_attn.shape == (batch_size, config['num_heads'], config['num_slots'], config['num_slots']), \
            f"Config {i} attention shape failed"
        print(f"  Config {i+1} (slots={config['num_slots']}, size={config['slot_size']}, heads={config['num_heads']}): PASS")

    print("\n" + "=" * 80)
    print("Relational Memory Core: ALL TESTS PASSED")
    print("=" * 80 + "\n")


def demonstrate_relational_reasoning():
    """展示關係記憶如何實現推理。"""
    print("=" * 80)
    print("Demonstrating Relational Reasoning Capabilities")
    print("=" * 80)

    np.random.seed(42)

    batch_size = 1
    num_slots = 4
    slot_size = 64
    num_heads = 2

    print("\nScenario: Memory slots representing entities that need to interact")
    print(f"  num_slots: {num_slots} (e.g., 4 objects being tracked)")
    print(f"  slot_size: {slot_size} (feature dimension)")
    print(f"  num_heads: {num_heads} (different types of relationships)")

    rm = RelationalMemory(
        num_slots=num_slots,
        slot_size=slot_size,
        num_heads=num_heads,
        use_gate=True,
        use_input_attention=True
    )

    # Initialize memory with distinct patterns for each slot
    memory = rm.reset_memory(batch_size)

    # Simulate making slots somewhat different
    for slot in range(num_slots):
        memory[0, slot, :] += np.random.randn(slot_size) * 0.5

    print("\n[Observation 1] Initial memory state")
    print(f"  Memory shape: {memory.shape}")
    print(f"  Memory initialized with distinct patterns per slot")

    # Forward pass to see attention patterns
    updated_memory, attn_weights = rm.forward(memory)

    print("\n[Observation 2] Attention patterns after one step")
    print(f"  Attention weights shape: {attn_weights.shape}")
    print(f"\n  Attention matrix (head 0):")
    print(f"  Rows = query slots, Cols = key slots")
    print(f"  Values show how much each slot attends to others\n")

    # Display attention for first head
    attn_head0 = attn_weights[0, 0]  # (num_slots, num_slots)
    for i in range(num_slots):
        row_str = "  Slot " + str(i) + " -> ["
        for j in range(num_slots):
            row_str += f"{attn_head0[i, j]:.3f}"
            if j < num_slots - 1:
                row_str += ", "
        row_str += "]"
        print(row_str)

    # Check which slots have high mutual attention
    print("\n[Observation 3] Relational interactions")
    threshold = 0.3
    for i in range(num_slots):
        for j in range(i + 1, num_slots):
            mutual_attn = attn_head0[i, j] + attn_head0[j, i]
            if mutual_attn > threshold:
                print(f"  Strong interaction between Slot {i} and Slot {j} (score: {mutual_attn:.3f})")

    # Simulate sequence of inputs
    print("\n[Observation 4] Evolution with inputs")
    memory_t = memory
    input_sequence = [np.random.randn(batch_size, 32) for _ in range(3)]

    for t, input_t in enumerate(input_sequence):
        memory_t, attn_t = rm.forward(memory_t, input_t)
        mean_attn = np.mean(attn_t[0, 0])
        print(f"  Step {t+1}: Mean attention = {mean_attn:.4f}")

    print("\n[Key Insights]")
    print("  1. Memory slots can attend to each other, enabling relational reasoning")
    print("  2. Different attention heads can capture different types of relations")
    print("  3. Memory evolves over time while maintaining multiple representations")
    print("  4. This enables reasoning about relationships between stored entities")
    print("  5. Unlike single-vector RNN states, can maintain distinct concepts simultaneously")

    print("\n" + "=" * 80 + "\n")


def main():
    """執行所有測試和展示。"""
    print("\n" + "=" * 80)
    print(" " * 20 + "RELATIONAL MEMORY CORE TEST SUITE")
    print(" " * 25 + "Paper 18: Relational RNN - Task P2-T1")
    print("=" * 80 + "\n")

    # Run tests
    test_layer_norm()
    test_gated_update()
    test_init_memory()
    test_relational_memory()
    demonstrate_relational_reasoning()

    print("=" * 80)
    print(" " * 25 + "ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\nSummary of Implementation:")
    print("  - layer_norm(): Normalizes activations for training stability")
    print("  - gated_update(): Controls information flow with learned gates")
    print("  - init_memory(): Initializes memory slots with small random values")
    print("  - RelationalMemory class: Core module with multi-head self-attention")
    print("\nKey Features:")
    print("  - Multi-head self-attention across memory slots")
    print("  - Residual connections for gradient flow")
    print("  - Layer normalization for stability")
    print("  - Optional gated updates for selective memory retention")
    print("  - Optional input attention for incorporating new information")
    print("\nRelational Reasoning Aspect:")
    print("  - Memory slots can attend to each other via self-attention")
    print("  - Enables modeling relationships between stored entities")
    print("  - Different attention heads capture different relational patterns")
    print("  - Maintains multiple distinct representations simultaneously")
    print("  - Superior to single-vector hidden states for complex reasoning")
    print("\nAll tests passed with batch=2, slots=4, slot_size=64, heads=2")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
