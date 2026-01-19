"""
關係 RNN 單元 (Relational RNN Cell) - 結合 LSTM 與關係記憶

本模組實作一個結合以下元件的 Relational RNN：
1. 用於序列處理的標準 LSTM
2. 使用多頭自注意力進行關係推理的關係記憶

論文 18：Relational RNN - 實作任務 P2-T2

架構:
- LSTM 處理序列輸入並維護隱藏狀態/單元狀態 (hidden/cell states)
- 關係記憶維護一組透過注意力互動的記憶槽
- LSTM 隱藏狀態被投影並用於更新關係記憶
- 記憶讀出 (memory readout) 與 LSTM 輸出結合以產生最終預測

僅使用 NumPy 的教育性實作。
"""

import numpy as np
from lstm_baseline import LSTMCell, xavier_initializer, orthogonal_initializer
from attention_mechanism import multi_head_attention, init_attention_params


class RelationalMemory:
    """
    使用多頭自注意力的關係記憶模組。

    記憶由一組透過注意力機制互動的槽組成。
    這允許模型同時維護和推理多個相關的資訊片段。

    架構:
        1. 記憶槽透過多頭自注意力互動
        2. 門機制 (gate mechanism) 控制記憶更新
        3. 殘差連接保留資訊
    """

    def __init__(self, num_slots=4, slot_size=64, num_heads=2, input_size=None):
        """
        初始化關係記憶。

        參數:
            num_slots: 記憶槽的數量
            slot_size: 每個記憶槽的維度
            num_heads: 注意力頭的數量
            input_size: 輸入到記憶的維度（如果為 None，則等於 slot_size）
        """
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.num_heads = num_heads
        self.input_size = input_size if input_size is not None else slot_size

        assert slot_size % num_heads == 0, \
            f"slot_size ({slot_size}) must be divisible by num_heads ({num_heads})"

        # 用於記憶互動的多頭注意力參數
        self.attn_params = init_attention_params(slot_size, num_heads)

        # 輸入投影：將輸入投影到記憶空間
        if self.input_size != slot_size:
            self.W_input = xavier_initializer((slot_size, self.input_size))
            self.b_input = np.zeros((slot_size, 1))
        else:
            self.W_input = None
            self.b_input = None

        # 用於控制記憶更新的門
        # 門決定要更新多少 vs. 保留多少現有記憶
        gate_input_size = slot_size + self.input_size
        self.W_gate = xavier_initializer((slot_size, gate_input_size))
        self.b_gate = np.zeros((slot_size, 1))

        # 更新投影：結合注意力輸出與輸入
        self.W_update = xavier_initializer((slot_size, slot_size))
        self.b_update = np.zeros((slot_size, 1))

    def forward(self, memory_prev, input_vec=None):
        """
        使用自注意力和可選的輸入更新記憶。

        參數:
            memory_prev: 先前的記憶狀態，形狀 (batch, num_slots, slot_size)
            input_vec: 可選的要納入的輸入，形狀 (batch, input_size)

        回傳:
            memory_new: 更新後的記憶，形狀 (batch, num_slots, slot_size)

        處理流程:
            1. 對記憶槽應用多頭自注意力
            2. 如果提供了輸入，投影它並加入記憶
            3. 應用門控更新以控制資訊流
            4. 殘差連接以保留現有記憶
        """
        batch_size = memory_prev.shape[0]

        # 步驟 1：對記憶槽進行多頭自注意力
        # memory_prev: (batch, num_slots, slot_size)
        # 自注意力：每個槽注意所有其他槽
        attended_memory, attn_weights = multi_head_attention(
            Q=memory_prev,
            K=memory_prev,
            V=memory_prev,
            num_heads=self.num_heads,
            W_q=self.attn_params['W_q'],
            W_k=self.attn_params['W_k'],
            W_v=self.attn_params['W_v'],
            W_o=self.attn_params['W_o']
        )
        # attended_memory: (batch, num_slots, slot_size)

        # 步驟 2：如果提供了輸入，則投影並納入
        if input_vec is not None:
            # input_vec: (batch, input_size)
            # 如果需要，投影到 slot_size
            if self.W_input is not None:
                # 重塑以進行矩陣乘法
                # input_vec: (batch, input_size) -> (input_size, batch)
                input_vec_T = input_vec.T  # (input_size, batch)
                # W_input @ input_vec_T: (slot_size, batch)
                projected_input = self.W_input @ input_vec_T + self.b_input
                # projected_input: (slot_size, batch) -> (batch, slot_size)
                projected_input = projected_input.T
            else:
                projected_input = input_vec
            # projected_input: (batch, slot_size)

            # 將投影後的輸入加到第一個記憶槽
            # 這是注入外部資訊的簡單方法
            attended_memory[:, 0, :] = attended_memory[:, 0, :] + projected_input

        # 步驟 3：應用帶有非線性的更新投影
        # 獨立處理每個槽
        # attended_memory: (batch, num_slots, slot_size)
        # 重塑為 (batch * num_slots, slot_size) 以進行處理
        attended_flat = attended_memory.reshape(batch_size * self.num_slots, self.slot_size)
        # attended_flat: (batch * num_slots, slot_size) -> (slot_size, batch * num_slots)
        attended_flat_T = attended_flat.T

        # 應用更新轉換
        # W_update @ attended_flat_T: (slot_size, batch * num_slots)
        updated_flat_T = np.tanh(self.W_update @ attended_flat_T + self.b_update)
        # updated_flat_T: (slot_size, batch * num_slots) -> (batch * num_slots, slot_size)
        updated_flat = updated_flat_T.T
        # 重塑回：(batch, num_slots, slot_size)
        updated_memory = updated_flat.reshape(batch_size, self.num_slots, self.slot_size)

        # 步驟 4：門控更新
        if input_vec is not None:
            # 計算門值
            # 對於每個槽，根據注意後的記憶和輸入決定要更新多少
            gates_list = []
            for slot_idx in range(self.num_slots):
                # 取得此槽的注意後記憶：(batch, slot_size)
                slot_attended = attended_memory[:, slot_idx, :]  # (batch, slot_size)

                # 與輸入串接以進行門控決策
                # gate_input: (batch, slot_size + input_size)
                gate_input = np.concatenate([slot_attended, input_vec], axis=1)
                # gate_input: (batch, slot_size + input_size) -> (slot_size + input_size, batch)
                gate_input_T = gate_input.T

                # 計算門：(slot_size, batch)
                gate_T = self._sigmoid(self.W_gate @ gate_input_T + self.b_gate)
                # gate_T: (slot_size, batch) -> (batch, slot_size)
                gate = gate_T.T
                gates_list.append(gate)

            # 堆疊門：(batch, num_slots, slot_size)
            gates = np.stack(gates_list, axis=1)
        else:
            # 沒有輸入，使用常數門值
            gates = np.ones((batch_size, self.num_slots, self.slot_size)) * 0.5

        # 步驟 5：應用門控殘差連接
        # memory_new = gate * updated + (1 - gate) * old
        memory_new = gates * updated_memory + (1 - gates) * memory_prev

        return memory_new

    @staticmethod
    def _sigmoid(x):
        """數值穩定的 sigmoid 函式。"""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )


class RelationalRNNCell:
    """
    結合 LSTM 與關係記憶的關係 RNN 單元。

    此單元透過以下步驟處理一個時間步：
    1. 對輸入執行 LSTM 以獲得隱藏狀態
    2. 使用 LSTM 隱藏狀態更新關係記憶
    3. 從記憶讀取並與 LSTM 輸出結合

    這種組合允許序列處理（LSTM）和關係推理（帶注意力的記憶）兩者並行。
    """

    def __init__(self, input_size, hidden_size, num_slots=4, slot_size=64, num_heads=2):
        """
        初始化關係 RNN 單元。

        參數:
            input_size: 輸入特徵的維度
            hidden_size: LSTM 隱藏狀態的維度
            num_slots: 關係記憶槽的數量
            slot_size: 每個記憶槽的維度
            num_heads: 記憶注意力頭的數量
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.num_heads = num_heads

        # 用於序列處理的 LSTM 單元
        self.lstm_cell = LSTMCell(input_size, hidden_size)

        # 帶有注意力的關係記憶
        self.memory = RelationalMemory(
            num_slots=num_slots,
            slot_size=slot_size,
            num_heads=num_heads,
            input_size=hidden_size  # 記憶接收 LSTM 隱藏狀態
        )

        # 從記憶到輸出貢獻的投影
        # 透過跨槽平均池化從記憶讀取
        self.W_memory_read = xavier_initializer((hidden_size, slot_size))
        self.b_memory_read = np.zeros((hidden_size, 1))

        # 結合 LSTM 輸出和記憶讀出
        self.W_combine = xavier_initializer((hidden_size, hidden_size * 2))
        self.b_combine = np.zeros((hidden_size, 1))

    def forward(self, x, h_prev, c_prev, memory_prev):
        """
        一個時間步的前向傳播。

        參數:
            x: 輸入，形狀 (batch, input_size)
            h_prev: 先前的 LSTM 隱藏狀態，形狀 (hidden_size, batch) 或 (batch, hidden_size)
            c_prev: 先前的 LSTM 單元狀態，形狀 (hidden_size, batch) 或 (batch, hidden_size)
            memory_prev: 先前的記憶，形狀 (batch, num_slots, slot_size)

        回傳:
            output: 結合後的輸出，形狀 (batch, hidden_size)
            h_new: 新的 LSTM 隱藏狀態，形狀 (hidden_size, batch)
            c_new: 新的 LSTM 單元狀態，形狀 (hidden_size, batch)
            memory_new: 新的記憶狀態，形狀 (batch, num_slots, slot_size)

        處理流程:
            1. LSTM 前向傳播：x -> h_new, c_new
            2. 使用 h_new 更新記憶：h_new -> memory_new
            3. 從記憶讀取（跨槽平均池化）
            4. 將 LSTM 隱藏狀態與記憶讀出結合
        """
        batch_size = x.shape[0]

        # 處理 h_prev 和 c_prev 的輸入形狀
        # LSTM 預期 (hidden_size, batch)
        if h_prev.ndim == 2 and h_prev.shape[0] == batch_size:
            # 轉換 (batch, hidden_size) -> (hidden_size, batch)
            h_prev = h_prev.T
        if c_prev.ndim == 2 and c_prev.shape[0] == batch_size:
            # 轉換 (batch, hidden_size) -> (hidden_size, batch)
            c_prev = c_prev.T

        # 步驟 1：LSTM 前向傳播
        # x: (batch, input_size)
        # h_prev, c_prev: (hidden_size, batch)
        h_new, c_new = self.lstm_cell.forward(x, h_prev, c_prev)
        # h_new, c_new: (hidden_size, batch)

        # 步驟 2：使用 LSTM 隱藏狀態更新關係記憶
        # h_new: (hidden_size, batch) -> (batch, hidden_size)
        h_new_for_memory = h_new.T

        # 以 LSTM 隱藏狀態作為輸入更新記憶
        memory_new = self.memory.forward(memory_prev, h_new_for_memory)
        # memory_new: (batch, num_slots, slot_size)

        # 步驟 3：從記憶讀取
        # 簡單策略：跨記憶槽平均池化
        memory_readout = np.mean(memory_new, axis=1)  # (batch, slot_size)

        # 將記憶讀出投影到 hidden_size
        # memory_readout: (batch, slot_size) -> (slot_size, batch)
        memory_readout_T = memory_readout.T
        # W_memory_read @ memory_readout_T: (hidden_size, batch)
        memory_contribution_T = self.W_memory_read @ memory_readout_T + self.b_memory_read
        # memory_contribution: (batch, hidden_size)
        memory_contribution = memory_contribution_T.T

        # 步驟 4：將 LSTM 隱藏狀態與記憶貢獻結合
        # h_new: (hidden_size, batch) -> (batch, hidden_size)
        h_new_batch_first = h_new.T

        # 串接 LSTM 隱藏狀態和記憶貢獻
        combined_input = np.concatenate([h_new_batch_first, memory_contribution], axis=1)
        # combined_input: (batch, hidden_size * 2)

        # 應用組合層
        # combined_input: (batch, hidden_size * 2) -> (hidden_size * 2, batch)
        combined_input_T = combined_input.T
        # W_combine @ combined_input_T: (hidden_size, batch)
        output_T = np.tanh(self.W_combine @ combined_input_T + self.b_combine)
        # output: (batch, hidden_size)
        output = output_T.T

        return output, h_new, c_new, memory_new

    def init_memory(self, batch_size):
        """
        將記憶初始化為零。

        參數:
            batch_size: 批次大小

        回傳:
            memory: 初始化後的記憶，形狀 (batch, num_slots, slot_size)
        """
        return np.zeros((batch_size, self.num_slots, self.slot_size))


class RelationalRNN:
    """
    用於序列處理的完整 Relational RNN。

    使用 RelationalRNNCell 處理序列並投影到輸出空間。
    """

    def __init__(self, input_size, hidden_size, output_size, num_slots=4, slot_size=64, num_heads=2):
        """
        初始化 Relational RNN。

        參數:
            input_size: 輸入特徵的維度
            hidden_size: LSTM 隱藏狀態的維度
            output_size: 輸出的維度
            num_slots: 記憶槽的數量
            slot_size: 每個記憶槽的維度
            num_heads: 注意力頭的數量
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.num_heads = num_heads

        # 關係 RNN 單元
        self.cell = RelationalRNNCell(
            input_size=input_size,
            hidden_size=hidden_size,
            num_slots=num_slots,
            slot_size=slot_size,
            num_heads=num_heads
        )

        # 輸出投影層
        self.W_out = xavier_initializer((output_size, hidden_size))
        self.b_out = np.zeros((output_size, 1))

    def forward(self, sequence, return_sequences=True, return_state=False):
        """
        通過 Relational RNN 處理序列。

        參數:
            sequence: 輸入序列，形狀 (batch, seq_len, input_size)
            return_sequences: 如果為 True，回傳所有時間步的輸出
            return_state: 如果為 True，回傳最終狀態

        回傳:
            outputs: 如果 return_sequences 為 True，形狀 (batch, seq_len, output_size)
                    否則形狀 (batch, output_size)
            如果 return_state=True，還會回傳 (h_final, c_final, memory_final)
        """
        batch_size, seq_len, _ = sequence.shape

        # 初始化狀態
        h = np.zeros((self.hidden_size, batch_size))
        c = np.zeros((self.hidden_size, batch_size))
        memory = self.cell.init_memory(batch_size)

        # 儲存輸出
        outputs = []

        # 處理序列
        for t in range(seq_len):
            # 取得時間 t 的輸入
            x_t = sequence[:, t, :]  # (batch, input_size)

            # 通過單元的前向傳播
            cell_output, h, c, memory = self.cell.forward(x_t, h, c, memory)
            # cell_output: (batch, hidden_size)
            # h, c: (hidden_size, batch)
            # memory: (batch, num_slots, slot_size)

            # 投影到輸出空間
            # cell_output: (batch, hidden_size) -> (hidden_size, batch)
            cell_output_T = cell_output.T
            # W_out @ cell_output_T: (output_size, batch)
            out_t_T = self.W_out @ cell_output_T + self.b_out
            # out_t: (batch, output_size)
            out_t = out_t_T.T

            outputs.append(out_t)

        # 準備回傳值
        if return_sequences:
            result = np.stack(outputs, axis=1)  # (batch, seq_len, output_size)
        else:
            result = outputs[-1]  # (batch, output_size)

        if return_state:
            # 以 batch-first 格式回傳狀態
            h_final = h.T  # (batch, hidden_size)
            c_final = c.T  # (batch, hidden_size)
            memory_final = memory  # (batch, num_slots, slot_size)
            return result, h_final, c_final, memory_final
        else:
            return result


# ============================================================================
# 測試函式
# ============================================================================

def test_relational_memory():
    """測試關係記憶模組。"""
    print("=" * 80)
    print("Testing Relational Memory Module")
    print("=" * 80)

    np.random.seed(42)

    # Test parameters
    batch_size = 2
    num_slots = 4
    slot_size = 64
    num_heads = 2
    input_size = 32

    print(f"\nParameters:")
    print(f"  batch_size: {batch_size}")
    print(f"  num_slots: {num_slots}")
    print(f"  slot_size: {slot_size}")
    print(f"  num_heads: {num_heads}")
    print(f"  input_size: {input_size}")

    # Create relational memory
    print(f"\n[Test 1] Creating RelationalMemory...")
    rel_mem = RelationalMemory(
        num_slots=num_slots,
        slot_size=slot_size,
        num_heads=num_heads,
        input_size=input_size
    )
    print(f"  RelationalMemory created successfully")

    # Test forward pass without input
    print(f"\n[Test 2] Forward pass without input...")
    memory = np.random.randn(batch_size, num_slots, slot_size) * 0.1
    memory_new = rel_mem.forward(memory, input_vec=None)

    print(f"  Input memory shape: {memory.shape}")
    print(f"  Output memory shape: {memory_new.shape}")
    assert memory_new.shape == (batch_size, num_slots, slot_size), \
        f"Shape mismatch: expected {(batch_size, num_slots, slot_size)}, got {memory_new.shape}"
    assert not np.isnan(memory_new).any(), "NaN detected in memory output"
    assert not np.isinf(memory_new).any(), "Inf detected in memory output"
    print(f"  Shape correct, no NaN/Inf")

    # Test forward pass with input
    print(f"\n[Test 3] Forward pass with input...")
    input_vec = np.random.randn(batch_size, input_size)
    memory_new_with_input = rel_mem.forward(memory, input_vec=input_vec)

    print(f"  Input vector shape: {input_vec.shape}")
    print(f"  Output memory shape: {memory_new_with_input.shape}")
    assert memory_new_with_input.shape == (batch_size, num_slots, slot_size)
    assert not np.isnan(memory_new_with_input).any(), "NaN detected"
    assert not np.isinf(memory_new_with_input).any(), "Inf detected"
    print(f"  Shape correct, no NaN/Inf")

    # Verify memory evolves
    print(f"\n[Test 4] Verifying memory evolution...")
    assert not np.allclose(memory_new_with_input, memory), \
        "Memory should change after forward pass"
    print(f"  Memory evolves correctly")

    # Test different inputs produce different outputs
    print(f"\n[Test 5] Different inputs produce different outputs...")
    input_vec_2 = np.random.randn(batch_size, input_size) * 2.0
    memory_new_2 = rel_mem.forward(memory, input_vec=input_vec_2)
    assert not np.allclose(memory_new_with_input, memory_new_2), \
        "Different inputs should produce different memory states"
    print(f"  Different inputs -> different outputs")

    print("\n" + "=" * 80)
    print("Relational Memory: ALL TESTS PASSED")
    print("=" * 80 + "\n")


def test_relational_rnn_cell():
    """測試關係 RNN 單元。"""
    print("=" * 80)
    print("Testing Relational RNN Cell")
    print("=" * 80)

    np.random.seed(42)

    # Test parameters
    batch_size = 2
    input_size = 32
    hidden_size = 64
    num_slots = 4
    slot_size = 64
    num_heads = 2

    print(f"\nParameters:")
    print(f"  batch_size: {batch_size}")
    print(f"  input_size: {input_size}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_slots: {num_slots}")
    print(f"  slot_size: {slot_size}")
    print(f"  num_heads: {num_heads}")

    # Create cell
    print(f"\n[Test 1] Creating RelationalRNNCell...")
    cell = RelationalRNNCell(
        input_size=input_size,
        hidden_size=hidden_size,
        num_slots=num_slots,
        slot_size=slot_size,
        num_heads=num_heads
    )
    print(f"  RelationalRNNCell created successfully")

    # Test single time step
    print(f"\n[Test 2] Single time step forward pass...")
    x = np.random.randn(batch_size, input_size)
    h_prev = np.zeros((batch_size, hidden_size))
    c_prev = np.zeros((batch_size, hidden_size))
    memory_prev = cell.init_memory(batch_size)

    output, h_new, c_new, memory_new = cell.forward(x, h_prev, c_prev, memory_prev)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  h_new shape: {h_new.shape}")
    print(f"  c_new shape: {c_new.shape}")
    print(f"  memory_new shape: {memory_new.shape}")

    # Verify shapes
    assert output.shape == (batch_size, hidden_size), \
        f"Output shape mismatch: expected {(batch_size, hidden_size)}, got {output.shape}"
    assert h_new.shape == (hidden_size, batch_size), \
        f"h_new shape mismatch: expected {(hidden_size, batch_size)}, got {h_new.shape}"
    assert c_new.shape == (hidden_size, batch_size), \
        f"c_new shape mismatch: expected {(hidden_size, batch_size)}, got {c_new.shape}"
    assert memory_new.shape == (batch_size, num_slots, slot_size), \
        f"memory_new shape mismatch: expected {(batch_size, num_slots, slot_size)}, got {memory_new.shape}"

    # Check for NaN/Inf
    assert not np.isnan(output).any(), "NaN in output"
    assert not np.isinf(output).any(), "Inf in output"
    assert not np.isnan(h_new).any(), "NaN in h_new"
    assert not np.isnan(c_new).any(), "NaN in c_new"
    assert not np.isnan(memory_new).any(), "NaN in memory_new"

    print(f"  All shapes correct, no NaN/Inf")

    # Test state evolution
    print(f"\n[Test 3] State evolution over multiple steps...")
    h = h_prev
    c = c_prev
    memory = memory_prev

    for step in range(3):
        x_t = np.random.randn(batch_size, input_size)
        output, h, c, memory = cell.forward(x_t, h, c, memory)
        print(f"  Step {step + 1}: output range [{output.min():.3f}, {output.max():.3f}]")

    print(f"  State evolution successful")

    # Verify memory evolves
    print(f"\n[Test 4] Verifying memory evolution...")
    assert not np.allclose(memory, memory_prev), \
        "Memory should evolve over time steps"
    print(f"  Memory evolves correctly")

    print("\n" + "=" * 80)
    print("Relational RNN Cell: ALL TESTS PASSED")
    print("=" * 80 + "\n")


def test_relational_rnn():
    """測試完整的 Relational RNN。"""
    print("=" * 80)
    print("Testing Relational RNN (Full Sequence Processor)")
    print("=" * 80)

    np.random.seed(42)

    # Test parameters (matching task specification)
    batch_size = 2
    seq_len = 10
    input_size = 32
    hidden_size = 64
    output_size = 16
    num_slots = 4
    slot_size = 64
    num_heads = 2

    print(f"\nParameters:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  input_size: {input_size}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  output_size: {output_size}")
    print(f"  num_slots: {num_slots}")
    print(f"  slot_size: {slot_size}")
    print(f"  num_heads: {num_heads}")

    # Create model
    print(f"\n[Test 1] Creating RelationalRNN...")
    model = RelationalRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_slots=num_slots,
        slot_size=slot_size,
        num_heads=num_heads
    )
    print(f"  RelationalRNN created successfully")

    # Create random sequence
    print(f"\n[Test 2] Processing sequence (return_sequences=True)...")
    sequence = np.random.randn(batch_size, seq_len, input_size)
    print(f"  Input sequence shape: {sequence.shape}")

    outputs = model.forward(sequence, return_sequences=True)
    print(f"  Output shape: {outputs.shape}")
    print(f"  Expected: ({batch_size}, {seq_len}, {output_size})")

    assert outputs.shape == (batch_size, seq_len, output_size), \
        f"Shape mismatch: expected {(batch_size, seq_len, output_size)}, got {outputs.shape}"
    assert not np.isnan(outputs).any(), "NaN detected in outputs"
    assert not np.isinf(outputs).any(), "Inf detected in outputs"
    print(f"  Shape correct, no NaN/Inf")

    # Test return_sequences=False
    print(f"\n[Test 3] Processing sequence (return_sequences=False)...")
    output_last = model.forward(sequence, return_sequences=False)
    print(f"  Output shape: {output_last.shape}")
    print(f"  Expected: ({batch_size}, {output_size})")

    assert output_last.shape == (batch_size, output_size), \
        f"Shape mismatch: expected {(batch_size, output_size)}, got {output_last.shape}"
    print(f"  Shape correct")

    # Test return_state=True
    print(f"\n[Test 4] Processing with state return...")
    outputs, h_final, c_final, memory_final = model.forward(
        sequence, return_sequences=True, return_state=True
    )

    print(f"  Outputs shape: {outputs.shape}")
    print(f"  h_final shape: {h_final.shape}")
    print(f"  c_final shape: {c_final.shape}")
    print(f"  memory_final shape: {memory_final.shape}")

    assert h_final.shape == (batch_size, hidden_size)
    assert c_final.shape == (batch_size, hidden_size)
    assert memory_final.shape == (batch_size, num_slots, slot_size)
    print(f"  All state shapes correct")

    # Test memory evolution over sequence
    print(f"\n[Test 5] Verifying memory evolution over sequence...")
    # Process same sequence again and track memory at each step
    h = np.zeros((hidden_size, batch_size))
    c = np.zeros((hidden_size, batch_size))
    memory = model.cell.init_memory(batch_size)

    memory_states = [memory.copy()]
    for t in range(seq_len):
        x_t = sequence[:, t, :]
        _, h, c, memory = model.cell.forward(x_t, h, c, memory)
        memory_states.append(memory.copy())

    # Check that memory changes over time
    memory_changes = []
    for t in range(1, len(memory_states)):
        change = np.linalg.norm(memory_states[t] - memory_states[t-1])
        memory_changes.append(change)

    print(f"  Memory change per step (first 5):")
    for t, change in enumerate(memory_changes[:5]):
        print(f"    Step {t+1}: {change:.4f}")

    assert all(change > 0 for change in memory_changes), \
        "Memory should change at each time step"
    print(f"  Memory evolves correctly over time")

    # Test different sequences produce different outputs
    print(f"\n[Test 6] Different sequences produce different outputs...")
    sequence_2 = np.random.randn(batch_size, seq_len, input_size) * 2.0
    outputs_2 = model.forward(sequence_2, return_sequences=True)

    assert not np.allclose(outputs, outputs_2), \
        "Different input sequences should produce different outputs"
    print(f"  Different inputs -> different outputs")

    print("\n" + "=" * 80)
    print("Relational RNN: ALL TESTS PASSED")
    print("=" * 80 + "\n")

    return model


def compare_with_lstm_baseline():
    """比較 Relational RNN 與 LSTM 基準。"""
    print("=" * 80)
    print("Comparison: Relational RNN vs. LSTM Baseline")
    print("=" * 80)

    from lstm_baseline import LSTM

    np.random.seed(42)

    # Common parameters
    batch_size = 2
    seq_len = 10
    input_size = 32
    hidden_size = 64
    output_size = 16

    # Create same input sequence for fair comparison
    sequence = np.random.randn(batch_size, seq_len, input_size)

    print(f"\nTest Configuration:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  input_size: {input_size}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  output_size: {output_size}")

    # LSTM Baseline
    print(f"\n[1] LSTM Baseline")
    lstm = LSTM(input_size, hidden_size, output_size)
    lstm_outputs = lstm.forward(sequence, return_sequences=True)

    print(f"  Output shape: {lstm_outputs.shape}")
    print(f"  Output range: [{lstm_outputs.min():.3f}, {lstm_outputs.max():.3f}]")
    print(f"  Output mean: {lstm_outputs.mean():.3f}")
    print(f"  Output std: {lstm_outputs.std():.3f}")

    # Count LSTM parameters
    lstm_params = lstm.get_params()
    lstm_param_count = sum(p.size for p in lstm_params.values())
    print(f"  Parameter count: {lstm_param_count:,}")

    # Relational RNN
    print(f"\n[2] Relational RNN")
    rel_rnn = RelationalRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_slots=4,
        slot_size=64,
        num_heads=2
    )
    rel_outputs = rel_rnn.forward(sequence, return_sequences=True)

    print(f"  Output shape: {rel_outputs.shape}")
    print(f"  Output range: [{rel_outputs.min():.3f}, {rel_outputs.max():.3f}]")
    print(f"  Output mean: {rel_outputs.mean():.3f}")
    print(f"  Output std: {rel_outputs.std():.3f}")

    # Estimate Relational RNN parameters (approximate)
    # LSTM + Memory attention + projections
    print(f"  Additional components:")
    print(f"    - Relational memory with {rel_rnn.num_slots} slots")
    print(f"    - Multi-head attention ({rel_rnn.num_heads} heads)")
    print(f"    - Memory update gates and projections")

    # Architecture comparison
    print(f"\n[3] Architecture Comparison")
    print(f"\n  LSTM Baseline:")
    print(f"    - Sequential processing only")
    print(f"    - Hidden state carries all information")
    print(f"    - No explicit relational reasoning")

    print(f"\n  Relational RNN:")
    print(f"    - Sequential processing (LSTM)")
    print(f"    + Relational memory (multi-head attention)")
    print(f"    - Memory slots can interact and specialize")
    print(f"    - Explicit relational reasoning capability")

    # Integration explanation
    print(f"\n[4] LSTM + Memory Integration")
    print(f"  How they interact:")
    print(f"    1. LSTM processes input sequentially")
    print(f"    2. LSTM hidden state updates relational memory")
    print(f"    3. Memory slots interact via self-attention")
    print(f"    4. Memory readout combined with LSTM output")
    print(f"    5. Combined representation used for predictions")

    print(f"\n  Benefits:")
    print(f"    - LSTM: temporal dependencies, sequential patterns")
    print(f"    - Memory: relational reasoning, entity tracking")
    print(f"    - Combined: both sequential and relational processing")

    print("\n" + "=" * 80)
    print("Comparison Complete")
    print("=" * 80 + "\n")


def main():
    """執行所有測試。"""
    print("\n" + "=" * 80)
    print(" " * 15 + "RELATIONAL RNN IMPLEMENTATION TEST SUITE")
    print(" " * 20 + "Paper 18: Relational RNN - Task P2-T2")
    print("=" * 80 + "\n")

    # Run all tests
    test_relational_memory()
    test_relational_rnn_cell()
    model = test_relational_rnn()
    compare_with_lstm_baseline()

    print("=" * 80)
    print(" " * 25 + "ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\nImplementation Summary:")
    print("  - RelationalMemory: Multi-head self-attention over memory slots")
    print("  - RelationalRNNCell: Combines LSTM + relational memory")
    print("  - RelationalRNN: Full sequence processor with output projection")
    print("  - All shapes verified")
    print("  - No NaN/Inf in forward passes")
    print("  - Memory evolution confirmed")
    print("  - Comparison with LSTM baseline complete")
    print("\nIntegration Approach:")
    print("  1. LSTM processes sequential input -> hidden state")
    print("  2. Hidden state updates relational memory via attention")
    print("  3. Memory slots interact through multi-head self-attention")
    print("  4. Memory readout (mean pooling) combined with LSTM output")
    print("  5. Combined representation projected to output space")
    print("\nKey Features:")
    print("  - Gated memory updates for controlled information flow")
    print("  - Residual connections preserve existing memory")
    print("  - Separate processing streams (sequential + relational)")
    print("  - Flexible memory size and attention heads")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
