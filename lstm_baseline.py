"""
用於關係型 RNN 比較的 LSTM 基準實作

本模組僅使用 NumPy 實作標準 LSTM（長短期記憶網路）。
實作內容包括：
- 適當的參數初始化（輸入權重使用 Xavier/He，遞迴權重使用正交初始化）
- 遺忘門偏差初始化為 1.0（幫助學習的標準技巧）
- LSTMCell 用於單一時間步處理
- LSTM 包裝器用於序列處理和輸出投影

論文 18：關係型 RNN 比較基準
"""

import numpy as np


def orthogonal_initializer(shape, gain=1.0):
    """
    使用正交初始化來初始化權重矩陣。
    這有助於防止遞迴連接中的梯度消失/爆炸問題。

    參數：
        shape：(rows, cols) 的元組
        gain：縮放因子（預設為 1.0）

    回傳：
        給定形狀的正交矩陣
    """
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return gain * q[:shape[0], :shape[1]]


def xavier_initializer(shape):
    """
    用於輸入權重的 Xavier/Glorot 初始化。
    有助於維持跨層激活值的變異數。

    參數：
        shape：(rows, cols) 的元組

    回傳：
        Xavier 初始化的矩陣
    """
    limit = np.sqrt(6.0 / (shape[0] + shape[1]))
    return np.random.uniform(-limit, limit, shape)


class LSTMCell:
    """
    具有遺忘門、輸入門和輸出門的標準 LSTM 單元。

    架構：
        f_t = sigmoid(W_f @ x_t + U_f @ h_{t-1} + b_f)  # 遺忘門（forget gate）
        i_t = sigmoid(W_i @ x_t + U_i @ h_{t-1} + b_i)  # 輸入門（input gate）
        c_tilde_t = tanh(W_c @ x_t + U_c @ h_{t-1} + b_c)  # 候選細胞狀態（candidate cell state）
        o_t = sigmoid(W_o @ x_t + U_o @ h_{t-1} + b_o)  # 輸出門（output gate）
        c_t = f_t * c_{t-1} + i_t * c_tilde_t  # 新的細胞狀態
        h_t = o_t * tanh(c_t)  # 新的隱藏狀態

    參數：
        input_size：輸入特徵的維度
        hidden_size：隱藏狀態和細胞狀態的維度
    """

    def __init__(self, input_size, hidden_size):
        """
        使用適當的初始化策略初始化 LSTM 參數。

        參數：
            input_size：整數，輸入特徵的維度
            hidden_size：整數，隱藏狀態和細胞狀態的維度
        """
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 遺忘門參數
        # 輸入權重：Xavier 初始化
        self.W_f = xavier_initializer((hidden_size, input_size))
        # 遞迴權重：正交初始化
        self.U_f = orthogonal_initializer((hidden_size, hidden_size))
        # 偏差：初始化為 1.0（幫助學習長期依賴的標準技巧）
        self.b_f = np.ones((hidden_size, 1))

        # 輸入門參數
        self.W_i = xavier_initializer((hidden_size, input_size))
        self.U_i = orthogonal_initializer((hidden_size, hidden_size))
        self.b_i = np.zeros((hidden_size, 1))

        # 細胞門參數（候選值）
        self.W_c = xavier_initializer((hidden_size, input_size))
        self.U_c = orthogonal_initializer((hidden_size, hidden_size))
        self.b_c = np.zeros((hidden_size, 1))

        # 輸出門參數
        self.W_o = xavier_initializer((hidden_size, input_size))
        self.U_o = orthogonal_initializer((hidden_size, hidden_size))
        self.b_o = np.zeros((hidden_size, 1))

    def forward(self, x, h_prev, c_prev):
        """
        單一時間步的前向傳播。

        參數：
            x：輸入，形狀 (batch_size, input_size) 或 (input_size, batch_size)
            h_prev：前一個隱藏狀態，形狀 (hidden_size, batch_size)
            c_prev：前一個細胞狀態，形狀 (hidden_size, batch_size)

        回傳：
            h：新的隱藏狀態，形狀 (hidden_size, batch_size)
            c：新的細胞狀態，形狀 (hidden_size, batch_size)
        """
        # 處理輸入形狀：將 (batch_size, input_size) 轉換為 (input_size, batch_size)
        if x.ndim == 2 and x.shape[1] == self.input_size:
            x = x.T  # 轉置為 (input_size, batch_size)

        # 確保 x 是二維的
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        # 確保 h_prev 和 c_prev 是二維的
        if h_prev.ndim == 1:
            h_prev = h_prev.reshape(-1, 1)
        if c_prev.ndim == 1:
            c_prev = c_prev.reshape(-1, 1)

        # 遺忘門：決定要從細胞狀態中丟棄哪些資訊
        f = self._sigmoid(self.W_f @ x + self.U_f @ h_prev + self.b_f)

        # 輸入門：決定要在細胞狀態中儲存哪些新資訊
        i = self._sigmoid(self.W_i @ x + self.U_i @ h_prev + self.b_i)

        # 候選細胞狀態：可能要加入的新資訊
        c_tilde = np.tanh(self.W_c @ x + self.U_c @ h_prev + self.b_c)

        # 輸出門：決定要輸出細胞狀態的哪些部分
        o = self._sigmoid(self.W_o @ x + self.U_o @ h_prev + self.b_o)

        # 更新細胞狀態：遺忘舊的 + 加入新的
        c = f * c_prev + i * c_tilde

        # 更新隱藏狀態：過濾後的細胞狀態
        h = o * np.tanh(c)

        return h, c

    @staticmethod
    def _sigmoid(x):
        """數值穩定的 sigmoid 函數。"""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )


class LSTM:
    """
    處理序列並產生輸出的 LSTM。

    這個包裝類別使用 LSTMCell 來處理輸入序列，
    並可選擇性地將隱藏狀態投影到輸出空間。

    參數：
        input_size：輸入特徵的維度
        hidden_size：隱藏狀態的維度
        output_size：輸出的維度（若為 None 則不進行投影）
    """

    def __init__(self, input_size, hidden_size, output_size=None):
        """
        初始化 LSTM，可選擇性地包含輸出投影。

        參數：
            input_size：整數，輸入特徵的維度
            hidden_size：整數，隱藏狀態的維度
            output_size：整數或 None，輸出的維度
                        若為 None，則輸出為隱藏狀態
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 建立 LSTM 單元
        self.cell = LSTMCell(input_size, hidden_size)

        # 可選的輸出投影層
        if output_size is not None:
            self.W_out = xavier_initializer((output_size, hidden_size))
            self.b_out = np.zeros((output_size, 1))
        else:
            self.W_out = None
            self.b_out = None

    def forward(self, sequence, return_sequences=True, return_state=False):
        """
        透過 LSTM 處理序列。

        參數：
            sequence：輸入序列，形狀 (batch_size, seq_len, input_size)
            return_sequences：布林值，若為 True 則回傳所有時間步的輸出，
                            若為 False 則只回傳最後一個輸出
            return_state：布林值，若為 True 則同時回傳最終的 (h, c) 狀態

        回傳：
            若 return_sequences=True 且 return_state=False：
                outputs：形狀 (batch_size, seq_len, output_size 或 hidden_size)
            若 return_sequences=False 且 return_state=False：
                output：形狀 (batch_size, output_size 或 hidden_size)
            若 return_state=True：
                outputs（或 output）、final_h、final_c
        """
        batch_size, seq_len, _ = sequence.shape

        # 初始化隱藏狀態和細胞狀態
        h = np.zeros((self.hidden_size, batch_size))
        c = np.zeros((self.hidden_size, batch_size))

        # 儲存每個時間步的輸出
        outputs = []

        # 處理序列
        for t in range(seq_len):
            # 取得時間 t 的輸入：(batch_size, input_size)
            x_t = sequence[:, t, :]

            # LSTM 前向傳播
            h, c = self.cell.forward(x_t, h, c)

            # 如果需要，投影到輸出空間
            if self.W_out is not None:
                # h 形狀：(hidden_size, batch_size)
                # output 形狀：(output_size, batch_size)
                out_t = self.W_out @ h + self.b_out
            else:
                out_t = h

            # 儲存輸出：轉置為 (batch_size, output_size 或 hidden_size)
            outputs.append(out_t.T)

        # 堆疊輸出
        if return_sequences:
            # 形狀：(batch_size, seq_len, output_size 或 hidden_size)
            result = np.stack(outputs, axis=1)
        else:
            # 只回傳最後一個輸出：(batch_size, output_size 或 hidden_size)
            result = outputs[-1]

        if return_state:
            # 回傳輸出和最終狀態
            # 將 h 和 c 轉置回 (batch_size, hidden_size)
            return result, h.T, c.T
        else:
            return result

    def get_params(self):
        """
        取得所有模型參數。

        回傳：
            參數名稱到陣列的字典
        """
        params = {
            'W_f': self.cell.W_f, 'U_f': self.cell.U_f, 'b_f': self.cell.b_f,
            'W_i': self.cell.W_i, 'U_i': self.cell.U_i, 'b_i': self.cell.b_i,
            'W_c': self.cell.W_c, 'U_c': self.cell.U_c, 'b_c': self.cell.b_c,
            'W_o': self.cell.W_o, 'U_o': self.cell.U_o, 'b_o': self.cell.b_o,
        }

        if self.W_out is not None:
            params['W_out'] = self.W_out
            params['b_out'] = self.b_out

        return params

    def set_params(self, params):
        """
        設定模型參數。

        參數：
            params：參數名稱到陣列的字典
        """
        self.cell.W_f = params['W_f']
        self.cell.U_f = params['U_f']
        self.cell.b_f = params['b_f']

        self.cell.W_i = params['W_i']
        self.cell.U_i = params['U_i']
        self.cell.b_i = params['b_i']

        self.cell.W_c = params['W_c']
        self.cell.U_c = params['U_c']
        self.cell.b_c = params['b_c']

        self.cell.W_o = params['W_o']
        self.cell.U_o = params['U_o']
        self.cell.b_o = params['b_o']

        if 'W_out' in params:
            self.W_out = params['W_out']
            self.b_out = params['b_out']


def test_lstm():
    """
    使用隨機資料測試 LSTM 實作。
    驗證項目：
    - 正確的輸出形狀
    - 無 NaN 或 Inf 值
    - 正確的狀態演化
    """
    print("="*60)
    print("Testing LSTM Implementation")
    print("="*60)

    # 測試參數
    batch_size = 2
    seq_len = 10
    input_size = 32
    hidden_size = 64
    output_size = 16

    # 建立隨機序列
    print(f"\n1. Creating random sequence...")
    print(f"   Shape: (batch={batch_size}, seq_len={seq_len}, input_size={input_size})")
    sequence = np.random.randn(batch_size, seq_len, input_size)

    # 測試 1：無輸出投影的 LSTM
    print(f"\n2. Testing LSTM without output projection...")
    lstm_no_proj = LSTM(input_size, hidden_size, output_size=None)

    outputs = lstm_no_proj.forward(sequence, return_sequences=True)
    print(f"   Output shape: {outputs.shape}")
    print(f"   Expected: ({batch_size}, {seq_len}, {hidden_size})")
    assert outputs.shape == (batch_size, seq_len, hidden_size), "Shape mismatch!"
    assert not np.isnan(outputs).any(), "NaN detected in outputs!"
    assert not np.isinf(outputs).any(), "Inf detected in outputs!"
    print(f"   ✓ Shape correct, no NaN/Inf")

    # 測試 2：有輸出投影的 LSTM
    print(f"\n3. Testing LSTM with output projection...")
    lstm_with_proj = LSTM(input_size, hidden_size, output_size=output_size)

    outputs = lstm_with_proj.forward(sequence, return_sequences=True)
    print(f"   Output shape: {outputs.shape}")
    print(f"   Expected: ({batch_size}, {seq_len}, {output_size})")
    assert outputs.shape == (batch_size, seq_len, output_size), "Shape mismatch!"
    assert not np.isnan(outputs).any(), "NaN detected in outputs!"
    assert not np.isinf(outputs).any(), "Inf detected in outputs!"
    print(f"   ✓ Shape correct, no NaN/Inf")

    # 測試 3：只回傳最後一個輸出
    print(f"\n4. Testing return_sequences=False...")
    output_last = lstm_with_proj.forward(sequence, return_sequences=False)
    print(f"   Output shape: {output_last.shape}")
    print(f"   Expected: ({batch_size}, {output_size})")
    assert output_last.shape == (batch_size, output_size), "Shape mismatch!"
    print(f"   ✓ Shape correct")

    # 測試 4：回傳狀態
    print(f"\n5. Testing return_state=True...")
    outputs, final_h, final_c = lstm_with_proj.forward(sequence, return_sequences=True, return_state=True)
    print(f"   Outputs shape: {outputs.shape}")
    print(f"   Final h shape: {final_h.shape}")
    print(f"   Final c shape: {final_c.shape}")
    assert final_h.shape == (batch_size, hidden_size), "Hidden state shape mismatch!"
    assert final_c.shape == (batch_size, hidden_size), "Cell state shape mismatch!"
    print(f"   ✓ All shapes correct")

    # 測試 5：驗證初始化屬性
    print(f"\n6. Verifying parameter initialization...")
    params = lstm_with_proj.get_params()

    # 檢查遺忘門偏差是否初始化為 1.0
    assert np.allclose(params['b_f'], 1.0), "Forget bias should be initialized to 1.0!"
    print(f"   ✓ Forget gate bias initialized to 1.0")

    # 檢查其他偏差是否為零
    assert np.allclose(params['b_i'], 0.0), "Input bias should be initialized to 0.0!"
    assert np.allclose(params['b_c'], 0.0), "Cell bias should be initialized to 0.0!"
    assert np.allclose(params['b_o'], 0.0), "Output bias should be initialized to 0.0!"
    print(f"   ✓ Other biases initialized to 0.0")

    # 檢查遞迴權重是否為正交（U @ U.T ≈ I）
    U_f = params['U_f']
    ortho_check = U_f @ U_f.T
    identity = np.eye(hidden_size)
    is_orthogonal = np.allclose(ortho_check, identity, atol=1e-5)
    print(f"   ✓ Recurrent weights are {'orthogonal' if is_orthogonal else 'approximately orthogonal'}")
    print(f"     Max deviation from identity: {np.max(np.abs(ortho_check - identity)):.6f}")

    # 測試 6：驗證狀態演化
    print(f"\n7. Testing state evolution...")
    # 建立具有模式的簡單序列
    simple_seq = np.ones((1, 5, input_size)) * 0.1
    outputs_1 = lstm_with_proj.forward(simple_seq, return_sequences=True)

    # 不同的輸入應該產生不同的輸出
    simple_seq_2 = np.ones((1, 5, input_size)) * 0.5
    outputs_2 = lstm_with_proj.forward(simple_seq_2, return_sequences=True)

    assert not np.allclose(outputs_1, outputs_2), "Different inputs should produce different outputs!"
    print(f"   ✓ State evolves correctly with different inputs")

    # 測試 7：單一時間步處理
    print(f"\n8. Testing single time step...")
    cell = LSTMCell(input_size, hidden_size)
    x = np.random.randn(batch_size, input_size)
    h_prev = np.zeros((hidden_size, batch_size))
    c_prev = np.zeros((hidden_size, batch_size))

    h, c = cell.forward(x, h_prev, c_prev)
    assert h.shape == (hidden_size, batch_size), "Hidden state shape mismatch!"
    assert c.shape == (hidden_size, batch_size), "Cell state shape mismatch!"
    assert not np.isnan(h).any(), "NaN in hidden state!"
    assert not np.isnan(c).any(), "NaN in cell state!"
    print(f"   ✓ Single step processing works correctly")

    # 總結
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
    print("\nLSTM Implementation Summary:")
    print(f"- Input size: {input_size}")
    print(f"- Hidden size: {hidden_size}")
    print(f"- Output size: {output_size}")
    print(f"- Forget bias initialized to 1.0 (helps long-term dependencies)")
    print(f"- Recurrent weights use orthogonal initialization")
    print(f"- Input weights use Xavier initialization")
    print(f"- No NaN/Inf in forward pass")
    print(f"- All output shapes verified")
    print("="*60)

    return lstm_with_proj


if __name__ == "__main__":
    # 執行測試
    np.random.seed(42)  # 確保可重現性
    model = test_lstm()

    print("\n" + "="*60)
    print("LSTM Baseline Ready for Comparison!")
    print("="*60)
