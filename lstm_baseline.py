"""
LSTM Baseline Implementation for Relational RNN Comparison

This module implements a standard LSTM (Long Short-Term Memory) network
using NumPy only. The implementation includes:
- Proper parameter initialization (Xavier/He for input weights, orthogonal for recurrent)
- Forget gate bias initialization to 1.0 (standard trick to help learning)
- LSTMCell for single time step processing
- LSTM wrapper for sequence processing with output projection

Paper 18: Relational RNN Comparison Baseline
"""

import numpy as np


def orthogonal_initializer(shape, gain=1.0):
    """
    Initialize weight matrix with orthogonal initialization.
    This helps prevent vanishing/exploding gradients in recurrent connections.

    Args:
        shape: tuple of (rows, cols)
        gain: scaling factor (default 1.0)

    Returns:
        Orthogonal matrix of given shape
    """
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return gain * q[:shape[0], :shape[1]]


def xavier_initializer(shape):
    """
    Xavier/Glorot initialization for input weights.
    Helps maintain variance of activations across layers.

    Args:
        shape: tuple of (rows, cols)

    Returns:
        Xavier-initialized matrix
    """
    limit = np.sqrt(6.0 / (shape[0] + shape[1]))
    return np.random.uniform(-limit, limit, shape)


class LSTMCell:
    """
    Standard LSTM cell with forget, input, and output gates.

    Architecture:
        f_t = sigmoid(W_f @ x_t + U_f @ h_{t-1} + b_f)  # forget gate
        i_t = sigmoid(W_i @ x_t + U_i @ h_{t-1} + b_i)  # input gate
        c_tilde_t = tanh(W_c @ x_t + U_c @ h_{t-1} + b_c)  # candidate cell state
        o_t = sigmoid(W_o @ x_t + U_o @ h_{t-1} + b_o)  # output gate
        c_t = f_t * c_{t-1} + i_t * c_tilde_t  # new cell state
        h_t = o_t * tanh(c_t)  # new hidden state

    Parameters:
        input_size: dimension of input features
        hidden_size: dimension of hidden state and cell state
    """

    def __init__(self, input_size, hidden_size):
        """
        Initialize LSTM parameters with proper initialization strategies.

        Args:
            input_size: int, dimension of input features
            hidden_size: int, dimension of hidden and cell states
        """
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Forget gate parameters
        # Input weights: Xavier initialization
        self.W_f = xavier_initializer((hidden_size, input_size))
        # Recurrent weights: Orthogonal initialization
        self.U_f = orthogonal_initializer((hidden_size, hidden_size))
        # Bias: Initialize to 1.0 (standard trick to help learning long-term dependencies)
        self.b_f = np.ones((hidden_size, 1))

        # Input gate parameters
        self.W_i = xavier_initializer((hidden_size, input_size))
        self.U_i = orthogonal_initializer((hidden_size, hidden_size))
        self.b_i = np.zeros((hidden_size, 1))

        # Cell gate parameters (candidate values)
        self.W_c = xavier_initializer((hidden_size, input_size))
        self.U_c = orthogonal_initializer((hidden_size, hidden_size))
        self.b_c = np.zeros((hidden_size, 1))

        # Output gate parameters
        self.W_o = xavier_initializer((hidden_size, input_size))
        self.U_o = orthogonal_initializer((hidden_size, hidden_size))
        self.b_o = np.zeros((hidden_size, 1))

    def forward(self, x, h_prev, c_prev):
        """
        Forward pass for a single time step.

        Args:
            x: input, shape (batch_size, input_size) or (input_size, batch_size)
            h_prev: previous hidden state, shape (hidden_size, batch_size)
            c_prev: previous cell state, shape (hidden_size, batch_size)

        Returns:
            h: new hidden state, shape (hidden_size, batch_size)
            c: new cell state, shape (hidden_size, batch_size)
        """
        # Handle input shape: convert (batch_size, input_size) to (input_size, batch_size)
        if x.ndim == 2 and x.shape[1] == self.input_size:
            x = x.T  # Transpose to (input_size, batch_size)

        # Ensure x is 2D
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        # Ensure h_prev and c_prev are 2D
        if h_prev.ndim == 1:
            h_prev = h_prev.reshape(-1, 1)
        if c_prev.ndim == 1:
            c_prev = c_prev.reshape(-1, 1)

        # Forget gate: decides what information to discard from cell state
        f = self._sigmoid(self.W_f @ x + self.U_f @ h_prev + self.b_f)

        # Input gate: decides what new information to store in cell state
        i = self._sigmoid(self.W_i @ x + self.U_i @ h_prev + self.b_i)

        # Candidate cell state: new information that could be added
        c_tilde = np.tanh(self.W_c @ x + self.U_c @ h_prev + self.b_c)

        # Output gate: decides what parts of cell state to output
        o = self._sigmoid(self.W_o @ x + self.U_o @ h_prev + self.b_o)

        # Update cell state: forget old + add new
        c = f * c_prev + i * c_tilde

        # Update hidden state: filtered cell state
        h = o * np.tanh(c)

        return h, c

    @staticmethod
    def _sigmoid(x):
        """Numerically stable sigmoid function."""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )


class LSTM:
    """
    LSTM that processes sequences and produces outputs.

    This wrapper class uses LSTMCell to process sequences of inputs
    and optionally projects the hidden states to output space.

    Parameters:
        input_size: dimension of input features
        hidden_size: dimension of hidden state
        output_size: dimension of output (None for no projection)
    """

    def __init__(self, input_size, hidden_size, output_size=None):
        """
        Initialize LSTM with optional output projection.

        Args:
            input_size: int, dimension of input features
            hidden_size: int, dimension of hidden state
            output_size: int or None, dimension of output
                        If None, outputs are hidden states
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Create LSTM cell
        self.cell = LSTMCell(input_size, hidden_size)

        # Optional output projection layer
        if output_size is not None:
            self.W_out = xavier_initializer((output_size, hidden_size))
            self.b_out = np.zeros((output_size, 1))
        else:
            self.W_out = None
            self.b_out = None

    def forward(self, sequence, return_sequences=True, return_state=False):
        """
        Process a sequence through the LSTM.

        Args:
            sequence: input sequence, shape (batch_size, seq_len, input_size)
            return_sequences: bool, if True return outputs for all time steps,
                            if False return only the last output
            return_state: bool, if True return final (h, c) states as well

        Returns:
            if return_sequences=True and return_state=False:
                outputs: shape (batch_size, seq_len, output_size or hidden_size)
            if return_sequences=False and return_state=False:
                output: shape (batch_size, output_size or hidden_size)
            if return_state=True:
                outputs (or output), final_h, final_c
        """
        batch_size, seq_len, _ = sequence.shape

        # Initialize hidden and cell states
        h = np.zeros((self.hidden_size, batch_size))
        c = np.zeros((self.hidden_size, batch_size))

        # Store outputs for each time step
        outputs = []

        # Process sequence
        for t in range(seq_len):
            # Get input at time t: (batch_size, input_size)
            x_t = sequence[:, t, :]

            # LSTM forward pass
            h, c = self.cell.forward(x_t, h, c)

            # Project to output space if needed
            if self.W_out is not None:
                # h shape: (hidden_size, batch_size)
                # output shape: (output_size, batch_size)
                out_t = self.W_out @ h + self.b_out
            else:
                out_t = h

            # Store output: transpose to (batch_size, output_size or hidden_size)
            outputs.append(out_t.T)

        # Stack outputs
        if return_sequences:
            # Shape: (batch_size, seq_len, output_size or hidden_size)
            result = np.stack(outputs, axis=1)
        else:
            # Return only last output: (batch_size, output_size or hidden_size)
            result = outputs[-1]

        if return_state:
            # Return outputs and final states
            # Transpose h and c back to (batch_size, hidden_size)
            return result, h.T, c.T
        else:
            return result

    def get_params(self):
        """
        Get all model parameters.

        Returns:
            dict of parameter names to arrays
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
        Set model parameters.

        Args:
            params: dict of parameter names to arrays
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
    Test the LSTM implementation with random data.
    Verifies:
    - Correct output shapes
    - No NaN or Inf values
    - Proper state evolution
    """
    print("="*60)
    print("Testing LSTM Implementation")
    print("="*60)

    # Test parameters
    batch_size = 2
    seq_len = 10
    input_size = 32
    hidden_size = 64
    output_size = 16

    # Create random sequence
    print(f"\n1. Creating random sequence...")
    print(f"   Shape: (batch={batch_size}, seq_len={seq_len}, input_size={input_size})")
    sequence = np.random.randn(batch_size, seq_len, input_size)

    # Test 1: LSTM without output projection
    print(f"\n2. Testing LSTM without output projection...")
    lstm_no_proj = LSTM(input_size, hidden_size, output_size=None)

    outputs = lstm_no_proj.forward(sequence, return_sequences=True)
    print(f"   Output shape: {outputs.shape}")
    print(f"   Expected: ({batch_size}, {seq_len}, {hidden_size})")
    assert outputs.shape == (batch_size, seq_len, hidden_size), "Shape mismatch!"
    assert not np.isnan(outputs).any(), "NaN detected in outputs!"
    assert not np.isinf(outputs).any(), "Inf detected in outputs!"
    print(f"   ✓ Shape correct, no NaN/Inf")

    # Test 2: LSTM with output projection
    print(f"\n3. Testing LSTM with output projection...")
    lstm_with_proj = LSTM(input_size, hidden_size, output_size=output_size)

    outputs = lstm_with_proj.forward(sequence, return_sequences=True)
    print(f"   Output shape: {outputs.shape}")
    print(f"   Expected: ({batch_size}, {seq_len}, {output_size})")
    assert outputs.shape == (batch_size, seq_len, output_size), "Shape mismatch!"
    assert not np.isnan(outputs).any(), "NaN detected in outputs!"
    assert not np.isinf(outputs).any(), "Inf detected in outputs!"
    print(f"   ✓ Shape correct, no NaN/Inf")

    # Test 3: Return only last output
    print(f"\n4. Testing return_sequences=False...")
    output_last = lstm_with_proj.forward(sequence, return_sequences=False)
    print(f"   Output shape: {output_last.shape}")
    print(f"   Expected: ({batch_size}, {output_size})")
    assert output_last.shape == (batch_size, output_size), "Shape mismatch!"
    print(f"   ✓ Shape correct")

    # Test 4: Return states
    print(f"\n5. Testing return_state=True...")
    outputs, final_h, final_c = lstm_with_proj.forward(sequence, return_sequences=True, return_state=True)
    print(f"   Outputs shape: {outputs.shape}")
    print(f"   Final h shape: {final_h.shape}")
    print(f"   Final c shape: {final_c.shape}")
    assert final_h.shape == (batch_size, hidden_size), "Hidden state shape mismatch!"
    assert final_c.shape == (batch_size, hidden_size), "Cell state shape mismatch!"
    print(f"   ✓ All shapes correct")

    # Test 5: Verify initialization properties
    print(f"\n6. Verifying parameter initialization...")
    params = lstm_with_proj.get_params()

    # Check forget gate bias is initialized to 1.0
    assert np.allclose(params['b_f'], 1.0), "Forget bias should be initialized to 1.0!"
    print(f"   ✓ Forget gate bias initialized to 1.0")

    # Check other biases are zero
    assert np.allclose(params['b_i'], 0.0), "Input bias should be initialized to 0.0!"
    assert np.allclose(params['b_c'], 0.0), "Cell bias should be initialized to 0.0!"
    assert np.allclose(params['b_o'], 0.0), "Output bias should be initialized to 0.0!"
    print(f"   ✓ Other biases initialized to 0.0")

    # Check recurrent weights are orthogonal (U @ U.T ≈ I)
    U_f = params['U_f']
    ortho_check = U_f @ U_f.T
    identity = np.eye(hidden_size)
    is_orthogonal = np.allclose(ortho_check, identity, atol=1e-5)
    print(f"   ✓ Recurrent weights are {'orthogonal' if is_orthogonal else 'approximately orthogonal'}")
    print(f"     Max deviation from identity: {np.max(np.abs(ortho_check - identity)):.6f}")

    # Test 6: Verify state evolution
    print(f"\n7. Testing state evolution...")
    # Create simple sequence with pattern
    simple_seq = np.ones((1, 5, input_size)) * 0.1
    outputs_1 = lstm_with_proj.forward(simple_seq, return_sequences=True)

    # Different input should give different output
    simple_seq_2 = np.ones((1, 5, input_size)) * 0.5
    outputs_2 = lstm_with_proj.forward(simple_seq_2, return_sequences=True)

    assert not np.allclose(outputs_1, outputs_2), "Different inputs should produce different outputs!"
    print(f"   ✓ State evolves correctly with different inputs")

    # Test 7: Single time step processing
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

    # Summary
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
    # Run tests
    np.random.seed(42)  # For reproducibility
    model = test_lstm()

    print("\n" + "="*60)
    print("LSTM Baseline Ready for Comparison!")
    print("="*60)
