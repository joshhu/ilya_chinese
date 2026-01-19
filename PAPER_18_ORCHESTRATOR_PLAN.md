# Paper 18: Relational RNN - Orchestrator Implementation Plan

## Overview

**Paper**: Relational Recurrent Neural Networks (Santoro et al.)
**Status**: Not yet implemented (8/30 remaining)
**Difficulty**: Intermediate
**Key Concepts**: Relational Memory, Self-Attention in RNN, Sequential Reasoning

## Implementation Objectives

1. Implement multi-head dot-product attention for memory
2. Build relational memory core
3. Create sequential reasoning tasks
4. Compare with standard LSTM baseline
5. Visualize memory interactions and attention patterns

---

## Atomic Task Breakdown

### Phase 1: Foundation & Setup (4 tasks - Run in Parallel)

**P1-T1**: Create notebook structure and imports
- Create `18_relational_rnn.ipynb`
- Add standard imports (numpy, matplotlib, scipy)
- Create cell structure with markdown headings
- Add paper citation and overview
- Deliverable: Empty notebook with structure

**P1-T2**: Implement multi-head dot-product attention mechanism
- Implement scaled dot-product attention function
- Implement multi-head attention wrapper
- Add query, key, value projections
- Include attention score calculation
- Deliverable: `multi_head_attention(Q, K, V, num_heads)` function

**P1-T3**: Implement standard LSTM baseline for comparison
- Create LSTM cell with gates (forget, input, output)
- Implement forward pass for sequences
- Add parameter initialization
- Include hidden state management
- Deliverable: `LSTM` class with forward method

**P1-T4**: Generate synthetic sequential reasoning dataset
- Create task: sort sequences with memory requirements
- Generate relational reasoning tasks (e.g., match pairs, track objects)
- Create simple bAbI-style QA tasks
- Add data preprocessing utilities
- Deliverable: `generate_reasoning_data()` function returning train/test splits

---

### Phase 2: Core Relational Memory Implementation (3 tasks - Run in Parallel)

**Dependencies**: Requires P1-T2 (attention mechanism)

**P2-T1**: Implement relational memory core module
- Build memory module using multi-head attention
- Implement memory update mechanism
- Add residual connections and layer norm
- Include memory row interactions via self-attention
- Deliverable: `RelationalMemory` class with `forward(input, memory)` method

**P2-T2**: Build relational RNN cell combining LSTM + relational memory
- Integrate LSTM hidden state with relational memory
- Implement gating between LSTM and memory
- Add memory read/write operations
- Include proper state management
- Deliverable: `RelationalRNNCell` class

**P2-T3**: Implement training utilities and loss functions
- Create sequence loss calculation
- Add gradient clipping utilities
- Implement learning rate schedule
- Add early stopping logic
- Deliverable: `train_step()` and `evaluate()` functions

---

### Phase 3: Training & Baseline Comparison (2 tasks - Run in Parallel)

**Dependencies**: Requires P1-T3, P1-T4, P2-T1, P2-T2, P2-T3

**P3-T1**: Train standard LSTM baseline
- Train LSTM on reasoning tasks
- Log training curves (loss, accuracy)
- Save best model parameters
- Record final test performance
- Deliverable: Trained LSTM baseline + performance metrics

**P3-T2**: Train relational RNN model
- Train Relational RNN on same tasks
- Match hyperparameters with baseline
- Log training curves
- Save best model parameters
- Deliverable: Trained Relational RNN + performance metrics

---

### Phase 4: Evaluation & Visualization (4 tasks - Run in Parallel)

**Dependencies**: Requires P3-T1, P3-T2

**P4-T1**: Generate comparative performance plots
- Plot training curves (LSTM vs Relational RNN)
- Create accuracy comparison bar charts
- Show convergence speed analysis
- Generate sample efficiency plots
- Deliverable: Performance comparison visualizations

**P4-T2**: Visualize attention patterns and memory interactions
- Extract attention weights from trained model
- Create attention heatmaps over time
- Visualize memory evolution during inference
- Show which memory slots are used when
- Deliverable: Attention visualization plots

**P4-T3**: Analyze reasoning capabilities
- Test on held-out complex reasoning examples
- Show failure cases for both models
- Demonstrate where relational memory helps
- Include qualitative analysis
- Deliverable: Reasoning analysis section with examples

**P4-T4**: Create ablation studies
- Test different numbers of attention heads (1, 2, 4, 8)
- Vary memory size
- Compare with/without residual connections
- Show impact of each component
- Deliverable: Ablation study results and plots

---

### Phase 5: Documentation & Polish (4 tasks - Run in Parallel)

**Dependencies**: Requires all previous phases

**P5-T1**: Write comprehensive markdown explanations
- Add theoretical background on relational reasoning
- Explain attention mechanism intuition
- Document architecture choices
- Include mathematical formulations
- Deliverable: Complete markdown documentation cells

**P5-T2**: Add code documentation and comments
- Document all functions with docstrings
- Add inline comments for complex operations
- Include shape annotations
- Add usage examples
- Deliverable: Fully documented code

**P5-T3**: Create summary and key insights section
- Summarize main findings
- Compare with paper results
- List key takeaways
- Suggest extensions and next steps
- Deliverable: Conclusion section

**P5-T4**: Run final tests and create checkpoint
- Execute full notebook end-to-end
- Verify all visualizations render correctly
- Check for errors and warnings
- Create clean output version
- Deliverable: Tested, error-free notebook

---

## Phase Summary

### Phase 1: Foundation & Setup
- **Tasks**: 4 (all parallel)
- **Dependencies**: None
- **Estimated Parallelism**: 4 subagents
- **Output**: Notebook structure, attention mechanism, LSTM baseline, synthetic data

### Phase 2: Core Implementation
- **Tasks**: 3 (all parallel after P1-T2)
- **Dependencies**: P1-T2
- **Estimated Parallelism**: 3 subagents
- **Output**: Relational memory core, Relational RNN cell, training utilities

### Phase 3: Training
- **Tasks**: 2 (parallel)
- **Dependencies**: P1-T3, P1-T4, P2-T1, P2-T2, P2-T3
- **Estimated Parallelism**: 2 subagents
- **Output**: Trained models (baseline and relational)

### Phase 4: Evaluation
- **Tasks**: 4 (all parallel)
- **Dependencies**: P3-T1, P3-T2
- **Estimated Parallelism**: 4 subagents
- **Output**: Visualizations, analysis, ablations

### Phase 5: Documentation
- **Tasks**: 4 (all parallel)
- **Dependencies**: All previous phases
- **Estimated Parallelism**: 4 subagents
- **Output**: Complete, documented, tested notebook

---

## Total Breakdown

- **Total Atomic Tasks**: 17
- **Total Phases**: 5
- **Maximum Parallelism**: 4 subagents (Phase 1, 4, 5)
- **Sequential Dependencies**: Phase 2 → Phase 3 → Phase 4 → Phase 5

---

## Success Criteria

Each subagent must deliver:

1. **Functional code**: Runs without errors
2. **Summary**: What was implemented and how
3. **Learnings**: Any deviations, design decisions, or insights
4. **Tests passed**: Code executes successfully
5. **Commit**: Clear commit message with task description

## Failure Protocol

- If a task fails 3 times, escalate to orchestrator
- Orchestrator will:
  1. Analyze root cause
  2. Replan task with simplified scope or alternative approach
  3. Potentially merge/split tasks
  4. Reassign to different subagent

---

## Quality Standards

- **Code**: NumPy-only (no PyTorch/TensorFlow)
- **Data**: Self-contained synthetic data generation
- **Visualizations**: Clear, publication-quality plots
- **Documentation**: Explain "why" not just "what"
- **Testing**: Verify shapes, ranges, convergence

---

## Implementation Notes

### Multi-Head Attention
```python
# Expected signature
def multi_head_attention(Q, K, V, num_heads=4):
    """
    Q: queries (batch, seq_len, d_model)
    K: keys (batch, seq_len, d_model)
    V: values (batch, seq_len, d_model)
    Returns: attended output (batch, seq_len, d_model)
    """
```

### Relational Memory
- Memory slots: 4-8 slots
- Each slot is a vector (e.g., 64-128 dim)
- Self-attention across memory slots
- Gates to update memory

### Sequential Reasoning Tasks
- **Task 1**: Track object positions over time
- **Task 2**: Remember and compare pairs
- **Task 3**: Simple bAbI-style QA (2-3 supporting facts)

### Comparison Metrics
- Accuracy on reasoning tasks
- Training convergence speed
- Sample efficiency
- Memory utilization

---

## Git Commit Strategy

After each task completion:

```
WIP: [Phase X Task Y] Brief description

- What was implemented
- Any design decisions
- Known limitations

Tests: Pass/Fail
Lint: Pass/Fail
```

Final commit:
```
feat: Implement Paper 18 - Relational RNN

Complete implementation of Relational Recurrent Neural Networks
- Multi-head attention for memory
- Relational memory core
- Sequential reasoning tasks
- Comparison with LSTM baseline
- Visualizations and ablations

Closes #18 (if issue tracking used)
```

---

## Begin Implementation

**Orchestrator**: Start with Phase 1, launch 4 parallel subagents for tasks P1-T1 through P1-T4.
