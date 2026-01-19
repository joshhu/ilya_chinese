# Synthetic Sequential Reasoning Dataset - Implementation Summary

**Task**: P1-T4 - Generate synthetic sequential reasoning dataset for Paper 18 (Relational RNN)

**Status**: Complete ✓

---

## Overview

This implementation provides three distinct sequential reasoning tasks designed to test memory and relational reasoning capabilities of neural network models. All tasks require the model to:
- **Remember** past events in a sequence
- **Track relationships** between entities
- **Reason** about temporal dependencies

---

## Task Descriptions

### Task 1: Object Tracking

**What it does**: Track multiple objects moving in a 2D grid over time and answer queries about their final positions.

**Why it requires relational reasoning**:
- Must maintain separate memory slots for each object
- Must track object identity across time steps
- Must reason about spatial relationships (positions in grid)
- Cannot solve through simple pattern matching - requires explicit memory

**Example**:
```
Objects move in 5x5 grid for 15 timesteps
- Object 0 moves: (1,2) -> (2,2) -> (2,3) -> ...
- Object 1 moves: (3,4) -> (3,3) -> (4,3) -> ...
- Object 2 moves: (0,0) -> (1,0) -> (1,1) -> ...

Query: "Where is Object 1?"
Answer: Final position of Object 1
```

**Data Shape**:
- Input: `(n_samples, seq_len+1, n_objects+2)`
  - One-hot object ID + normalized (x,y) coordinates
  - Last timestep is query (object ID only)
- Output: `(n_samples, 2)` - normalized (x,y) position

**Key Properties**:
- Deterministic: Same sequence always gives same answer
- Scalable: Can increase grid size, objects, or sequence length
- Tests memory: Must remember last position of queried object

---

### Task 2: Pair Matching

**What it does**: Show pairs of elements, then query one element and ask for its paired element.

**Why it requires relational reasoning**:
- Must bind pairs together in memory (relational structure)
- Must retrieve correct pair given partial information
- Must distinguish between multiple pairs shown in sequence
- Cannot use positional encoding alone - needs associative memory

**Example**:
```
Show pairs:
- Timestep 0-1: (8, 16) - element 8 paired with 16
- Timestep 2-3: (11, 6) - element 11 paired with 6

Query (timestep 4): "What was paired with 8?"
Answer: 16
```

**Data Shape**:
- Input: `(n_samples, seq_len, vocab_size+1)`
  - One-hot element encoding
  - +1 dimension for query marker
- Output: `(n_samples, vocab_size)` - one-hot answer

**Key Properties**:
- Tests associative memory: Must create and retrieve bindings
- Scalable: Can vary vocabulary size and number of pairs
- Relational: The relationship between pairs is key information

---

### Task 3: Simple bAbI-style QA

**What it does**: Track entities and their locations/properties over time, then answer questions requiring multi-hop reasoning.

**Why it requires relational reasoning**:
- Must track multiple entities simultaneously
- Must update entity states over time (locations change)
- Must perform multi-hop reasoning (e.g., "John has ball" + "John in kitchen" = "ball in kitchen")
- Requires understanding of entity-property-location relations

**Example**:
```
Facts:
1. Entity0 went to Loc3
2. Entity2 went to Loc3
3. Entity2 went to Loc3
4. Entity0 went to Loc3
5. Entity2 grabbed Object

Question: "Where is Object?"
Answer: Loc3 (because Entity2 has it and is in Loc3)
```

**Data Shape**:
- Input: `(n_samples, max_facts+1, n_entities+n_locations+3)`
  - One-hot entity + location encoding
  - Fact type markers (movement vs. grab)
  - Question marker
- Output: `(n_samples, n_locations)` - one-hot location answer

**Key Properties**:
- Multi-hop reasoning: Must combine multiple facts
- State tracking: Entity locations change over time
- Temporal ordering: Order of facts matters
- Relational graph: Entities, objects, and locations form a knowledge graph

---

## Implementation Details

### Core Functions

1. **`generate_object_tracking()`**: Generates object tracking sequences
2. **`generate_pair_matching()`**: Generates pair association sequences
3. **`generate_babi_simple()`**: Generates QA sequences with facts
4. **`create_train_test_split()`**: Splits data for training/testing
5. **`create_batches()`**: Creates mini-batches with optional shuffling
6. **`visualize_example()`**: Visualizes examples from each task type

### Data Utilities

- **Train/test splitting**: Configurable test ratio with reproducible seeding
- **Batch creation**: Efficient mini-batch generation with shuffling
- **Normalization**: Optional min-max or standard normalization

### Visualization

Each task has custom visualization showing:
- **Input sequence heatmap**: Shows temporal structure
- **Task-specific view**:
  - Tracking: Object trajectories on grid
  - Matching: Textual representation of pairs
  - QA: Textual representation of facts and questions

---

## Testing & Validation

### Solvability Tests
✓ All tasks verified to be deterministic and solvable
✓ Output positions/answers match expected values from sequence
✓ No random or inconsistent answers

### Statistical Analysis
- **Task 1**: Position distribution uniform across grid
- **Task 2**: All vocabulary elements used, balanced distribution
- **Task 3**: Location answers reasonably distributed

### Complexity Scaling
- **Task 1**: Complexity increases with number of objects
- **Task 2**: Difficulty scales with vocabulary size
- **Task 3**: Sequence length grows with number of facts

---

## Why These Tasks Require Relational Reasoning

### 1. Cannot be solved by simple pattern matching
- Answers depend on specific relationships in the sequence
- Same input patterns can have different answers in different contexts
- Requires maintaining structured representations

### 2. Require explicit memory
- Must remember information from earlier in sequence
- Cannot rely solely on local context
- Need to maintain multiple pieces of information simultaneously

### 3. Involve entity relationships
- **Tracking**: Object-position relationships
- **Matching**: Element-element bindings
- **QA**: Entity-location-object relations

### 4. Need temporal reasoning
- Order of events matters
- State changes over time
- Must integrate information across timesteps

---

## Generated Files

### Core Implementation
- **`reasoning_tasks.py`**: Main implementation (690 lines)
  - All 3 task generators
  - Data utilities
  - Visualization functions
  - Built-in testing

### Testing & Validation
- **`test_reasoning_tasks.py`**: Extended testing (252 lines)
  - Solvability demonstrations
  - Statistical analysis
  - Difficulty scaling analysis
  - Multiple example generation

### Visualizations
- **`task_tracking_example.png`**: Object tracking visualization
- **`task_matching_example.png`**: Pair matching visualization
- **`task_babi_example.png`**: QA task visualization
- **`task_difficulty_scaling.png`**: Complexity scaling plots
- **`tracking_ex[1-3].png`**: Additional tracking examples
- **`matching_ex[1-3].png`**: Additional matching examples
- **`babi_ex[1-3].png`**: Additional QA examples

---

## Usage Example

```python
from reasoning_tasks import (
    generate_object_tracking,
    generate_pair_matching,
    generate_babi_simple,
    create_train_test_split,
    create_batches
)

# Generate datasets
X_track, y_track, meta_track = generate_object_tracking(n_samples=1000)
X_match, y_match, meta_match = generate_pair_matching(n_samples=1000)
X_qa, y_qa, meta_qa = generate_babi_simple(n_samples=1000)

# Split into train/test
X_train, X_test, y_train, y_test = create_train_test_split(X_track, y_track)

# Create batches for training
for X_batch, y_batch in create_batches(X_train, y_train, batch_size=32):
    # Train your model here
    pass
```

---

## Next Steps

These datasets are ready to be used for:

1. **Training LSTM baseline** (P3-T1)
2. **Training Relational RNN** (P3-T2)
3. **Comparing performance** (P4-T1)
4. **Analyzing attention patterns** (P4-T2)

The tasks are designed to be:
- **Simple enough** to learn with reasonable compute
- **Complex enough** to differentiate architectures
- **Interpretable** for analysis and visualization
- **Scalable** for difficulty adjustment

---

## Quality Metrics

✓ **All tests pass**: Shapes, values, and distributions verified
✓ **Deterministic**: Same seed produces same data
✓ **Balanced**: No obvious biases in answer distribution
✓ **Varied**: Multiple examples show good diversity
✓ **Documented**: Comprehensive comments and docstrings
✓ **Visualized**: Clear plots for understanding tasks

---

## Conclusion

Successfully implemented three distinct sequential reasoning tasks that require:
- **Memory** to retain past events
- **Relational reasoning** to bind entities and properties
- **Temporal understanding** to process sequences correctly

All tasks are deterministic, solvable, scalable, and ready for model training and evaluation.
