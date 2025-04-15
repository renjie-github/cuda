# Diverse Beam Search Implementation

This document provides detailed information about the CUDA-accelerated diverse beam search implementation.

## Algorithm Overview

Diverse beam search is an extension of standard beam search that promotes diversity among the generated sequences by:
1. Dividing beams into groups
2. Applying diversity penalties between groups
3. Maintaining group-specific scores and indices

## Implementation Details

### 1. CUDA Kernel Design

The diverse beam search kernel (`process_diverse_beam_search_kernel`) is designed with the following features:

```cuda
__global__ void process_diverse_beam_search_kernel(
    const int64_t* input_ids,
    const float* next_scores,
    const int64_t* next_tokens,
    const int64_t* next_indices,
    float* next_beam_scores,
    int64_t* next_beam_tokens,
    int64_t* next_beam_indices,
    const int batch_size,
    const int num_beams,
    const int num_beam_groups,
    const int vocab_size,
    const int pad_token_id,
    const int eos_token_id,
    const float diversity_penalty
)
```

Key design points:
- 2D grid layout: `(batch_size, num_beam_groups)`
- Thread block size: `num_beams / num_beam_groups`
- Shared memory for group scores
- Efficient diversity penalty computation

### 2. Memory Management

The implementation uses several memory optimization techniques:

1. **Group-based Memory Allocation**:
   - Each group has its own memory space
   - Contiguous memory layout for efficient access
   - Shared memory for temporary scores

2. **Diversity Penalty Application**:
   - Penalties applied in parallel
   - Efficient score updates
   - Memory-efficient token tracking

3. **Batch Processing**:
   - Parallel processing across batches
   - Efficient memory transfers
   - Optimized tensor operations

### 3. Performance Optimizations

1. **Thread Block Configuration**:
   ```python
   const dim3 blocks(batch_size, num_beam_groups);
   const dim3 threads(num_beams / num_beam_groups);
   ```

2. **Memory Access Patterns**:
   - Coalesced memory access
   - Shared memory usage
   - Efficient score updates

3. **Computation Optimization**:
   - Parallel group processing
   - Efficient diversity penalty application
   - Optimized beam selection

## Usage Guide

### 1. Initialization

```python
from cuda_beam_search.diverse import CUDADiverseBeamSearchScorer

scorer = CUDADiverseBeamSearchScorer(
    batch_size=4,
    num_beams=6,  # Must be divisible by num_beam_groups
    num_beam_groups=2,
    device=torch.device("cuda"),
    length_penalty=1.0,
    do_early_stopping=False,
    num_beam_hyps_to_keep=1,
    diversity_penalty=0.5
)
```

### 2. Generation Loop

```python
for step in range(max_length):
    # Get model outputs
    outputs = model(input_ids, attention_mask=attention_mask)
    next_token_logits = outputs.logits[:, -1, :]
    
    # Apply temperature and other sampling parameters
    next_token_logits = next_token_logits / temperature
    next_token_scores = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
    
    # Process with diverse beam search
    next_beam_scores, next_beam_tokens, next_beam_indices = scorer.process(
        input_ids=input_ids,
        next_scores=next_token_scores,
        next_tokens=next_tokens,
        next_indices=next_indices,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Update input_ids and attention_mask
    input_ids = torch.cat([
        input_ids[beam_next_indices, :],
        beam_next_tokens.unsqueeze(1)
    ], dim=-1)
    
    attention_mask = torch.cat([
        attention_mask[beam_next_indices, :],
        torch.ones((batch_size * num_beams, 1), device=device)
    ], dim=-1)
```

### 3. Finalization

```python
decoded, best_scores = scorer.finalize(
    input_ids=input_ids,
    final_beam_scores=beam_scores.view(batch_size, num_beams),
    final_beam_tokens=beam_next_tokens.view(batch_size, num_beams),
    final_beam_indices=beam_next_indices.view(batch_size, num_beams),
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id
)
```

## Performance Considerations

1. **Batch Size**:
   - Larger batch sizes improve GPU utilization
   - Memory usage increases linearly with batch size
   - Optimal batch size depends on GPU memory

2. **Number of Beams**:
   - Must be divisible by number of beam groups
   - More beams increase diversity but reduce speed
   - Balance between diversity and performance

3. **Diversity Penalty**:
   - Higher values increase diversity
   - Too high values may reduce quality
   - Optimal value depends on task

4. **Memory Usage**:
   - Monitor GPU memory usage
   - Adjust batch size and number of beams
   - Use memory-efficient operations

## Troubleshooting

1. **Build Issues**:
   - Ensure CUDA Toolkit version matches PyTorch
   - Check GPU compatibility
   - Verify CUDA installation

2. **Runtime Issues**:
   - Check tensor shapes and types
   - Verify device placement
   - Monitor GPU memory usage

3. **Performance Issues**:
   - Optimize batch size
   - Adjust number of beams
   - Fine-tune diversity penalty

## Future Improvements

1. **Algorithm Enhancements**:
   - Dynamic diversity penalty
   - Adaptive beam group size
   - Improved scoring mechanisms

2. **Performance Optimizations**:
   - Better memory management
   - Enhanced parallel processing
   - Optimized kernel design

3. **Feature Additions**:
   - Support for more sampling methods
   - Additional diversity metrics
   - Enhanced batch processing 