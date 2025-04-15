# CUDA Beam Search

A CUDA-accelerated implementation of beam search and diverse beam search algorithms, optimized for large language models.

## Features

### Standard Beam Search
- High-performance CUDA implementation
- Support for batch processing
- Configurable number of beams
- Length penalty for sequence normalization
- Temperature scaling for logits
- Early stopping support
- Memory-efficient implementation using shared memory
- Optimized memory access patterns
- Efficient thread utilization

### Diverse Beam Search
- Group-based beam processing
- Configurable diversity penalty
- Support for multiple beam groups
- Shared memory optimization
- Length penalty and temperature scaling
- Early stopping support
- Memory-efficient implementation
- Parallel processing of beam groups

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cuda-beam-search.git
cd cuda-beam-search
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Build the CUDA extensions:
```bash
python setup.py build_ext --inplace
```

## Usage

### Standard Beam Search

```python
from cuda_beam_search import CUDABeamSearchScorer
import torch

# Initialize the scorer
scorer = CUDABeamSearchScorer(
    batch_size=4,
    num_beams=5,
    device=torch.device("cuda"),
    length_penalty=1.0,  # Optional: length normalization
    temperature=1.0,     # Optional: temperature scaling
    early_stopping=True, # Optional: enable early stopping
    max_steps=100        # Optional: maximum generation steps
)

# Process a step
next_beam_scores, next_beam_tokens, next_beam_indices = scorer.process(
    input_ids,
    next_scores,
    next_tokens,
    next_indices,
    pad_token_id,
    eos_token_id
)

# Finalize the search
sequences, sequence_scores = scorer.finalize(
    input_ids,
    final_beam_scores,
    final_beam_tokens,
    final_beam_indices,
    pad_token_id,
    eos_token_id
)
```

### Diverse Beam Search

```python
from cuda_beam_search import CUDADiverseBeamSearchScorer
import torch

# Initialize the scorer
scorer = CUDADiverseBeamSearchScorer(
    batch_size=4,
    num_beams=6,
    num_beam_groups=3,
    device=torch.device("cuda"),
    diversity_penalty=0.5,  # Optional: diversity penalty
    length_penalty=1.0,     # Optional: length normalization
    temperature=1.0,        # Optional: temperature scaling
    early_stopping=True,    # Optional: enable early stopping
    max_steps=100           # Optional: maximum generation steps
)

# Process a step
next_beam_scores, next_beam_tokens, next_beam_indices = scorer.process(
    input_ids,
    next_scores,
    next_tokens,
    next_indices,
    pad_token_id,
    eos_token_id
)

# Finalize the search
sequences, sequence_scores = scorer.finalize(
    input_ids,
    final_beam_scores,
    final_beam_tokens,
    final_beam_indices,
    pad_token_id,
    eos_token_id
)
```

## Performance

The CUDA implementation provides significant speedup over CPU implementations, especially for:
- Large batch sizes
- Large number of beams
- Long sequences
- Multiple beam groups (diverse beam search)

### Performance Optimizations
- Shared memory usage for faster data access
- Memory coalescing for efficient memory access patterns
- Grid-stride loop pattern for optimal parallel processing
- Early stopping support to reduce unnecessary computation
- Temperature scaling and length penalty implemented in CUDA
- Efficient thread block configuration
- Optimized shared memory allocation

## Requirements

- Python 3.7+
- PyTorch 1.9.0+
- CUDA 11.0+
- NVIDIA GPU with compute capability 6.0+

## Project Structure

```
cuda_beam_search/
├── src/
│   ├── beam_search.py           # Standard beam search implementation
│   ├── cuda_beam_search.cu      # CUDA implementation for standard beam search
│   ├── diverse/
│   │   ├── beam_search.py       # Diverse beam search implementation
│   │   ├── cuda_beam_search.cu  # CUDA implementation for diverse beam search
│   │   └── __init__.py          # Diverse beam search module exports
│   └── __init__.py              # Main module exports
├── tests/
│   ├── test_beam_search.py      # Tests for standard beam search
│   └── diverse/
│       └── test_beam_search.py  # Tests for diverse beam search
├── docs/
│   ├── implementation.md        # Implementation details
│   └── diverse/
│       └── implementation.md    # Diverse beam search implementation details
└── examples/
    ├── gpt2_generation.py       # Example usage with GPT-2
    └── requirements.txt         # Example dependencies
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 