# CUDA Beam Search Implementation

This project implements CUDA-accelerated beam search algorithms, including both standard beam search and diverse beam search, based on the implementation from the Hugging Face Transformers library. The implementation is designed to be efficient and scalable for large language models.

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
└── docs/
    ├── implementation.md        # Implementation details for standard beam search
    └── diverse/
        └── implementation.md    # Implementation details for diverse beam search
```

## Prerequisites

- CUDA Toolkit (version 11.0 or higher)
- Python 3.8 or higher
- PyTorch with CUDA support
- NVIDIA GPU with compute capability 6.0 or higher

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cuda_beam_search.git
cd cuda_beam_search
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Build the CUDA extensions:
```bash
python setup.py build_ext --inplace
python setup_diverse.py build_ext --inplace
```

## Running the Tests

To verify both implementations, run the test suites:

```bash
# Test standard beam search
pytest tests/

# Test diverse beam search
pytest tests/diverse/
```

## Usage

### Standard Beam Search

```python
from cuda_beam_search import CUDABeamSearchScorer

# Initialize the beam search scorer
scorer = CUDABeamSearchScorer(
    batch_size=2,
    num_beams=3,
    device=torch.device("cuda"),
    length_penalty=1.0,
    do_early_stopping=False,
    num_beam_hyps_to_keep=1
)

# Process one step of beam search
next_beam_scores, next_beam_tokens, next_beam_indices = scorer.process(
    input_ids=input_ids,
    next_scores=next_scores,
    next_tokens=next_tokens,
    next_indices=next_indices,
    pad_token_id=0,
    eos_token_id=1
)
```

### Diverse Beam Search

```python
from cuda_beam_search.diverse import CUDADiverseBeamSearchScorer

# Initialize the diverse beam search scorer
scorer = CUDADiverseBeamSearchScorer(
    batch_size=2,
    num_beams=6,  # Must be divisible by num_beam_groups
    num_beam_groups=2,
    device=torch.device("cuda"),
    length_penalty=1.0,
    do_early_stopping=False,
    num_beam_hyps_to_keep=1,
    diversity_penalty=0.5  # Controls the strength of diversity
)

# Process one step of diverse beam search
next_beam_scores, next_beam_tokens, next_beam_indices = scorer.process(
    input_ids=input_ids,
    next_scores=next_scores,
    next_tokens=next_tokens,
    next_indices=next_indices,
    pad_token_id=0,
    eos_token_id=1
)
```

## Performance Comparison

Both CUDA implementations provide significant speedup compared to their CPU counterparts, especially for large batch sizes and number of beams. The exact performance improvement depends on your hardware configuration.

### Key Features

1. **Standard Beam Search**:
   - Efficient parallel processing of multiple beams
   - Memory-optimized data structures
   - Support for early stopping and length penalties

2. **Diverse Beam Search**:
   - Group-based beam search with diversity penalties
   - Parallel processing of beam groups
   - Configurable diversity strength
   - Efficient memory access patterns

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 