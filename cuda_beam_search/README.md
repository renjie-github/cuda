# CUDA Beam Search

A CUDA-accelerated implementation of beam search and diverse beam search for language models.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cuda_beam_search.git
cd cuda_beam_search
```

2. Install the package:
```bash
pip install -e .
```

## Usage

### Standard Beam Search

```python
from cuda_beam_search import CUDABeamSearchScorer
import torch

# Initialize the scorer
scorer = CUDABeamSearchScorer(
    batch_size=1,
    num_beams=5,
    device=torch.device("cuda"),
    length_penalty=1.0,
    do_early_stopping=False,
    num_beam_hyps_to_keep=1
)

# Use in your generation loop
next_beam_scores, next_beam_tokens, next_beam_indices = scorer.process(
    input_ids=input_ids,
    next_scores=next_scores,
    next_tokens=next_tokens,
    next_indices=next_indices,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id
)
```

### Diverse Beam Search

```python
from cuda_beam_search import CUDADiverseBeamSearchScorer
import torch

# Initialize the scorer
scorer = CUDADiverseBeamSearchScorer(
    batch_size=1,
    num_beams=6,  # Must be divisible by num_beam_groups
    num_beam_groups=2,
    device=torch.device("cuda"),
    length_penalty=1.0,
    do_early_stopping=False,
    num_beam_hyps_to_keep=1,
    diversity_penalty=0.5
)

# Use in your generation loop
next_beam_scores, next_beam_tokens, next_beam_indices = scorer.process(
    input_ids=input_ids,
    next_scores=next_scores,
    next_tokens=next_tokens,
    next_indices=next_indices,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id
)
```

## Example with GPT-2

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from cuda_beam_search import CUDADiverseBeamSearchScorer

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2').cuda()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Initialize diverse beam search
scorer = CUDADiverseBeamSearchScorer(
    batch_size=1,
    num_beams=6,
    num_beam_groups=2,
    device=torch.device("cuda"),
    diversity_penalty=0.5
)

# Prepare input
input_text = "The future of artificial intelligence"
input_ids = tokenizer.encode(input_text, return_tensors='pt').cuda()

# Generation loop
for _ in range(50):  # max_length
    outputs = model(input_ids)
    next_token_logits = outputs.logits[:, -1, :]
    
    # Get top tokens and scores
    next_token_scores = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
    next_token_scores, next_tokens = torch.topk(next_token_scores, 2 * scorer.num_beams, dim=1)
    next_indices = next_tokens // tokenizer.vocab_size
    
    # Process with beam search
    next_beam_scores, next_beam_tokens, next_beam_indices = scorer.process(
        input_ids=input_ids,
        next_scores=next_token_scores,
        next_tokens=next_tokens,
        next_indices=next_indices,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Update input_ids
    input_ids = torch.cat([
        input_ids[next_beam_indices.view(-1), :],
        next_beam_tokens.view(-1, 1)
    ], dim=-1)
    
    if scorer._done.all():
        break

# Decode the results
decoded, best_scores = scorer.finalize(
    input_ids=input_ids,
    final_beam_scores=next_beam_scores,
    final_beam_tokens=next_beam_tokens,
    final_beam_indices=next_beam_indices,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id
)

# Print the generated text
for sequence in decoded:
    print(tokenizer.decode(sequence, skip_special_tokens=True))
```

## Requirements

- CUDA-capable GPU
- PyTorch with CUDA support
- Python 3.6+
- NVIDIA CUDA Toolkit

## License

MIT License 