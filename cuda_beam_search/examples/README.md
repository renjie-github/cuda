# GPT-2 Generation with CUDA Beam Search

This example demonstrates how to use the CUDA-accelerated beam search implementations with GPT-2 for text generation.

## Features

- Integration with GPT-2 model from Hugging Face Transformers
- Support for both standard and diverse beam search
- CUDA-accelerated generation
- Configurable generation parameters

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Make sure you have built the CUDA extensions:
```bash
cd ..
python setup.py build_ext --inplace
python setup_diverse.py build_ext --inplace
cd examples
```

## Usage

Run the example script:
```bash
python gpt2_generation.py
```

The script will:
1. Load the GPT-2 model and tokenizer
2. Generate text using standard beam search
3. Generate text using diverse beam search
4. Print the results for comparison

## Parameters

The `generate_with_beam_search` function accepts the following parameters:

- `model`: The GPT-2 model instance
- `tokenizer`: The GPT-2 tokenizer instance
- `prompt`: The input text prompt
- `max_length`: Maximum length of generated sequences (default: 50)
- `num_beams`: Number of beams for beam search (default: 5)
- `num_return_sequences`: Number of sequences to return (default: 1)
- `device`: Device to run on (default: "cuda")
- `use_diverse`: Whether to use diverse beam search (default: False)
- `num_beam_groups`: Number of beam groups for diverse beam search (default: 2)
- `diversity_penalty`: Strength of diversity penalty (default: 0.5)

## Example Output

The script will generate output similar to:

```
Standard Beam Search Results:
Sequence 1: The future of artificial intelligence is bright and promising. With advances in machine learning...

Sequence 2: The future of artificial intelligence holds great potential for transforming various industries...

Sequence 3: The future of artificial intelligence will be shaped by continued research and development...

Diverse Beam Search Results:
Sequence 1: The future of artificial intelligence is bright and promising. With advances in machine learning...

Sequence 2: The future of artificial intelligence presents both opportunities and challenges. Ethical considerations...

Sequence 3: The future of artificial intelligence could revolutionize healthcare, transportation, and other sectors...
```

## Performance

The CUDA implementation provides significant speedup compared to the CPU implementation, especially for:
- Large batch sizes
- Many beams
- Long sequences
- Multiple beam groups (in diverse beam search)

## Notes

- The diverse beam search requires the number of beams to be divisible by the number of beam groups
- Higher diversity penalty values will result in more diverse outputs
- The example uses GPT-2 small, but the code can be adapted for other models 