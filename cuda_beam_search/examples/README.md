# CUDA Beam Search Examples

This directory contains example implementations demonstrating how to use the CUDA-accelerated beam search and diverse beam search with language models.

## GPT-2 Generation Example

The `gpt2_generation.py` script demonstrates how to use both standard and diverse beam search with the GPT-2 model.

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Build CUDA extensions:
```bash
cd ..
python setup.py build_ext --inplace
cd examples
```

### Usage

1. Run the example:
```bash
python gpt2_generation.py
```

2. The script will:
   - Load the GPT-2 model and tokenizer
   - Generate text using standard beam search
   - Generate text using diverse beam search
   - Print the results

### Code Walkthrough

1. **Model Loading**:
```python
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to("cuda")
```

2. **Standard Beam Search**:
```python
standard_results = generate_with_beam_search(
    model=model,
    tokenizer=tokenizer,
    prompts=prompts,
    max_length=50,
    num_beams=5,
    num_return_sequences=2,
    batch_size=2,
    use_diverse=False
)
```

3. **Diverse Beam Search**:
```python
diverse_results = generate_with_beam_search(
    model=model,
    tokenizer=tokenizer,
    prompts=prompts,
    max_length=50,
    num_beams=6,
    num_return_sequences=2,
    batch_size=2,
    use_diverse=True,
    num_beam_groups=2,
    diversity_penalty=0.5
)
```

### Parameters

- `prompts`: List of input prompts
- `max_length`: Maximum sequence length
- `num_beams`: Number of beams for search
- `num_return_sequences`: Number of sequences to return per prompt
- `batch_size`: Number of prompts to process in parallel
- `temperature`: Sampling temperature
- `top_k`: Top-k sampling parameter
- `top_p`: Top-p (nucleus) sampling parameter
- `repetition_penalty`: Penalty for repeated tokens
- `no_repeat_ngram_size`: Size of n-grams to prevent repeating
- `use_diverse`: Whether to use diverse beam search
- `num_beam_groups`: Number of beam groups for diverse search
- `diversity_penalty`: Strength of diversity penalty

### Output

The script will output generated sequences for each prompt, showing both standard and diverse beam search results:

```
Standard Beam Search Results:

Prompt 1: The future of artificial intelligence
Sequence 1: The future of artificial intelligence is bright and full of potential...
Sequence 2: The future of artificial intelligence holds many possibilities...

Diverse Beam Search Results:

Prompt 1: The future of artificial intelligence
Sequence 1: The future of artificial intelligence is bright and full of potential...
Sequence 2: The future of artificial intelligence raises important ethical questions...
```

### Performance Tips

1. **Batch Size**:
   - Larger batch sizes improve GPU utilization
   - Optimal size depends on GPU memory
   - Start with small batches and increase

2. **Number of Beams**:
   - More beams increase quality but reduce speed
   - For diverse search, must be divisible by groups
   - Balance between quality and performance

3. **Diversity Parameters**:
   - Adjust `diversity_penalty` based on task
   - More groups increase diversity
   - Higher penalties increase diversity

### Troubleshooting

1. **CUDA Errors**:
   - Check GPU memory availability
   - Verify CUDA installation
   - Ensure proper device placement

2. **Performance Issues**:
   - Monitor GPU utilization
   - Check memory usage
   - Optimize batch size

3. **Quality Issues**:
   - Adjust temperature
   - Modify diversity penalty
   - Change number of beams

### Extending the Example

1. **Different Models**:
   - Replace GPT-2 with other models
   - Adjust tokenizer and model loading
   - Modify generation parameters

2. **Custom Parameters**:
   - Add new sampling methods
   - Implement custom scoring
   - Add new diversity metrics

3. **Batch Processing**:
   - Implement custom batching
   - Add progress tracking
   - Include error handling 