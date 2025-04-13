import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from cuda_beam_search import CUDABeamSearchScorer
from cuda_beam_search.diverse import CUDADiverseBeamSearchScorer
from typing import List, Union, Optional
import numpy as np

def generate_with_beam_search(
    model,
    tokenizer,
    prompts: Union[str, List[str]],
    max_length: int = 50,
    num_beams: int = 5,
    num_return_sequences: int = 1,
    device: str = "cuda",
    use_diverse: bool = False,
    num_beam_groups: int = 2,
    diversity_penalty: float = 0.5,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    batch_size: int = 1,
    pad_to_multiple_of: Optional[int] = None,
) -> List[List[str]]:
    """
    Generate text using beam search with batch processing support.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: Single prompt string or list of prompts
        max_length: Maximum length of generated sequences
        num_beams: Number of beams for beam search
        num_return_sequences: Number of sequences to return per prompt
        device: Device to run on
        use_diverse: Whether to use diverse beam search
        num_beam_groups: Number of beam groups for diverse beam search
        diversity_penalty: Strength of diversity penalty
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        repetition_penalty: Penalty for repeated tokens
        no_repeat_ngram_size: Size of n-grams to prevent repeating
        batch_size: Number of prompts to process in parallel
        pad_to_multiple_of: Pad sequences to multiple of this number
    
    Returns:
        List of lists of generated sequences, one list per input prompt
    """
    # Convert single prompt to list
    if isinstance(prompts, str):
        prompts = [prompts]
    
    # Process prompts in batches
    all_generated_sequences = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        # Encode the batch of prompts
        encodings = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            pad_to_multiple_of=pad_to_multiple_of
        )
        input_ids = encodings.input_ids.to(device)
        attention_mask = encodings.attention_mask.to(device)
        batch_size = input_ids.shape[0]
        
        # Initialize the appropriate beam search scorer
        if use_diverse:
            scorer = CUDADiverseBeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                num_beam_groups=num_beam_groups,
                device=torch.device(device),
                length_penalty=1.0,
                do_early_stopping=False,
                num_beam_hyps_to_keep=num_return_sequences,
                diversity_penalty=diversity_penalty
            )
        else:
            scorer = CUDABeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device=torch.device(device),
                length_penalty=1.0,
                do_early_stopping=False,
                num_beam_hyps_to_keep=num_return_sequences
            )
        
        # Initialize beam search variables
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)
        
        # Expand input_ids and attention_mask to match the number of beams
        input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, -1).reshape(batch_size * num_beams, -1)
        attention_mask = attention_mask.unsqueeze(1).expand(batch_size, num_beams, -1).reshape(batch_size * num_beams, -1)
        
        # Track generated sequences for n-gram repetition penalty
        generated_sequences = [[] for _ in range(batch_size * num_beams)]
        
        # Generation loop
        for step in range(max_length):
            # Get model outputs
            outputs = model(input_ids, attention_mask=attention_mask)
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size * num_beams):
                    for previous_token in set(generated_sequences[i]):
                        next_token_logits[i, previous_token] /= repetition_penalty
            
            # Apply top-k and top-p filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Get probabilities
            next_token_scores = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
            
            # Reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
            
            # Get top tokens
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )
            
            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size
            
            # Process with beam search
            beam_scores, beam_next_tokens, beam_next_indices = scorer.process(
                input_ids=input_ids,
                next_scores=next_token_scores,
                next_tokens=next_tokens,
                next_indices=next_indices,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Update input_ids and attention_mask
            beam_next_tokens = beam_next_tokens.view(-1)
            beam_next_indices = beam_next_indices.view(-1)
            
            # Update generated sequences
            for i in range(batch_size * num_beams):
                generated_sequences[i].append(beam_next_tokens[i].item())
            
            input_ids = torch.cat([
                input_ids[beam_next_indices, :],
                beam_next_tokens.unsqueeze(1)
            ], dim=-1)
            
            attention_mask = torch.cat([
                attention_mask[beam_next_indices, :],
                torch.ones((batch_size * num_beams, 1), device=device)
            ], dim=-1)
            
            # Check if all beams are done
            if scorer._done.all():
                break
        
        # Finalize beam search
        decoded, best_scores = scorer.finalize(
            input_ids=input_ids,
            final_beam_scores=beam_scores.view(batch_size, num_beams),
            final_beam_tokens=beam_next_tokens.view(batch_size, num_beams),
            final_beam_indices=beam_next_indices.view(batch_size, num_beams),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # Decode the generated sequences
        batch_generated_sequences = []
        for sequence in decoded:
            text = tokenizer.decode(sequence, skip_special_tokens=True)
            batch_generated_sequences.append(text)
        
        # Group sequences by prompt
        for j in range(batch_size):
            start_idx = j * num_return_sequences
            end_idx = start_idx + num_return_sequences
            all_generated_sequences.append(batch_generated_sequences[start_idx:end_idx])
    
    return all_generated_sequences

def main():
    # Load model and tokenizer
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to("cuda")
    
    # Example prompts
    prompts = [
        "The future of artificial intelligence",
        "Climate change is a pressing issue",
        "The impact of technology on society"
    ]
    
    # Generate with standard beam search
    print("Standard Beam Search Results:")
    standard_results = generate_with_beam_search(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_length=50,
        num_beams=5,
        num_return_sequences=2,
        batch_size=2,  # Process 2 prompts at a time
        use_diverse=False
    )
    
    for i, (prompt, results) in enumerate(zip(prompts, standard_results)):
        print(f"\nPrompt {i+1}: {prompt}")
        for j, result in enumerate(results):
            print(f"Sequence {j+1}: {result}")
    
    # Generate with diverse beam search
    print("\nDiverse Beam Search Results:")
    diverse_results = generate_with_beam_search(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_length=50,
        num_beams=6,  # Must be divisible by num_beam_groups
        num_return_sequences=2,
        batch_size=2,  # Process 2 prompts at a time
        use_diverse=True,
        num_beam_groups=2,
        diversity_penalty=0.5
    )
    
    for i, (prompt, results) in enumerate(zip(prompts, diverse_results)):
        print(f"\nPrompt {i+1}: {prompt}")
        for j, result in enumerate(results):
            print(f"Sequence {j+1}: {result}")

if __name__ == "__main__":
    main() 