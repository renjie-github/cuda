import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from cuda_beam_search import CUDABeamSearchScorer
from cuda_beam_search.diverse import CUDADiverseBeamSearchScorer

def generate_with_beam_search(
    model,
    tokenizer,
    prompt,
    max_length=50,
    num_beams=5,
    num_return_sequences=1,
    device="cuda",
    use_diverse=False,
    num_beam_groups=2,
    diversity_penalty=0.5
):
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
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
    
    # Expand input_ids to match the number of beams
    input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, -1).reshape(batch_size * num_beams, -1)
    
    # Generation loop
    for step in range(max_length):
        # Get model outputs
        outputs = model(input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        
        # Apply temperature and get probabilities
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
        
        # Update input_ids
        beam_next_tokens = beam_next_tokens.view(-1)
        beam_next_indices = beam_next_indices.view(-1)
        
        input_ids = torch.cat([
            input_ids[beam_next_indices, :],
            beam_next_tokens.unsqueeze(1)
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
    generated_sequences = []
    for sequence in decoded:
        text = tokenizer.decode(sequence, skip_special_tokens=True)
        generated_sequences.append(text)
    
    return generated_sequences

def main():
    # Load model and tokenizer
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to("cuda")
    
    # Example prompt
    prompt = "The future of artificial intelligence"
    
    # Generate with standard beam search
    print("Standard Beam Search Results:")
    standard_results = generate_with_beam_search(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_length=50,
        num_beams=5,
        num_return_sequences=3,
        use_diverse=False
    )
    for i, result in enumerate(standard_results):
        print(f"Sequence {i+1}: {result}\n")
    
    # Generate with diverse beam search
    print("\nDiverse Beam Search Results:")
    diverse_results = generate_with_beam_search(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_length=50,
        num_beams=6,  # Must be divisible by num_beam_groups
        num_return_sequences=3,
        use_diverse=True,
        num_beam_groups=2,
        diversity_penalty=0.5
    )
    for i, result in enumerate(diverse_results):
        print(f"Sequence {i+1}: {result}\n")

if __name__ == "__main__":
    main() 