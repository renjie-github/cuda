import torch
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import BeamSearchScorer, DiverseBeamSearchScorer
from cuda_beam_search import CUDABeamSearchScorer
from cuda_beam_search.diverse import CUDADiverseBeamSearchScorer
from typing import List, Dict
import numpy as np

def benchmark_standard_beam_search(
    model,
    tokenizer,
    prompts: List[str],
    num_beams: int = 5,
    max_length: int = 50,
    batch_size: int = 1,
    num_runs: int = 10
) -> Dict[str, float]:
    """Benchmark standard beam search implementations."""
    results = {"transformers": [], "cuda": []}
    
    # Transformers implementation
    for _ in range(num_runs):
        start_time = time.time()
        
        # Initialize scorer
        scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            device=model.device
        )
        
        # Process prompts
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=model.device)
            beam_scores[:, 1:] = -1e9
            
            for _ in range(max_length):
                outputs = model(input_ids)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_scores = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
                next_token_scores = next_token_scores + beam_scores.view(-1).unsqueeze(1)
                
                next_token_scores = next_token_scores.view(batch_size, num_beams * next_token_scores.size(-1))
                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
                )
                
                next_indices = next_tokens // next_token_scores.size(-1)
                next_tokens = next_tokens % next_token_scores.size(-1)
                
                beam_scores, beam_next_tokens, beam_next_indices = scorer.process(
                    input_ids=input_ids,
                    next_scores=next_token_scores,
                    next_tokens=next_tokens,
                    next_indices=next_indices,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                input_ids = torch.cat([
                    input_ids[beam_next_indices, :],
                    beam_next_tokens.unsqueeze(1)
                ], dim=-1)
        
        results["transformers"].append(time.time() - start_time)
    
    # CUDA implementation
    for _ in range(num_runs):
        start_time = time.time()
        
        # Initialize scorer
        scorer = CUDABeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            device=model.device
        )
        
        # Process prompts
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=model.device)
            beam_scores[:, 1:] = -1e9
            
            for _ in range(max_length):
                outputs = model(input_ids)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_scores = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
                next_token_scores = next_token_scores + beam_scores.view(-1).unsqueeze(1)
                
                next_token_scores = next_token_scores.view(batch_size, num_beams * next_token_scores.size(-1))
                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
                )
                
                next_indices = next_tokens // next_token_scores.size(-1)
                next_tokens = next_tokens % next_token_scores.size(-1)
                
                beam_scores, beam_next_tokens, beam_next_indices = scorer.process(
                    input_ids=input_ids,
                    next_scores=next_token_scores,
                    next_tokens=next_tokens,
                    next_indices=next_indices,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                input_ids = torch.cat([
                    input_ids[beam_next_indices, :],
                    beam_next_tokens.unsqueeze(1)
                ], dim=-1)
        
        results["cuda"].append(time.time() - start_time)
    
    return {
        "transformers": {
            "mean": np.mean(results["transformers"]),
            "std": np.std(results["transformers"])
        },
        "cuda": {
            "mean": np.mean(results["cuda"]),
            "std": np.std(results["cuda"])
        }
    }

def benchmark_diverse_beam_search(
    model,
    tokenizer,
    prompts: List[str],
    num_beams: int = 6,
    num_beam_groups: int = 2,
    max_length: int = 50,
    batch_size: int = 1,
    num_runs: int = 10
) -> Dict[str, float]:
    """Benchmark diverse beam search implementations."""
    results = {"transformers": [], "cuda": []}
    
    # Transformers implementation
    for _ in range(num_runs):
        start_time = time.time()
        
        # Initialize scorer
        scorer = DiverseBeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            device=model.device
        )
        
        # Process prompts
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=model.device)
            beam_scores[:, 1:] = -1e9
            
            for _ in range(max_length):
                outputs = model(input_ids)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_scores = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
                next_token_scores = next_token_scores + beam_scores.view(-1).unsqueeze(1)
                
                next_token_scores = next_token_scores.view(batch_size, num_beams * next_token_scores.size(-1))
                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
                )
                
                next_indices = next_tokens // next_token_scores.size(-1)
                next_tokens = next_tokens % next_token_scores.size(-1)
                
                beam_scores, beam_next_tokens, beam_next_indices = scorer.process(
                    input_ids=input_ids,
                    next_scores=next_token_scores,
                    next_tokens=next_tokens,
                    next_indices=next_indices,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                input_ids = torch.cat([
                    input_ids[beam_next_indices, :],
                    beam_next_tokens.unsqueeze(1)
                ], dim=-1)
        
        results["transformers"].append(time.time() - start_time)
    
    # CUDA implementation
    for _ in range(num_runs):
        start_time = time.time()
        
        # Initialize scorer
        scorer = CUDADiverseBeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            device=model.device
        )
        
        # Process prompts
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=model.device)
            beam_scores[:, 1:] = -1e9
            
            for _ in range(max_length):
                outputs = model(input_ids)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_scores = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
                next_token_scores = next_token_scores + beam_scores.view(-1).unsqueeze(1)
                
                next_token_scores = next_token_scores.view(batch_size, num_beams * next_token_scores.size(-1))
                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
                )
                
                next_indices = next_tokens // next_token_scores.size(-1)
                next_tokens = next_tokens % next_token_scores.size(-1)
                
                beam_scores, beam_next_tokens, beam_next_indices = scorer.process(
                    input_ids=input_ids,
                    next_scores=next_token_scores,
                    next_tokens=next_tokens,
                    next_indices=next_indices,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                input_ids = torch.cat([
                    input_ids[beam_next_indices, :],
                    beam_next_tokens.unsqueeze(1)
                ], dim=-1)
        
        results["cuda"].append(time.time() - start_time)
    
    return {
        "transformers": {
            "mean": np.mean(results["transformers"]),
            "std": np.std(results["transformers"])
        },
        "cuda": {
            "mean": np.mean(results["cuda"]),
            "std": np.std(results["cuda"])
        }
    }

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
    
    # Benchmark standard beam search
    print("Benchmarking Standard Beam Search...")
    standard_results = benchmark_standard_beam_search(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        num_beams=5,
        max_length=50,
        batch_size=2
    )
    
    print("\nStandard Beam Search Results:")
    print(f"Transformers: {standard_results['transformers']['mean']:.4f} ± {standard_results['transformers']['std']:.4f} seconds")
    print(f"CUDA: {standard_results['cuda']['mean']:.4f} ± {standard_results['cuda']['std']:.4f} seconds")
    print(f"Speedup: {standard_results['transformers']['mean'] / standard_results['cuda']['mean']:.2f}x")
    
    # Benchmark diverse beam search
    print("\nBenchmarking Diverse Beam Search...")
    diverse_results = benchmark_diverse_beam_search(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        num_beams=6,
        num_beam_groups=2,
        max_length=50,
        batch_size=2
    )
    
    print("\nDiverse Beam Search Results:")
    print(f"Transformers: {diverse_results['transformers']['mean']:.4f} ± {diverse_results['transformers']['std']:.4f} seconds")
    print(f"CUDA: {diverse_results['cuda']['mean']:.4f} ± {diverse_results['cuda']['std']:.4f} seconds")
    print(f"Speedup: {diverse_results['transformers']['mean'] / diverse_results['cuda']['mean']:.2f}x")

if __name__ == "__main__":
    main() 