import torch
import pytest
from src.beam_search import BeamSearchScorer
from cuda_beam_search.diverse import CUDADiverseBeamSearchScorer

def test_beam_search_basic():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    num_beams = 3
    vocab_size = 10
    max_length = 5
    
    # Initialize beam search scorer
    scorer = BeamSearchScorer(
        batch_size=batch_size,
        num_beams=num_beams,
        device=device,
        length_penalty=1.0,
        do_early_stopping=False,
        num_beam_hyps_to_keep=1
    )
    
    # Create dummy input data
    input_ids = torch.randint(0, vocab_size, (batch_size * num_beams, max_length), device=device)
    next_scores = torch.randn((batch_size, vocab_size), device=device)
    next_tokens = torch.randint(0, vocab_size, (batch_size, vocab_size), device=device)
    next_indices = torch.randint(0, num_beams, (batch_size, vocab_size), device=device)
    
    # Test process step
    next_beam_scores, next_beam_tokens, next_beam_indices = scorer.process(
        input_ids=input_ids,
        next_scores=next_scores,
        next_tokens=next_tokens,
        next_indices=next_indices,
        pad_token_id=0,
        eos_token_id=1
    )
    
    # Verify shapes
    assert next_beam_scores.shape == (batch_size, num_beams)
    assert next_beam_tokens.shape == (batch_size, num_beams)
    assert next_beam_indices.shape == (batch_size, num_beams)
    
    # Test finalize step
    final_beam_scores = torch.randn((batch_size, num_beams), device=device)
    final_beam_tokens = torch.randint(0, vocab_size, (batch_size, num_beams), device=device)
    final_beam_indices = torch.randint(0, num_beams, (batch_size, num_beams), device=device)
    
    decoded, best_scores = scorer.finalize(
        input_ids=input_ids,
        final_beam_scores=final_beam_scores,
        final_beam_tokens=final_beam_tokens,
        final_beam_indices=final_beam_indices,
        pad_token_id=0,
        eos_token_id=1
    )
    
    # Verify shapes
    assert decoded.shape[0] == batch_size * scorer.num_beam_hyps_to_keep
    assert best_scores.shape[0] == batch_size * scorer.num_beam_hyps_to_keep

if __name__ == "__main__":
    pytest.main([__file__]) 

scorer = CUDADiverseBeamSearchScorer(
    batch_size=2,
    num_beams=6,
    num_beam_groups=2,
    device=torch.device("cuda"),
    diversity_penalty=0.5
) 