import torch
import pytest
from cuda_beam_search import CUDABeamSearchScorer

def test_beam_search_basic():
    batch_size = 2
    num_beams = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    scorer = CUDABeamSearchScorer(
        batch_size=batch_size,
        num_beams=num_beams,
        device=device
    )
    
    # Test input tensors
    input_ids = torch.randint(0, 100, (batch_size * num_beams, 5), device=device)
    next_scores = torch.randn((batch_size, num_beams * 100), device=device)
    next_tokens = torch.randint(0, 100, (batch_size, 2 * num_beams), device=device)
    next_indices = torch.randint(0, num_beams, (batch_size, 2 * num_beams), device=device)
    
    # Test process method
    beam_scores, beam_tokens, beam_indices = scorer.process(
        input_ids=input_ids,
        next_scores=next_scores,
        next_tokens=next_tokens,
        next_indices=next_indices,
        pad_token_id=0,
        eos_token_id=1
    )
    
    assert beam_scores.shape == (batch_size, num_beams)
    assert beam_tokens.shape == (batch_size, num_beams)
    assert beam_indices.shape == (batch_size, num_beams)

def test_beam_search_batch_processing():
    batch_size = 4
    num_beams = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    scorer = CUDABeamSearchScorer(
        batch_size=batch_size,
        num_beams=num_beams,
        device=device
    )
    
    # Test larger batch
    input_ids = torch.randint(0, 100, (batch_size * num_beams, 10), device=device)
    next_scores = torch.randn((batch_size, num_beams * 100), device=device)
    next_tokens = torch.randint(0, 100, (batch_size, 2 * num_beams), device=device)
    next_indices = torch.randint(0, num_beams, (batch_size, 2 * num_beams), device=device)
    
    beam_scores, beam_tokens, beam_indices = scorer.process(
        input_ids=input_ids,
        next_scores=next_scores,
        next_tokens=next_tokens,
        next_indices=next_indices,
        pad_token_id=0,
        eos_token_id=1
    )
    
    assert beam_scores.shape == (batch_size, num_beams)
    assert beam_tokens.shape == (batch_size, num_beams)
    assert beam_indices.shape == (batch_size, num_beams)

def test_beam_search_early_stopping():
    batch_size = 2
    num_beams = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    scorer = CUDABeamSearchScorer(
        batch_size=batch_size,
        num_beams=num_beams,
        device=device,
        do_early_stopping=True
    )
    
    input_ids = torch.randint(0, 100, (batch_size * num_beams, 5), device=device)
    next_scores = torch.randn((batch_size, num_beams * 100), device=device)
    next_tokens = torch.randint(0, 100, (batch_size, 2 * num_beams), device=device)
    next_indices = torch.randint(0, num_beams, (batch_size, 2 * num_beams), device=device)
    
    beam_scores, beam_tokens, beam_indices = scorer.process(
        input_ids=input_ids,
        next_scores=next_scores,
        next_tokens=next_tokens,
        next_indices=next_indices,
        pad_token_id=0,
        eos_token_id=1
    )
    
    assert beam_scores.shape == (batch_size, num_beams)
    assert beam_tokens.shape == (batch_size, num_beams)
    assert beam_indices.shape == (batch_size, num_beams) 