from typing import Tuple
import torch
from ..beam_search import BeamSearchScorer

try:
    from .cuda_beam_search import process_diverse_beam_search
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

class DiverseBeamSearchScorer(BeamSearchScorer):
    def __init__(
        self,
        batch_size: int,
        num_beams: int,
        num_beam_groups: int,
        device: torch.device,
        length_penalty: float = 1.0,
        do_early_stopping: bool = False,
        num_beam_hyps_to_keep: int = 1,
        diversity_penalty: float = 0.5,
    ):
        super().__init__(
            batch_size=batch_size,
            num_beams=num_beams,
            device=device,
            length_penalty=length_penalty,
            do_early_stopping=do_early_stopping,
            num_beam_hyps_to_keep=num_beam_hyps_to_keep,
        )
        self.num_beam_groups = num_beam_groups
        self.diversity_penalty = diversity_penalty
        self._done = torch.zeros(batch_size, dtype=torch.bool, device=device)

    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        pad_token_id: int,
        eos_token_id: int,
    ) -> Tuple[torch.Tensor]:
        if CUDA_AVAILABLE and input_ids.is_cuda:
            next_beam_scores, next_beam_tokens, next_beam_indices = process_diverse_beam_search(
                input_ids,
                next_scores,
                next_tokens,
                next_indices,
                self.num_beam_groups,
                self.diversity_penalty,
                pad_token_id,
                eos_token_id
            )
            return next_beam_scores, next_beam_tokens, next_beam_indices
        else:
            return super().process(
                input_ids,
                next_scores,
                next_tokens,
                next_indices,
                pad_token_id,
                eos_token_id
            )

# Alias for backward compatibility
CUDADiverseBeamSearchScorer = DiverseBeamSearchScorer 