import torch
from .beam_search import BeamSearchScorer
try:
    from .cuda_beam_search import process_beam_search as process_beam_search_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

class CUDABeamSearchScorer(BeamSearchScorer):
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
            next_beam_scores, next_beam_tokens, next_beam_indices = process_beam_search_cuda(
                input_ids,
                next_scores,
                next_tokens,
                next_indices,
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