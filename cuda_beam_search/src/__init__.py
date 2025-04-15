from typing import Tuple, Optional
import torch
from .beam_search import BeamSearchScorer
from .diverse.beam_search import CUDADiverseBeamSearchScorer
try:
    from .cuda_beam_search import process_beam_search as process_beam_search_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

class CUDABeamSearchScorer(BeamSearchScorer):
    def __init__(
        self,
        batch_size: int,
        num_beams: int,
        device: torch.device,
        length_penalty: float = 1.0,
        temperature: float = 1.0,
        early_stopping: bool = False,
        max_steps: Optional[int] = None
    ):
        super().__init__(batch_size, num_beams, device)
        self.length_penalty = length_penalty
        self.temperature = temperature
        self.early_stopping = early_stopping
        self.max_steps = max_steps
        self._done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self._scores = torch.zeros((batch_size, num_beams), dtype=torch.float32, device=device)
        self._beam_indices = torch.zeros((batch_size, num_beams), dtype=torch.int64, device=device)
        self._beam_tokens = torch.zeros((batch_size, num_beams), dtype=torch.int64, device=device)
        self._step = 0

    def process(
        self,
        input_ids: torch.Tensor,
        next_scores: torch.Tensor,
        next_tokens: torch.Tensor,
        next_indices: torch.Tensor,
        pad_token_id: int,
        eos_token_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA is not available or the CUDA extension is not properly built.")
        
        if self.early_stopping and self._step >= self.max_steps:
            return self._scores, self._beam_tokens, self._beam_indices

        # Ensure tensors are on the correct device
        input_ids = input_ids.to(self.device)
        next_scores = next_scores.to(self.device)
        next_tokens = next_tokens.to(self.device)
        next_indices = next_indices.to(self.device)

        # Process with CUDA kernel
        next_beam_scores, next_beam_tokens, next_beam_indices = process_beam_search_cuda(
            input_ids,
            next_scores,
            next_tokens,
            next_indices,
            pad_token_id,
            eos_token_id,
            self.length_penalty,
            self.temperature
        )

        # Update internal state
        self._scores = next_beam_scores
        self._beam_tokens = next_beam_tokens
        self._beam_indices = next_beam_indices
        self._step += 1

        # Check for early stopping
        if self.early_stopping:
            eos_mask = (next_beam_tokens == eos_token_id)
            self._done = self._done | eos_mask.any(dim=1)
            if self._done.all():
                return self._scores, self._beam_tokens, self._beam_indices

        return next_beam_scores, next_beam_tokens, next_beam_indices

    def finalize(
        self,
        input_ids: torch.Tensor,
        final_beam_scores: torch.Tensor,
        final_beam_tokens: torch.Tensor,
        final_beam_indices: torch.Tensor,
        pad_token_id: int,
        eos_token_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA is not available or the CUDA extension is not properly built.")

        # Ensure tensors are on the correct device
        input_ids = input_ids.to(self.device)
        final_beam_scores = final_beam_scores.to(self.device)
        final_beam_tokens = final_beam_tokens.to(self.device)
        final_beam_indices = final_beam_indices.to(self.device)

        # Finalize with CUDA kernel
        return process_beam_search_cuda(
            input_ids,
            final_beam_scores,
            final_beam_tokens,
            final_beam_indices,
            pad_token_id,
            eos_token_id
        )

class CUDADiverseBeamSearchScorer(CUDADiverseBeamSearchScorer):
    def __init__(
        self,
        batch_size: int,
        num_beams: int,
        num_beam_groups: int,
        device: torch.device,
        diversity_penalty: float = 0.0,
        length_penalty: float = 1.0,
        temperature: float = 1.0,
        early_stopping: bool = False,
        max_steps: Optional[int] = None
    ):
        if num_beams % num_beam_groups != 0:
            raise ValueError("`num_beams` should be divisible by `num_beam_groups`")
        
        super().__init__(batch_size, num_beams, num_beam_groups, device, diversity_penalty, length_penalty, temperature, early_stopping, max_steps)

    def process(
        self,
        input_ids: torch.Tensor,
        next_scores: torch.Tensor,
        next_tokens: torch.Tensor,
        next_indices: torch.Tensor,
        pad_token_id: int,
        eos_token_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA is not available or the CUDA extension is not properly built.")
        
        if self.early_stopping and self._step >= self.max_steps:
            return self._scores, self._beam_tokens, self._beam_indices

        # Ensure tensors are on the correct device
        input_ids = input_ids.to(self.device)
        next_scores = next_scores.to(self.device)
        next_tokens = next_tokens.to(self.device)
        next_indices = next_indices.to(self.device)

        # Process with CUDA kernel
        next_beam_scores, next_beam_tokens, next_beam_indices = process_beam_search_cuda(
            input_ids,
            next_scores,
            next_tokens,
            next_indices,
            pad_token_id,
            eos_token_id,
            self.diversity_penalty,
            self.length_penalty,
            self.temperature
        )

        # Update internal state
        self._scores = next_beam_scores
        self._beam_tokens = next_beam_tokens
        self._beam_indices = next_beam_indices
        self._step += 1

        # Check for early stopping
        if self.early_stopping:
            eos_mask = (next_beam_tokens == eos_token_id)
            self._done = self._done | eos_mask.any(dim=1)
            if self._done.all():
                return self._scores, self._beam_tokens, self._beam_indices

        return next_beam_scores, next_beam_tokens, next_beam_indices

    def finalize(
        self,
        input_ids: torch.Tensor,
        final_beam_scores: torch.Tensor,
        final_beam_tokens: torch.Tensor,
        final_beam_indices: torch.Tensor,
        pad_token_id: int,
        eos_token_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA is not available or the CUDA extension is not properly built.")

        # Ensure tensors are on the correct device
        input_ids = input_ids.to(self.device)
        final_beam_scores = final_beam_scores.to(self.device)
        final_beam_tokens = final_beam_tokens.to(self.device)
        final_beam_indices = final_beam_indices.to(self.device)

        # Finalize with CUDA kernel
        return process_beam_search_cuda(
            input_ids,
            final_beam_scores,
            final_beam_tokens,
            final_beam_indices,
            pad_token_id,
            eos_token_id
        ) 