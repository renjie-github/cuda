import torch
import numpy as np
from typing import List, Tuple, Optional
from ..beam_search import BeamSearchScorer, BeamHypotheses

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
        diversity_penalty: float = 0.0,
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
        self.group_size = self.num_beams // self.num_beam_groups
        
        if self.num_beams % self.num_beam_groups != 0:
            raise ValueError(
                f"Number of beams ({self.num_beams}) should be divisible by number of beam groups ({self.num_beam_groups})"
            )
        
        self._beam_hyps = [
            [BeamHypotheses(
                num_beams=self.group_size,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping,
            ) for _ in range(self.num_beam_groups)]
            for _ in range(batch_size)
        ]

    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        pad_token_id: int,
        eos_token_id: int,
    ) -> Tuple[torch.Tensor]:
        cur_len = input_ids.shape[-1]
        batch_size = len(self._beam_hyps)
        
        next_beam_scores = torch.zeros((batch_size, self.num_beams), dtype=next_scores.dtype, device=self.device)
        next_beam_tokens = torch.zeros((batch_size, self.num_beams), dtype=next_tokens.dtype, device=self.device)
        next_beam_indices = torch.zeros((batch_size, self.num_beams), dtype=next_indices.dtype, device=self.device)

        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # Process each beam group separately
            for group_idx in range(self.num_beam_groups):
                group_start_idx = group_idx * self.group_size
                group_end_idx = (group_idx + 1) * self.group_size
                
                # Get scores for this group
                group_scores = next_scores[batch_idx].clone()
                
                # Apply diversity penalty
                if group_idx > 0 and self.diversity_penalty > 0.0:
                    for prev_group_idx in range(group_idx):
                        prev_group_start = prev_group_idx * self.group_size
                        prev_group_end = (prev_group_idx + 1) * self.group_size
                        
                        # Get tokens from previous group
                        prev_tokens = next_beam_tokens[batch_idx, prev_group_start:prev_group_end]
                        
                        # Apply diversity penalty to scores
                        for prev_token in prev_tokens:
                            # Find the index of the token in next_tokens
                            token_mask = next_tokens[batch_idx] == prev_token
                            if token_mask.any():
                                group_scores[token_mask] -= self.diversity_penalty
                
                # Select top tokens for this group
                group_scores, group_indices = torch.topk(group_scores, self.group_size)
                group_tokens = next_tokens[batch_idx, group_indices]
                
                # Store results
                next_beam_scores[batch_idx, group_start_idx:group_end_idx] = group_scores
                next_beam_tokens[batch_idx, group_start_idx:group_end_idx] = group_tokens
                next_beam_indices[batch_idx, group_start_idx:group_end_idx] = group_indices

        return next_beam_scores, next_beam_tokens, next_beam_indices

    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.LongTensor,
        final_beam_indices: torch.LongTensor,
        pad_token_id: int,
        eos_token_id: int,
    ) -> Tuple[torch.LongTensor]:
        batch_size = len(self._beam_hyps)

        # Finalize all open beam hypotheses
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                continue

            for group_idx in range(self.num_beam_groups):
                group_start_idx = group_idx * self.group_size
                group_end_idx = (group_idx + 1) * self.group_size
                
                for beam_id in range(self.group_size):
                    batch_beam_idx = batch_idx * self.num_beams + group_start_idx + beam_id
                    final_score = final_beam_scores[batch_idx, group_start_idx + beam_id].item()
                    final_tokens = input_ids[batch_beam_idx]
                    beam_hyp[group_idx].add(final_tokens, final_score)

        # Select the best hypotheses
        sent_lengths = torch.zeros(batch_size * self.num_beam_hyps_to_keep, dtype=torch.long, device=self.device)
        best = []
        best_scores = torch.zeros(batch_size * self.num_beam_hyps_to_keep, device=self.device, dtype=torch.float32)

        # Retrieve best hypotheses from each group
        for i, hypotheses in enumerate(self._beam_hyps):
            group_best = []
            for group_hyp in hypotheses:
                sorted_hyps = sorted(group_hyp.beams, key=lambda x: x[0])
                if sorted_hyps:
                    group_best.append(sorted_hyps.pop())
            
            # Sort all group bests and take top k
            sorted_group_best = sorted(group_best, key=lambda x: x[0], reverse=True)
            for j in range(min(self.num_beam_hyps_to_keep, len(sorted_group_best))):
                best_hyp_tuple = sorted_group_best[j]
                best_score = best_hyp_tuple[0]
                best_hyp = best_hyp_tuple[1]
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)
                best.append(best_hyp)
                best_scores[i * self.num_beam_hyps_to_keep + j] = best_score

        # Prepare for adding eos
        sent_max_len = min(sent_lengths.max().item() + 1, 1024)
        decoded = torch.zeros((batch_size * self.num_beam_hyps_to_keep, sent_max_len), dtype=torch.long, device=self.device)
        
        # Shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            decoded.fill_(pad_token_id)

        # Fill with hypotheses and eos_token_id if the latter fits in
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < sent_max_len:
                decoded[i, sent_lengths[i]] = eos_token_id

        return decoded, best_scores 