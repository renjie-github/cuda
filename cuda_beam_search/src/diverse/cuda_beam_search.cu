#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void process_diverse_beam_search_kernel(
    const int64_t* input_ids,
    const float* next_scores,
    const int64_t* next_tokens,
    const int64_t* next_indices,
    float* next_beam_scores,
    int64_t* next_beam_tokens,
    int64_t* next_beam_indices,
    const int batch_size,
    const int num_beams,
    const int num_beam_groups,
    const int vocab_size,
    const int pad_token_id,
    const int eos_token_id,
    const float diversity_penalty
) {
    const int batch_idx = blockIdx.x;
    const int group_idx = blockIdx.y;
    const int beam_idx = threadIdx.x;
    
    const int group_size = num_beams / num_beam_groups;
    if (beam_idx >= group_size) return;
    
    // Get scores for this group
    float* group_scores = new float[vocab_size];
    for (int i = 0; i < vocab_size; i++) {
        group_scores[i] = next_scores[batch_idx * vocab_size + i];
    }
    
    // Apply diversity penalty
    if (group_idx > 0 && diversity_penalty > 0.0) {
        for (int prev_group = 0; prev_group < group_idx; prev_group++) {
            for (int prev_beam = 0; prev_beam < group_size; prev_beam++) {
                const int prev_idx = batch_idx * num_beams + prev_group * group_size + prev_beam;
                const int64_t prev_token = next_beam_tokens[prev_idx];
                
                // Find where this token appears in next_tokens
                for (int i = 0; i < vocab_size; i++) {
                    if (next_tokens[batch_idx * vocab_size + i] == prev_token) {
                        group_scores[i] -= diversity_penalty;
                    }
                }
            }
        }
    }
    
    // Find the best token for this beam
    float best_score = -1e9;
    int64_t best_token = pad_token_id;
    int64_t best_index = 0;
    
    for (int i = 0; i < vocab_size; i++) {
        if (group_scores[i] > best_score) {
            best_score = group_scores[i];
            best_token = next_tokens[batch_idx * vocab_size + i];
            best_index = next_indices[batch_idx * vocab_size + i];
        }
    }
    
    // Store the results
    const int out_idx = batch_idx * num_beams + group_idx * group_size + beam_idx;
    next_beam_scores[out_idx] = best_score;
    next_beam_tokens[out_idx] = best_token;
    next_beam_indices[out_idx] = best_index;
    
    delete[] group_scores;
}

std::vector<torch::Tensor> process_diverse_beam_search_cuda(
    torch::Tensor input_ids,
    torch::Tensor next_scores,
    torch::Tensor next_tokens,
    torch::Tensor next_indices,
    int num_beam_groups,
    float diversity_penalty,
    int pad_token_id,
    int eos_token_id
) {
    CHECK_INPUT(input_ids);
    CHECK_INPUT(next_scores);
    CHECK_INPUT(next_tokens);
    CHECK_INPUT(next_indices);
    
    const int batch_size = next_scores.size(0);
    const int num_beams = input_ids.size(0) / batch_size;
    const int vocab_size = next_scores.size(1);
    
    auto next_beam_scores = torch::zeros({batch_size, num_beams}, 
        torch::dtype(torch::kFloat32).device(input_ids.device()));
    auto next_beam_tokens = torch::zeros({batch_size, num_beams}, 
        torch::dtype(torch::kInt64).device(input_ids.device()));
    auto next_beam_indices = torch::zeros({batch_size, num_beams}, 
        torch::dtype(torch::kInt64).device(input_ids.device()));
    
    const dim3 blocks(batch_size, num_beam_groups);
    const dim3 threads(num_beams / num_beam_groups);
    
    process_diverse_beam_search_kernel<<<blocks, threads>>>(
        input_ids.data_ptr<int64_t>(),
        next_scores.data_ptr<float>(),
        next_tokens.data_ptr<int64_t>(),
        next_indices.data_ptr<int64_t>(),
        next_beam_scores.data_ptr<float>(),
        next_beam_tokens.data_ptr<int64_t>(),
        next_beam_indices.data_ptr<int64_t>(),
        batch_size,
        num_beams,
        num_beam_groups,
        vocab_size,
        pad_token_id,
        eos_token_id,
        diversity_penalty
    );
    
    return {next_beam_scores, next_beam_tokens, next_beam_indices};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("process_diverse_beam_search", &process_diverse_beam_search_cuda, "Process diverse beam search (CUDA)");
} 