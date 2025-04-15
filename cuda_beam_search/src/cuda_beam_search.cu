#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#define CHECK_CUDA_DETAILED(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor. Got device: " + x.device().str())
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA_DETAILED(x); CHECK_CONTIGUOUS(x)

// Constants for optimal performance
constexpr int MAX_BLOCK_SIZE = 1024;
constexpr int SHARED_MEM_SIZE = 32768; // 32KB shared memory

__global__ void process_beam_search_kernel(
    const int64_t* input_ids,
    const float* next_scores,
    const int64_t* next_tokens,
    const int64_t* next_indices,
    float* next_beam_scores,
    int64_t* next_beam_tokens,
    int64_t* next_beam_indices,
    const int batch_size,
    const int num_beams,
    const int vocab_size,
    const int pad_token_id,
    const int eos_token_id,
    const float length_penalty,
    const float temperature
) {
    extern __shared__ float shared_scores[];
    const int batch_idx = blockIdx.x;
    const int beam_idx = threadIdx.x;
    
    if (beam_idx >= num_beams) return;
    
    // Use shared memory for better performance
    float* local_scores = &shared_scores[beam_idx * vocab_size];
    
    // Load scores into shared memory
    for (int i = beam_idx; i < vocab_size; i += blockDim.x) {
        const int token_idx = batch_idx * vocab_size + i;
        local_scores[i] = next_scores[token_idx] / temperature; // Apply temperature scaling
    }
    __syncthreads();
    
    // Find the best token for this beam
    float best_score = -1e9;
    int64_t best_token = pad_token_id;
    int64_t best_index = 0;
    
    for (int i = 0; i < vocab_size; i++) {
        const float score = local_scores[i];
        if (score > best_score) {
            best_score = score;
            best_token = next_tokens[batch_idx * vocab_size + i];
            best_index = next_indices[batch_idx * vocab_size + i];
        }
    }
    
    // Apply length penalty
    best_score = best_score / powf((5.0f + 1.0f) / 6.0f, length_penalty);
    
    // Store the results
    const int out_idx = batch_idx * num_beams + beam_idx;
    next_beam_scores[out_idx] = best_score;
    next_beam_tokens[out_idx] = best_token;
    next_beam_indices[out_idx] = best_index;
}

std::vector<torch::Tensor> process_beam_search_cuda(
    torch::Tensor input_ids,
    torch::Tensor next_scores,
    torch::Tensor next_tokens,
    torch::Tensor next_indices,
    int pad_token_id,
    int eos_token_id,
    float length_penalty = 1.0f,
    float temperature = 1.0f
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
    
    const dim3 blocks(batch_size);
    const dim3 threads(num_beams);
    const size_t shared_mem_size = num_beams * vocab_size * sizeof(float);
    
    process_beam_search_kernel<<<blocks, threads, shared_mem_size>>>(
        input_ids.data_ptr<int64_t>(),
        next_scores.data_ptr<float>(),
        next_tokens.data_ptr<int64_t>(),
        next_indices.data_ptr<int64_t>(),
        next_beam_scores.data_ptr<float>(),
        next_beam_tokens.data_ptr<int64_t>(),
        next_beam_indices.data_ptr<int64_t>(),
        batch_size,
        num_beams,
        vocab_size,
        pad_token_id,
        eos_token_id,
        length_penalty,
        temperature
    );
    
    return {next_beam_scores, next_beam_tokens, next_beam_indices};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("process_beam_search", &process_beam_search_cuda, "Process beam search (CUDA)");
} 