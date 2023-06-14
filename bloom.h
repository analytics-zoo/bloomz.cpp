#include "ggml.h"

#include "utils.h"

struct bloom_hparams {
    int32_t n_vocab = 32000;
    int32_t n_ctx   = 512;   // this is provided as user input?
    int32_t n_embd  = 4096;
    int32_t n_mult  = 256;
    int32_t n_head  = 32;
    int32_t n_layer = 32;
    int32_t f16     = 1;
};

struct bloom_layer {
    // normalization
    struct ggml_tensor * attention_norm;
    struct ggml_tensor * attention_norm_b;

    // attention
    struct ggml_tensor * query_key_value;
    struct ggml_tensor * query_key_value_b;
    struct ggml_tensor * wo;
    struct ggml_tensor * wo_b;

    // normalization
    struct ggml_tensor * ffn_norm;
    struct ggml_tensor * ffn_norm_b;

    // ff
    struct ggml_tensor * w1;
    struct ggml_tensor * w1_b;
    struct ggml_tensor * w2;
    struct ggml_tensor * w2_b;
};

struct bloom_model {
    bloom_hparams hparams;

    struct ggml_tensor * tok_embeddings;
    struct ggml_tensor * norm;
    struct ggml_tensor * norm_b;

    struct ggml_tensor * output_norm;
    struct ggml_tensor * output_norm_b;
    struct ggml_tensor * output;


    std::vector<bloom_layer> layers;

    // key + value memory
    struct ggml_tensor * memory_k;
    struct ggml_tensor * memory_v;

    //
    struct ggml_context * ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};


// load the model's weights from a file
bool bloom_model_load(const std::string & fname, bloom_model & model, gpt_vocab & vocab, int n_ctx);

// evaluate the transformer
//
//   - model:     the model
//   - n_threads: number of threads to use
//   - n_past:    the context size so far
//   - embd_inp:  the embeddings of the tokens in the context
//   - embd_w:    the predicted logits for the next token
//
// The GPT-J model requires about 16MB of memory per input token.
//
bool bloom_eval(
        const bloom_model & model,
        const int n_threads,
        const int n_past,
        const std::vector<gpt_vocab::id> & embd_inp,
              std::vector<float>         & embd_w,
              std::vector<float>         & embeddings,
              size_t                     & mem_per_token,
              bool logits_all = false,
              bool embed = false);
