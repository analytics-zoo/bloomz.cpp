#include "bloom.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#include <signal.h>
#endif

struct ChatContext {
    bloom_model model;
    gpt_vocab vocab;
    size_t mem_per_token = 0;
    std::vector<gpt_vocab::id> cached_tokens;
    std::vector<float> logits;
    std::vector<float> embeddings;
};

// load the model's weights from a file
bool bloom_model_load(const std::string & fname, bloom_model & model, gpt_vocab & vocab, int n_ctx) {
    printf("%s: loading model from '%s' - please wait ...\n", "loading bigdl-llm model", fname.c_str());

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", "loading bigdl-llm model", fname.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char *) &magic, sizeof(magic));
        if (magic != 0x67676d6c) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", "loading bigdl-llm model", fname.c_str());
            return false;
        }
    }

    int n_ff = 0;
    int n_parts = 0;

    // load hparams
    {
        auto & hparams = model.hparams;

        fin.read((char *) &hparams.n_vocab, sizeof(hparams.n_vocab));
        //fin.read((char *) &hparams.n_ctx,   sizeof(hparams.n_ctx));
        fin.read((char *) &hparams.n_embd,  sizeof(hparams.n_embd));
        fin.read((char *) &hparams.n_mult,  sizeof(hparams.n_mult));
        fin.read((char *) &hparams.n_head,  sizeof(hparams.n_head));
        fin.read((char *) &hparams.n_layer, sizeof(hparams.n_layer));
        fin.read((char *) &hparams.f16,     sizeof(hparams.f16));

        hparams.n_ctx = n_ctx;

        n_ff = ((4*hparams.n_embd + hparams.n_mult - 1)/hparams.n_mult)*hparams.n_mult;
        // n_parts = BLOOM_N_PARTS.at(hparams.n_embd);
        n_parts = 1;

        printf("%s: n_vocab = %d\n", "loading bigdl-llm model", hparams.n_vocab);
        printf("%s: n_ctx   = %d\n", "loading bigdl-llm model", hparams.n_ctx);
        printf("%s: n_embd  = %d\n", "loading bigdl-llm model", hparams.n_embd);
        printf("%s: n_mult  = %d\n", "loading bigdl-llm model", hparams.n_mult);
        printf("%s: n_head  = %d\n", "loading bigdl-llm model", hparams.n_head);
        printf("%s: n_layer = %d\n", "loading bigdl-llm model", hparams.n_layer);
        printf("%s: f16     = %d\n", "loading bigdl-llm model", hparams.f16);
        printf("%s: n_ff    = %d\n", "loading bigdl-llm model", n_ff);
        printf("%s: n_parts = %d\n", "loading bigdl-llm model", n_parts);
    }

    // load vocab
    {
        const int32_t n_vocab = model.hparams.n_vocab;

        if (n_vocab != model.hparams.n_vocab) {
            fprintf(stderr, "%s: invalid model file '%s' (bad vocab size %d != %d)\n",
                    "loading bigdl-llm model", fname.c_str(), n_vocab, model.hparams.n_vocab);
            return false;
        }

        std::string word;
        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            fin.read((char *) &len, sizeof(len));

            word.resize(len);
            fin.read((char *) word.data(), len);

            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;

            //if (i < 30000) {
            //    printf("%s: vocab[%d] = '%s'\n", __func__, i, word.c_str());
            //}
        }

        for (auto const & kv : vocab.id_to_token) {
            const std::string & word = kv.second;
            if (word.size() > 0) {
                if (word[0] == ' ') {
                    if (word.size() > 1) {
                        vocab.space_words[(uint8_t)word[1]].push_back(word);
                    }
                } else {
                    vocab.words[(uint8_t)word[0]].push_back(word);
                }
            }
        }
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    ggml_type wtype = GGML_TYPE_COUNT;
    switch (model.hparams.f16) {
        case 0: wtype = GGML_TYPE_F32;  break;
        case 1: wtype = GGML_TYPE_F16;  break;
        case 2: wtype = GGML_TYPE_Q4_0; break;
        case 3: wtype = GGML_TYPE_Q4_1; break;
        default:
                {
                    fprintf(stderr, "%s: invalid model file '%s' (bad f16 value %d)\n",
                            "loading bigdl-llm model", fname.c_str(), model.hparams.f16);
                    return false;
                }
    }

    const ggml_type wtype2 = GGML_TYPE_F32;

    auto & ctx = model.ctx;

    size_t ctx_size = 0;

    {
        const auto & hparams = model.hparams;

        const int64_t n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;

        ctx_size += n_embd*n_vocab*ggml_type_sizef(wtype); // tok_embeddings

        ctx_size += n_embd*ggml_type_sizef(GGML_TYPE_F32); // norm
        ctx_size += n_embd*ggml_type_sizef(GGML_TYPE_F32); // norm_b

        ctx_size += n_embd*ggml_type_sizef(GGML_TYPE_F32); // output_norm
        ctx_size += n_embd*ggml_type_sizef(GGML_TYPE_F32); // output_norm_b

        ctx_size += n_embd*n_vocab*ggml_type_sizef(wtype); // output

        ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // attention_norm
        ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // attention_norm_b

        ctx_size += n_layer*(3*n_embd*n_embd*ggml_type_sizef(wtype)); // query_key_value
        ctx_size += n_layer*(3*n_embd*ggml_type_sizef(GGML_TYPE_F32)); // query_key_value_b
        ctx_size += n_layer*(n_embd*n_embd*ggml_type_sizef(wtype)); // wo
        ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // wo_b

        ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // ffn_norm
        ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // ffn_norm_b

        ctx_size += n_layer*(n_ff*n_embd*ggml_type_sizef(wtype)); // w1
        ctx_size += n_layer*(n_ff*ggml_type_sizef(GGML_TYPE_F32)); // w1_b
        ctx_size += n_layer*(n_ff*n_embd*ggml_type_sizef(wtype)); // w2
        ctx_size += n_layer*(n_ff*ggml_type_sizef(GGML_TYPE_F32)); // w2_b

        ctx_size += n_ctx*n_layer*n_embd*ggml_type_sizef(GGML_TYPE_F32); // memory_k
        ctx_size += n_ctx*n_layer*n_embd*ggml_type_sizef(GGML_TYPE_F32); // memory_v

        ctx_size += (5 + 10*n_layer)*256; // object overhead TODO:

        printf("%s: ggml ctx size = %6.2f MB\n", "loading bigdl-llm model", ctx_size/(1024.0*1024.0));
    }

    // create the ggml context
    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", "loading bigdl-llm model");
            return false;
        }
    }

    // prepare memory for the weights
    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;

        model.layers.resize(n_layer);

        model.tok_embeddings = ggml_new_tensor_2d(ctx, wtype, n_embd, n_vocab);
        model.norm   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        model.norm_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

        model.output_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        model.output_norm_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        model.output = ggml_new_tensor_2d(ctx, wtype,         n_embd, n_vocab);

        // map by name
        model.tensors["tok_embeddings.weight"] = model.tok_embeddings;
        model.tensors["norm.weight"]   = model.norm;
        model.tensors["norm.bias"]   = model.norm_b;

        model.tensors["output_norm.weight"] = model.output_norm;
        model.tensors["output_norm.bias"] = model.output_norm_b;
        model.tensors["output.weight"] = model.output;

        for (int i = 0; i < n_layer; ++i) {
            auto & layer = model.layers[i];

            layer.attention_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.attention_norm_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.query_key_value = ggml_new_tensor_2d(ctx, wtype, n_embd, 3*n_embd);
            layer.query_key_value_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3*n_embd);
            layer.wo = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.wo_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.ffn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.ffn_norm_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.w1 = ggml_new_tensor_2d(ctx, wtype, n_embd,   n_ff);
            layer.w1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_ff);
            layer.w2 = ggml_new_tensor_2d(ctx, wtype,   n_ff, n_embd);
            layer.w2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            // map by name
            model.tensors["layers." + std::to_string(i) + ".attention_norm.weight"] = layer.attention_norm;
            model.tensors["layers." + std::to_string(i) + ".attention_norm.bias"] = layer.attention_norm_b;

            model.tensors["layers." + std::to_string(i) + ".attention.query_key_value.weight"] = layer.query_key_value;
            model.tensors["layers." + std::to_string(i) + ".attention.query_key_value.bias"] = layer.query_key_value_b;
            model.tensors["layers." + std::to_string(i) + ".attention.wo.weight"] = layer.wo;
            model.tensors["layers." + std::to_string(i) + ".attention.wo.bias"] = layer.wo_b;

            model.tensors["layers." + std::to_string(i) + ".ffn_norm.weight"] = layer.ffn_norm;
            model.tensors["layers." + std::to_string(i) + ".ffn_norm.bias"] = layer.ffn_norm_b;

            model.tensors["layers." + std::to_string(i) + ".feed_forward.w1.weight"] = layer.w1;
            model.tensors["layers." + std::to_string(i) + ".feed_forward.w1.bias"] = layer.w1_b;
            model.tensors["layers." + std::to_string(i) + ".feed_forward.w2.weight"] = layer.w2;
            model.tensors["layers." + std::to_string(i) + ".feed_forward.w2.bias"] = layer.w2_b;
        }
    }

    // key + value memory
    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;

        const int n_mem      = n_layer*n_ctx;
        const int n_elements = n_embd*n_mem;

        model.memory_k = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
        model.memory_v = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);

        const size_t memory_size = ggml_nbytes(model.memory_k) + ggml_nbytes(model.memory_v);

        printf("%s: memory_size = %8.2f MB, n_mem = %d\n", "loading bigdl-llm model", memory_size/1024.0/1024.0, n_mem);
    }

    const size_t file_offset = fin.tellg();

    fin.close();

    std::vector<uint8_t> tmp;

    for (int i = 0; i < n_parts; ++i) {
        const int part_id = i;
        //const int part_id = n_parts - i - 1;

        std::string fname_part = fname;
        if (i > 0) {
            fname_part += "." + std::to_string(i);
        }

        printf("%s: loading model part %d/%d from '%s'\n", "loading bigdl-llm model", i+1, n_parts, fname_part.c_str());

        fin = std::ifstream(fname_part, std::ios::binary);
        fin.seekg(file_offset);

        // load weights
        {
            int n_tensors = 0;
            size_t total_size = 0;

            printf("%s: ", "loading bigdl-llm model");

            while (true) {
                int32_t n_dims;
                int32_t length;
                int32_t ftype;

                fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
                fin.read(reinterpret_cast<char *>(&length), sizeof(length));
                fin.read(reinterpret_cast<char *>(&ftype),  sizeof(ftype));

                if (fin.eof()) {
                    break;
                }

                int64_t nelements = 1;
                int32_t ne[2] = { 1, 1 };
                for (int i = 0; i < n_dims; ++i) {
                    fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
                    nelements *= ne[i];
                }

                std::string name(length, 0);
                fin.read(&name[0], length);

                if (model.tensors.find(name.data()) == model.tensors.end()) {
                    fprintf(stderr, "%s: unknown tensor '%s' in model file\n", "loading bigdl-llm model", name.data());
                    return false;
                }

                // split_type = 0: split by columns
                // split_type = 1: split by rows
                int split_type = 0;

                // split_type = 0:
                // regex:
                //   - tok_embeddings.*
                //   - layers.*.attention.wo.weight
                //   - layers.*.feed_forward.w2.weight

                // split_type = 1:
                // regex:
                //   - output.*
                //   - layers.*.attention.wq.weight
                //   - layers.*.attention.wk.weight
                //   - layers.*.attention.wv.weight
                //   - layers.*.feed_forward.w1.weight
                //   - layers.*.feed_forward.w3.weight
                if (name.find("tok_embeddings") != std::string::npos) {
                    split_type = 0;
                } else if (name.find("layers") != std::string::npos) {
                    if (name.find("attention.wo.weight") != std::string::npos) {
                        split_type = 0;
                    } else if (name.find("feed_forward.w2.weight") != std::string::npos) {
                        split_type = 0;
                    } else {
                        split_type = 1;
                    }
                } else if (name.find("output") != std::string::npos) {
                    split_type = 1;
                }

                auto tensor = model.tensors[name.data()];

                if (n_dims == 1) {
                    if (ggml_nelements(tensor) != nelements) {
                        fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                        return false;
                    }
                } else {
                    if (ggml_nelements(tensor)/n_parts != nelements) {
                        fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                        return false;
                    }
                }

                if (n_dims == 1) {
                    if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
                        fprintf(stderr,
                                "%s: tensor '%s' has wrong shape in model file: got [%ld, %ld], expected [%d, %d]\n",
                                __func__, name.data(), tensor->ne[0], tensor->ne[1], ne[0], ne[1]);
                        return false;
                    }
                } else {
                    if (split_type == 0) {
                        if (tensor->ne[0]/n_parts != ne[0] || tensor->ne[1] != ne[1]) {
                            fprintf(stderr,
                                    "%s: tensor '%s' has wrong shape in model file: got [%ld, %ld], expected [%d, %d]\n",
                                    __func__, name.data(), tensor->ne[0] / n_parts, tensor->ne[1], ne[0], ne[1]);
                            return false;
                        }
                    } else {
                        if (tensor->ne[0] != ne[0] || tensor->ne[1]/n_parts != ne[1]) {
                            fprintf(stderr,
                                    "%s: tensor '%s' has wrong shape in model file: got [%ld, %ld], expected [%d, %d]\n",
                                    __func__, name.data(), tensor->ne[0], tensor->ne[1] / n_parts, ne[0], ne[1]);
                            return false;
                        }
                    }
                }

                if (0) {
                    static const char * ftype_str[] = { "f32", "f16", "q4_0", "q4_1", };
                    printf("%24s - [%5d, %5d], type = %6s, split = %d\n", name.data(), ne[0], ne[1], ftype_str[ftype], split_type);
                }

                size_t bpe = 0;

                switch (ftype) {
                    case 0: bpe = ggml_type_size(GGML_TYPE_F32);  break;
                    case 1: bpe = ggml_type_size(GGML_TYPE_F16);  break;
                    case 2: bpe = ggml_type_size(GGML_TYPE_Q4_0); assert(ne[0] % 64 == 0); break;
                    case 3: bpe = ggml_type_size(GGML_TYPE_Q4_1); assert(ne[0] % 64 == 0); break;
                    default:
                            {
                                fprintf(stderr, "%s: unknown ftype %d in model file\n", __func__, ftype);
                                return false;
                            }
                };

                if (n_dims == 1 || n_parts == 1) {
                    if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                        fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                                __func__, name.data(), ggml_nbytes(tensor), nelements*bpe);
                        return false;
                    }

                    if (part_id == 0) {
                        fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));
                    } else {
                        fin.seekg(ggml_nbytes(tensor), std::ios::cur);
                    }

                    total_size += ggml_nbytes(tensor);
                } else {
                    if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)/n_parts) {
                        fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                                __func__, name.data(), ggml_nbytes(tensor)/n_parts, nelements*bpe);
                        return false;
                    }

                    if (split_type == 0) {
                        const int np0 = ne[0];

                        const size_t row_size = (tensor->ne[0]/ggml_blck_size(tensor->type))*ggml_type_size(tensor->type);
                        assert(row_size == tensor->nb[1]);

                        for (int i1 = 0; i1 < ne[1]; ++i1) {
                            const size_t offset_row = i1*row_size;
                            const size_t offset = offset_row + ((part_id*np0)/ggml_blck_size(tensor->type))*ggml_type_size(tensor->type);
                            fin.read(reinterpret_cast<char *>(tensor->data) + offset, row_size/n_parts);
                        }
                    } else {
                        const int np1 = ne[1];

                        const size_t row_size = (tensor->ne[0]/ggml_blck_size(tensor->type))*ggml_type_size(tensor->type);

                        for (int i1 = 0; i1 < ne[1]; ++i1) {
                            const size_t offset_row = (i1 + part_id*np1)*row_size;
                            fin.read(reinterpret_cast<char *>(tensor->data) + offset_row, row_size);
                        }
                    }

                    total_size += ggml_nbytes(tensor)/n_parts;
                }

                //printf("%42s - [%5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ftype == 0 ? "float" : "f16", ggml_nbytes(tensor)/1024.0/1024.0);
                if (++n_tensors % 8 == 0) {
                    printf(".");
                    fflush(stdout);
                }
            }

            printf(" done\n");

            printf("%s: model size = %8.2f MB / num tensors = %d\n", "loading bigdl-llm model", total_size/1024.0/1024.0, n_tensors);
        }

        fin.close();
    }

    return true;
}

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
              bool logits_all,
              bool embed) {

    const int64_t N = embd_inp.size();

    const auto & hparams = model.hparams;

    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx   = hparams.n_ctx;
    const int n_head  = hparams.n_head;
    const int n_vocab = hparams.n_vocab;

    const int d_key = n_embd/n_head;

    static size_t buf_size = 512ul*10000*10000;     // todo!
    static void * buf = malloc(buf_size);

    if (mem_per_token > 0 && mem_per_token*N > buf_size) {
        const size_t buf_size_new = 1.1*(mem_per_token*N); // add 10% to account for ggml object overhead
        //printf("\n%s: reallocating buffer from %zu to %zu bytes\n", __func__, buf_size, buf_size_new);

        // reallocate
        buf_size = buf_size_new;
        buf = realloc(buf, buf_size);
        if (buf == nullptr) {
            fprintf(stderr, "%s: failed to allocate %zu bytes\n", __func__, buf_size);
            return false;
        }
    }

    struct ggml_init_params params = {
        buf_size,
        buf,
        false
    };

    struct ggml_context * ctx0 = ggml_init(params);
    ggml_cgraph gf = {};
    gf.n_threads = n_threads;

    struct ggml_tensor * embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    memcpy(embd->data, embd_inp.data(), N*ggml_element_size(embd));

    struct ggml_tensor * inpL = ggml_get_rows(ctx0, model.tok_embeddings, embd);

    // word embeddings norm
    {
        inpL = ggml_norm(ctx0, inpL);
        inpL = ggml_mul(ctx0, ggml_repeat(ctx0, model.norm, inpL), inpL);
        inpL = ggml_add(ctx0, ggml_repeat(ctx0, model.norm_b, inpL), inpL);
    }

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL; //TODO: copy?

        struct ggml_tensor * cur;

        // norm
        {
            cur = ggml_norm(ctx0, inpL);

            // cur = attention_norm*cur
            cur = ggml_mul(ctx0,
                        ggml_repeat(ctx0, model.layers[il].attention_norm, cur),
                        cur);
            cur = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].attention_norm_b, cur), cur);
        }

        // attn
        {
            cur = ggml_mul_mat(ctx0,model.layers[il].query_key_value, cur);

            cur = ggml_add(ctx0,
                    ggml_repeat(ctx0, model.layers[il].query_key_value_b, cur),
                    cur);
        }

        // cur = ggml_debug(ctx0, cur);

        // self-attention
        {
            struct ggml_tensor * Qcur = ggml_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 0*sizeof(float)*n_embd);
            struct ggml_tensor * Kcur = ggml_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 1*sizeof(float)*n_embd); //TODO: float or fp16?
            struct ggml_tensor * Vcur = ggml_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 2*sizeof(float)*n_embd);

            // store key and value to memory
            if (N >= 1) {
                struct ggml_tensor * k = ggml_view_1d(ctx0, model.memory_k, N*n_embd, (ggml_element_size(model.memory_k)*n_embd)*(il*n_ctx + n_past));
                struct ggml_tensor * v = ggml_view_1d(ctx0, model.memory_v, N*n_embd, (ggml_element_size(model.memory_v)*n_embd)*(il*n_ctx + n_past));

                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Vcur, v));
            }

            // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
            struct ggml_tensor * Q =
                ggml_permute(ctx0,
                            ggml_cpy(ctx0, Qcur,
                                ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd/n_head, n_head, N)),
                        0, 2, 1, 3);

            // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
            struct ggml_tensor * K =
                ggml_permute(ctx0, ggml_reshape_3d(ctx0,
                                ggml_view_1d(ctx0, model.memory_k, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_k)*n_embd),
                                n_embd/n_head, n_head, n_past + N),
                        0, 2, 1, 3);

            // K * Q
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            struct ggml_tensor * KQ_scaled =
                ggml_scale(ctx0,
                        KQ,
                        ggml_new_f32(ctx0, 1.0f/sqrt(float(n_embd)/n_head))
                        );

            // Alibi
            // KQ_scaled_alibi = KQ_scaled + alibi_bias //TODO: optimize
            struct ggml_tensor * KQ_scaled_alibi = ggml_alibi(ctx0, KQ_scaled, n_past, n_head, 8.0);

            // KQ_masked = mask_past(KQ_scaled)
            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf(ctx0, KQ_scaled_alibi, n_past);

            // KQ = soft_max(KQ_masked)
            struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);

            // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
            struct ggml_tensor *V_trans =
                    ggml_cpy(ctx0,
                             ggml_permute(ctx0,
                                          ggml_reshape_3d(ctx0,
                                                          ggml_view_1d(ctx0, model.memory_v, (n_past + N) * n_embd,
                                                                       il * n_ctx * ggml_element_size(model.memory_v) *
                                                                       n_embd),
                                                          n_embd / n_head, n_head, n_past + N),
                                          1, 2, 0, 3),
                             ggml_new_tensor_3d(ctx0, model.memory_v->type, n_past + N, n_embd / n_head, n_head));
            // KQV = transpose(V) * KQ_soft_max
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V_trans, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            // cur = KQV_merged.contiguous().view(n_embd, N)
            cur = ggml_cpy(ctx0,
                    KQV_merged,
                    ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));

            // projection
            cur = ggml_mul_mat(ctx0,
                    model.layers[il].wo,
                    cur);
            cur = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].wo_b, cur), cur);
        }

        struct ggml_tensor * inpFF = ggml_add(ctx0, cur, inpSA);

        // feed-forward network
        {
            // norm
            {
                cur = ggml_norm(ctx0, inpFF);

                // cur = ffn_norm*cur + ffn_norm_b
                cur = ggml_mul(ctx0,
                        ggml_repeat(ctx0, model.layers[il].ffn_norm, cur),
                        cur);
                cur = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].ffn_norm_b, cur), cur);
            }

            cur = ggml_mul_mat(ctx0,
                    model.layers[il].w1,
                    cur);
            cur = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].w1_b, cur), cur);

            cur = ggml_gelu(ctx0, cur);

            cur = ggml_mul_mat(ctx0,
                    model.layers[il].w2,
                    cur);
            cur = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].w2_b, cur), cur);
        }

        cur  = ggml_add(ctx0, cur, inpFF);

        // input for next layer
        inpL = cur;
    }

    // used at the end to optionally extract the embeddings
    struct ggml_tensor * embedding_tensor = NULL;

    // norm
    {
        inpL = ggml_norm(ctx0, inpL);

        // inpL = norm*inpL
        inpL = ggml_mul(ctx0,
                    ggml_repeat(ctx0, model.output_norm, inpL),
                    inpL);

        inpL = ggml_add(ctx0, ggml_repeat(ctx0, model.output_norm_b, inpL), inpL);

        embedding_tensor = inpL;
    }

    // lm_head
    {
        inpL = ggml_mul_mat(ctx0, model.output, inpL);
    }

    // logits -> probs
    //inpL = ggml_soft_max(ctx0, inpL);

    // run the computation
    ggml_build_forward_expand(&gf, inpL);
    ggml_graph_compute       (ctx0, &gf);

    //if (n_past%100 == 0) {
    //    ggml_graph_print   (&gf);
    //    ggml_graph_dump_dot(&gf, NULL, "gpt-2.dot");
    //}

    //embd_w.resize(n_vocab*N);
    //memcpy(embd_w.data(), ggml_get_data(inpL), sizeof(float)*n_vocab*N);

    // return result for just the last token
    if (!logits_all) {
        embd_w.resize(n_vocab);
        memcpy(embd_w.data(), (float *) ggml_get_data(inpL) + (n_vocab*(N-1)), sizeof(float)*n_vocab);
    } else {
        embd_w.resize(n_vocab * N);
        memcpy(embd_w.data(), (float *) ggml_get_data(inpL), sizeof(float)*n_vocab*N);
    }

    if (embed) {
        embeddings.resize(n_embd);
        memcpy(embeddings.data(), (float *)ggml_get_data(embedding_tensor) + (n_embd*(N - 1)), sizeof(float)*n_embd);
    }

    if (mem_per_token == 0) {
        mem_per_token = ggml_used_mem(ctx0)/N;
    }
    //printf("used_mem = %zu\n", ggml_used_mem(ctx0));

    ggml_free(ctx0);

    return true;
}

extern "C" ChatContext* bloom_load(const char * fname, int n_ctx, int n_threads) {
    ChatContext * ctx = new ChatContext{};

    // init model and vocab
    bool res = bloom_model_load(fname, ctx->model, ctx->vocab, n_ctx);
    if (!res) {
        return 0;
    }

    // determine the required inference memory per token:
    res = bloom_eval(ctx->model,
                     n_threads,
                     0,
                     { 0, 1, 2, 3 },
                     ctx->logits,
                     ctx->embeddings,
                     ctx->mem_per_token);
    if (!res) {
        return 0;
    }

    return ctx;
}

extern "C" void bloom_free(ChatContext* ctx) {
    delete ctx;
}

int inference(gpt_params & params,
              const bloom_model & model,
              const gpt_vocab & vocab,
              size_t mem_per_token,
              std::vector<gpt_vocab::id>& tokens,
              std::vector<gpt_vocab::id>& last_n_tokens,
              int n_past,
              char * dst) {
    ggml_time_init();
    const int64_t t_start_us = ggml_time_us();

    int64_t t_sample_us  = 0;
    int64_t t_eval_us = 0;
    int64_t t_predict_us = 0;
    size_t n_past_init = n_past;

    std::mt19937 rng(params.seed);
    std::vector<float> logits, embeddings;

    while (n_past < tokens.size()) {
        // eval input prompt
        const int64_t t_start_eval_us = ggml_time_us();

        int n = std::min((size_t)params.n_batch, tokens.size() - n_past);
        std::vector<gpt_vocab::id> embd(tokens.cbegin() + n_past,
                                        tokens.cbegin() + n_past + n);
        if (!bloom_eval(model,
                        params.n_threads,
                        n_past,
                        embd,
                        logits,
                        embeddings,
                        mem_per_token)) {
            // todo: better error handling
            printf("Failed to predict\n");
            return 1;
        }
        n_past += n;

        t_eval_us += ggml_time_us() - t_start_eval_us;
    }

    int n_predict = 0;
    while (1) {
        {
            // sample next token
            const int64_t t_start_sample_us = ggml_time_us();

            gpt_vocab::id id = bloom_sample_top_p(vocab,
                                                  logits.data() + (logits.size() - model.hparams.n_vocab),
                                                  last_n_tokens,
                                                  params.repeat_penalty,
                                                  params.top_p,
                                                  params.top_k,
                                                  params.temp,
                                                  rng);
            last_n_tokens.erase(last_n_tokens.begin());
            last_n_tokens.push_back(id);
            ++n_predict;

            const auto& word = vocab.id_to_token.find(id)->second;
            strcpy(dst, word.c_str());
            dst += word.size();

            t_sample_us += ggml_time_us() - t_start_sample_us;
        }
        if (last_n_tokens.back() == 2 || n_predict >= params.n_predict) {
            // end of text token or reach the token number limit
            break;
        }

        {
            // predict the next token
            const int64_t t_start_predict_us = ggml_time_us();

            std::vector<gpt_vocab::id> embd(1, last_n_tokens.back());
            if (!bloom_eval(model,
                            params.n_threads,
                            n_past,
                            embd,
                            logits,
                            embeddings,
                            mem_per_token)) {
                // todo: better error handling
                printf("Failed to predict\n");
                return -1;
            }
            tokens.push_back(last_n_tokens.back());
            ++n_past;

            t_predict_us += ggml_time_us() - t_start_predict_us;
        }
    }

    // report timing
    {
        const int64_t t_end_us = ggml_time_us();

        int n_prompt = tokens.size() - n_predict - n_past_init;
        n_predict = n_predict - 1;

        printf("\n\n");
        printf("%s:    mem per token = %8zu bytes\n", __func__, mem_per_token);
        printf("%s:      sample time = %8.2f ms\n", __func__, t_sample_us/1000.0f);
        printf("%s: evel prompt time = %8.2f ms / %d tokens / %.2f ms per token\n", __func__, t_eval_us/1000.0f, n_prompt, t_eval_us/1000.0f/n_prompt);
        printf("%s:     predict time = %8.2f ms / %d tokens / %.2f ms per token\n", __func__, t_predict_us/1000.0f, n_predict, t_predict_us/1000.0f/n_predict);
        printf("%s:       total time = %8.2f ms\n", __func__, (t_end_us - t_start_us)/1000.0f);
    }

    return 0;
}

extern "C" int bloom_run(ChatContext *ctx,
                         int32_t seed,
                         int32_t n_threads,
                         int32_t n_batch,
                         int32_t n_predict,
                         bool match_str,
                         const char* prompt,
                         char* dst)
{
    gpt_params params;
    params.seed = seed < 0 ? time(NULL) : seed;
    params.n_threads = n_threads > 0 ? n_threads : params.n_threads;
    params.n_batch = n_batch > 0 ? n_batch : params.n_batch;

    std::vector<gpt_vocab::id> & cached_tokens = ctx->cached_tokens;

    int n_past = 0;
    if (match_str) {
        int n_chars = 0, match = true;
        for (int id : cached_tokens) {
            const std::string & str = ctx->vocab.id_to_token[id];
            for (int j = 0; j < str.size(); ++j) {
                if (prompt[n_chars + j] == '\0' || prompt[n_chars + j] != str[j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                n_past += 1;
                n_chars += str.size();
            } else {
                break;
            }
        }
        n_past = prompt[n_chars] == '\0' ? n_past-1 : n_past;
        cached_tokens.resize(n_past);

        params.prompt = std::string(prompt + n_chars);
        printf("n_past: %d, n_chars: %d, prompt: %s\n", n_past, n_chars, params.prompt.c_str());
        std::vector<gpt_vocab::id> new_tokens = bloom_tokenize(ctx->vocab, params.prompt, false);
        cached_tokens.insert(cached_tokens.end(), new_tokens.begin(), new_tokens.end());
    } else {
        params.prompt = std::string(prompt);
        std::vector<gpt_vocab::id> input_tokens = bloom_tokenize(ctx->vocab, params.prompt, false);

        while (n_past < cached_tokens.size() && n_past < input_tokens.size()) {
            if (cached_tokens[n_past] == input_tokens[n_past]) {
                ++n_past;
            } else {
                break;
            }
        }
        n_past = std::min(n_past, (int)input_tokens.size() - 1);
        // printf("n_past: %d\n", n_past);

        cached_tokens.swap(input_tokens);
    }

    params.n_predict = std::min(n_predict, ctx->model.hparams.n_ctx - (int)cached_tokens.size());

    std::vector<gpt_vocab::id> last_n_tokens{};
    int n_tokens = cached_tokens.size();
    if (n_tokens >= params.repeat_last_n) {
        for (int i = n_tokens - params.repeat_last_n; i < n_tokens; ++i) {
            last_n_tokens.push_back(cached_tokens[i]);
        }
    } else {
        last_n_tokens.resize(params.repeat_last_n - n_tokens, 0);
        for (int i = 0; i < n_tokens; ++i) {
            last_n_tokens.push_back(cached_tokens[i]);
        }
    }

    strcpy(dst, prompt);
    dst += strlen(prompt);

    int ret = inference(params,
                        ctx->model,
                        ctx->vocab,
                        ctx->mem_per_token,
                        cached_tokens,
                        last_n_tokens,
                        n_past,
                        dst);
    if (ret < 0) {
        dst[0] = '\0';
        return -1;
    }

    return 0;
}

extern "C" void c_free(void * p) {
    free(p);
}

extern "C" int32_t* tokenize_api(ChatContext *ctx,
                                 const char *prompt,
                                 bool bos,
                                 int32_t *len) {
    std::vector<gpt_vocab::id> tokens = bloom_tokenize(ctx->vocab, prompt, bos);
    int32_t *c_tokens = (int32_t*)malloc(sizeof(int32_t) * tokens.size());
    std::copy(tokens.cbegin(), tokens.cend(), c_tokens);
    *len = tokens.size();
    return c_tokens;
}

extern "C" char* detokenize_api(ChatContext *ctx,
                                int32_t token_id) {
    const gpt_vocab::token& word = ctx->vocab.id_to_token[token_id];
    char *dst = (char *)malloc(sizeof(char) * (word.size() + 1));
    strcpy(dst, word.c_str());
    return dst;
}

static bool eval_internal(ChatContext *ctx,
                          int32_t *tokens,
                          int32_t token_num,
                          int32_t n_threads,
                          int32_t n_batch,
                          bool logits_all = false,
                          bool embed = false) {
    gpt_params params;
    params.n_threads = n_threads > 0 ? n_threads : params.n_threads;
    params.n_batch = n_batch > 0 ? n_batch : params.n_batch;

    int n_past = 0;
    std::vector<gpt_vocab::id> & cached_tokens = ctx->cached_tokens;
    std::vector<gpt_vocab::id> input_tokens(tokens, tokens + token_num);
    if (!logits_all) {
        while (n_past < cached_tokens.size() && n_past < token_num) {
            if (cached_tokens[n_past] == input_tokens[n_past]) {
                ++n_past;
            } else {
                break;
            }
        }
        n_past = std::min(n_past, token_num - 1);
    }

    // printf("n_past: %d\n", n_past);

    while (n_past < input_tokens.size()) {
        // eval input prompt
        int n = std::min((size_t)params.n_batch, input_tokens.size() - n_past);
        std::vector<gpt_vocab::id> embd(input_tokens.cbegin() + n_past,
                                        input_tokens.cbegin() + n_past + n);
        if (!bloom_eval(ctx->model,
                        params.n_threads,
                        n_past,
                        embd,
                        ctx->logits,
                        ctx->embeddings,
                        ctx->mem_per_token,
                        logits_all,
                        embed)) {
            // todo: better error handling
            printf("Failed to predict\n");
            return false;
        }
        n_past += n;
    }

    cached_tokens.swap(input_tokens);
    return true;
}

extern "C" float* eval_api(ChatContext *ctx,
                           int32_t *tokens,
                           int32_t token_num,
                           int32_t seed,
                           int32_t n_threads,
                           int32_t n_batch,
                           int64_t* len) {
    bool status = eval_internal(ctx, tokens, token_num, n_threads, n_batch, true);
    assert(status);
    *len = ctx->logits.size();
    return ctx->logits.data();
}

extern "C" float* embed_api(ChatContext *ctx,
                            int32_t *tokens,
                            int32_t token_num,
                            int32_t seed,
                            int32_t n_threads,
                            int32_t n_batch,
                            int64_t* len) {
    bool status = eval_internal(ctx, tokens, token_num, n_threads, n_batch, false, true);
    assert(status);
    *len = ctx->embeddings.size();
    return ctx->embeddings.data();
}


extern "C" int32_t forward_api(ChatContext *ctx,
                               int32_t *tokens,
                               int32_t token_num,
                               int32_t seed,
                               int32_t n_threads,
                               int32_t n_batch) {
    bool status = eval_internal(ctx, tokens, token_num, n_threads, n_batch);
    assert(status);

    gpt_params params;
    params.seed = seed < 0 ? time(NULL) : seed;

    std::mt19937 rng(seed);
    std::vector<gpt_vocab::id> & cached_tokens = ctx->cached_tokens;
    std::vector<gpt_vocab::id> last_n_tokens{};
    int n_tokens = cached_tokens.size();
    if (n_tokens >= params.repeat_last_n) {
        for (int i = n_tokens - params.repeat_last_n; i < n_tokens; ++i) {
            last_n_tokens.push_back(cached_tokens[i]);
        }
    } else {
        last_n_tokens.resize(params.repeat_last_n - n_tokens, 0);
        for (int i = 0; i < n_tokens; ++i) {
            last_n_tokens.push_back(cached_tokens[i]);
        }
    }

    gpt_vocab::id id = bloom_sample_top_p(ctx->vocab,
                                          ctx->logits.data() + (ctx->logits.size() - ctx->model.hparams.n_vocab),
                                          last_n_tokens,
                                          params.repeat_penalty,
                                          params.top_p,
                                          params.top_k,
                                          params.temp,
                                          rng);
    return id;
}
