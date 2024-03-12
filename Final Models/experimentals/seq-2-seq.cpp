#include <torch/torch.h>

// Head class
class HeadImpl : public torch::nn::Module {
public:
    HeadImpl(int64_t d_embd, int64_t n_head, float dropout, int64_t block_size)
        : key(torch::nn::Linear(d_embd, d_embd / n_head)),
          query(torch::nn::Linear(d_embd, d_embd / n_head)),
          value(torch::nn::Linear(d_embd, d_embd / n_head)),
          tril(torch::tril(torch::ones({block_size, block_size}))),
          dropout(torch::nn::Dropout(dropout)) {
        register_module("key", key);
        register_module("query", query);
        register_module("value", value);
    }

    torch::Tensor forward(torch::Tensor x) {
        auto [B, T, C] = x.sizes();
        auto key = this->key(x);  // (B,T,hs)
        auto query = this->query(x);  // (B,T,hs)

        // Compute attention scores ("affinities")
        auto weights = torch::matmul(query, key.transpose(-2,-1)) * std::pow(key.size(-1), -0.5);
        weights.masked_fill_(this->tril[:T, :T] == 0, std::numeric_limits<float>::lowest());
        weights = torch::softmax(weights, -1);
        weights = this->dropout(weights);

        // Perform the weighted aggregation of the values
        auto value = this->value(x);  // (B,T,hs)
        auto out = torch::matmul(weights, value);  // (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out;
    }

private:
    torch::nn::Linear key, query, value;
    torch::Tensor tril;
    torch::nn::Dropout dropout;
};

TORCH_MODULE(Head);

// MultiHeadAttention class
class MultiHeadAttentionImpl : public torch::nn::Module {
public:
    MultiHeadAttentionImpl(int64_t d_embd, int64_t n_head, float dropout, int64_t block_size)
        : heads(register_module("heads", torch::nn::ModuleList<Head>(n_head))),
          proj(torch::nn::Linear(n_head * (d_embd / n_head), d_embd)),
          dropout(torch::nn::Dropout(dropout)) {}

    torch::Tensor forward(torch::Tensor x) {
        auto out = torch::cat(torch::TensorList{head->forward(x) for (auto head : this->heads)}, -1);
        out = this->dropout(out);
        return out;
    }

private:
    torch::nn::ModuleList<Head> heads;
    torch::nn::Linear proj;
    torch::nn::Dropout dropout;
};

TORCH_MODULE(MultiHeadAttention);

// FeedForward class
class FeedForwardImpl : public torch::nn::Module {
public:
    FeedForwardImpl(int64_t d_embd)
        : fc1(torch::nn::Linear(d_embd, 4*d_embd)),
          fc2(torch::nn::Linear(4*d_embd, d_embd)) {}

    torch::Tensor forward(torch::Tensor x) {
        x = torch::gelu(this->fc1(x));  // GELU instead of ReLU
        x = this->fc2(x);
        return x;
    }

private:
    torch::nn::Linear fc1, fc2;
};

TORCH_MODULE(FeedForward);


// EncoderLayer class
class EncoderLayerImpl : public torch::nn::Module {
public:
    EncoderLayerImpl(int64_t d_embd, int64_t n_head, float dropout, int64_t block_size)
        : s_att(register_module("s_att", MultiHeadAttention(d_embd, n_head, dropout, block_size))),
          ffwd(register_module("ffwd", FeedForward(d_embd))) {}

    torch::Tensor forward(torch::Tensor src, torch::Tensor src_mask = torch::Tensor()) {
        auto src2 = this->s_att(src);
        src = src + src2;
        src = torch::layer_norm(src, src.norm({src.dim() - 1}, true));
        
        src2 = this->ffwd(src);
        src = src + src2;
        src = torch::layer_norm(src, src.norm({src.dim() - 1}, true));
        
        return src;
    }

private:
    MultiHeadAttention s_att;
    FeedForward ffwd;
};

TORCH_MODULE(EncoderLayer);

// DecoderLayer class
class DecoderLayerImpl : public torch::nn::Module {
public:
    DecoderLayerImpl(int64_t d_embd, int64_t n_head, float dropout, int64_t block_size)
        : s_att(register_module("s_att", MultiHeadAttention(d_embd, n_head, dropout, block_size))),
          enc_att(register_module("enc_att", EncoderDecoderAttention(d_embd, n_head, dropout, block_size))),
          ffwd(register_module("ffwd", FeedForward(d_embd))) {}

    torch::Tensor forward(torch::Tensor trg, torch::Tensor enc_src, torch::Tensor trg_mask = torch::Tensor(), torch::Tensor src_mask = torch::Tensor()) {
        auto trg2 = this->s_att(trg);
        trg = trg + trg2;
        trg = torch::layer_norm(trg, trg.norm({trg.dim() - 1}, true));

        trg2 = this->enc_att(trg, enc_src, enc_src);
        trg = trg + trg2;
        trg = torch::layer_norm(trg, trg.norm({trg.dim() - 1}, true));

        trg2 = this->ffwd(trg);
        trg = trg + trg2;
        trg = torch::layer_norm(trg, trg.norm({trg.dim() - 1}, true));

        return trg;
    }

private:
    MultiHeadAttention s_att;
    EncoderDecoderAttention enc_att;
    FeedForward ffwd;
};

TORCH_MODULE(DecoderLayer);

// Transformer class
class TransformerImpl : public torch::nn::Module {
public:
    TransformerImpl(int64_t block_size, int64_t vocab_size, int64_t n_layers, int64_t d_embd, int64_t n_head, float dropout)
        : d_embd(d_embd),
          block_size(block_size),
          token_embd(torch::nn::Embedding(vocab_size, d_embd)),
          pos_embd(torch::nn::Embedding(block_size, d_embd)),
          enc_layer(register_module("enc_layer", torch::nn::ModuleList<EncoderLayer>(n_layers))),
          dec_layer(register_module("dec_layer", torch::nn::ModuleList<DecoderLayer>(n_layers))),
          norm_final(torch::nn::LayerNorm(d_embd)),
          lm_head(torch::nn::Linear(d_embd, vocab_size)),
          fc_out(torch::nn::Linear(d_embd, vocab_size)),
          dropout(torch::nn::Dropout(dropout)) {
        register_module("token_embd", token_embd);
        register_module("pos_embd", pos_embd);
    }

    // make_src_mask function
    torch::Tensor make_src_mask(torch::Tensor src) {
        auto src_mask = (src != src_pad_idx).unsqueeze(1).unsqueeze(2);
        return src_mask;
    }

    // make_trg_mask function
    torch::Tensor make_trg_mask(torch::Tensor trg) {
        auto trg_pad_mask = (trg != trg_pad_idx).unsqueeze(1).unsqueeze(2);
        auto trg_len = trg.size(1);
        auto trg_sub_mask = torch::tril(torch::ones({trg_len, trg_len}, trg.device())).to(torch::kBool);
        auto trg_mask = trg_pad_mask & trg_sub_mask;
        return trg_mask;
    }

    // generate function
    std::tuple<torch::Tensor, torch::Tensor> generate(torch::Tensor idx, int max_tokens=50) {
        for (int i = 0; i < max_tokens; ++i) {
            auto idx_cond = idx.slice(1, -block_size, -1);
            auto result = this->forward(idx_cond);
            auto logits = std::get<0>(result).slice(1, -1, -1);
            auto loss = std::get<1>(result);
            auto probs = torch::softmax(logits, -1);
            auto idx_next = torch::multinomial(probs, 1);
            idx = torch::cat({idx, idx_next}, 1);
        }
        return {idx, torch::Tensor()};
    }


private:
    int64_t d_embd, block_size;
    torch::nn::Embedding token_embd, pos_embd;
    torch::nn::ModuleList<EncoderLayer> enc_layer;
    torch::nn::ModuleList<DecoderLayer> dec_layer;
    torch::nn::LayerNorm norm_final;
    torch::nn::Linear lm_head, fc_out;
    torch::nn::Dropout dropout;
};

TORCH_MODULE(Transformer);