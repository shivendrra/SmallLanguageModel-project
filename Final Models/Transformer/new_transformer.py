import torch
import numpy as np

class nn:
    class Modules:
        """ neural network modules """
        class TokenEmbeddings:
            def __init__(self, vocab_size, n_embd):
                self.vocab_size = vocab_size
                self.n_embd = n_embd
                self.token_embeddings = self.forward()

            def forward(self):
                token_embeddings = torch.randn(vocab_size, n_embd)
                token_embeddings /= torch.norm(token_embeddings, dim=1, keepdim=True)
                return token_embeddings

        class PositionalEmbeddings:
            def __init__(self, block_size, n_embd):
                self.block_size = block_size
                self.n_embd = n_embd
                self.positional_embeddings = self.forward()

            def forward(self):
                position_embeddings = torch.zeros(block_size, n_embd)
                for pos in range(block_size):
                    for i in range(0, n_embd, 2):
                        position = torch.tensor(pos, dtype=torch.float)
                        exponent = torch.tensor(i / n_embd, dtype=torch.float)
                        position_embeddings[pos, i] = torch.sin(position / (1e4 ** exponent))
                        position_embeddings[pos, i+1] = torch.cos(position / (1e4 ** exponent))
                return position_embeddings

        class Sequential:
            def __init__(self, *modules):
                self.modules = modules

            def process(self, x):
                for module in self.modules:
                    x = module(x)
                return x
        
        class LinearLayer:
            def __init__(self, input_dim, output_dim, bias=True):
                self.input_dim = input_dim
                self.output_dim = output_dim
                self.weights = torch.randn(output_dim, input_dim)
                self.bias = torch.zeros(output_dim) if bias else None
            
            def forward(self, x):
                output = torch.matmul(x, self.weights.t())
                if self.bias is not None:
                    output += self.bias
                return output

        class LayerNorm:
            def __init__(self, n_features, eps=1e-5):
                self.n_features = n_features
                self.eps = eps
                self.gamma = torch.ones(n_features)
                self.beta = torch.zeros(n_features)

            def forward(self, x):
                # mean and standard deviation along the feature dimension
                mean = torch.mean(x, dim=-1, keepdim=True)
                std = torch.std(x, dim=-1, keepdim=True)
                
                # normalization
                x_normalized = (x - mean) / (std + self.eps)
                
                # scale and shift
                y = self.gamma * x_normalized + self.beta
                return y
            
        class Dropout:
            def __init__(self, dropout_prob):
                self.dropout_prob = dropout_prob
                self.mask = None
            
            def forward(self, x, training=True):
                if training:
                    self.mask = (np.random.rand(*x.shape) < self.dropout_prob) / self.dropout_prob
                    return x * self.mask
                else:
                    return x
        
        class ModuleList:
            def __init__(self, modules):
                self.modules = modules

            def append(self, module):
                self.modules.append(module)

            def __getitem__(self, index):
                return self.modules[index]

            def __len__(self):
                return len(self.modules)

    
    class Functionals:
        """ functionals """
        class CrossEntropy:
            def __init__(self, logits, targets, ignore_index=None):
                self.logits = logits
                self.targets = targets
                self.ignore_idx = ignore_index

            def forward(self):
                logits = np.array(self.logits)
                targets = np.array(self.targets)

                if self.ignore_idx is not None:
                    mask = (targets != self.ignore_idx)
                    logits = logits[mask]
                    targets = targets[mask]

                exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                softmax = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
                cross_entropy = -np.log(softmax[np.arange(len(targets)), targets])
                mean_loss = np.mean(cross_entropy)
                return mean_loss
    
        class Softmax:
            def __call__(self, x, axis=-1):
                x -= np.max(x, axis=axis, keepdims=True)
                exp_x = np.exp(x)
                softmax_probs = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
                return softmax_probs

        class GELU:
            @staticmethod
            def error_function(x):
                """Error function (erf) implementation using NumPy."""
                return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

            @staticmethod
            def forward(x):
                """Gaussian Error Linear Unit (GELU) activation function."""
                return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

            def __call__(self, x):
                return self.forward(x)

#---------------------

class SingleHead(nn.Modules):
    def __init__(self, n_embd, n_head, dropout, block_size):
        head_size = n_embd // n_head
        super().__init__()
        self.key = nn.Modules.LinearLayer(n_embd, head_size, bias=False)
        self.query = nn.Modules.LinearLayer(n_embd, head_size, bias=False)
        self.value = nn.Modules.LinearLayer(n_embd, head_size, bias=False)
        # self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Modules.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
    
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * (C // self.tril.size(-1))**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = nn.Functionals.Softmax(wei, dim=-1)
        wei = self.dropout.forward(wei)
    
        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention:
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        self.head = nn.Modules.ModuleList([SingleHead(n_embd, n_head, dropout, block_size) for _ in range(n_head)])
        self.proj = nn.Modules.LinearLayer(n_head * (n_embd // n_head), n_embd)
        self.dropout = nn.Modules.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.head], dim=-1)
        out = self.dropout.forward(out)

        return out
class FeedForward(nn.Modules):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Modules.Sequential(
            nn.Modules.LinearLayer(n_embd, 4*n_embd),
            nn.Functionals.GELU(),
            nn.Modules.LinearLayer(4*n_embd, n_embd),
            nn.Modules.Dropout(dropout)
            )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Modules):
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.attention = MultiHeadAttention(n_head, head_size, dropout, block_size)
        self.feedForward = FeedForward(n_embd, dropout)
        self.linear1 = nn.Modules.LayerNorm(n_embd)
        self.linear2 = nn.Modules.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.attention(self.linear1.forward(x))
        x = x + self.feedForward(self.linear2.forward(x))
        return x

class CustomTransformerModel(nn.Modules):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.token_embedding_table = nn.Modules.TokenEmbeddings(vocab_size, n_embd).forward()
        self.position_embedding_table = nn.Modules.PositionalEmbeddings(block_size, n_embd).forward()
        self.blocks = nn.Modules.Sequential(*[Block(n_embd, n_head, dropout, block_size) for _ in range(n_layer)])
        self.linear_final = nn.Modules.LayerNorm(n_embd)
        self.lm_head = nn.Modules.LinearLayer(n_embd, vocab_size)
        self.init_weights()

    def init_weights(self):
        for module in [self.linear_final, self.lm_head]:
            # initialize weights
            torch.nn.init.normal_(module.weights, mean=0.0, std=0.02)
            # initialize bias if present
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        tok_emb = self.token_embedding_table[idx]  # (B, T, C)
        pos_emb = self.position_embedding_table[:tok_emb.shape[1]]  # (T, C)

        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x) # (B,T,C)
        x = self.linear_final(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = nn.Functionals.CrossEntropy(logits, targets, ignore_index=-52)
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.7, top_k=None):
        generated_sequence = idx.clone()

        for _ in range(max_new_tokens):
            idx_cond = generated_sequence[:, -self.block_size:]
            logits, _ = self(idx_cond)

            # temperature scaling
            scaled_logits = logits / temperature

            # top-k sampling
            if top_k is not None:
                sorted_logits, sorted_indices = torch.sort(scaled_logits[:, -1, :], descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_k
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                for idx in range(sorted_indices_to_remove.size(0)):
                    scaled_logits[idx, -1, sorted_indices_to_remove[idx]] = float('-inf')

            probs = nn.Functionals.Softmax(scaled_logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_sequence = torch.cat((generated_sequence, next_token), dim=1)

        return generated_sequence

batch_size = 64
vocab_size = 483
block_size = 128
n_embd = 8
n_head = 4
n_layer = 4
dropout = 0.1
idx = torch.randint(0, vocab_size, (batch_size, block_size))

model = CustomTransformerModel(vocab_size, block_size, n_embd, n_head, n_layer, dropout)
token_embeddings = model.token_embedding_table
position_embeddings = model.position_embedding_table
output = model.forward(idx)

print("inputs: ", idx.shape)
print("Token Embeddings Shape:", token_embeddings.shape)
print("Positional Embeddings Shape:", position_embeddings.shape)
print("Output Shape:", output.shape)