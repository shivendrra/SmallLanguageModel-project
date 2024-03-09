import torch
import json
import os
os.chdir('D:/Machine Learning/SLM-Project')

with open('Final Models/Transformer/hyperparams.json', 'r', encoding='utf-8') as file:
    params = json.load(file)

class TrainBiGramModel:
    def __init__(self, model, optimizer, train_data, val_data, batch_size, block_size):
        self.max_iters = params['max_iters']
        self.eval_interval = params['eval_interval']
        self.eval_iters = params['eval_iters']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.optimizer = optimizer
        self.train_data = train_data
        self.val_data = val_data
        self.block_size = block_size
        self.batch_size = batch_size

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + self.block_size + 1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y

    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = self.get_batch(split)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
            self.model.train()
        return out

    def train_model(self):
        for iter in range(self.max_iters):
            if iter % self.eval_interval == 0 or iter == self.max_iters - 1:
                losses = self.estimate_loss()
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            xb, yb = self.get_batch('train')
            # For the bi-gram model, we need to consider only the last two tokens
            xb = xb[:, -2:, :]
            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
