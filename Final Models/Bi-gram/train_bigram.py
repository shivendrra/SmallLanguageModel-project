import torch

def get_batch(data, block_size, batch_size, device):
  if len(data) < block_size:
      # Handle the case when len(data) is less than block_size
    print("Warning!!: Data length is less than block_size. { Skipping batch }")
    return None, None
  
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i + block_size] for i in ix])
  y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y

@torch.no_grad()
def estimate_loss(model, eval_iters, train_data, val_data, block_size, device):
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    data = train_data if split == 'train' else val_data
    for k in range(eval_iters):
      X, Y = get_batch(data, block_size, 1, device)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

def train_model(model, optimizer, max_iters, eval_interval, eval_iters, train_data, val_data, block_size, batch_size, device):
  train_losses = []
  for iter in range(max_iters):
    
    if iter % eval_interval == 0 or iter == max_iters - 1:
      losses = estimate_loss(model, eval_iters, train_data, val_data, block_size, device)
      print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    xb, yb = get_batch(train_data, block_size, batch_size, device)
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
  
  return iter, {'train': sum(train_losses) / len(train_losses)}