"""
simple character rnn from Karpathy's blog
"""

import numpy as np

def random_init(num_rows, num_cols):
    return np.random.rand(num_rows, num_cols)*0.01

def zero_init(num_rows, num_cols):
    return np.zeros((num_rows, num_cols))

class DataReader:
    def __init__(self, path, seq_length):
        self.fp = open(path, "r")
        self.data = self.fp.read()
        chars = list(set(self.data))
        self.char_to_ix = {ch:i for (i,ch) in enumerate(chars)}
        self.ix_to_char = {i:ch for (i,ch) in enumerate(chars)}
        self.data_size = len(self.data)
        self.vocab_size = len(chars)
        self.pointer = 0
        self.seq_length = seq_length

    def next_batch(self):
        input_start = self.pointer
        input_end = self.pointer + self.seq_length
        inputs = [self.char_to_ix[ch] for ch in self.data[input_start:input_end]]
        targets = [self.char_to_ix[ch] for ch in self.data[input_start+1:input_end+1]]
        self.pointer += self.seq_length
        if self.pointer + self.seq_length + 1 >= self.data_size:
            # reset pointer
            self.pointer = 0
        return inputs, targets

    def just_started(self):
        return self.pointer == 0

    def close(self):
        self.fp.close()

class SimpleRNN:
    def __init__(self, hidden_size, vocab_size, seq_length, learning_rate):
        # hyper parameters
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate

        # model parameters
        self.Wxh = random_init(hidden_size, vocab_size) # input to hidden
        self.Whh = random_init(hidden_size, hidden_size) # hidden to hidden
        self.Why = random_init(vocab_size, hidden_size) # hidden to output
        self.bh = zero_init(hidden_size, 1) # bias for hidden layer
        self.by = zero_init(vocab_size, 1) # bias for output

        # memory vars for adagrad
        self.mWxh = np.zeros_like(self.Wxh)
        self.mWhh = np.zeros_like(self.Whh)
        self.mWhy = np.zeros_like(self.Why)
        self.mbh = np.zeros_like(self.bh)
        self.mby = np.zeros_like(self.by)


    def forward(self, inputs, hprev):
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        for t in range(len(inputs)):
            xs[t] = zero_init(self.vocab_size,1)
            xs[t][inputs[t]] = 1 # one hot encoding , 1-of-k
            hs[t] = np.tanh(np.dot(self.Wxh,xs[t]) + np.dot(self.Whh,hs[t-1]) + self.bh) # hidden state
            ys[t] = np.dot(self.Why,hs[t]) + self.by # unnormalised log probs for next char
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probs for next char
        return xs, hs, ps

    def backward(self, xs, hs, ps, targets):
        # backward pass: compute gradients going backwards
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])
        for t in reversed(range(self.seq_length)):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1 # backprop into y
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dhnext # backprop into h
            dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh non-linearity
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            dhnext = np.dot(self.Whh.T, dhraw)
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
        return dWxh, dWhh, dWhy, dbh, dby

    def loss(self, ps, targets):
        """loss for a sequence"""
        return sum(-np.log(ps[t][targets[t],0]) for t in range(self.seq_length))

    def update_model(self, dWxh, dWhh, dWhy, dbh, dby):
        # parameter update with adagrad
        for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
                                  [dWxh, dWhh, dWhy, dbh, dby],
                                  [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]):
            mem += dparam*dparam
            param += -self.learning_rate*dparam/np.sqrt(mem+1e-8) # adagrad update

    def sample(self, h, seed_ix, n):
        """
        sample a sequence of integers from the model
        h is memory state, seed_ix is seed letter from the first time step
        """
        x = zero_init(self.vocab_size, 1)
        x[seed_ix] = 1
        ixes = []
        for t in range(n):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            p = np.exp(y)/np.sum(np.exp(y))
            ix = np.random.choice(range(self.vocab_size), p = p.ravel())
            x = zero_init(self.vocab_size,1)
            x[ix] = 1
            ixes.append(ix)
        return ixes

    def train(self, data_reader):
        iter_num = 0
        smooth_loss = -np.log(1.0/data_reader.vocab_size)*self.seq_length
        while True:
            if data_reader.just_started():
                hprev = zero_init(self.hidden_size,1)
            inputs, targets = data_reader.next_batch()
            xs, hs, ps = self.forward(inputs, hprev)
            dWxh, dWhh, dWhy, dbh, dby = self.backward(xs, hs, ps, targets)
            loss = self.loss(ps, targets)
            self.update_model(dWxh, dWhh, dWhy, dbh, dby)
            smooth_loss = smooth_loss*0.999 + loss*0.001
            hprev = hs[self.seq_length-1]
            if not iter_num%500:
                sample_ix = self.sample(hprev, inputs[0], 200)
                print("".join(data_reader.ix_to_char[ix] for ix in sample_ix))
                print("\n\niter :%d, loss:%f"%(iter_num, smooth_loss))
            iter_num += 1




if __name__ == "__main__":
    seq_length = 25
    data_reader = DataReader("input.txt", seq_length)
    rnn = SimpleRNN(hidden_size=100, vocab_size=data_reader.vocab_size,seq_length=seq_length,learning_rate=1e-1)
    rnn.train(data_reader)