"""Ordinary neural networks trained by gradient descent: the baseline for comparison.

Only numpy. One hidden layer of tanh units, sigmoid output, binary
cross-entropy, full-batch Adam, fixed hyperparameters (not tuned per task):
- `MLP` for tasks whose output is a function of one input pattern;
- `RNN` (Elman) for tasks with memory: it reads x(t) and predicts the output
  that answers x(0..t), trained by backpropagation through time;
- `GRU` (experiment 32): the same task with a gated recurrent unit.
"""
import numpy as np


class Adam:
    def __init__(self, params, lr):
        self.p, self.lr, self.t = params, lr, 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for p, g, m, v in zip(self.p, grads, self.m, self.v):
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g * g
            p -= self.lr * (m / (1 - b1 ** self.t)) / (np.sqrt(v / (1 - b2 ** self.t)) + eps)


def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


class MLP:
    def __init__(self, n_in, hidden=64, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1 / np.sqrt(n_in), (n_in, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, 1 / np.sqrt(hidden), hidden)
        self.b2 = np.zeros(1)

    def _forward(self, X):
        H = np.tanh(X @ self.W1 + self.b1)
        return H, _sigmoid(H @ self.w2 + self.b2[0])

    def fit(self, X, y, steps=2000, lr=0.01):
        X, y = np.asarray(X, float), np.asarray(y, float)
        opt = Adam([self.W1, self.b1, self.w2, self.b2], lr)
        for _ in range(steps):
            H, p = self._forward(X)
            d = (p - y) / len(y)                      # d loss / d logit
            dH = np.outer(d, self.w2) * (1 - H * H)
            opt.step([X.T @ dH, dH.sum(0), H.T @ d, np.array([d.sum()])])
        return self

    def predict(self, X):
        return (self._forward(np.asarray(X, float))[1] >= 0.5).astype(int)


class RNN:
    def __init__(self, n_in, hidden=64, seed=0):
        rng = np.random.default_rng(seed)
        self.Wx = rng.normal(0, 1 / np.sqrt(n_in), (n_in, hidden))
        self.Wh = rng.normal(0, 1 / np.sqrt(hidden), (hidden, hidden))
        self.b = np.zeros(hidden)
        self.wo = rng.normal(0, 1 / np.sqrt(hidden), hidden)
        self.bo = np.zeros(1)

    def _forward(self, X):
        """X: (batch, time, n_in). Returns hidden states (time + 1) and outputs."""
        B, T, _ = X.shape
        Hs = [np.zeros((B, len(self.b)))]
        for t in range(T):
            Hs.append(np.tanh(X[:, t] @ self.Wx + Hs[-1] @ self.Wh + self.b))
        P = _sigmoid(np.stack([h @ self.wo for h in Hs[1:]], 1) + self.bo[0])
        return Hs, P

    def fit(self, X, Y, steps=3000, lr=0.005, clip=5.0):
        """X: (batch, time, n_in); Y: (batch, time) targets."""
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        B, T, _ = X.shape
        params = [self.Wx, self.Wh, self.b, self.wo, self.bo]
        opt = Adam(params, lr)
        for _ in range(steps):
            Hs, P = self._forward(X)
            D = (P - Y) / Y.size                       # d loss / d logits
            gWx, gWh, gb = np.zeros_like(self.Wx), np.zeros_like(self.Wh), np.zeros_like(self.b)
            gwo = sum(Hs[t + 1].T @ D[:, t] for t in range(T))
            dh = np.zeros_like(Hs[0])
            for t in reversed(range(T)):
                dh = dh + np.outer(D[:, t], self.wo)
                dz = dh * (1 - Hs[t + 1] ** 2)
                gWx += X[:, t].T @ dz
                gWh += Hs[t].T @ dz
                gb += dz.sum(0)
                dh = dz @ self.Wh.T
            grads = [gWx, gWh, gb, gwo, np.array([D.sum()])]
            norm = np.sqrt(sum((g * g).sum() for g in grads))
            if norm > clip:
                grads = [g * (clip / norm) for g in grads]
            opt.step(grads)
        return self

    def predict(self, X):
        return (self._forward(np.asarray(X, float))[1] >= 0.5).astype(int)


class GRU:
    """Gated recurrent unit (experiment 32): reads x(t), predicts the output
    that answers x(0..t); trained like RNN by backpropagation through time."""

    def __init__(self, n_in, hidden=64, seed=0):
        rng = np.random.default_rng(seed)
        sx, sh = 1 / np.sqrt(n_in), 1 / np.sqrt(hidden)
        self.W = [rng.normal(0, sx, (n_in, hidden)) for _ in range(3)]      # z, r, n
        self.U = [rng.normal(0, sh, (hidden, hidden)) for _ in range(3)]
        self.b = [np.zeros(hidden) for _ in range(3)]
        self.wo = rng.normal(0, sh, hidden)
        self.bo = np.zeros(1)

    def params(self):
        return self.W + self.U + self.b + [self.wo, self.bo]

    def _forward(self, X):
        B, T, _ = X.shape
        Wz, Wr, Wn = self.W
        Uz, Ur, Un = self.U
        bz, br, bn = self.b
        Hs, cache = [np.zeros((B, len(bz)))], []
        for t in range(T):
            h, x = Hs[-1], X[:, t]
            z = _sigmoid(x @ Wz + h @ Uz + bz)
            r = _sigmoid(x @ Wr + h @ Ur + br)
            n = np.tanh(x @ Wn + (r * h) @ Un + bn)
            Hs.append((1 - z) * n + z * h)
            cache.append((z, r, n))
        P = _sigmoid(np.stack([h @ self.wo for h in Hs[1:]], 1) + self.bo[0])
        return Hs, cache, P

    def grads(self, X, Y):
        """(cross-entropy loss, gradients in the order of params())."""
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        B, T, _ = X.shape
        Hs, cache, P = self._forward(X)
        Pc = np.clip(P, 1e-12, 1 - 1e-12)
        loss = -np.mean(Y * np.log(Pc) + (1 - Y) * np.log(1 - Pc))
        D = (P - Y) / Y.size
        gW = [np.zeros_like(w) for w in self.W]
        gU = [np.zeros_like(u) for u in self.U]
        gb = [np.zeros_like(b) for b in self.b]
        gwo = sum(Hs[t + 1].T @ D[:, t] for t in range(T))
        Uz, Ur, Un = self.U
        dh = np.zeros_like(Hs[0])
        for t in reversed(range(T)):
            dh = dh + np.outer(D[:, t], self.wo)
            h, x = Hs[t], X[:, t]
            z, r, n = cache[t]
            dan = dh * (1 - z) * (1 - n * n)
            daz = dh * (h - n) * z * (1 - z)
            drh = dan @ Un.T
            dar = drh * h * r * (1 - r)
            for k, (da, src) in enumerate(((daz, h), (dar, h), (dan, r * h))):
                gW[k] += x.T @ da
                gU[k] += src.T @ da
                gb[k] += da.sum(0)
            dh = dh * z + drh * r + daz @ Uz.T + dar @ Ur.T
        return loss, gW + gU + gb + [gwo, np.array([D.sum()])]

    def fit(self, X, Y, steps=3000, lr=0.005, clip=5.0):
        opt = Adam(self.params(), lr)
        for _ in range(steps):
            grads = self.grads(X, Y)[1]
            norm = np.sqrt(sum((g * g).sum() for g in grads))
            if norm > clip:
                grads = [g * (clip / norm) for g in grads]
            opt.step(grads)
        return self

    def predict(self, X):
        return (self._forward(np.asarray(X, float))[2] >= 0.5).astype(int)
