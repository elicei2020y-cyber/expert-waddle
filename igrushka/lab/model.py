"""Модель под испытанием, квантование и модели порчи.

Модель символьная: число рождается ПО ЦИФРЕ, как в настоящей языковой
модели. Внимания нет -- это окно фиксированной длины, и выдавать её
за языковую модель нельзя. Проверяется на ней ФОРМА порчи, не её величина.
"""
import numpy as np, math, random, os
from core import make_report, render, parse, KEYS8, DEC, rnd

VOC = sorted(set("0123456789.= \n") | set("ndfMsetp"))
S2I = {c: i for i, c in enumerate(VOC)}
V = len(VOC)
K, E, H = 64, len(VOC), 384
EYE = np.eye(V)
CACHE = os.path.join(os.path.dirname(__file__), "model.npz")


def corpus(n=60, seed=3):
    r = random.Random(seed)
    seen, out = set(), []
    while len(out) < n:
        rep = make_report(r)
        if rep["n"] in seen:
            continue
        s = render(rep)
        if all(c in S2I for c in s):
            seen.add(rep["n"]); out.append((rep, s))
    return out


def _xy(data):
    X, Y = [], []
    for _, s in data:
        pad = " " * K + s
        for i in range(len(s)):
            X.append([S2I[c] for c in pad[i:i + K]])
            Y.append(S2I[s[i]])
    return np.array(X), np.array(Y)


def _fwd(p, X):
    e = EYE[X].reshape(len(X), -1)
    h = np.maximum(e @ p["w1"] + p["b1"], 0)
    z = h @ p["w2"] + p["b2"]
    z -= z.max(-1, keepdims=True)
    ex = np.exp(z)
    return e, h, ex / ex.sum(-1, keepdims=True)


def train(data=None, steps=2800, lr=0.4, bs=256, seed=7, verbose=True):
    rng = np.random.default_rng(seed)
    data = data or corpus()
    X, Y = _xy(data)
    p = dict(w1=rng.normal(0, math.sqrt(2 / (K * E)), (K * E, H)),
             b1=np.zeros(H),
             w2=rng.normal(0, math.sqrt(2 / H), (H, V)),
             b2=np.zeros(V))
    for t in range(steps):
        i = rng.integers(0, len(X), bs)
        xb, yb = X[i], Y[i]
        e, h, sm = _fwd(p, xb)
        d = sm.copy(); d[np.arange(bs), yb] -= 1; d /= bs
        gw2 = h.T @ d; gb2 = d.sum(0)
        dh = (d @ p["w2"].T) * (h > 0)
        p["w2"] -= lr * gw2; p["b2"] -= lr * gb2
        p["w1"] -= lr * (e.T @ dh); p["b1"] -= lr * dh.sum(0)
        if verbose and t % 700 == 0:
            _, _, s2 = _fwd(p, X[:1500])
            nll = -np.log(s2[np.arange(1500), Y[:1500]] + 1e-12).mean()
            print(f"    шаг {t:>5}  кросс-энтропия {nll:.4f}", flush=True)
    return p


def load(force=False):
    """Обученная модель из кэша; при отсутствии -- обучить и сохранить."""
    if os.path.exists(CACHE) and not force:
        return dict(np.load(CACHE))
    print("  обучение модели (около 4 минут, один раз)...")
    p = train()
    np.savez(CACHE, **p)
    return p


def generate(p, prefix, maxlen=70):
    s = prefix
    for _ in range(maxlen):
        pad = (" " * K + s)[-K:]
        _, _, sm = _fwd(p, np.array([[S2I[c] for c in pad]]))
        c = VOC[int(sm[0].argmax())]
        if c == "\n":
            break
        s += c
    return s


def emit(p, data):
    """Разобранные отчёты, порождённые моделью по префиксу n=..."""
    return [parse(generate(p, f"n={rep['n']:g} ")) for rep, _ in data]


# --------------------------------------------------------- квантование
def q_det(w, bits):
    if bits >= 16:
        return w.copy()
    q = 2 ** (bits - 1) - 1
    sc = np.abs(w).max() / q
    return w.copy() if sc == 0 else np.clip(np.round(w / sc), -q - 1, q) * sc


def q_sto(w, bits, g):
    if bits >= 16:
        return w.copy()
    q = 2 ** (bits - 1) - 1
    sc = np.abs(w).max() / q
    if sc == 0:
        return w.copy()
    x = w / sc
    fl = np.floor(x)
    return np.clip(fl + (g.random(x.shape) < (x - fl)), -q - 1, q) * sc


def quantized(p, bits, stochastic=None, which=("w1", "w2")):
    f = (lambda w: q_sto(w, bits, stochastic)) if stochastic is not None \
        else (lambda w: q_det(w, bits))
    return {k: (f(v) if k in which else v.copy()) for k, v in p.items()}


# -------------------------------------------------------- модели порчи
def corrupt_prop(rep, u, rng, lo=0.08, hi=0.6):
    """Пропорциональный сдвиг -- модель, на которой строился раздел о кодах."""
    out = dict(rep)
    for _ in range(60):
        f = rng.uniform(lo, hi) * rng.choice([-1, 1])
        if u in ("n", "df"):
            nv = max(4, rep[u] + max(1, round(abs(f) * rep[u])) * (1 if f > 0 else -1))
        elif u == "p":
            nv = rnd(min(0.999, max(1e-5, rep[u] * (1 + f))), DEC[u])
        elif u == "M":
            nv = rnd(min(7.0, max(1.0, rep[u] * (1 + f))), DEC[u])
        else:
            nv = rnd(rep[u] * (1 + f), DEC[u])
        if nv != rep[u]:
            out[u] = nv
            return out
    return None


def corrupt_digit(rep, u, rng):
    """Подмена одной цифры -- то, что делает потокенное порождение."""
    s = f"{rep[u]:g}"
    pos = [i for i, c in enumerate(s) if c.isdigit()]
    if not pos:
        return None
    for _ in range(40):
        i = rng.choice(pos)
        c = rng.choice([d for d in "0123456789" if d != s[i]])
        cand = s[:i] + c + s[i + 1:]
        try:
            nv = rnd(float(cand), DEC[u])
        except ValueError:
            continue
        if nv != rep[u] and (u not in ("n", "df") or nv >= 4):
            out = dict(rep); out[u] = nv
            return out
    return None


CORRUPTIONS = {"prop": corrupt_prop, "digit": corrupt_digit}
