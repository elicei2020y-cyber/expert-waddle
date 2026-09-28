"""Двадцать рычагов против порчи от квантования -- и их отстрел.

Рабочая точка: 4 бита. Там символьная модель ещё держит формат
(112/120), но плотность порчи 4.28 из 8 -- то есть порча уже плотная
и локализация не работает. Если рычаг что-то стоит, он виден здесь.

Мера одна и та же для всех: сколько отчётов воспроизведены ТОЧНО так,
как их выдала неквантованная модель, и какова плотность расхождений.
Знаменатель печатается всегда.
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import KEYS8, parse, analyse, CATALOGS, rnd, DEC, make_report
import model as M

BITS = 4
SEED = 101


# ---------- рычаги на весах -------------------------------------------
def q_perchannel(w, bits, g=None):
    """13. Масштаб на столбец, а не на тензор."""
    q = 2 ** (bits - 1) - 1
    sc = np.abs(w).max(axis=0, keepdims=True) / q
    sc = np.where(sc == 0, 1.0, sc)
    return np.clip(np.round(w / sc), -q - 1, q) * sc


def q_outlier(w, bits, frac=0.01):
    """16. Верхняя доля весов по модулю остаётся в полной точности."""
    q = 2 ** (bits - 1) - 1
    a = np.abs(w)
    thr = np.quantile(a, 1 - frac)
    keep = a >= thr
    body = w[~keep]
    sc = np.abs(body).max() / q if body.size else 1.0
    out = w.copy()
    out[~keep] = np.clip(np.round(body / sc), -q - 1, q) * sc
    return out


def bias_correct(p, pq, data):
    """12. Сдвиг средней активации, внесённый квантованием, гасится смещениями."""
    X, _ = M._xy(data)
    X = X[:2000]
    e = M.EYE[X].reshape(len(X), -1)
    h0 = np.maximum(e @ p["w1"] + p["b1"], 0)
    h1 = np.maximum(e @ pq["w1"] + pq["b1"], 0)
    out = {k: v.copy() for k, v in pq.items()}
    out["b1"] = pq["b1"] + (e @ p["w1"] + p["b1"] - (e @ pq["w1"] + pq["b1"])).mean(0)
    z0 = h0 @ p["w2"] + p["b2"]
    z1 = h1 @ pq["w2"] + pq["b2"]
    out["b2"] = pq["b2"] + (z0 - z1).mean(0)
    return out


# ---------- декодирование ---------------------------------------------
def gen_ensemble(ps, prefix, maxlen=70):
    """11. Усреднение распределений по нескольким стохастическим квантованиям."""
    s = prefix
    for _ in range(maxlen):
        pad = (" " * M.K + s)[-M.K:]
        x = np.array([[M.S2I[c] for c in pad]])
        sm = sum(M._fwd(p, x)[2][0] for p in ps) / len(ps)
        c = M.VOC[int(sm.argmax())]
        if c == "\n":
            break
        s += c
    return s


def gen_conf(p, prefix, maxlen=70, thr=0.6):
    """10. Запись уверенности: минимальная вероятность выбранного символа."""
    s = prefix
    low = 0
    for _ in range(maxlen):
        pad = (" " * M.K + s)[-M.K:]
        _, _, sm = M._fwd(p, np.array([[M.S2I[c] for c in pad]]))
        i = int(sm[0].argmax())
        if sm[0][i] < thr:
            low += 1
        c = M.VOC[i]
        if c == "\n":
            break
        s += c
    return s, low


# ---------- рычаги на выходе ------------------------------------------
FREE = ("n", "M", "sd")


def rederive(rep):
    """5. Производные величины не берутся у модели, а считаются."""
    if not all(k in rep for k in FREE):
        return rep
    import math
    from core import MU0, p_two_tailed
    n, Mv, sd = rep["n"], rep["M"], rep["sd"]
    if n < 2 or sd <= 0:
        return rep
    se = sd / math.sqrt(n)
    t = (Mv - MU0) / se
    out = dict(rep)
    out["df"] = rnd(n - 1, 0)
    out["se"] = rnd(se, DEC["se"])
    out["t"] = rnd(t, DEC["t"])
    out["d"] = rnd((Mv - MU0) / sd, DEC["d"])
    try:
        out["p"] = rnd(p_two_tailed(abs(t), n - 1), DEC["p"])
    except Exception:
        pass
    return out


def intersect(a, b):
    """7. Оставить только то, в чём два уровня квантования согласны."""
    return {k: v for k, v in a.items() if b.get(k) == v}


# ---------- прогон ----------------------------------------------------
def measure(name, reports, base):
    exact = dens = n = 0
    for bs, got in zip(base, reports):
        if len(bs) != 8:
            continue
        n += 1
        if len(got) != 8:
            dens += 8
            continue
        chg = sum(1 for k in KEYS8 if got.get(k) != bs[k])
        dens += chg
        exact += (chg == 0)
    return dict(name=name, n=n, exact=exact, dens=dens / n if n else float("nan"))


def run():
    p = M.load()
    data = M.corpus()
    base = M.emit(p, data)
    g = np.random.default_rng(SEED)
    plain = {k: (M.q_sto(v, BITS, g) if k in ("w1", "w2") else v.copy())
             for k, v in p.items()}
    rows = []

    def emit_with(pp):
        return [parse(M.generate(pp, f"n={r['n']:g} ")) for r, _ in data]

    rows.append(measure("0. без рычага (4 бита)", emit_with(plain), base))

    pc = {k: (q_perchannel(v, BITS) if k in ("w1", "w2") else v.copy())
          for k, v in p.items()}
    rows.append(measure("13. масштаб на столбец", emit_with(pc), base))

    ol = {k: (q_outlier(v, BITS) if k in ("w1", "w2") else v.copy())
          for k, v in p.items()}
    rows.append(measure("16. 1% выбросов в fp", emit_with(ol), base))

    h2 = {k: (M.q_det(v, 8) if k == "w2" else
              (M.q_det(v, BITS) if k == "w1" else v.copy())) for k, v in p.items()}
    rows.append(measure("14. голова в 8 битах", emit_with(h2), base))

    h1 = {k: (M.q_det(v, 8) if k == "w1" else
              (M.q_det(v, BITS) if k == "w2" else v.copy())) for k, v in p.items()}
    rows.append(measure("15. первый слой в 8 битах", emit_with(h1), base))

    bc = bias_correct(p, plain, data)
    rows.append(measure("12. коррекция смещений", emit_with(bc), base))

    ens = [{k: (M.q_sto(v, BITS, np.random.default_rng(200 + i))
                if k in ("w1", "w2") else v.copy()) for k, v in p.items()}
           for i in range(3)]
    rows.append(measure("11. ансамбль 3 квантований",
                        [parse(gen_ensemble(ens, f"n={r['n']:g} ")) for r, _ in data],
                        base))

    g2 = np.random.default_rng(SEED + 1)
    other = {k: (M.q_sto(v, BITS, g2) if k in ("w1", "w2") else v.copy())
             for k, v in p.items()}
    a, b = emit_with(plain), emit_with(other)
    rows.append(measure("7. пересечение двух уровней",
                        [rederive(intersect(x, y)) for x, y in zip(a, b)], base))

    rows.append(measure("5. производные пересчитаны",
                        [rederive(x) for x in a], base))

    print(f"  {'рычаг':<30}{'точно':>16}{'плотность':>12}")
    print("  " + "-" * 58)
    for r in rows:
        frac = "{}/{}".format(r["exact"], r["n"])
        print("  {:<30}{:>16}{:>12.2f}".format(r["name"], frac, r["dens"]))


if __name__ == "__main__":
    run()
