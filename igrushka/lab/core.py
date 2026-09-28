"""Ядро лаборатории: отчёты, отношения, интервалы, сигнатуры, охранник.

Всё сравнения обрываются на арифметике или строках. Ни одна функция здесь
не выносит суждения о правдоподобии.
"""
import math, random, re
from itertools import combinations
import sys
from tstats import p_two_tailed, t_from_p

MU0 = 4.0
DEC = {"n": 0, "df": 0, "M": 2, "sd": 2, "se": 2, "t": 2, "p": 3, "d": 2,
       "ci_lo": 2, "ci_hi": 2}
KEYS8 = ["n", "df", "M", "sd", "se", "t", "p", "d"]


def rnd(v, k):
    """Округление половины вверх, а не банковское."""
    return math.floor(v * 10 ** k + 0.5) / 10 ** k


def box(name, val):
    h = 0.5 * 10 ** (-DEC[name])
    return (val - h, val + h)


from functools import lru_cache

@lru_cache(maxsize=4096)
def _crit(k):
    return t_from_p(0.05, max(k / 8.0, 1.0))

def crit(df):
    """Критическое значение с кэшем по сетке в 1/8 степени свободы."""
    return _crit(round(df * 8))


# ---------------------------------------------------------------- отчёты
def make_report(rng, nmin=12, nmax=160, with_ci=False):
    """Внутренне согласованный отчёт по шкале Лайкерта 1..7."""
    while True:
        n = rng.randint(nmin, nmax)
        items = [rng.randint(1, 7) for _ in range(n)]
        M = sum(items) / n
        sd = math.sqrt(sum((x - M) ** 2 for x in items) / (n - 1))
        if sd < 0.3:
            continue
        se = sd / math.sqrt(n)
        t = (M - MU0) / se
        if not (0.8 < abs(t) < 12):
            continue
        p = p_two_tailed(abs(t), n - 1)
        if p < 1e-4:
            continue
        r = dict(n=n, df=n - 1, M=M, sd=sd, se=se, t=t, p=p, d=(M - MU0) / sd)
        if with_ci:
            c = crit(n - 1)
            r["ci_lo"] = M - c * se
            r["ci_hi"] = M + c * se
        return {k: rnd(v, DEC[k]) for k, v in r.items()}


def render(rep, keys=KEYS8):
    return " ".join(f"{k}={rep[k]:g}" for k in keys) + "\n"


PAIR = re.compile(r"([A-Za-z_]+)\s*=\s*([-+]?\d*\.?\d+)")


def parse(txt, keys=KEYS8):
    got = {}
    for m in PAIR.finditer(txt):
        k = m.group(1)
        if k in keys and k not in got:
            try:
                got[k] = rnd(float(m.group(2)), DEC[k])
            except ValueError:
                pass
    return got


# ------------------------------------------------------------ отношения
POOL = {
    "R1": (("df", "n"), lambda v: v["df"] - (v["n"] - 1)),
    "R2": (("t", "M", "se"), lambda v: v["t"] - (v["M"] - MU0) / v["se"]),
    "R3": (("p", "t", "df"), lambda v: v["p"] - p_two_tailed(abs(v["t"]), max(v["df"], 1.0))),
    "R4": (("ci_lo", "M", "se", "df"), lambda v: v["ci_lo"] - (v["M"] - crit(v["df"]) * v["se"])),
    "R5": (("ci_hi", "M", "se", "df"), lambda v: v["ci_hi"] - (v["M"] + crit(v["df"]) * v["se"])),
    "R6": (("se", "sd", "n"), lambda v: v["se"] - v["sd"] / math.sqrt(max(v["n"], 1e-9))),
    "R7": (("d", "M", "sd"), lambda v: v["d"] - (v["M"] - MU0) / v["sd"]),
    "R8": (("d", "t", "n"), lambda v: v["d"] - v["t"] / math.sqrt(max(v["n"], 1e-9))),
}
INVOLVES = {r: set(nm) for r, (nm, _) in POOL.items()}

CATALOGS = {
    "current": ["R1", "R2", "R3"],              # то, что построено в проекте
    "with_ci": ["R1", "R2", "R3", "R4", "R5"],  # +доверительный интервал
    "K1": ["R2", "R6", "R7", "R8"],             # лучший по выводу на 6 величинах
    "K2": ["R1", "R2", "R3", "R6", "R7", "R8"], # на 8 величинах
    "full": list(POOL),
}


def sig(v, catalog):
    return {r for r in catalog if v in INVOLVES[r]}


def variables(catalog):
    return sorted({v for r in catalog for v in INVOLVES[r]})


def localizable(catalog):
    """Теорема: u локализуема <=> S(u) не вложено ни в какое S(v), v != u."""
    V = variables(catalog)
    return {u: not any(sig(u, catalog) <= sig(w, catalog) for w in V if w != u)
            for u in V}


def holds(rname, rep):
    """Угловой метод по интервалам от точности публикации."""
    names, f = POOL[rname]
    if any(k not in rep or rep[k] is None for k in names):
        return True
    bs = [box(nm, rep[nm]) for nm in names]
    lo = hi = None
    for mask in range(1 << len(names)):
        v = {nm: (b[1] if (mask >> i) & 1 else b[0])
             for i, (nm, b) in enumerate(zip(names, bs))}
        try:
            r = f(v)
        except (ZeroDivisionError, ValueError):
            return True
        if not math.isfinite(r):
            return True
        lo = r if lo is None else min(lo, r)
        hi = r if hi is None else max(hi, r)
    return lo <= 0 <= hi


BRACKET = {"n": (3.0, 1e5), "df": (2.0, 1e5), "p": (1e-6, 0.9999),
           "M": (1.0, 7.0), "sd": (1e-3, 10.0), "se": (1e-4, 10.0),
           "t": (-1e3, 1e3), "d": (-1e3, 1e3), "ci_lo": (-50., 50.),
           "ci_hi": (-50., 50.)}
SKIPPED = [0, 0]   # [пропущено углов, всего] -- смещение проверки, называется в отчётах


def solve_interval(rname, v, rep):
    """Интервал значений v, при которых отношение выполнимо."""
    names, f = POOL[rname]
    others = [nm for nm in names if nm != v]
    if any(nm not in rep for nm in others):
        return None, None
    bs = [box(nm, rep[nm]) for nm in others]
    lo_b, hi_b = BRACKET[v]
    lo = hi = None
    for mask in range(1 << len(others)):
        SKIPPED[1] += 1
        vals = {nm: (b[1] if (mask >> i) & 1 else b[0])
                for i, (nm, b) in enumerate(zip(others, bs))}

        def g(x):
            vv = dict(vals); vv[v] = x
            try:
                r = f(vv)
            except (ZeroDivisionError, ValueError):
                return None
            return r if math.isfinite(r) else None
        a, b_ = lo_b, hi_b
        ga, gb = g(a), g(b_)
        if ga is None or gb is None or ga * gb > 0:
            SKIPPED[0] += 1
            continue
        for _ in range(34):
            if b_ - a < 1e-6 * max(1.0, abs(a)):
                break
            m = 0.5 * (a + b_); gm = g(m)
            if gm is None:
                break
            if ga * gm <= 0: b_, gb = m, gm
            else: a, ga = m, gm
        root = 0.5 * (a + b_)
        lo = root if lo is None else min(lo, root)
        hi = root if hi is None else max(hi, root)
    return lo, hi


def agrees(v, rep, catalog):
    """Согласны ли причастные отношения насчёт значения v."""
    lo, hi = -math.inf, math.inf
    for r in catalog:
        if v not in INVOLVES[r]:
            continue
        a, b = solve_interval(r, v, rep)
        if a is None:
            continue
        lo, hi = max(lo, a), min(hi, b)
        if lo > hi:
            return False
    return True


def min_cover(viol, present):
    """Нижняя граница числа порч: наименьшее покрытие нарушенных отношений."""
    if not viol:
        return 0
    pool = [v for v in present if any(v in INVOLVES[r] for r in viol)]
    for k in range(1, len(pool) + 1):
        if any(all(any(v in INVOLVES[r] for v in c) for r in viol)
               for c in combinations(pool, k)):
            return k
    return len(pool)


def analyse(rep, catalog=CATALOGS["K2"], strict=False):
    """Полный разбор: нарушения, кандидаты, нижняя граница."""
    viol = {r for r in catalog if not holds(r, rep)}
    present = [v for v in variables(catalog) if v in rep and rep[v] is not None]
    cands = [v for v in present if sig(v, catalog) and not (viol - sig(v, catalog))]
    if strict and len(cands) > 0:
        cands = [v for v in cands if agrees(v, rep, catalog)]
    return dict(viol=sorted(viol), cands=cands,
                lower_bound=min_cover(viol, present))
