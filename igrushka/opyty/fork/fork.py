"""Две линии от точки дефекта: отмотанная и оставленная.

ЗАМЫСЕЛ ЗАКАЗЧИКА. При обнаружении дефекта раздвоить порождение: одна
ветвь отматывается и перепорождается, другая продолжается с ошибкой.
Сравнить, что вышло. Прямая проверка того, стоит ли отмотка своей цены.

ОБЛАСТЬ ДЕЙСТВИЯ ЗАМЕРА. Подложка -- символьная модель лаборатории,
про язык и смысл она не скажет ничего. Но механизм под проверкой в ней
настоящий: порождение авторегрессивное, поля выдаются по очереди
n df M sd se t p d, и испорченное значение попадает в контекст всех
следующих. Именно распространение ошибки и цена отмотки и меряются.

ЗАВИСИМОСТИ (они же граф отмотки):
    df <- n          (R1)
    se <- sd, n      (R6)
    t  <- M, se      (R2)
    p  <- t, df      (R3)
    d  <- M, sd      (R7)  либо  t, n  (R8)
    M, sd -- свободные, ни из чего не следуют

ПРЕДСКАЗАНИЯ, ЗАПИСАННЫЕ ДО ПРОГОНА.

  П1. Ошибка распространяется: в оставленной ветви поля после дефекта
      портятся заметно чаще, чем до него.

  П2. Отмотка с ЖАДНЫМ перепорождением и без пометки не даст НИЧЕГО:
      контекст тот же, выбор детерминирован, выйдет тот же знак в знак
      результат. Пометка -- не украшение, она единственное, что меняет
      контекст.

  П3. Отмотка к происхождению зависимости побьёт отмотку на один шаг,
      потому что причина часто лежит раньше места обнаружения.
"""
import sys, os, numpy as np
from collections import Counter
sys.path.insert(0, __import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "lab")))
from core import KEYS8, DEC, rnd, box, holds, CATALOGS, POOL, INVOLVES
import model as M
from levers import q_perchannel

CAT = CATALOGS["K2"]
ORDER = ["n", "df", "M", "sd", "se", "t", "p", "d"]
POS = {k: i for i, k in enumerate(ORDER)}


def emit_field(pp, s, key, ban=(), maxlen=12):
    """Дописать значение поля. ban -- значения, которые запрещено выдать
    (операционное содержание пометки «это уже оказалось неверным»)."""
    for attempt in range(1 + len(ban)):
        val = ""
        skip = attempt
        for _ in range(maxlen):
            pad = (" " * M.K + s + val)[-M.K:]
            _, _, sm = M._fwd(pp, np.array([[M.S2I[c] for c in pad]]))
            probs = sm[0].copy()
            if not val and skip:
                order = np.argsort(-probs)
                i = int(order[min(skip, len(order) - 1)])
            else:
                i = int(probs.argmax())
            c = M.VOC[i]
            if c in " \n":
                break
            val += c
        try:
            v = rnd(float(val), DEC[key])
        except ValueError:
            return None, val
        if v not in ban:
            return v, val
    return v, val


def checkable(cur):
    """Отношения, все переменные которых уже выданы."""
    return [r for r in CAT if all(k in cur for k in INVOLVES[r])]


def origin_of(rname, cur):
    """Куда отматывать: самое раннее ПОРОЖДАЕМОЕ поле нарушенного отношения.

    n исключено: оно задано промптом, а не порождено, откатываться к нему
    некуда. На этом опыт и споткнулся -- отмотка ушла к n и обвалилась.
    """
    gen = [k for k in INVOLVES[rname] if k != "n"]
    return min(gen, key=lambda k: POS[k]) if gen else None


def run_one(pp, n0, policy, max_attempts=3, budget=60):
    """policy: none | rewind1 | rewind1_mark | origin_mark
    budget -- жёсткий предел шагов порождения, иначе цикл."""
    s = f"n={n0:g} "
    cur = {"n": float(n0)}
    hist = {"n": s}
    banned = {k: set() for k in ORDER}      # пометки: что уже оказалось неверным
    attempts = {k: 0 for k in ORDER}        # счётчик попыток на позицию
    regen = faults = steps = 0
    i = 1
    while i < len(ORDER):
        steps += 1
        if steps > budget:
            break
        k = ORDER[i]
        s_field = s + f"{k}="
        v, raw = emit_field(pp, s_field, k, ban=banned[k])
        if v is None:
            return None, regen, faults
        cur[k] = v
        s = s_field + f"{raw} "
        hist[k] = s

        bad = [r for r in checkable(cur) if not holds(r, cur)]
        if not bad or policy == "none":
            i += 1
            continue

        faults += 1
        tgt = k if policy in ("rewind1", "rewind1_mark") else origin_of(bad[0], cur)
        if tgt is None or POS[tgt] < 1:
            i += 1
            continue
        attempts[tgt] += 1
        if attempts[tgt] > max_attempts:
            i += 1                      # сторож: сдаёмся на этой позиции, идём дальше
            continue
        if policy in ("rewind1_mark", "origin_mark") and tgt in cur:
            banned[tgt].add(cur[tgt])   # пометка «это значение неверно»
        j = POS[tgt]
        s = hist[ORDER[j - 1]]
        for kk in ORDER[j:]:
            cur.pop(kk, None)
        regen += 1
        i = j
    return cur, regen, faults


def score(cur, t):
    if cur is None or len(cur) < 8:
        return None
    return sum(1 for k in KEYS8 if cur.get(k) == t[k])


def main():
    p = M.load()
    truth = [t for t, _ in M.corpus()]
    print("подложка: символьная модель лаборатории, 60 отчётов")
    print("дефекты наводятся квантованием весов\n")
    for bits in (4, 3):
        pp = {k: (q_perchannel(v, bits) if k in ("w1", "w2") else v.copy())
              for k, v in p.items()}
        print(f"=== {bits} бита ===")
        print(f"{'политика':<36s}{'верных':>9s}{'ровно 8':>8s}"
              f"{'без выдачи':>9s}{'перепорожд':>12s}")
        base = None
        for pol, name in (("none", "оставленная линия (без вмешательства)"),
                          ("rewind1", "отмотка на 1 шаг, БЕЗ пометки"),
                          ("rewind1_mark", "отмотка на 1 шаг, с пометкой"),
                          ("origin_mark", "отмотка к происхождению, с пометкой")):
            tot = ok8 = rg = n = 0
            dropped = 0
            for t in truth:
                cur, regen, faults = run_one(pp, t["n"], pol)
                sc = score(cur, t)
                if sc is None:
                    dropped += 1
                    continue
                n += 1; tot += sc; ok8 += int(sc == 8); rg += regen
            if n == 0:
                print(f"{name:<34s}  разобрать не удалось"); continue
            print(f"{name:<36s}{tot/n:9.2f}{ok8:8d}{dropped:9d}{rg/n:12.2f}")
        print()


if __name__ == "__main__":
    main()
