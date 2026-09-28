"""Что даёт проверяющему БОГАТАЯ запись: журнал действий и их результатов.

Прежние опыты давали ему восемь чисел и шесть отношений. В настоящей
системе у него будет список действий, полученные данные, заявленные
причины-следствия и локальные контексты. Каждая пара «действие ->
результат» есть новое отношение для сверки, и её можно проверять
БУКВАЛЬНО, не дожидаясь, пока закроется арифметика.

Здесь запись растёт: k наблюдений, каждое сообщает истинное значение
одного поля -- это модель "инструмент вернул значение, оно в журнале".
Меряется, как с ростом k меняются:
   -- доля обнаруженных неверных полей
   -- ОПОЗДАНИЕ обнаружения (главная величина: она задаёт глубину отмотки)
   -- восстановление отмоткой
   -- и цена: ложные срабатывания (здесь их ноль по устройству --
      сверка буквальная; в настоящей записи это перестанет быть так)
"""
import sys, numpy as np
from collections import Counter
sys.path.insert(0, __import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "lab")))
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from core import KEYS8, DEC, rnd, holds, CATALOGS, INVOLVES
import model as M
from levers import q_perchannel
import fork as F

CAT = CATALOGS["K2"]
ORDER = F.ORDER
POS = F.POS
GEN = ORDER[1:]          # порождаемые поля


def run_with_record(pp, t, rec_keys, policy="origin_mark",
                    max_attempts=3, budget=60):
    """rec_keys -- поля, истинное значение которых лежит в журнале действий.
    Проверка против журнала буквальная и срабатывает сразу при выдаче."""
    s = f"n={t['n']:g} "
    cur = {"n": float(t["n"])}
    hist = {"n": s}
    banned = {k: set() for k in ORDER}
    attempts = {k: 0 for k in ORDER}
    regen = steps = 0
    flagged_at = {}
    i = 1
    while i < len(ORDER):
        steps += 1
        if steps > budget:
            break
        k = ORDER[i]
        sf = s + f"{k}="
        v, raw = F.emit_field(pp, sf, k, ban=banned[k])
        if v is None:
            return None, regen, flagged_at
        cur[k] = v; s = sf + f"{raw} "; hist[k] = s

        # 1) сверка с журналом действий -- мгновенная, буквальная
        rec_fault = (k in rec_keys and cur[k] != t[k])
        # 2) арифметика -- только когда отношение закрылось
        bad = [r for r in F.checkable(cur) if not holds(r, cur)]

        if not (rec_fault or bad):
            i += 1; continue
        if rec_fault:
            tgt = k                       # источник назван записью, гадать не надо
            flagged_at.setdefault(k, i)
        else:
            tgt = F.origin_of(bad[0], cur)
            for x in INVOLVES[bad[0]]:
                flagged_at.setdefault(x, i)
        if tgt is None or POS[tgt] < 1:
            i += 1; continue
        attempts[tgt] += 1
        if attempts[tgt] > max_attempts:
            i += 1; continue
        banned[tgt].add(cur.get(tgt))
        jj = POS[tgt]
        s = hist[ORDER[jj - 1]]
        for kk in ORDER[jj:]:
            cur.pop(kk, None)
        regen += 1
        i = jj
    return cur, regen, flagged_at


def main(bits=3, seed=5):
    p = M.load()
    pp = {k: (q_perchannel(v, bits) if k in ("w1", "w2") else v.copy())
          for k, v in p.items()}
    truth = [t for t, _ in M.corpus()]
    rng = np.random.default_rng(seed)
    print(f"{bits} бита, 60 отчётов. k -- сколько полей покрыто журналом действий")
    print(f"{'k':>2s}{'верных из 8':>13s}{'молчание':>10s}"
          f"{'обнаружено':>12s}{'опоздание':>11s}{'перепорожд':>12s}")
    for k in (0, 1, 2, 3, 4, 5, 6, 7):
        tot = n = drop = rg = 0
        seen = wrong = 0
        lags = []
        for t in truth:
            rec = set(rng.choice(GEN, size=k, replace=False)) if k else set()
            cur, regen, flg = run_with_record(pp, t, rec)
            rg += regen
            if cur is None or len(cur) < 8:
                drop += 1; continue
            n += 1
            tot += sum(1 for kk in KEYS8 if cur.get(kk) == t[kk])
            bad = [kk for kk in KEYS8 if cur.get(kk) != t[kk]]
            wrong += len(bad)
            for kk in bad:
                if kk in flg:
                    seen += 1
                    lags.append(flg[kk] - POS[kk])
        if n == 0:
            print(f"{k:>2d}   нет выдачи"); continue
        lag = np.median(lags) if lags else float("nan")
        print(f"{k:>2d}{tot/n:13.2f}{drop:10d}{seen/max(wrong,1):11.0%}"
              f"{lag:11.1f}{rg/len(truth):12.2f}")
    print()
    print("k=0 -- прежний голодный паёк. Рост k -- пополнение записи.")


if __name__ == "__main__":
    main()
