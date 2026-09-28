"""Засорённый контекст: видит ли проверка ошибку и докуда достаёт отмотка.

ДВА ВОПРОСА.

(A) СЛЕПОТА И ОПОЗДАНИЕ. Поле видно проверке только когда закрылось
    отношение, в которое оно входит. Свободные M и sd не выводятся ни
    из чего, поэтому в момент порождения не проверяемы вовсе -- дефект
    в них обнаружится лишь когда испортит зависимое, если испортит.
    Измеряется: какая доля неверных полей была хоть когда-то отмечена
    и на сколько шагов позже своего порождения.

(B) ГОРИЗОНТ ОТМОТКИ. Контекст засоряется ДО того, как проверяющий
    включился: первые j полей уже неверны. Три политики:
      own   -- отматывать можно только то, что порождено под надзором
      full  -- отматывать можно и засорённую часть, оспаривая её
      none  -- не вмешиваться
    Вопрос: достаёт ли отмотка до чужого мусора, и что это даёт.
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


def blindness(bits=3):
    """(A) Какая доля неверных полей вообще отмечается и с каким опозданием."""
    p = M.load()
    pp = {k: (q_perchannel(v, bits) if k in ("w1", "w2") else v.copy())
          for k, v in p.items()}
    truth = [t for t, _ in M.corpus()]
    seen = Counter(); total = Counter(); lag = []
    for t in truth:
        cur, _, _ = F.run_one(pp, t["n"], "none")
        if cur is None or len(cur) < 8:
            continue
        wrong = [k for k in KEYS8 if cur.get(k) != t[k]]
        for k in wrong:
            total[k] += 1
        # когда каждое отношение закрывается и что оно отмечает
        flagged = {}
        acc = {}
        for i, k in enumerate(ORDER):
            acc[k] = cur[k]
            for r in CAT:
                if all(x in acc for x in INVOLVES[r]) and not holds(r, acc):
                    for x in INVOLVES[r]:
                        flagged.setdefault(x, i)
        for k in wrong:
            if k in flagged:
                seen[k] += 1
                lag.append(flagged[k] - POS[k])
    print("(A) СЛЕПОТА ПРОВЕРКИ, 3 бита")
    print(f"{'поле':<6s}{'неверно раз':>13s}{'отмечено':>10s}{'доля':>8s}")
    for k in ORDER:
        if total[k]:
            print(f"{k:<6s}{total[k]:13d}{seen[k]:10d}{seen[k]/total[k]:8.0%}")
    ts, ss = sum(total.values()), sum(seen.values())
    print(f"{'всего':<6s}{ts:13d}{ss:10d}{ss/max(ts,1):8.0%}")
    if lag:
        lag = np.array(lag)
        print(f"опоздание обнаружения: медиана {np.median(lag):.0f} шаг(ов), "
              f"максимум {lag.max()}, доля мгновенных {np.mean(lag<=0):.0%}")
    print()


def run_from_polluted(pp, t, j, policy, max_attempts=3, budget=60):
    """Первые j полей уже неверны. Порождаем остальное.
    policy: none | own | full"""
    cur = {}
    s = f"n={t['n']:g} "
    cur["n"] = float(t["n"])
    hist = {"n": s}
    # засорение: поля 1..j-1 подменены неверными значениями
    for k in ORDER[1:j]:
        bad = rnd(t[k] + 3 * 10 ** (-DEC[k]), DEC[k])
        cur[k] = bad
        s += f"{k}={bad:g} "
        hist[k] = s
    banned = {k: set() for k in ORDER}
    attempts = {k: 0 for k in ORDER}
    floor = 1 if policy == "full" else j      # докуда разрешено отматывать
    regen = steps = 0
    i = j
    while i < len(ORDER):
        steps += 1
        if steps > budget:
            break
        k = ORDER[i]
        sf = s + f"{k}="
        v, raw = F.emit_field(pp, sf, k, ban=banned[k])
        if v is None:
            return None, regen
        cur[k] = v; s = sf + f"{raw} "; hist[k] = s
        bad = [r for r in F.checkable(cur) if not holds(r, cur)]
        if not bad or policy == "none":
            i += 1; continue
        tgt = F.origin_of(bad[0], cur)
        if tgt is None or POS[tgt] < floor:
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
    return cur, regen


def horizon(bits=3, j=3):
    print(f"(B) ГОРИЗОНТ ОТМОТКИ, {bits} бита, засорено полей до позиции {j}")
    print(f"   подменены: {', '.join(ORDER[1:j])}")
    p = M.load()
    pp = {k: (q_perchannel(v, bits) if k in ("w1", "w2") else v.copy())
          for k, v in p.items()}
    truth = [t for t, _ in M.corpus()]
    print(f"{'политика':<38s}{'верных полей':>13s}{'молчание':>10s}{'перепорожд':>12s}")
    for pol, name in (("none", "не вмешиваться"),
                      ("own", "отмотка только своего"),
                      ("full", "отмотка в засорённое тоже")):
        tot = n = rg = drop = 0
        for t in truth:
            cur, regen = run_from_polluted(pp, t, j, pol)
            rg += regen
            if cur is None or len(cur) < 8:
                drop += 1; continue
            n += 1
            tot += sum(1 for k in KEYS8 if cur.get(k) == t[k])
        if n == 0:
            print(f"{name:<38s}  нет выдачи"); continue
        print(f"{name:<38s}{tot/n:13.2f}{drop:10d}{rg/len(truth):12.2f}")
    print()


if __name__ == "__main__":
    blindness(3)
    horizon(3, 3)
    horizon(3, 5)
