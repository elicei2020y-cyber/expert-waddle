"""Распределение Стьюдента без внешних зависимостей.

Почему не scipy, хотя он в среде есть: ядро верификатора — то место,
где цепочка вывода обязана оканчиваться на арифметике. Собственная
реализация делает рантайм независимым от окружения, а scipy при этом
остаётся оракулом в тестах (test_tstats.py). Две независимые
реализации, сходящиеся в пределах измеренной точности, — это проверка,
а одна, взятая на веру, — нет.

О ТОЧНОСТИ, ЧЕСТНО. Максимальное расхождение с scipy — 1.0e-10
относительных, и оно моё, а не scipy: два независимых пути внутри
scipy (special.betainc и 2*stats.t.sf) согласуются между собой до
7e-13. Источник локализован — сокращение при вычитании двух lgamma
порядка 3.8e4 при больших a (df >= 1000): абсолютная погрешность
логарифма становится относительной после exp. Итерации непрерывной
дроби и переход на log1p ничего не меняют, потому что дробь сходится,
а дело в префакторе.

Не исправляется сознательно. Интервал сравнения в L4 задаётся
точностью записи в статье: при трёх знаках после запятой его
полуширина 5e-4, то есть на шесть порядков грубее этой погрешности.
Асимптотическое разложение для lgamma(a+1/2) - lgamma(a) убрало бы
расхождение, но добавило бы собственный непроверенный код в то самое
место, которому верификатор обязан доверять. Порог в тестах поэтому
привязан к требованию приложения, а не подогнан под результат.

Двусторонний p выражается через регуляризованную неполную бету
напрямую, без промежуточной CDF:

    p = I_x(nu/2, 1/2),   где x = nu / (nu + t^2)

Отсюда же берутся обратные решения, нужные для локализации (L5).
"""

import math
from typing import Optional

_EPS = 3e-16
_FPMIN = 1e-300
_MAXIT = 300


def _betacf(a: float, b: float, x: float) -> float:
    """Непрерывная дробь для неполной беты, алгоритм Ленца."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, _MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            return h
    return h


def betai(a: float, b: float, x: float) -> float:
    """Регуляризованная неполная бета I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    # log(x) через log1p: при x, близком к единице (а это типичный
    # случай — x = df/(df+t^2) при большом df), прямой log теряет
    # значащие цифры, а log1p нет.
    lb = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
          + a * math.log1p(x - 1.0) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lb) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lb) * _betacf(b, a, 1.0 - x) / b


def p_two_tailed(t: float, df: float) -> float:
    """Двусторонний p для t-статистики при df степенях свободы."""
    if df <= 0:
        raise ValueError("df должно быть положительным")
    t = abs(t)
    return betai(df / 2.0, 0.5, df / (df + t * t))


def p_normal_limit(t: float) -> float:
    """Предел двустороннего p при df -> бесконечность.

    Жёсткая нижняя граница: ни при каких df двусторонний p для данного
    |t| не опустится ниже. Из этого получается бесплатное оправдание
    df при локализации — если сообщённый p ниже предела, никакое df
    не согласует тройку, значит врёт t или p.
    """
    return math.erfc(abs(t) / math.sqrt(2.0))


def t_from_p(p: float, df: float) -> Optional[float]:
    """|t| по двустороннему p и df. p монотонно убывает по |t|."""
    if not (0.0 < p < 1.0) or df <= 0:
        return None
    lo, hi = 0.0, 1.0
    while p_two_tailed(hi, df) > p:
        hi *= 2.0
        if hi > 1e6:
            return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if p_two_tailed(mid, df) > p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def df_from(t: float, p: float, df_max: float = 1e7) -> Optional[float]:
    """df по t и двустороннему p. Возвращает None, если решения нет.

    None — содержательный ответ, а не сбой: при p ниже нормального
    предела для данного |t| гипотеза «врёт df» отвергается целиком.
    """
    if not (0.0 < p < 1.0):
        return None
    t = abs(t)
    if p < p_normal_limit(t):
        return None                      # недостижимо ни при каком df
    lo, hi = 1e-6, df_max
    if p_two_tailed(t, lo) < p:
        return None                      # недостижимо и снизу
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if p_two_tailed(t, mid) > p:
            lo = mid
        else:
            hi = mid
    out = 0.5 * (lo + hi)
    return None if out >= df_max * 0.999 else out
