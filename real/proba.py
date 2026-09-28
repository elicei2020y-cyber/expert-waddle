"""Быстрая проба (разведка): работает ли отмотка с пометкой вообще.

20 отчётов, только исходные веса, три политики: А (не вмешиваться),
Б (отмотка последнего написанного поля, без пометки),
В (отмотка последнего написанного поля, с пометкой).
Бюджет: 3 перепорождения на отчёт. Подсказка, формулы, порядок полей,
пример -- без изменений (см. real/etap0.py).

Порядок полей: df, t, se, d. Проверка t закрывается только после se
(так задано в ZADANIE.md) -- поэтому при дефекте в t "последнее написанное
поле" на момент обнаружения -- это se, а не t. Политики Б/В в этой пробе
рассчитаны буквально: отматывают именно последнее написанное поле, даже
если реальный источник дефекта -- t. Это намеренное свойство наивной
политики (см. П4 в PREDICTIONS.md, происхождение по графу -- отдельная
политика Г, не в этой пробе).
"""
import json, math, re, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
FIELDS = ["df", "t", "se", "d"]
BUDGET = 3
NUM_RE = re.compile(r"[-+]?\d*\.?\d+")

SYSTEM = (
    "Ты вычисляешь статистику одновыборочного t-теста. Дано: n (размер выборки), "
    "M (выборочное среднее), sd (выборочное стандартное отклонение), mu = 4.00 "
    "(проверяемое среднее). Формулы:\n"
    "df = n - 1\n"
    "se = sd / sqrt(n)\n"
    "t = (M - mu) / se\n"
    "d = (M - mu) / sd\n"
    "Выводи ровно четыре строки в порядке df, t, se, d, каждую в формате "
    "\"имя = число\" с двумя знаками после запятой (df -- целое без знаков "
    "после запятой). Ничего кроме этих строк не пиши."
)
EXAMPLE_IN = "n = 25, M = 4.80, sd = 1.50, mu = 4.00"
EXAMPLE_OUT = "df = 24\nt = 2.67\nse = 0.30\nd = 0.53"


def truth_of(inp):
    n, M, sd, mu = inp["n"], inp["M"], inp["sd"], inp["mu"]
    df = n - 1
    se = sd / math.sqrt(n)
    return dict(df=df, se=se, t=(M - mu) / se, d=(M - mu) / sd)


def legit_df(n, v):
    return abs(v - (n - 1)) <= 0.5


def legit_se(sd, n, v):
    return abs(v - sd / math.sqrt(n)) <= 0.005


def legit_t(M, mu, se_report, se_true, v):
    a = abs(v - (M - mu) / se_report) <= 0.005 if se_report else False
    b = abs(v - (M - mu) / se_true) <= 0.005
    return a or b


def legit_d(M, mu, sd, t_report, n, v):
    a = abs(v - (M - mu) / sd) <= 0.005
    b = abs(v - t_report / math.sqrt(n)) <= 0.005 if t_report is not None else False
    return a or b


def build_prompt(tok, inp, body):
    user = f"n = {inp['n']}, M = {inp['M']:.2f}, sd = {inp['sd']:.2f}, mu = {inp['mu']:.2f}"
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": EXAMPLE_IN},
        {"role": "assistant", "content": EXAMPLE_OUT},
        {"role": "user", "content": user},
    ]
    prefix = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prefix + body


def gen_value(model, tok, prompt, ban, max_new=12):
    """Дописать число после prompt (который уже заканчивается 'имя = ').

    Жадно по каждому токену, кроме самого первого генерируемого токена:
    если это повторная попытка (skip = число уже забаненных значений),
    первый токен берётся skip-м по вероятности, а не argmax -- иначе при
    жадном декодировании с тем же префиксом выйдет тот же результат
    (баг, найденный в этой пробе: раньше при бане просто возвращалось то
    же значение). Дальше по строке -- обычный argmax, как договорено в
    ZADANIE.md (жадное декодирование).
    """
    ids = tok(prompt, return_tensors="pt").input_ids
    skip = len(ban)
    text = ""
    with torch.no_grad():
        out = model(ids, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1]
        for step in range(max_new):
            if step == 0 and skip:
                order = torch.argsort(logits, descending=True)
                tid = int(order[min(skip, len(order) - 1)])
            else:
                tid = int(torch.argmax(logits))
            piece = tok.decode([tid], skip_special_tokens=True)
            if tid == tok.eos_token_id or "\n" in piece:
                break
            text += piece
            if len(text) > max_new * 4:
                break
            out = model(torch.tensor([[tid]]), past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1]
    m = NUM_RE.search(text)
    if m is None:
        return None, text
    try:
        v = round(float(m.group(0)), 4)
    except ValueError:
        return None, text
    return v, text


def run_report(model, tok, inp, policy, log):
    """policy: A | B_nomark | C_mark"""
    truth = truth_of(inp)
    body = ""
    vals = {}       # текущие значения полей
    closed = {}      # что уже прошло проверку (bool) для df, se, d; t особый
    banned = {f: set() for f in FIELDS}
    regen_budget = BUDGET
    events = []

    def emit(field):
        nonlocal body
        p = build_prompt(tok, inp, body + f"{field} =")
        v, raw = gen_value(model, tok, p, banned[field])
        vals[field] = v
        body += f"{field} = {raw.split(chr(10))[0].strip()}\n" if v is not None else f"{field} = ?\n"
        return v

    # df
    v = emit("df")
    df_ok = (v is not None) and legit_df(inp["n"], v)

    # t (проверка отложена)
    emit("t")

    # se
    v_se = emit("se")
    se_ok = (v_se is not None) and legit_se(inp["sd"], inp["n"], v_se)
    t_ok = (vals["t"] is not None) and legit_t(inp["M"], inp["mu"], v_se, truth["se"], vals["t"])

    # -- политика: если есть дефект (se или t), отмотать ПОСЛЕДНЕЕ НАПИСАННОЕ (se) --
    if policy != "A" and (not se_ok or not t_ok) and regen_budget > 0:
        target = "se"
        attempts = 0
        while (not se_ok or not t_ok) and regen_budget > 0:
            attempts += 1
            regen_budget -= 1
            if policy == "C_mark":
                reason = []
                if not se_ok:
                    reason.append(f"не сходится se = sd/√n (было {vals['se']})")
                if not t_ok:
                    reason.append(f"t не сошлось при данном se (было {vals['t']})")
                mark = f"[проверка: поле se отозвано — {'; '.join(reason)}]\n"
                banned["se"].add(vals["se"])
            else:
                mark = ""
            body_before = body
            # обрезать body до конца поля t (то есть убрать старую строку se)
            lines = body.split("\n")
            # lines: [..., "df = X", "t = Y", "se = Z", ""]
            body = "\n".join(lines[:2]) + "\n" + mark
            v_se = emit("se")
            events.append(dict(attempt=attempts, mark=mark.strip(), new_se=v_se))
            se_ok = (v_se is not None) and legit_se(inp["sd"], inp["n"], v_se)
            t_ok = (vals["t"] is not None) and legit_t(inp["M"], inp["mu"], v_se, truth["se"], vals["t"])

    # d
    v_d = emit("d")
    d_ok = (v_d is not None) and legit_d(inp["M"], inp["mu"], inp["sd"], vals["t"], inp["n"], v_d)

    if policy != "A" and not d_ok and regen_budget > 0:
        attempts = 0
        while not d_ok and regen_budget > 0:
            attempts += 1
            regen_budget -= 1
            if policy == "C_mark":
                mark = f"[проверка: поле d отозвано — не сходится d = (M-mu)/sd (было {vals['d']})]\n"
                banned["d"].add(vals["d"])
            else:
                mark = ""
            lines = body.split("\n")
            body = "\n".join(lines[:3]) + "\n" + mark
            v_d = emit("d")
            events.append(dict(attempt=attempts, mark=mark.strip(), new_d=v_d))
            d_ok = (v_d is not None) and legit_d(inp["M"], inp["mu"], inp["sd"], vals["t"], inp["n"], v_d)

    used_regens = BUDGET - regen_budget
    silence = (used_regens >= BUDGET) and not (se_ok and t_ok and d_ok)
    correct = dict(df=df_ok, t=t_ok, se=se_ok, d=d_ok)
    log.append(dict(inp=inp, vals=dict(vals), truth=truth, body=body,
                     regens=used_regens, events=events, silence=silence,
                     correct=correct))
    if silence:
        return None, used_regens
    return correct, used_regens


def main():
    with open("real/proba_inputs.json") as f:
        inputs = json.load(f)

    print(f"загрузка модели {MODEL_ID} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.eval()
    print(f"загрузка: {time.time()-t0:.1f} c\n")

    results = {}
    logs = {}
    for policy in ("A", "B_nomark", "C_mark"):
        t0 = time.time()
        log = []
        field_ok = {f: 0 for f in FIELDS}
        n_delivered = 0
        n_silent = 0
        regen_total = 0
        for inp in inputs:
            correct, regens = run_report(model, tok, inp, policy, log)
            regen_total += regens
            if correct is None:
                n_silent += 1
                continue
            n_delivered += 1
            for f in FIELDS:
                if correct[f]:
                    field_ok[f] += 1
        dt = time.time() - t0
        results[policy] = dict(field_ok=field_ok, n_delivered=n_delivered,
                                n_silent=n_silent, regen_total=regen_total,
                                time=dt, per_report=dt / len(inputs))
        logs[policy] = log
        print(f"=== политика {policy} === время {dt:.1f} c ({dt/len(inputs):.2f} c/отчёт)")
        print(f"  выдано: {n_delivered}/{len(inputs)}  молчание: {n_silent}")
        print(f"  верно по полям: {field_ok}  (сумма {sum(field_ok.values())}/{4*len(inputs)})")
        print(f"  перепорождений всего: {regen_total}\n")

    with open("real/proba_log.json", "w") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2, default=str)

    # сравнение Б и А знак в знак
    print("=== стенд: Б против А, знак в знак ===")
    mism = 0
    for i, (la, lb) in enumerate(zip(logs["A"], logs["B_nomark"])):
        if la["vals"] != lb["vals"]:
            mism += 1
            print(f"  отчёт {i}: A={la['vals']} B={lb['vals']}")
    print(f"расхождений: {mism}/{len(inputs)}")

    # спасено / повреждено для В (и Б) относительно А, по каждому полю
    for policy in ("B_nomark", "C_mark"):
        print(f"\n=== {policy} относительно А (было при А / стало при {policy}) ===")
        saved = damaged = 0
        for la, lp in zip(logs["A"], logs[policy]):
            for f in FIELDS:
                a_ok = la["correct"][f]
                p_ok = lp["correct"][f]
                if not a_ok and p_ok:
                    saved += 1
                elif a_ok and not p_ok:
                    damaged += 1
        print(f"  спасено: {saved}  повреждено: {damaged}")
        results[policy]["saved_vs_A"] = saved
        results[policy]["damaged_vs_A"] = damaged

    with open("real/proba_summary.json", "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in results.items()}, f, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()
