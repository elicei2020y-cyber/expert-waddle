"""Этап 0в: порядок полей df, se, t, d (исправление постановки — прежний
порядок df,t,se,d делал t невычислимым и неспасаемым, см. отчёт этапа 0б).

Следствие (отклонение от ZADANIE.md, зафиксировать в отчёте): опоздание
обнаружения дефекта пропадает -- каждое поле проверяется сразу после
печати. Отмотка "на одно поле" и "к происхождению" в этой постановке
совпадают, поэтому политика Г здесь не нужна (не запускается).

15 отчётов, Qwen2.5-1.5B-Instruct. Сначала политика А (гейт: >=50% из
60 полей). Если прошло -- без остановки: Б (без пометки), В-текст
(пометка, строго жадное декодирование, без обхода), В-запрет (пометка +
обход первого токена, как в igrushka/lab/model.py).
"""
import json, math, re, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
FIELDS = ["df", "se", "t", "d"]     # <-- естественный порядок
FIELD_BUDGET = 3
NUM_RE = re.compile(r"[-+]?\d*\.?\d+")
GATE_MIN_FRAC = 0.5
INPUTS_PATH = "real/proba_inputs.json"
N_REPORTS = 15

SYSTEM = (
    "Ты вычисляешь статистику одновыборочного t-теста. Дано: n (размер выборки), "
    "M (выборочное среднее), sd (выборочное стандартное отклонение), mu = 4.00 "
    "(проверяемое среднее). Формулы:\n"
    "df = n - 1\n"
    "se = sd / sqrt(n)\n"
    "t = (M - mu) / se\n"
    "d = (M - mu) / sd\n"
    "Выводи ровно четыре строки в порядке df, se, t, d, каждую в формате "
    "\"имя = число\" с двумя знаками после запятой (df -- целое без знаков "
    "после запятой). Ничего кроме этих строк не пиши."
)

EXAMPLES = [
    dict(n=25, M=4.80, sd=1.50, mu=4.00, out="df = 24\nse = 0.30\nt = 2.67\nd = 0.53"),
    dict(n=16, M=3.60, sd=0.80, mu=4.00, out="df = 15\nse = 0.20\nt = -2.00\nd = -0.50"),
    dict(n=100, M=5.20, sd=3.00, mu=4.00, out="df = 99\nse = 0.30\nt = 4.00\nd = 0.40"),
]

FORMULA = {"df": "df = n − 1", "se": "se = sd/√n", "t": "t = (M − mu)/se", "d": "d = (M − mu)/sd"}


def fmt_user(inp):
    return f"n = {inp['n']}, M = {inp['M']:.2f}, sd = {inp['sd']:.2f}, mu = {inp['mu']:.2f}"


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


def check_field(field, inp, truth, vals):
    if field == "df":
        return legit_df(inp["n"], vals["df"]) if vals["df"] is not None else False
    if field == "se":
        return legit_se(inp["sd"], inp["n"], vals["se"]) if vals["se"] is not None else False
    if field == "t":
        return legit_t(inp["M"], inp["mu"], vals["se"], truth["se"], vals["t"]) if vals["t"] is not None else False
    if field == "d":
        return legit_d(inp["M"], inp["mu"], inp["sd"], vals["t"], inp["n"], vals["d"]) if vals["d"] is not None else False


def build_messages(inp):
    messages = [{"role": "system", "content": SYSTEM}]
    for ex in EXAMPLES:
        messages.append({"role": "user", "content": fmt_user(ex)})
        messages.append({"role": "assistant", "content": ex["out"]})
    messages.append({"role": "user", "content": fmt_user(inp)})
    return messages


def build_prompt(tok, inp, body):
    prefix = tok.apply_chat_template(build_messages(inp), tokenize=False, add_generation_prompt=True)
    return prefix + body


def gen_value(model, tok, prompt, force_skip=0, max_new=12):
    ids = tok(prompt, return_tensors="pt").input_ids
    text = ""
    with torch.no_grad():
        out = model(ids, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1]
        for step in range(max_new):
            if step == 0 and force_skip:
                order = torch.argsort(logits, descending=True)
                tid = int(order[min(force_skip, len(order) - 1)])
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


def run_report(model, tok, inp, policy):
    """policy: A | B_nomark | V_text | V_ban.
    Каждое поле проверяется сразу после печати (постановка df,se,t,d) --
    отмотка "на одно поле" = отмотка "к происхождению": Г не нужна."""
    truth = truth_of(inp)
    body = ""
    vals = {}
    events = []
    regen_total = 0

    def emit(field, force_skip=0):
        nonlocal body
        p = build_prompt(tok, inp, body + f"{field} = ")
        v, raw = gen_value(model, tok, p, force_skip=force_skip)
        vals[field] = v
        line = raw.split("\n")[0].strip()
        body += f"{field} = {line}\n" if v is not None else f"{field} = ?\n"
        return v

    correct = {}
    for field in FIELDS:
        emit(field)
        ok = check_field(field, inp, truth, vals)
        if policy != "A" and not ok:
            for attempt in range(1, FIELD_BUDGET + 1):
                prev = vals[field]
                mark_txt = f"поле {field} отозвано — не сходится {FORMULA[field]} (было {prev})"
                lines = body.split("\n")
                idx = FIELDS.index(field)
                body = "\n".join(lines[:idx]) + ("\n" if idx else "")
                if policy in ("V_text", "V_ban"):
                    body += f"[проверка: {mark_txt}]\n"
                force_skip = attempt if policy == "V_ban" else 0
                new_v = emit(field, force_skip=force_skip)
                regen_total += 1
                events.append(dict(field=field, attempt=attempt, prev=prev, new=new_v,
                                    mark=mark_txt if policy != "B_nomark" else None,
                                    changed=(new_v != prev)))
                ok = check_field(field, inp, truth, vals)
                if ok:
                    break
        correct[field] = ok

    silence = policy != "A" and any(
        not correct[f] and sum(1 for e in events if e["field"] == f) >= FIELD_BUDGET
        for f in FIELDS
    )
    return dict(inp=inp, vals=dict(vals), truth=truth, body=body, correct=correct,
                events=events, regen_total=regen_total, silence=silence)


def main():
    with open(INPUTS_PATH) as f:
        inputs = json.load(f)[:N_REPORTS]

    print(f"загрузка модели {MODEL_ID} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.eval()
    print(f"загрузка: {time.time()-t0:.1f} c\n")

    t_start = time.time()
    print(f"=== ГЕЙТ: политика А, {N_REPORTS} отчётов (порядок df,se,t,d) ===")
    t0 = time.time()
    log_A = [run_report(model, tok, inp, "A") for inp in inputs]
    time_A = time.time() - t0
    field_ok = {f: sum(1 for r in log_A if r["correct"][f]) for f in FIELDS}
    total_ok = sum(field_ok.values())
    frac = total_ok / (4 * N_REPORTS)
    print(f"время: {time_A:.1f} c ({time_A/N_REPORTS:.2f} c/отчёт)")
    print(f"верно по полям: {field_ok} -> {total_ok}/{4*N_REPORTS} = {frac:.1%}")

    results = {"A": dict(log=log_A, time=time_A, field_ok=field_ok, frac=frac)}

    if frac < GATE_MIN_FRAC:
        print(f"\nКРИТЕРИЙ НЕ ПРОЙДЕН ({frac:.1%} < {GATE_MIN_FRAC:.0%}). СТОП.")
        save(results)
        return

    print(f"\nКРИТЕРИЙ ПРОЙДЕН ({frac:.1%} >= {GATE_MIN_FRAC:.0%}). Продолжаю без остановки.\n")

    for policy in ("B_nomark", "V_text", "V_ban"):
        t0 = time.time()
        log = [run_report(model, tok, inp, policy) for inp in inputs]
        dt = time.time() - t0
        field_ok = {f: sum(1 for r in log if r["correct"][f]) for f in FIELDS}
        n_silent = sum(1 for r in log if r["silence"])
        regen_total = sum(r["regen_total"] for r in log)
        print(f"=== {policy} === время {dt:.1f} c ({dt/N_REPORTS:.2f} c/отчёт)")
        print(f"  верно по полям: {field_ok}  молчание: {n_silent}/{N_REPORTS}  перепорожд: {regen_total}")
        results[policy] = dict(log=log, time=dt, field_ok=field_ok, n_silent=n_silent, regen_total=regen_total)

    mism = sum(1 for la, lb in zip(log_A, results["B_nomark"]["log"]) if la["vals"] != lb["vals"])
    print(f"\nБ против А знак в знак: расхождений {mism}/{N_REPORTS}")
    results["B_nomark"]["mismatch_vs_A"] = mism

    for policy in ("B_nomark", "V_text", "V_ban"):
        saved = damaged = 0
        for la, lp in zip(log_A, results[policy]["log"]):
            for f in FIELDS:
                a_ok, p_ok = la["correct"][f], lp["correct"][f]
                if not a_ok and p_ok:
                    saved += 1
                elif a_ok and not p_ok:
                    damaged += 1
        results[policy]["saved_vs_A"] = saved
        results[policy]["damaged_vs_A"] = damaged
        print(f"{policy} относительно А: спасено {saved}  повреждено {damaged}")

    # В-текст: доля случаев, когда ПЕРВАЯ попытка после пометки дала значение,
    # ОТЛИЧНОЕ от значения того же поля при А (это и есть признак чтения пометки,
    # а не свойство жадного декодирования -- текст промпта уже другой).
    first_attempts = [e for r in results["V_text"]["log"] for e in r["events"] if e["attempt"] == 1]
    changed_vs_A = 0
    for r in results["V_text"]["log"]:
        a_row = next(la for la in log_A if la["inp"] == r["inp"])
        for e in r["events"]:
            if e["attempt"] != 1:
                continue
            a_val = a_row["vals"][e["field"]]
            if e["new"] != a_val:
                changed_vs_A += 1
    print(f"\nВ-текст: на первой попытке после пометки дала значение != А в "
          f"{changed_vs_A}/{len(first_attempts)} случаях (это и есть чтение пометки)")
    results["V_text"]["changed_vs_A_first_attempt"] = changed_vs_A
    results["V_text"]["first_attempts_total"] = len(first_attempts)

    save(results)
    print(f"\nВсего времени: {time.time()-t_start:.0f} c")


def save(results):
    def strip(r):
        return dict(vals=r["vals"], truth=r["truth"], correct=r["correct"],
                    events=r["events"], regen_total=r["regen_total"],
                    silence=r["silence"], body=r["body"], inp=r["inp"])
    out = {k: {kk: (vv if kk != "log" else [strip(r) for r in vv]) for kk, vv in v.items()}
           for k, v in results.items()}
    with open("real/proba0c_log.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()
