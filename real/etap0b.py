"""Этап 0б: гейт на Qwen2.5-1.5B-Instruct (политика А, 20 отчётов) и, без
остановки при прохождении критерия, быстрая проба на 4 политики:
А, Б (без пометки), В-текст (пометка, жёсткое жадное декодирование,
без обхода), В-запрет (пометка + обход первого токена, как в игрушке).

Изменения против ZADANIE.md (см. отчёт):
  - подсказка: три решённых примера вместо одного (разные n, se, знак t, d) --
    чтобы модели нечего было тупо скопировать;
  - модель: Qwen2.5-1.5B-Instruct вместо 0.5B (по критерию этапа 0б: фон
    0.5B был ниже 50%/80 полей);
  - бюджет на поле (df, se, d -- по 3 попытки НА КАЖДОЕ поле отдельно,
    не общий бюджет 3 на отчёт), как в igrushka/lab/model.py (max_attempts);
  - пометка расщеплена на В-текст (только текст, декодирование строго
    жадное) и В-запрет (текст + принудительный обход первого токена).
"""
import json, math, re, time, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
FIELDS = ["df", "t", "se", "d"]
FIELD_BUDGET = 3   # попыток НА ПОЛЕ (df, se, d -- каждое отдельно)
NUM_RE = re.compile(r"[-+]?\d*\.?\d+")
GATE_MIN_FRAC = 0.5   # >=50% из 80 полей на политике А -> продолжать без остановки
TIME_BUDGET_S = 30 * 60
INPUTS_PATH = "real/proba_inputs.json"

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

# три решённых примера с разными n, se, знаком t, разными d
EXAMPLES = [
    dict(n=25, M=4.80, sd=1.50, mu=4.00, out="df = 24\nt = 2.67\nse = 0.30\nd = 0.53"),
    dict(n=16, M=3.60, sd=0.80, mu=4.00, out="df = 15\nt = -2.00\nse = 0.20\nd = -0.50"),
    dict(n=100, M=5.20, sd=3.00, mu=4.00, out="df = 99\nt = 4.00\nse = 0.30\nd = 0.40"),
]


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


def build_messages(inp):
    messages = [{"role": "system", "content": SYSTEM}]
    for ex in EXAMPLES:
        messages.append({"role": "user", "content": fmt_user(ex)})
        messages.append({"role": "assistant", "content": ex["out"]})
    messages.append({"role": "user", "content": fmt_user(inp)})
    return messages


def build_prompt(tok, inp, body):
    messages = build_messages(inp)
    prefix = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prefix + body


def gen_value(model, tok, prompt, force_skip=0, max_new=12):
    """Жадно, кроме первого генерируемого токена при force_skip>0: тогда
    первый токен берётся force_skip-м по вероятности (обход, как в игрушке)."""
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
    """policy: A | B_nomark | V_text | V_ban
    Бюджет на поле (df, se, d -- по FIELD_BUDGET каждое, независимо).
    se -- единственная доступная цель отмотки и при дефекте в t (t пишется
    раньше se и проверяется только после него -- см. ZADANIE.md)."""
    truth = truth_of(inp)
    body = ""
    vals = {}
    events = []
    regen_total = 0

    def emit(field, force_skip=0):
        nonlocal body
        p = build_prompt(tok, inp, body + f"{field} =")
        v, raw = gen_value(model, tok, p, force_skip=force_skip)
        vals[field] = v
        line = raw.split("\n")[0].strip()
        body += f"{field} = {line}\n" if v is not None else f"{field} = ?\n"
        return v

    def rewind_field(field, mark_lines, attempt_idx):
        """Обрезать body до конца поля перед `field`, вставить пометку
        (если политика её предполагает) и заново породить `field`."""
        nonlocal body
        order_idx = FIELDS.index(field)
        lines = body.split("\n")
        # lines[:order_idx] -- строки полей ДО field (df[,t][,se])
        body = "\n".join(lines[:order_idx]) + ("\n" if order_idx else "")
        if policy in ("V_text", "V_ban"):
            body += "[проверка: " + "; ".join(mark_lines) + "]\n"
        force_skip = attempt_idx if policy == "V_ban" else 0
        return emit(field, force_skip=force_skip)

    # ---- df ----
    v = emit("df")
    df_ok = (v is not None) and legit_df(inp["n"], v)
    if policy != "A" and not df_ok:
        for attempt in range(1, FIELD_BUDGET + 1):
            prev = vals["df"]
            mark = [f"поле df отозвано — не сходится df = n − 1 (было {prev})"]
            new_v = rewind_field("df", mark, attempt)
            regen_total += 1
            events.append(dict(field="df", attempt=attempt, prev=prev, new=new_v,
                                mark=mark[0] if policy != "B_nomark" else None,
                                changed=(new_v != prev)))
            df_ok = (new_v is not None) and legit_df(inp["n"], new_v)
            if df_ok:
                break

    # ---- t (проверка отложена до se) ----
    emit("t")

    # ---- se (+ отложенная проверка t) ----
    v_se = emit("se")
    se_ok = (v_se is not None) and legit_se(inp["sd"], inp["n"], v_se)
    t_ok = (vals["t"] is not None) and legit_t(inp["M"], inp["mu"], v_se, truth["se"], vals["t"])
    if policy != "A" and not (se_ok and t_ok):
        for attempt in range(1, FIELD_BUDGET + 1):
            prev = vals["se"]
            if not se_ok:
                mark = [f"поле se отозвано — не сходится se = sd/√n (было {prev})"]
            else:
                mark = [f"поле t не сошлось: t = (M − mu)/se (было t = {vals['t']}); "
                        f"se — ближайшее доступное поле для отмотки, само se установлено верно (было se = {prev})"]
            new_v = rewind_field("se", mark, attempt)
            regen_total += 1
            events.append(dict(field="se", attempt=attempt, prev=prev, new=new_v,
                                mark=mark[0] if policy != "B_nomark" else None,
                                changed=(new_v != prev)))
            se_ok = (new_v is not None) and legit_se(inp["sd"], inp["n"], new_v)
            t_ok = (vals["t"] is not None) and legit_t(inp["M"], inp["mu"], new_v, truth["se"], vals["t"])
            if se_ok and t_ok:
                break

    # ---- d ----
    v_d = emit("d")
    d_ok = (v_d is not None) and legit_d(inp["M"], inp["mu"], inp["sd"], vals["t"], inp["n"], v_d)
    if policy != "A" and not d_ok:
        for attempt in range(1, FIELD_BUDGET + 1):
            prev = vals["d"]
            mark = [f"поле d отозвано — не сходится d = (M − mu)/sd (было {prev})"]
            new_v = rewind_field("d", mark, attempt)
            regen_total += 1
            events.append(dict(field="d", attempt=attempt, prev=prev, new=new_v,
                                mark=mark[0] if policy != "B_nomark" else None,
                                changed=(new_v != prev)))
            d_ok = (new_v is not None) and legit_d(inp["M"], inp["mu"], inp["sd"], vals["t"], inp["n"], new_v)
            if d_ok:
                break

    correct = dict(df=df_ok, t=t_ok, se=se_ok, d=d_ok)
    # молчание: поле реально исчерпало СВОЙ бюджет (по каждому полю отдельно)
    silence = (not df_ok and regen_total_field(events, "df") >= FIELD_BUDGET) or \
              (not (se_ok and t_ok) and regen_total_field(events, "se") >= FIELD_BUDGET) or \
              (not d_ok and regen_total_field(events, "d") >= FIELD_BUDGET)
    return dict(inp=inp, vals=dict(vals), truth=truth, body=body, correct=correct,
                events=events, regen_total=regen_total, silence=silence and policy != "A")


def regen_total_field(events, field):
    return sum(1 for e in events if e["field"] == field)


def main():
    with open(INPUTS_PATH) as f:
        inputs = json.load(f)

    print(f"загрузка модели {MODEL_ID} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.eval()
    load_time = time.time() - t0
    print(f"загрузка: {load_time:.1f} c\n")

    t_start = time.time()

    # ---- ГЕЙТ: политика А, 20 отчётов ----
    print("=== ГЕЙТ: политика А, 20 отчётов ===")
    log_A = []
    t0 = time.time()
    for inp in inputs:
        res = run_report(model, tok, inp, "A")
        log_A.append(res)
    time_A = time.time() - t0
    field_ok = {f: sum(1 for r in log_A if r["correct"][f]) for f in FIELDS}
    total_ok = sum(field_ok.values())
    frac = total_ok / (4 * len(inputs))
    print(f"время: {time_A:.1f} c ({time_A/len(inputs):.2f} c/отчёт)")
    print(f"верно по полям: {field_ok} -> {total_ok}/{4*len(inputs)} = {frac:.1%}")

    results = {"A": dict(log=log_A, time=time_A, field_ok=field_ok, frac=frac)}

    if frac < GATE_MIN_FRAC:
        print(f"\nКРИТЕРИЙ НЕ ПРОЙДЕН ({frac:.1%} < {GATE_MIN_FRAC:.0%}). СТОП.")
        save(results, inputs)
        return

    print(f"\nКРИТЕРИЙ ПРОЙДЕН ({frac:.1%} >= {GATE_MIN_FRAC:.0%}). Продолжаю без остановки.")

    # оценка стоимости политик с отмоткой по смоук-тесту: ~2.9x цены А
    # (se-блок почти всегда тратит весь бюджет из-за t, плюс d-блок при дефекте)
    per_report_A = time_A / len(inputs)
    est_per_report_rewind = per_report_A * 3.0
    elapsed = time.time() - t_start
    remaining = TIME_BUDGET_S - elapsed
    n_use = inputs
    if len(inputs) * est_per_report_rewind * 3 > remaining:
        n_use = inputs[:15]
        print(f"по оценке (3 политики х {est_per_report_rewind:.1f} c/отчёт) не укладываюсь в "
              f"{remaining:.0f} c -- сокращаю до {len(n_use)} отчётов для Б/В-текст/В-запрет")
    print(f"истекло {elapsed:.0f} c из {TIME_BUDGET_S} c бюджета пробы, для Б/В-текст/В-запрет N={len(n_use)}")

    for policy in ("B_nomark", "V_text", "V_ban"):
        elapsed = time.time() - t_start
        remaining = TIME_BUDGET_S - elapsed
        if remaining < 60:
            print(f"бюджет времени пробы почти исчерпан ({elapsed:.0f} c) -- останавливаю дальнейшие политики")
            break
        if len(n_use) * est_per_report_rewind > remaining and len(n_use) > 15:
            n_use = inputs[:15]
            print(f"по факту не укладываюсь -- сокращаю до {len(n_use)} отчётов")
        t0 = time.time()
        log = []
        for inp in n_use:
            log.append(run_report(model, tok, inp, policy))
        dt = time.time() - t0
        field_ok = {f: sum(1 for r in log if r["correct"][f]) for f in FIELDS}
        n_silent = sum(1 for r in log if r["silence"])
        regen_total = sum(r["regen_total"] for r in log)
        print(f"\n=== {policy} === N={len(n_use)} время {dt:.1f} c ({dt/len(n_use):.2f} c/отчёт)")
        print(f"  верно по полям: {field_ok}  молчание: {n_silent}/{len(n_use)}  перепорожд: {regen_total}")
        results[policy] = dict(log=log, time=dt, field_ok=field_ok, n_silent=n_silent,
                                regen_total=regen_total, n_used=len(n_use))

    # Б против А знак в знак (на пересечении использованных входов)
    if "B_nomark" in results:
        nb = results["B_nomark"]["n_used"]
        mism = sum(1 for la, lb in zip(log_A[:nb], results["B_nomark"]["log"]) if la["vals"] != lb["vals"])
        print(f"\nБ против А знак в знак: расхождений {mism}/{nb}")
        results["B_nomark"]["mismatch_vs_A"] = mism

    for policy in ("B_nomark", "V_text", "V_ban"):
        if policy not in results:
            continue
        nb = results[policy]["n_used"]
        saved = damaged = 0
        for la, lp in zip(log_A[:nb], results[policy]["log"]):
            for f in FIELDS:
                a_ok, p_ok = la["correct"][f], lp["correct"][f]
                if not a_ok and p_ok:
                    saved += 1
                elif a_ok and not p_ok:
                    damaged += 1
        results[policy]["saved_vs_A"] = saved
        results[policy]["damaged_vs_A"] = damaged
        print(f"{policy} относительно А: спасено {saved}  повреждено {damaged}")

    if "V_text" in results:
        changed = sum(1 for r in results["V_text"]["log"] for e in r["events"] if e["changed"])
        total_attempts = sum(1 for r in results["V_text"]["log"] for e in r["events"])
        print(f"\nВ-текст: модель сменила значение после пометки в {changed}/{total_attempts} попытках")
        results["V_text"]["changed_after_mark"] = changed
        results["V_text"]["total_attempts"] = total_attempts

    save(results, inputs)
    print(f"\nВсего времени на пробу: {time.time()-t_start:.0f} c")


def save(results, inputs):
    def strip(r):
        return dict(vals=r["vals"], truth=r["truth"], correct=r["correct"],
                    events=r["events"], regen_total=r["regen_total"],
                    silence=r["silence"], body=r["body"], inp=r["inp"])
    out = {}
    for k, v in results.items():
        out[k] = {kk: (vv if kk != "log" else [strip(r) for r in vv]) for kk, vv in v.items()}
    with open("real/proba1_5b_log.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()
