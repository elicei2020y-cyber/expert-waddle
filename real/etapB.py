"""Задача Б: переписывание из текста (арифметика закрыта).

Абзац с результатами двух групп -> 9 полей по порядку:
n_A, M_A, SD_A, n_B, M_B, SD_B, df, t, p -- каждое строкой "имя = число".
Проверка: точное строковое совпадение с тем, как число напечатано в
тексте (не численное сравнение с допуском).

Дефекты даёт квантование (qguard.py) весов nn.Linear внутри
model.model.layers. Модель: Qwen2.5-0.5B-Instruct.
"""
import json, math, os, random, re, sys, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qguard.qguard import quantize

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
FIELDS = ["n_A", "M_A", "SD_A", "n_B", "M_B", "SD_B", "df", "t", "p"]
FIELD_BUDGET = 3
NUM_RE = re.compile(r"[-+]?\d*\.?\d+")
Q_CACHE = os.path.expanduser("~/q_cache")
N_REPORTS = 20
TIME_BUDGET_S = 45 * 60

SYSTEM = (
    "Тебе дан абзац с результатами сравнения двух групп (A и B). Перепиши "
    "из текста девять чисел, ровно в этом порядке, каждое строкой "
    "\"имя = число\":\n"
    "n_A = ...\nM_A = ...\nSD_A = ...\nn_B = ...\nM_B = ...\nSD_B = ...\n"
    "df = ...\nt = ...\np = ...\n"
    "Ничего не считай -- только переписывай числа из текста дословно, "
    "в исходном формате (n и df -- целые без точки, M и SD и t -- с двумя "
    "знаками после запятой, p -- с тремя). Группы A и B могут быть "
    "упомянуты в любом порядке -- определяй по букве (A или B), не по "
    "порядку появления в тексте. Ничего кроме девяти строк не пиши."
)

TEMPLATES = [
    "В группе A (n = {nA}) среднее составило {MA} (SD = {SDA}), в группе B "
    "(n = {nB}) — {MB} (SD = {SDB}); t({df}) = {t}, p = {p}.",
    "В группе B (n = {nB}) среднее составило {MB} (SD = {SDB}), в группе A "
    "(n = {nA}) — {MA} (SD = {SDA}); t({df}) = {t}, p = {p}.",
    "Среднее в группе A (n={nA}) равнялось {MA}, SD={SDA}; в группе B "
    "(n={nB}) — {MB}, SD={SDB}. t({df})={t}, p={p}.",
    "Результаты по группе B: n={nB}, M={MB}, SD={SDB}. По группе A: "
    "n={nA}, M={MA}, SD={SDA}. t-критерий: t({df}) = {t}, p = {p}.",
]


def fmt_pair(v):
    return dict(nA=v["nA"], MA=f"{v['MA']:.2f}", SDA=f"{v['SDA']:.2f}",
                nB=v["nB"], MB=f"{v['MB']:.2f}", SDB=f"{v['SDB']:.2f}",
                df=v["df"], t=f"{v['t']:.2f}", p=f"{v['p']:.3f}")


def make_text(v, tpl_idx):
    return TEMPLATES[tpl_idx].format(**fmt_pair(v))


def truth_of(v):
    s = fmt_pair(v)
    return dict(n_A=str(s["nA"]), M_A=s["MA"], SD_A=s["SDA"],
                n_B=str(s["nB"]), M_B=s["MB"], SD_B=s["SDB"],
                df=str(s["df"]), t=s["t"], p=s["p"])


EXAMPLES = [
    dict(v=dict(nA=48, MA=5.31, SDA=1.27, nB=52, MB=4.86, SDB=1.44, df=98, t=1.66, p=0.100), tpl=0),
    dict(v=dict(nA=30, MA=6.10, SDA=0.85, nB=35, MB=5.40, SDB=1.10, df=63, t=2.45, p=0.017), tpl=1),
    dict(v=dict(nA=22, MA=3.75, SDA=1.05, nB=25, MB=4.20, SDB=0.95, df=45, t=-1.35, p=0.184), tpl=2),
]


def gen_inputs(seed, k):
    rng = random.Random(seed)
    out = []
    for _ in range(k):
        nA = rng.randint(15, 90)
        nB = rng.randint(15, 90)
        MA = round(rng.uniform(2, 7), 2)
        MB = round(rng.uniform(2, 7), 2)
        SDA = round(rng.uniform(0.4, 3.0), 2)
        SDB = round(rng.uniform(0.4, 3.0), 2)
        t = round(rng.uniform(-3, 3), 2)
        p = round(rng.uniform(0.001, 0.999), 3)
        v = dict(nA=nA, MA=MA, SDA=SDA, nB=nB, MB=MB, SDB=SDB, df=nA + nB - 2, t=t, p=p)
        tpl = rng.randrange(len(TEMPLATES))
        out.append(dict(v=v, tpl=tpl, text=make_text(v, tpl)))
    return out


def build_messages(item):
    messages = [{"role": "system", "content": SYSTEM}]
    for ex in EXAMPLES:
        messages.append({"role": "user", "content": make_text(ex["v"], ex["tpl"])})
        t = truth_of(ex["v"])
        out = "\n".join(f"{f} = {t[f]}" for f in FIELDS)
        messages.append({"role": "assistant", "content": out})
    messages.append({"role": "user", "content": item["text"]})
    return messages


def build_prompt(tok, item, body):
    prefix = tok.apply_chat_template(build_messages(item), tokenize=False, add_generation_prompt=True)
    return prefix + body


def gen_value_str(model, tok, prompt, force_skip=0, max_new=14):
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
    return (m.group(0) if m else None), text


def run_report(model, tok, item, policy):
    truth = truth_of(item["v"])
    body = ""
    vals = {}
    events = []

    def emit(field, force_skip=0):
        nonlocal body
        p = build_prompt(tok, item, body + f"{field} =")
        v, raw = gen_value_str(model, tok, p, force_skip=force_skip)
        vals[field] = v
        line = raw.split("\n")[0].strip()
        body += f"{field} = {line}\n"
        return v

    correct = {}
    for field in FIELDS:
        emit(field)
        ok = (vals[field] == truth[field])
        if policy != "A" and not ok:
            for attempt in range(1, FIELD_BUDGET + 1):
                prev = vals[field]
                group = "A" if field.endswith("_A") else ("B" if field.endswith("_B") else None)
                if group:
                    mark_txt = (f"поле {field} отозвано — значение должно быть взято из текста "
                                f"дословно, для группы {group} (было {prev})")
                else:
                    mark_txt = f"поле {field} отозвано — значение должно быть взято из текста дословно (было {prev})"
                lines = body.split("\n")
                idx = FIELDS.index(field)
                body = "\n".join(lines[:idx]) + ("\n" if idx else "")
                if policy in ("V_text", "V_ban"):
                    body += f"[проверка: {mark_txt}]\n"
                force_skip = attempt if policy == "V_ban" else 0
                new_v = emit(field, force_skip=force_skip)
                events.append(dict(field=field, attempt=attempt, prev=prev, new=new_v,
                                    mark=mark_txt if policy != "B_nomark" else None,
                                    changed=(new_v != prev)))
                ok = (new_v == truth[field])
                if ok:
                    break
        correct[field] = ok

    silence = policy != "A" and any(
        not correct[f] and sum(1 for e in events if e["field"] == f) >= FIELD_BUDGET
        for f in FIELDS
    )
    return dict(item=dict(v=item["v"], tpl=item["tpl"], text=item["text"]), vals=dict(vals),
                truth=truth, body=body, correct=correct, events=events, silence=silence)


def quantize_model_(model, bits):
    n = 0
    for layer in model.model.layers:
        for name, module in layer.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data.numpy().astype(np.float64)
                Wq = quantize(W.T, bits).T
                module.weight.data = torch.from_numpy(Wq.astype(np.float32).copy())
                n += 1
    return n


def load_quantized(model_id, bits, cache_dir=Q_CACHE):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"qwen0.5b_{bits}bit.pt")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()
    if bits >= 16:
        return model
    if os.path.exists(path):
        sd = torch.load(path, map_location="cpu")
        model.load_state_dict(sd)
        print(f"  ({bits} бит: загружено из кэша {path})")
        return model
    t0 = time.time()
    n = quantize_model_(model, bits)
    print(f"  ({bits} бит: квантовано {n} слоёв за {time.time()-t0:.1f} c)")
    torch.save(model.state_dict(), path)
    return model


def run_policy(model, tok, inputs, policy, label):
    t0 = time.time()
    log = [run_report(model, tok, it, policy) for it in inputs]
    dt = time.time() - t0
    field_ok = {f: sum(1 for r in log if r["correct"][f]) for f in FIELDS}
    total = sum(field_ok.values())
    n_silent = sum(1 for r in log if r["silence"])
    regen_total = sum(len(r["events"]) for r in log)
    print(f"=== {label} === N={len(inputs)} время {dt:.1f} c ({dt/len(inputs):.2f} c/отчёт)")
    print(f"  верно по полям: {field_ok} -> {total}/{9*len(inputs)} = {total/(9*len(inputs)):.1%}")
    print(f"  молчание: {n_silent}/{len(inputs)}  перепорожд: {regen_total}")
    return dict(log=log, time=dt, field_ok=field_ok, total=total, frac=total/(9*len(inputs)),
                n_silent=n_silent, regen_total=regen_total)


def save(results, path="real/etapB_log.json"):
    def strip(r):
        return r
    out = {k: {kk: (vv if kk != "log" else vv) for kk, vv in v.items()} for k, v in results.items()}
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)


def main():
    inputs = gen_inputs(seed=987654, k=N_REPORTS)
    with open("real/etapB_inputs.json", "w") as f:
        json.dump(inputs, f, ensure_ascii=False, indent=2)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    t_start = time.time()

    results = {}
    print("=== исходные веса ===")
    model = load_quantized(MODEL_ID, 16)
    results["orig_A"] = run_policy(model, tok, inputs, "A", "исходные веса, А")

    print("\n=== 4 бита ===")
    model4 = load_quantized(MODEL_ID, 4)
    results["4bit_A"] = run_policy(model4, tok, inputs, "A", "4 бита, А")

    levels_tested = {"orig": results["orig_A"], "4bit": results["4bit_A"]}
    run_3bit = results["4bit_A"]["frac"] >= 0.90
    if run_3bit:
        print(f"\n4 бита: неверно {1-results['4bit_A']['frac']:.1%} < 10% -> квантую 3 бита (медленно)")
        t0 = time.time()
        model3 = load_quantized(MODEL_ID, 3)
        print(f"(3 бита готово, суммарно с квантованием {time.time()-t0:.1f} c)")
        results["3bit_A"] = run_policy(model3, tok, inputs, "A", "3 бита, А")
        levels_tested["3bit"] = results["3bit_A"]

    # выбрать уровень, где верно от 50% до 90%
    chosen = None
    for name, r in levels_tested.items():
        if 0.50 <= r["frac"] <= 0.90:
            chosen = name
            break
    print(f"\nУровни: " + ", ".join(f"{k}={v['frac']:.1%}" for k, v in levels_tested.items()))
    if chosen is None:
        print("НИ ОДИН уровень не попал в диапазон 50-90% -- нужна ваша команда. СТОП.")
        save(results)
        return
    print(f"Выбран уровень: {chosen} ({levels_tested[chosen]['frac']:.1%}) -- продолжаю без остановки.\n")

    model_map = {"orig": model, "4bit": model4}
    if "3bit" in levels_tested:
        model_map["3bit"] = model3
    chosen_model = model_map[chosen]

    n_use = inputs
    elapsed = time.time() - t_start
    remaining = TIME_BUDGET_S - elapsed
    per_report_A = levels_tested[chosen]["time"] / len(inputs)
    est_rewind = per_report_A * 3.0
    if len(inputs) * est_rewind * 3 > remaining:
        n_use = inputs[:max(10, int(remaining / (3 * est_rewind)))]
        print(f"по оценке сокращаю до {len(n_use)} отчётов для Б/В-текст/В-запрет "
              f"(осталось {remaining:.0f} c)")

    for policy, key in (("B_nomark", "B"), ("V_text", "V_text"), ("V_ban", "V_ban")):
        elapsed = time.time() - t_start
        remaining = TIME_BUDGET_S - elapsed
        if remaining < 60:
            print("бюджет времени почти исчерпан -- останавливаю")
            break
        results[key] = run_policy(chosen_model, tok, n_use, policy, f"{chosen}, {policy}")

    if "B" in results:
        nb = len(n_use)
        chosen_A_log = levels_tested[chosen]["log"][:nb]
        mism = sum(1 for la, lb in zip(chosen_A_log, results["B"]["log"]) if la["vals"] != lb["vals"])
        print(f"\nБ против А знак в знак: расхождений {mism}/{nb}")
        results["B"]["mismatch_vs_A"] = mism

        for key in ("B", "V_text", "V_ban"):
            if key not in results:
                continue
            saved = damaged = 0
            for la, lp in zip(chosen_A_log, results[key]["log"]):
                for f in FIELDS:
                    a_ok, p_ok = la["correct"][f], lp["correct"][f]
                    if not a_ok and p_ok:
                        saved += 1
                    elif a_ok and not p_ok:
                        damaged += 1
            results[key]["saved_vs_A"] = saved
            results[key]["damaged_vs_A"] = damaged
            print(f"{key} относительно А: спасено {saved}  повреждено {damaged}")

        if "V_text" in results:
            changed = 0
            total_first = 0
            for r, la in zip(results["V_text"]["log"], chosen_A_log):
                for e in r["events"]:
                    if e["attempt"] != 1:
                        continue
                    total_first += 1
                    if e["new"] != la["vals"][e["field"]]:
                        changed += 1
            print(f"В-текст: на первой попытке после пометки != А в {changed}/{total_first} случаях")
            results["V_text"]["changed_vs_A_first_attempt"] = changed
            results["V_text"]["first_attempts_total"] = total_first

    results["chosen_level"] = chosen
    results["n_use"] = len(n_use)
    save(results)
    print(f"\nВсего времени: {time.time()-t_start:.0f} c")


if __name__ == "__main__":
    main()
