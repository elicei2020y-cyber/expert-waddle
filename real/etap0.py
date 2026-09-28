"""Этап 0: установка, проба скорости на 5 отчётах, политика А (не вмешиваться).

Модель: Qwen/Qwen2.5-0.5B-Instruct, float32, CPU, жадное декодирование.
Поля: df, t, se, d -- в этом порядке, каждое строкой "имя = число".
"""
import random, re, time, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MU = 4.00
FIELDS = ["df", "t", "se", "d"]

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


def gen_inputs(seed, k):
    rng = random.Random(seed)
    out = []
    for _ in range(k):
        n = rng.randint(8, 200)
        M = round(rng.uniform(2, 6), 2)
        sd = round(rng.uniform(0.5, 3), 2)
        out.append(dict(n=n, M=M, sd=sd, mu=MU))
    return out


def truth(inp):
    n, M, sd, mu = inp["n"], inp["M"], inp["sd"], inp["mu"]
    df = n - 1
    se = sd / (n ** 0.5)
    t = (M - mu) / se
    d = (M - mu) / sd
    return dict(df=df, t=t, se=se, d=d)


NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


def gen_field(model, tok, prompt_ids, max_new=12):
    """Дописать 'имя = ' и дать модели дописать число до перевода строки."""
    ids = tok(prompt_ids, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    new_tokens = out[0][ids.shape[1]:]
    text = tok.decode(new_tokens, skip_special_tokens=True)
    line = text.split("\n")[0]
    m = NUM_RE.search(line)
    return (m.group(0) if m else None), text


def check(field, val_str, t):
    if val_str is None:
        return False, "не разобралось"
    try:
        v = float(val_str)
    except ValueError:
        return False, "не число"
    tv = t[field]
    tol = 0.005
    if field == "df":
        ok = abs(v - tv) <= 0.5
    elif field == "se":
        ok = abs(v - tv) <= tol
    elif field == "d":
        ok = abs(v - tv) <= tol
    elif field == "t":
        ok = abs(v - tv) <= tol
    else:
        ok = False
    return ok, v


def build_chat_prefix(tok, inp, done_fields):
    user = f"n = {inp['n']}, M = {inp['M']:.2f}, sd = {inp['sd']:.2f}, mu = {inp['mu']:.2f}"
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": EXAMPLE_IN},
        {"role": "assistant", "content": EXAMPLE_OUT},
        {"role": "user", "content": user},
    ]
    prefix = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    body = ""
    for f in done_fields:
        body += f"{f} = {done_fields[f]}\n"
    return prefix + body + f"{FIELDS[len(done_fields)]} = " if len(done_fields) < len(FIELDS) else prefix + body


def run_report(model, tok, inp):
    t = truth(inp)
    done = {}
    results = {}
    for f in FIELDS:
        prefix = build_chat_prefix(tok, inp, done)
        raw, full_text = gen_field(model, tok, prefix)
        ok, v = check(f, raw, t)
        results[f] = (raw, ok, v, t[f])
        done[f] = raw if raw is not None else "?"
    return results


def main():
    print(f"загрузка токенизатора и модели: {MODEL_ID} (float32, cpu)")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()
    print(f"загрузка заняла {time.time()-t0:.1f} c")

    inputs = gen_inputs(seed=12345, k=5)
    total_fields = 0
    total_ok = 0
    total_parsed = 0
    per_report_time = []

    for i, inp in enumerate(inputs):
        t0 = time.time()
        res = run_report(model, tok, inp)
        dt = time.time() - t0
        per_report_time.append(dt)
        print(f"\n--- отчёт {i+1}: n={inp['n']} M={inp['M']} sd={inp['sd']} ({dt:.2f} c) ---")
        for f in FIELDS:
            raw, ok, v, tv = res[f]
            total_fields += 1
            if raw is not None:
                total_parsed += 1
            if ok:
                total_ok += 1
            print(f"  {f}: модель={raw!r} истина={tv:.4f} {'OK' if ok else 'НЕВЕРНО'}")

    print("\n=== ИТОГО (политика А, 5 отчётов, исходные веса) ===")
    print(f"среднее время на отчёт: {sum(per_report_time)/len(per_report_time):.2f} c")
    print(f"разобрано как число: {total_parsed}/{total_fields}")
    print(f"верно: {total_ok}/{total_fields}")


if __name__ == "__main__":
    main()
