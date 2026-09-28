"""Задача Б, круг 2 (стенд исправлен): В-текст-стирание против В-текст (КБ6).

4 бита (кэш), бюджет 3 на поле. Порядок:
  0. проверка сохранения JSONL на 1 отчёте;
  1. Б на 5 отчётах -- только проверить Б=А знак в знак после исправления;
  2. А, В-текст, В-текст-стирание на 20 отчётах (или 15 при нехватке
     времени), по-отчётный лог пишется в JSONL сразу же.
"""
import json, os, sys, time
sys.path.insert(0, "real")
import etapB as B

TIME_BUDGET_S = 45 * 60
N_REPORTS = 20
JSONL_PATH = "real/etapB2_log.jsonl"


def check_save_on_one_report(model, tok, inputs):
    if os.path.exists(JSONL_PATH):
        os.remove(JSONL_PATH)
    B.run_policy(model, tok, inputs[:1], "A", "проверка сохранения (1 отчёт)", jsonl_path=JSONL_PATH)
    with open(JSONL_PATH) as f:
        lines = f.readlines()
    assert len(lines) == 1, f"ожидали 1 строку в JSONL, нашли {len(lines)}"
    rec = json.loads(lines[0])
    assert "vals" in rec and "correct" in rec and "policy" in rec, f"неполная запись: {rec.keys()}"
    print("проверка сохранения: OK (1 строка JSONL, поля на месте)")
    os.remove(JSONL_PATH)


def main():
    inputs = json.load(open("real/etapB_inputs.json"))
    tok = B.AutoTokenizer.from_pretrained(B.MODEL_ID)
    model4 = B.load_quantized(B.MODEL_ID, 4)

    check_save_on_one_report(model4, tok, inputs)

    t_start = time.time()

    # Б на 5 отчётах -- только сверить со знак-в-знак после исправления стенда
    a5 = B.run_policy(model4, tok, inputs[:5], "A", "4 бита, А (5 отчётов, контроль)")
    b5 = B.run_policy(model4, tok, inputs[:5], "B_nomark", "4 бита, Б (5 отчётов, контроль)")
    mism5 = sum(1 for la, lb in zip(a5["log"], b5["log"]) if la["vals"] != lb["vals"])
    print(f"Б против А знак в знак (контроль, 5 отчётов): расхождений {mism5}/5")

    n_use = inputs[:N_REPORTS]
    results = {"A_check5": mism5}
    results["A"] = B.run_policy(model4, tok, n_use, "A", "4 бита, А (по-отчётно)", jsonl_path=JSONL_PATH)

    elapsed = time.time() - t_start
    remaining = TIME_BUDGET_S - elapsed
    per_report_A = results["A"]["time"] / len(n_use)
    est_vtext = per_report_A * 2.3
    if len(n_use) * est_vtext * 2 > remaining:
        n_use = inputs[:15]
        print(f"по оценке сокращаю до {len(n_use)} отчётов для В-текст/В-текст-стирание "
              f"(осталось {remaining:.0f} c)")
        results["A"] = B.run_policy(model4, tok, n_use, "A", "4 бита, А (по-отчётно, N=15)",
                                     jsonl_path=JSONL_PATH)

    for policy, key in (("V_text", "V_text"), ("V_text_erase", "V_text_erase")):
        elapsed = time.time() - t_start
        remaining = TIME_BUDGET_S - elapsed
        if remaining < 60:
            print("бюджет времени почти исчерпан -- останавливаю")
            break
        results[key] = B.run_policy(model4, tok, n_use, policy, f"4 бита, {policy}",
                                     jsonl_path=JSONL_PATH)

    A_log = results["A"]["log"]
    for key in ("V_text", "V_text_erase"):
        if key not in results:
            continue
        saved = damaged = 0
        for la, lp in zip(A_log, results[key]["log"]):
            for f in B.FIELDS:
                a_ok, p_ok = la["correct"][f], lp["correct"][f]
                if not a_ok and p_ok:
                    saved += 1
                elif a_ok and not p_ok:
                    damaged += 1
        results[key]["saved_vs_A"] = saved
        results[key]["damaged_vs_A"] = damaged
        n_reports_fully_ok = sum(1 for r in results[key]["log"] if all(r["correct"].values()))
        n_delivered = sum(1 for r in results[key]["log"] if not r["silence"])
        results[key]["n_reports_fully_ok"] = n_reports_fully_ok
        results[key]["n_delivered"] = n_delivered
        print(f"{key} относительно А: спасено {saved}  повреждено {damaged}  "
              f"выдано {n_delivered}/{len(n_use)}  полностью верных {n_reports_fully_ok}")

    results["n_use"] = len(n_use)
    B.save(results, path="real/etapB2_log.json")
    print(f"\nВсего времени: {time.time()-t_start:.0f} c")


if __name__ == "__main__":
    main()
