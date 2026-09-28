"""Задача Б, круг 2: В-текст-стирание против В-текст (КБ6).

4 бита (кэш), 20 отчётов (те же, что и при исходном А), бюджет 3 на
поле. Политики: А (для точной попольной базы сравнения — агрегат уже
известен, но нужен по-отчётный лог), В-текст (как раньше), В-текст-
стирание (пометка видна только пока перепорождается её поле).
"""
import json, sys, time
sys.path.insert(0, "real")
import etapB as B

TIME_BUDGET_S = 45 * 60
N_REPORTS = 20


def main():
    inputs = json.load(open("real/etapB_inputs.json"))
    tok = B.AutoTokenizer.from_pretrained(B.MODEL_ID)
    model4 = B.load_quantized(B.MODEL_ID, 4)

    t_start = time.time()
    n_use = inputs[:N_REPORTS]

    results = {}
    results["A"] = B.run_policy(model4, tok, n_use, "A", "4 бита, А (по-отчётно)")

    elapsed = time.time() - t_start
    remaining = TIME_BUDGET_S - elapsed
    per_report_A = results["A"]["time"] / len(n_use)
    est_vtext = per_report_A * 2.3  # по прошлому прогону ~2.2-2.3x цены A
    if len(n_use) * est_vtext * 2 > remaining:
        n_use = inputs[:15]
        print(f"по оценке сокращаю до {len(n_use)} отчётов для В-текст/В-текст-стирание "
              f"(осталось {remaining:.0f} c)")
        results["A"] = B.run_policy(model4, tok, n_use, "A", "4 бита, А (по-отчётно, N=15)")

    for policy, key in (("V_text", "V_text"), ("V_text_erase", "V_text_erase")):
        elapsed = time.time() - t_start
        remaining = TIME_BUDGET_S - elapsed
        if remaining < 60:
            print("бюджет времени почти исчерпан -- останавливаю")
            break
        results[key] = B.run_policy(model4, tok, n_use, policy, f"4 бита, {policy}")

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
