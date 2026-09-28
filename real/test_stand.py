"""Маленький тест стенда: три отмотки подряд на разных полях -- в
собранном тексте должны быть все записанные поля по порядку и не
больше одной пометки видно за раз. Без модели: gen_value_str
подменяется сценарием.

    python3 real/test_stand.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import etapB as B

CALLS = []


def fake_gen_value_str(model, tok, prompt, force_skip=0, max_new=14):
    CALLS.append(prompt)
    # посчитать пометки, видимые ПРЯМО СЕЙЧАС в этом промпте
    n_marks = prompt.count("[проверка:")
    assert n_marks <= 1, f"больше одной пометки одновременно видно: {n_marks}\n{prompt!r}"
    step = len(CALLS)
    # сценарий: n_A, SD_A, df нуждаются в отмотке (неверно на первой
    # попытке, верно со второй); остальные сразу верны.
    script = {
        1: "999",     # n_A первая попытка -- неверно
        2: None,      # -> будет подставлено по имени поля ниже
    }
    return None, "placeholder"


def main():
    item = dict(v=dict(nA=30, MA=6.24, SDA=1.06, nB=87, MB=5.82, SDB=1.69, df=117, t=-0.30, p=0.651),
                tpl=0, text="...")
    truth = B.truth_of(item["v"])

    # управляемый сценарий через замену gen_value_str: у полей n_A, SD_A,
    # df первая попытка -- заведомо неверное число, вторая -- верное.
    wrong_once = {"n_A", "SD_A", "df"}
    attempt_count = {}

    def scripted(model, tok, prompt, force_skip=0, max_new=14):
        n_marks = prompt.count("[проверка:")
        if current_policy[0] == "V_text_erase":
            # для стирания в любой момент видна максимум ОДНА (текущая,
            # транзитная) пометка -- предыдущие уже стёрты
            assert n_marks <= 1, f"[V_text_erase] больше одной пометки одновременно: {n_marks}"
        # определить, какое поле сейчас генерируется -- по хвосту промпта
        field = prompt.rstrip().split("\n")[-1].split(" =")[0]
        attempt_count[field] = attempt_count.get(field, 0) + 1
        if field in wrong_once and attempt_count[field] == 1:
            return "999", "999"
        return truth[field], truth[field]

    current_policy = [None]
    B.gen_value_str = scripted
    B.build_prompt = lambda tok, item, body: body  # без реального токенизатора/чат-шаблона
    for policy in ("V_text", "V_text_erase"):
        current_policy[0] = policy
        attempt_count.clear()
        r = B.run_report(None, None, item, policy)
        body = r["body"]
        lines = [l for l in body.split("\n") if l]
        field_lines = [l for l in lines if not l.startswith("[проверка:")]
        got_fields = [l.split(" =")[0] for l in field_lines]
        assert got_fields == B.FIELDS, f"[{policy}] поля не по порядку или выпали: {got_fields}"
        n_marks_total = sum(1 for l in lines if l.startswith("[проверка:"))
        if policy == "V_text":
            assert n_marks_total == len(wrong_once), \
                f"[V_text] ожидали {len(wrong_once)} постоянных пометки, нашли {n_marks_total}"
        else:
            assert n_marks_total == 0, f"[V_text_erase] пометки не должны оставаться в теле, нашли {n_marks_total}"
        assert all(r["correct"].values()), f"[{policy}] не все поля сошлись после отмотки: {r['correct']}"
        print(f"{policy}: OK -- поля по порядку {got_fields}, постоянных пометок {n_marks_total}")

    print("\nвсе проверки прошли")


if __name__ == "__main__":
    main()
