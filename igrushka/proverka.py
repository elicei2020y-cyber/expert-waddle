"""Эталон игрушки: fork.py должен давать те же числа, что в docs/igrushka_rezultaty.md.

    python3 igrushka/proverka.py
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
out = subprocess.run([sys.executable, os.path.join(HERE, "opyty", "fork", "fork.py")],
                     capture_output=True, text=True).stdout

ETALON = [
    ("оставленная линия (без вмешательства)", "7.52      53        0        0.00"),
    ("отмотка на 1 шаг, БЕЗ пометки", "5.27      21        0        8.25"),
    ("отмотка на 1 шаг, с пометкой", "8.00      22       38        0.05"),
    ("отмотка к происхождению, с пометкой", "6.12      23       26        1.24"),
]
bad = [f"  {k}: ждали «{v}»" for k, v in ETALON if not any(k in l and v in l for l in out.splitlines())]
if bad:
    print("РАСХОЖДЕНИЕ с эталоном игрушки:"); print("\n".join(bad)); print(out); sys.exit(1)
print("совпало: игрушка воспроизводит эталонные числа (4 проверки)")
