"""Pick the cut size K from the ranker's OOF recall table: the smallest K in (8, 10, 12, 15) whose learned recall is at least the
raw (baseline) top-15 recall in BOTH countries; 15 if none. Writes work/rk_k.txt and prints the decision."""
import re
import sys
from pathlib import Path

log = Path(sys.argv[1]).read_text(encoding="utf-8")
tab = {}
for line in log.splitlines():
    m = re.match(r"OOF (\w+): (.*)", line)
    if not m:
        continue
    d = {}
    for k, raw, learned in re.findall(r"K=(\d+): raw ([\d.]+) learned ([\d.]+)", m.group(2)):
        d[int(k)] = (float(raw), float(learned))
    tab[m.group(1)] = d
pick = 15
for k in (8, 10, 12, 15):
    if tab and all(d[k][1] >= d[15][0] for d in tab.values()):
        pick = k
        break
for c, d in tab.items():
    print(c, "raw@15 %.4f" % d[15][0], " ".join("learned@%d %.4f" % (k, d[k][1]) for k in (8, 10, 12, 15)))
print("chosen K =", pick)
Path(sys.argv[2]).write_text(str(pick))
