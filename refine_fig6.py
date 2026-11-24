import os

p = r"E:\All EEG\replot_publication_ready_figures_v2.py"
with open(p, "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if "ax1.tick_params(labelsize=17)" in l and "ax1.set_xlabel" in lines[i-2]:
        lines.insert(i, "ax1.set_ylim(0, 1.55)\n")
        print(f"Inserted ylim at line {i}")
        break

for i, l in enumerate(lines):
    if "ax1.text(0.04, 0.44," in l:
        lines[i] = l.replace("0.04, 0.44,", "0.04, 0.58,")
        print(f"Updated text position at line {i}")
        break

with open(p, "w", encoding="utf-8") as f:
    f.writelines(lines)

print("Updated replot_publication_ready_figures_v2.py successfully")
