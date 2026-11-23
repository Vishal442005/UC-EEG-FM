import os
from PIL import Image

fig_dir = r"E:\All EEG\Figures"
if os.path.exists(fig_dir):
    for f in sorted(os.listdir(fig_dir)):
        if f.endswith('.png'):
            p = os.path.join(fig_dir, f)
            im = Image.open(p)
            print(f"{f:50s} : {im.size} : {os.path.getsize(p):,} bytes")
else:
    print(f"Directory {fig_dir} does not exist")
