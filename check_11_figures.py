import os
from PIL import Image

fig_dir = r"E:\All EEG\Figures"
files = [
    "Figure_1_Master_Architecture.png",
    "Figure_2_Cross_Disorder_Latent_Manifold.png",
    "Figure_3_Clinical_ROC_Curves.png",
    "Figure_4_Confusion_Matrices.png",
    "Figure_5_Few_Shot_Sample_Efficiency.png",
    "Figure_6_Feature_Saliency_and_Neurobiology.png",
    "Figure_7_Topographical_Scalp_Activation_Maps.png",
    "Figure_8_Cross_Disorder_Neurobiology_and_Connectomes.png",
    "Figure_9_Clinical_Decision_Support_System.png",
    "Figure_10_Clinical_Benchmark_Generalization.png",
    "Figure_11_Pareto_Frontier_and_Edge_Latency.png"
]

print("Checking presence and size of all 11 manuscript figures:")
for f in files:
    p = os.path.join(fig_dir, f)
    if os.path.exists(p):
        im = Image.open(p)
        print(f"  [OK] {f:55s} : {im.size} : {os.path.getsize(p):,} bytes")
    else:
        print(f"  [MISSING] {f}")
