import os
from PIL import Image

fig_dir = r"E:\All EEG\Figures"
brain_dir = r"C:\Users\Visha\.gemini\antigravity\brain\a487a487-156f-4781-ba99-0b2ec1ba776a"

for f in [
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
]:
    src = os.path.join(fig_dir, f)
    dst = os.path.join(brain_dir, f)
    if os.path.exists(src):
        # copy
        import shutil
        shutil.copy2(src, dst)
        im = Image.open(src)
        print(f"Copied {f} to brain artifact: size = {im.size}")
