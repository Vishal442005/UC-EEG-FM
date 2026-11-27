import cv2
import numpy as np

for num, name in [
    (1, "Figure_1_Master_Architecture.png"),
    (2, "Figure_2_Cross_Disorder_Latent_Manifold.png"),
    (3, "Figure_3_Clinical_ROC_Curves.png"),
    (4, "Figure_4_Confusion_Matrices.png"),
    (5, "Figure_5_Few_Shot_Sample_Efficiency.png"),
    (6, "Figure_6_Feature_Saliency_and_Neurobiology.png"),
    (7, "Figure_7_Topographical_Scalp_Activation_Maps.png"),
    (8, "Figure_8_Cross_Disorder_Neurobiology_and_Connectomes.png"),
    (9, "Figure_9_Clinical_Decision_Support_System.png"),
    (10, "Figure_10_Clinical_Benchmark_Generalization.png"),
    (11, "Figure_11_Pareto_Frontier_and_Edge_Latency.png")
]:
    p = f"C:\\Users\\Visha\\Desktop\\NewAlcohol\\UC_EEG_FM\\Figures\\{name}"
    im = cv2.imread(p)
    if im is not None:
        h, w, c = im.shape
        top_edge = np.mean(im[:15, :, :])
        bottom_edge = np.mean(im[-15:, :, :])
        left_edge = np.mean(im[:, :15, :])
        right_edge = np.mean(im[:, -15:, :])
        print(f"Fig {num:2d} ({w}x{h}): top={top_edge:.1f}, bot={bottom_edge:.1f}, left={left_edge:.1f}, right={right_edge:.1f}")
