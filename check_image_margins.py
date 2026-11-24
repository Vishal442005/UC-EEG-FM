import cv2
import numpy as np

# Load Figure 9 and check borders
im9 = cv2.imread(r"E:\All EEG\Figures\Figure_9_Clinical_Decision_Support_System.png")
h, w, c = im9.shape
print(f"Figure 9: shape = ({w}, {h})")

# Check if margins have non-white pixels (cropping test)
top_edge = np.mean(im9[:20, :, :])
bottom_edge = np.mean(im9[-20:, :, :])
left_edge = np.mean(im9[:, :20, :])
right_edge = np.mean(im9[:, -20:, :])
print(f"Figure 9 Edges (255 = clean white margin): top={top_edge:.1f}, bottom={bottom_edge:.1f}, left={left_edge:.1f}, right={right_edge:.1f}")

# Check Figure 10
im10 = cv2.imread(r"E:\All EEG\Figures\Figure_10_Clinical_Benchmark_Generalization.png")
h, w, c = im10.shape
print(f"\nFigure 10: shape = ({w}, {h})")
top_edge = np.mean(im10[:20, :, :])
bottom_edge = np.mean(im10[-20:, :, :])
left_edge = np.mean(im10[:, :20, :])
right_edge = np.mean(im10[:, -20:, :])
print(f"Figure 10 Edges: top={top_edge:.1f}, bottom={bottom_edge:.1f}, left={left_edge:.1f}, right={right_edge:.1f}")

# Check Figure 11
im11 = cv2.imread(r"E:\All EEG\Figures\Figure_11_Pareto_Frontier_and_Edge_Latency.png")
h, w, c = im11.shape
print(f"\nFigure 11: shape = ({w}, {h})")
top_edge = np.mean(im11[:20, :, :])
bottom_edge = np.mean(im11[-20:, :, :])
left_edge = np.mean(im11[:, :20, :])
right_edge = np.mean(im11[:, -20:, :])
print(f"Figure 11 Edges: top={top_edge:.1f}, bottom={bottom_edge:.1f}, left={left_edge:.1f}, right={right_edge:.1f}")
