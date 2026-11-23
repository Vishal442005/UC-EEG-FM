import os
import shutil

dirs = [
    r"E:\All EEG\Figures",
    r"E:\All EEG\latex_paper\figures",
    r"C:\Users\Visha\Desktop\NewAlcohol\Figures",
    r"C:\Users\Visha\.gemini\antigravity\brain\a487a487-156f-4781-ba99-0b2ec1ba776a"
]

copies = [
    ("Figure_8_Clinical_Decision_Support_System.png", "Figure_9_Clinical_Decision_Support_System.png"),
    ("Figure_8_Clinical_Decision_Support_System.pdf", "Figure_9_Clinical_Decision_Support_System.pdf"),
    ("Figure_9_Clinical_Benchmark_Generalization.png", "Figure_10_Clinical_Benchmark_Generalization.png"),
    ("Figure_9_Clinical_Benchmark_Generalization.pdf", "Figure_10_Clinical_Benchmark_Generalization.pdf"),
    ("Figure_10_Pareto_Frontier_and_Edge_Latency.png", "Figure_11_Pareto_Frontier_and_Edge_Latency.png"),
    ("Figure_10_Pareto_Frontier_and_Edge_Latency.pdf", "Figure_11_Pareto_Frontier_and_Edge_Latency.pdf")
]

for d in dirs:
    if os.path.exists(d):
        for src, dst in copies:
            src_p = os.path.join(d, src)
            dst_p = os.path.join(d, dst)
            if os.path.exists(src_p):
                shutil.copy2(src_p, dst_p)
                print(f"Copied {src} -> {dst} in {d}")

print("=== Figure filename synchronization completed successfully ===")
