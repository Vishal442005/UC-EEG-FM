import json
import numpy as np

with open(r"E:\All EEG\Models\phase4_transfer_and_saliency.json", "r") as f:
    bench4 = json.load(f)

obs_sim = bench4['cross_disorder_alignment']['cosine_similarity']
null_mean = bench4['cross_disorder_alignment']['null_mean']
null_std = bench4['cross_disorder_alignment']['null_std']
print(f"obs_sim = {obs_sim}, null_mean = {null_mean}, null_std = {null_std}")
