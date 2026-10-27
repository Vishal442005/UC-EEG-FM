"""
Downloads the 10 selected landmark and latest (2021-2026) papers on EEG foundation models,
alcoholism classification, Alzheimer's detection, and seizure prediction into:
E:\All EEG\Literature_Review\Papers\
"""

import os
import sys
import time
import json
import requests
import xml.etree.ElementTree as ET

OUT_DIR = r"E:\All EEG\Literature_Review\Papers"
os.makedirs(OUT_DIR, exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
}

PAPERS = [
    {
        "id": "2305.10351",
        "file": "01_Yang_2023_BIOT_Biosignal_Transformer.pdf",
        "scope": "Cross-Disorder EEG Foundation Model / Seizure & Abnormality Detection",
        "year": 2023
    },
    {
        "id": "2405.18765",
        "file": "02_Jiang_2024_LaBraM_Large_Brain_Model.pdf",
        "scope": "Large Brain Foundation Model / Multi-Hospital Clinical EEG Classification",
        "year": 2024
    },
    {
        "id": "2101.12037",
        "file": "03_Kostas_2021_BENDR_Transformers_Biosignal.pdf",
        "scope": "Contrastive Self-Supervised EEG Transformer / Multi-Task Decoding",
        "year": 2021
    },
    {
        "id": "2311.03764",
        "file": "04_Cui_2023_NeuroGPT_Foundation_Model_EEG.pdf",
        "scope": "Generative Foundation Model for EEG Classification",
        "year": 2023
    },
    {
        "id": "2512.08959",
        "file": "05_Wang_2025_EEGBench_Clinical_Foundation_Models.pdf",
        "scope": "Clinical Benchmarking Suite for EEG Foundation Models",
        "year": 2025
    },
    {
        "id": "2604.10293",
        "file": "06_Silva_2026_Validation_Strategy_Alcoholism_EEG.pdf",
        "scope": "Alcohol Use Disorder Detection / Subject Leakage & Validation Strategies",
        "year": 2026
    },
    {
        "id": "2605.15009",
        "file": "07_Li_2026_DeepTokenEEG_MCI_Alzheimers_Classification.pdf",
        "scope": "Tokenized EEG Features for Alzheimer's & MCI Classification",
        "year": 2026
    },
    {
        "id": "2608.29755",
        "file": "08_GraM_Diff_2026_Alzheimers_Diagnosis.pdf",
        "scope": "Graph-Mamba Diffusion for EEG-based Alzheimer's Disease Diagnosis",
        "year": 2026
    },
    {
        "id": "2601.06016",
        "file": "09_Ahmed_2026_LookAroundNet_Seizure_Detection.pdf",
        "scope": "Transformers for Clinically Viable Seizure Detection (CHB-MIT / TUH)",
        "year": 2026
    },
    {
        "id": "2606.02166",
        "file": "10_EEGFuseFormer_2026_Seizure_Onset_Prediction.pdf",
        "scope": "Spatial-Temporal Transformer Feature Fusion for Seizure Prediction",
        "year": 2026
    }
]

print("=" * 85)
print(f"DOWNLOADING {len(PAPERS)} RESEARCH PAPERS ON CLINICAL EEG DETECTION/CLASSIFICATION (2021-2026)")
print("=" * 85)

results = []

for idx, p in enumerate(PAPERS, 1):
    pid = p["id"]
    fname = p["file"]
    dest_path = os.path.join(OUT_DIR, fname)
    pdf_url = f"https://arxiv.org/pdf/{pid}.pdf"
    api_url = f"https://export.arxiv.org/api/query?id_list={pid}"
    
    print(f"\n[{idx}/{len(PAPERS)}] Fetching metadata for arXiv:{pid} ...", end="", flush=True)
    
    title = ""
    authors = []
    summary = ""
    published = ""
    
    try:
        r_meta = requests.get(api_url, headers=HEADERS, timeout=10)
        if r_meta.status_code == 200:
            tree = ET.fromstring(r_meta.content)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            entry = tree.find('atom:entry', ns)
            if entry is not None:
                title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
                summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
                published = entry.find('atom:published', ns).text.strip()[:10]
                authors = [a.find('atom:name', ns).text.strip() for a in entry.findall('atom:author', ns)]
                print(" [DONE]")
        else:
            print(f" [HTTP {r_meta.status_code}]")
    except Exception as e:
        print(f" [Error: {e}]")

    print(f"  Title: {title}")
    print(f"  Authors: {', '.join(authors[:3]) + (' et al.' if len(authors) > 3 else '')} ({published})")
    print(f"  Scope: {p['scope']}")

    # Download PDF
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 100000:
        sz_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"  PDF Status: Already exists on disk ({sz_mb:.2f} MB)")
        pdf_size = os.path.getsize(dest_path)
    else:
        print(f"  Downloading PDF from: {pdf_url} ...", end="", flush=True)
        try:
            r_pdf = requests.get(pdf_url, headers=HEADERS, timeout=30)
            if r_pdf.status_code == 200 and len(r_pdf.content) > 50000:
                with open(dest_path, "wb") as f_out:
                    f_out.write(r_pdf.content)
                sz_mb = len(r_pdf.content) / (1024 * 1024)
                print(f" [SUCCESS - {sz_mb:.2f} MB]")
                pdf_size = len(r_pdf.content)
            else:
                print(f" [FAILED - Status {r_pdf.status_code}, len {len(r_pdf.content)}]")
                pdf_size = 0
        except Exception as e:
            print(f" [FAILED - {e}]")
            pdf_size = 0

    p_info = {
        "index": idx,
        "arxiv_id": pid,
        "title": title,
        "authors": authors,
        "published": published,
        "year": p["year"],
        "scope": p["scope"],
        "local_file": fname,
        "file_path": dest_path,
        "size_bytes": pdf_size,
        "size_mb": round(pdf_size / (1024 * 1024), 2),
        "url": pdf_url,
        "summary": summary
    }
    results.append(p_info)
    time.sleep(1.5) # respectful delay

manifest_path = os.path.join(OUT_DIR, "downloaded_papers_catalog.json")
with open(manifest_path, "w", encoding="utf-8") as f_out:
    json.dump(results, f_out, indent=2)

print("\n" + "=" * 85)
print(f"SUMMARY: {sum(1 for r in results if r['size_bytes'] > 100000)}/{len(PAPERS)} PAPERS SUCCESSFULLY DOWNLOADED")
print(f"Manifest written to: {manifest_path}")
print("=" * 85)
