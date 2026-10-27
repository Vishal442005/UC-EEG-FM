"""
Script to download top 10 relevant peer-reviewed/arXiv papers (2021-2026)
on EEG Foundation Models, Alcoholism Detection, Alzheimer's Classification,
and Pediatric Epilepsy/Seizure Detection.
"""

import os
import sys
import time
import ssl
import json
import urllib.request
import xml.etree.ElementTree as ET
import certifi

OUT_DIR = r"E:\All EEG\Literature_Review\Papers"
os.makedirs(OUT_DIR, exist_ok=True)

ctx = ssl.create_default_context(cafile=certifi.where())

# Targeted papers (2021-2026)
PAPER_IDS = [
    ("2305.18342", "Yang_2023_BIOT_Biosignal_Transformer.pdf"),
    ("2405.18765", "Jiang_2024_LaBraM_Large_Brain_Model.pdf"),
    ("2101.12037", "Kostas_2021_BENDR_Transformers_Biosignal.pdf"),
    ("2311.03764", "Cui_2023_NeuroGPT_Foundation_Model_EEG.pdf"),
    ("2512.08959", "Wang_2025_EEGBench_Clinical_Foundation_Models.pdf"),
    ("2604.10293", "Silva_2026_Validation_Strategy_Alcoholism_EEG.pdf"),
    ("2605.15009", "Li_2026_DeepTokenEEG_MCI_Alzheimers_Classification.pdf"),
    ("2601.06016", "Ahmed_2026_LookAroundNet_Transformers_Seizure_Detection.pdf"),
    ("2302.14367", "Wang_2023_BrainBERT_Self_Supervised_EEG.pdf"),
    ("2606.00815", "OmniEEG_2026_OmniEEGBench_Evaluation_Benchmark.pdf")
]

print("=" * 80)
print(f"DOWNLOADING {len(PAPER_IDS)} RELEVANT EEG RESEARCH PAPERS (2021-2026)")
print("=" * 80)

download_manifest = []

for idx, (arxiv_id, out_filename) in enumerate(PAPER_IDS, 1):
    dest_path = os.path.join(OUT_DIR, out_filename)
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    api_url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    
    print(f"\n[{idx}/{len(PAPER_IDS)}] Processing arXiv ID: {arxiv_id}")
    
    # 1. Fetch metadata
    meta = {"arxiv_id": arxiv_id, "file_name": out_filename, "url": pdf_url}
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
            tree = ET.fromstring(resp.read())
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            entry = tree.find('atom:entry', ns)
            if entry is not None:
                title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
                summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
                published = entry.find('atom:published', ns).text.strip()[:10]
                authors = [a.find('atom:name', ns).text.strip() for a in entry.findall('atom:author', ns)]
                meta["title"] = title
                meta["published"] = published
                meta["authors"] = authors
                meta["summary"] = summary
                print(f"  Title: {title}")
                print(f"  Published: {published} | Authors: {', '.join(authors[:3])} et al.")
    except Exception as e:
        print(f"  [WARN] Metadata fetch error: {e}")
        meta["title"] = out_filename
        meta["published"] = "Unknown"
        meta["authors"] = []
        meta["summary"] = ""

    time.sleep(1.0) # rate limit

    # 2. Download PDF
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 100000:
        print(f"  [SKIPPED] Already downloaded ({os.path.getsize(dest_path)/(1024*1024):.2f} MB)")
        meta["size_bytes"] = os.path.getsize(dest_path)
    else:
        print(f"  Downloading PDF from: {pdf_url} ...", flush=True)
        try:
            pdf_req = urllib.request.Request(
                pdf_url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            )
            with urllib.request.urlopen(pdf_req, context=ctx, timeout=30) as p_resp:
                content = p_resp.read()
                with open(dest_path, "wb") as f_out:
                    f_out.write(content)
            size_mb = os.path.getsize(dest_path) / (1024 * 1024)
            print(f"  [SUCCESS] Saved: {out_filename} ({size_mb:.2f} MB)")
            meta["size_bytes"] = len(content)
        except Exception as e:
            print(f"  [ERROR] Failed to download {pdf_url}: {e}")
            meta["size_bytes"] = 0

    download_manifest.append(meta)
    time.sleep(3.0) # Respect arXiv rate limits

manifest_path = os.path.join(OUT_DIR, "papers_manifest.json")
with open(manifest_path, "w", encoding="utf-8") as f_mf:
    json.dump(download_manifest, f_mf, indent=2)

print("\n" + "=" * 80)
print(f"DOWNLOAD COMPLETE. Manifest written to: {manifest_path}")
print("=" * 80)
