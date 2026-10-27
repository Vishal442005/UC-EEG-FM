import requests
import re
import os

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
}

dest_dir = r"E:\All EEG\Papers"
os.makedirs(dest_dir, exist_ok=True)
dest_file = os.path.join(dest_dir, "Zhang_2021_NatureBiomedEng_Psychiatric_Subtypes_EEG.pdf")

# 1. Try Maastricht pure repository
print("Checking Maastricht University repository...")
try:
    url = "https://pure.maastrichtuniversity.nl/en/publications/identification-of-psychiatric-disorder-subtypes-from-functional-c"
    r = requests.get(url, headers=HEADERS, timeout=10)
    links = [m for m in re.findall(r'href="([^"]+)"', r.text) if '.pdf' in m.lower() or 'download' in m.lower()]
    print("Found links:", links)
    for l in links:
        if not l.startswith('http'):
            l = 'https://pure.maastrichtuniversity.nl' + l
        r_pdf = requests.get(l, headers=HEADERS, timeout=15)
        if r_pdf.status_code == 200 and r_pdf.content[:4] == b'%PDF':
            with open(dest_file, "wb") as f:
                f.write(r_pdf.content)
            print(f"SUCCESS! Saved from {l}: {len(r_pdf.content)/(1024*1024):.2f} MB")
            break
except Exception as e:
    print("Maastricht error:", e)

if not os.path.exists(dest_file) or os.path.getsize(dest_file) < 50000:
    print("Trying alternative open access endpoints for Nature BME...")
    # Check arXiv or bioRxiv for Zhang et al. or another Nature BME / Nature portfolio paper
    # E.g. "Identification of psychiatric disorder subtypes from functional connectivity patterns in resting-state electroencephalography" on bioRxiv
    try:
        r_biorxiv = requests.get("https://api.biorxiv.org/details/biorxiv/10.1101/2020.06.29.178970", headers=HEADERS, timeout=10)
        data = r_biorxiv.json()
        print("bioRxiv details:", data.get('messages'))
    except Exception as e:
        print("bioRxiv error:", e)
