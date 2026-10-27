import urllib.request
import urllib.parse
import json
import ssl
import certifi

ctx = ssl.create_default_context(cafile=certifi.where())

# Search Europe PMC for Open Access Nature portfolio papers on EEG classification/detection/transformers
query = 'JOURNAL:"Nature Communications" AND EEG AND (transformer OR classification OR detection OR "foundation model" OR seizure OR Alzheimer) AND OPEN_ACCESS:y'
url = f'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={urllib.parse.quote(query)}&format=json&pageSize=10&sort=P_PDATE_D%20desc'

req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
        data = json.loads(resp.read().decode())
        results = data.get('resultList', {}).get('result', [])
        print(f"Found {len(results)} Nature Communications / Nature portfolio papers:\n")
        for idx, r in enumerate(results, 1):
            print(f"[{idx}] {r.get('title')}")
            print(f"    Journal: {r.get('journalTitle')} | Year: {r.get('pubYear')}")
            print(f"    DOI: {r.get('doi')} | PMCID: {r.get('pmcid')}")
            print(f"    Fulltext URL: {r.get('fullTextUrlList', {}).get('fullTextUrl', [{}])[0].get('url', 'N/A')}\n")
except Exception as e:
    print('Error:', e)
