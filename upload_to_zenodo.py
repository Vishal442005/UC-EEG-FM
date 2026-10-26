"""
Zenodo Automated Deposit & Archival Tool for UC-EEG-FM.
Supports uploading model weights, benchmark reports, and figures to Zenodo REST API.
"""

import os
import sys
import json
import requests

ZENODO_API_URL = "https://zenodo.org/api/deposit/depositions"
ZENODO_SANDBOX_URL = "https://sandbox.zenodo.org/api/deposit/depositions"

def main():
    token = os.environ.get("ZENODO_ACCESS_TOKEN")
    use_sandbox = os.environ.get("ZENODO_SANDBOX", "true").lower() == "true"
    base_url = ZENODO_SANDBOX_URL if use_sandbox else ZENODO_API_URL
    
    print("=" * 70)
    print("ZENODO REPRODUCIBLE RESEARCH DEPOSIT TOOL")
    print(f"Target Endpoint: {'Sandbox (Test)' if use_sandbox else 'Production (Permanent DOI)'}")
    print("=" * 70)
    
    if not token:
        print("\n[INFO] ZENODO_ACCESS_TOKEN not detected in environment variables.")
        print("To deposit automatically via API:")
        print("  1. Create a Zenodo personal access token at: https://zenodo.org/account/settings/applications/tokens/new")
        print("  2. In PowerShell, set: $env:ZENODO_ACCESS_TOKEN=\"your_token_here\"")
        print("  3. Run: python upload_to_zenodo.py")
        print("\nAlternatively, connect your GitHub repository directly to Zenodo via: https://zenodo.org/account/settings/github/")
        print("Zenodo will automatically archive every release and assign a permanent citable DOI!\n")
        return

    headers = {"Authorization": f"Bearer {token}"}
    
    # 1. Load Metadata
    with open("zenodo.json", "r") as f:
        metadata = json.load(f)
        
    payload = {"metadata": metadata}
    
    print("Creating deposition container on Zenodo...")
    r = requests.post(base_url, params={'access_token': token}, json=payload)
    if r.status_code not in [200, 201]:
        print(f"Failed to create deposition: {r.status_code} - {r.text}")
        return
        
    dep_data = r.json()
    dep_id = dep_data['id']
    bucket_url = dep_data['links']['bucket']
    print(f"Deposition created successfully! Deposition ID: {dep_id}")
    print(f"Reserved DOI: {dep_data.get('metadata', {}).get('prereserve_doi', {}).get('doi', 'Pending')}")
    
    # 2. Upload Core Model & Artifact Files
    files_to_upload = [
        os.path.join("Models", "uc_eeg_fm_backbone.pt"),
        os.path.join("Models", "latent_manifold_embeddings.npz"),
        os.path.join("Models", "uc_eeg_fm_backbone.torchscript.pt"),
        os.path.join("Models", "uc_eeg_fm_backbone.onnx"),
        os.path.join("Figures", "Figure_2_Cross_Disorder_Latent_Manifold.png"),
        os.path.join("Figures", "Figure_3_Clinical_ROC_Curves.png"),
        os.path.join("Figures", "Figure_7_Topographical_Scalp_Activation_Maps.png"),
        os.path.join("Figures", "Figure_8_Clinical_Decision_Support_System.png"),
        "MANUSCRIPT_DRAFT.md",
        "SUPPLEMENTARY_INFORMATION.md",
        "master_cohort_metadata.tsv"
    ]
    
    print("\nUploading core assets to Zenodo storage bucket...")
    for file_path in files_to_upload:
        if os.path.exists(file_path):
            fname = os.path.basename(file_path)
            print(f"  Uploading {fname} ({os.path.getsize(file_path)/(1024*1024):.2f} MB)...", end="", flush=True)
            with open(file_path, "rb") as fp:
                ur = requests.put(f"{bucket_url}/{fname}", data=fp, params={'access_token': token})
                if ur.status_code in [200, 201]:
                    print(" [DONE]")
                else:
                    print(f" [FAILED: {ur.status_code}]")
                    
    print(f"\nDeposition successfully prepared! Review and publish at: https://{'sandbox.' if use_sandbox else ''}zenodo.org/deposit/{dep_id}")

if __name__ == "__main__":
    main()
