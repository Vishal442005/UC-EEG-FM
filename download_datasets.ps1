# PowerShell script to download ~20 GB of Nature‑compliant EEG datasets onto the E: drive
# Destination folder
$dest = "E:\All EEG\Raw_Datasets"
New-Item -ItemType Directory -Force -Path $dest | Out-Null

# 1. CHB‑MIT Scalp EEG (PhysioNet) – ~2 GB
Write-Host "Downloading CHB‑MIT Scalp EEG…"
Invoke-WebRequest -Uri "https://physionet.org/files/chbmit/1.0.0/chbmit.tar.gz" -OutFile "$dest\chbmit.tar.gz"

# 2. TUH Seizure Corpus – small public subset (~5 GB)
Write-Host "Downloading TUH small subset (Abnormal)…"
Invoke-WebRequest -Uri "https://isip.piconepress.com/projects/tuh_eeg/download/tuh_eeg_abnormal.tar.gz" -OutFile "$dest\tuh_abnormal.tar.gz"

# 3. Bonn University EEG Benchmark – ~0.1 GB
Write-Host "Downloading Bonn Benchmark…"
Invoke-WebRequest -Uri "https://zenodo.org/record/2585655/files/EEG%20Bonn%20Dataset.zip" -OutFile "$dest\bonn_dataset.zip"

# 4. Bern‑Barcelona Intracranial EEG – ~0.2 GB
Write-Host "Downloading Bern‑Barcelona Intracranial EEG…"
Invoke-WebRequest -Uri "https://zenodo.org/record/2585656/files/BernBarcelonaEEG.zip" -OutFile "$dest\bern_barcelona.zip"

# 5. Siena Scalp EEG (PhysioNet) – ~1 GB
Write-Host "Downloading Siena Scalp EEG…"
Invoke-WebRequest -Uri "https://physionet.org/files/siena-scalp-eeg/1.0.0/siena-scalp-eeg.tar.gz" -OutFile "$dest\siena_scalp.tar.gz"

# 6. OpenNeuro ds005907 – Alcohol 3‑armed bandit (~5 GB)
Write-Host "Downloading OpenNeuro ds005907 (Alcohol Bandit)…"
Invoke-WebRequest -Uri "https://openneuro.org/crn/download?dataset=ds005907" -OutFile "$dest\ds005907.zip"

# 7. OpenNeuro ds004515 – Alcohol Imagery Reinforcement Learning (~4 GB)
Write-Host "Downloading OpenNeuro ds004515 (Alcohol Imagery)…"
Invoke-WebRequest -Uri "https://openneuro.org/crn/download?dataset=ds004515" -OutFile "$dest\ds004515.zip"

# 8. OpenNeuro ds008851 – AUD Cue‑Reactivity (~3 GB)
Write-Host "Downloading OpenNeuro ds008851 (AUD Cue‑Reactivity)…"
Invoke-WebRequest -Uri "https://openneuro.org/crn/download?dataset=ds008851" -OutFile "$dest\ds008851.zip"

Write-Host "All download commands queued. Execute this script in PowerShell to fetch ~20 GB of open‑access, BIDS‑compatible datasets suitable for a Nature‑level study."