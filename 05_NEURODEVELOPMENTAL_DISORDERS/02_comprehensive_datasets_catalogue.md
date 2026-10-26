# Comprehensive EEG Datasets Catalogue: Neurodevelopmental Disorders

This catalogue details open-access electroencephalography (EEG) datasets dedicated to Attention Deficit Hyperactivity Disorder (ADHD), Autism Spectrum Disorder (ASD), and pediatric neurodevelopment.

---

## 1. IEEE Dataport: ADHD EEG Dataset (Nasrabadi Cohort)

* **Repository:** [IEEE Dataport - EEG Data for ADHD / Control Children](https://ieee-dataport.org/open-access/eeg-data-adhd-control-children)
* **Primary Investigator:** Prof. Ali M. Nasrabadi, Department of Biomedical Engineering, Shahed University, Tehran, Iran.
* **Subjects & Demographics:**
  * Total $N = 121$ children (ages 7–12 years):
    * **ADHD Cohort:** $N = 61$ children (48 boys, 13 girls) diagnosed with ADHD by certified child psychiatrists based on DSM-IV criteria. None had taken psychostimulant medication for $\ge 48$ hours prior to recording.
    * **Control Cohort:** $N = 60$ age- and sex-matched typically developing children (50 boys, 10 girls) with no history of psychiatric or neurological disorders.
* **Recording Specifications:**
  * **Montage:** 19 scalp channels standard International 10-20 layout:
    `Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2` (referenced to linked earlobes `A1+A2`).
  * **Sampling Frequency:** $f_s = 128\text{ Hz}$.
* **Experimental Paradigm:**
  * Continuous visual attention task designed to engage sustained executive focus:
  * Children were shown pictures of cartoon characters and instructed to count the total number of characters appearing on screen. The number of characters per image varied pseudo-randomly between 5 and 16, requiring continuous focused visual search.
* **Data Organization:**
  * Distributed as raw MATLAB structures (`.mat`), containing labeled matrices for each participant.
* **Access Status:** Open Access (IEEE DataPort Open).

---

## 2. Child Mind Institute: Healthy Brain Network (HBN)

* **Repository:** [Healthy Brain Network Data Portal](https://healthybrainnetwork.org/) / [FCP-INDI HBN S3 Archive](http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/)
* **Institution:** Child Mind Institute, New York City, USA.
* **Cohort Scale & Significance:**
  * One of the largest pediatric neuroimaging and electrophysiological biobanks in the world.
  * Over **$2,500+$ children and adolescents** (ages 5.0–21.0 years) from the greater New York area.
  * Deep clinical and psychiatric phenotyping: DSM-5 diagnostic consensus for ADHD (Inattentive, Hyperactive, Combined), ASD (ADOS-2, ADI-R), Learning Disorders, Anxiety, and Depression.
* **Recording Specifications:**
  * **System:** High-density 128-channel Geodesic Sensor Net (EGI Net Station).
  * **Sampling Frequency:** $f_s = 500\text{ Hz}$.
  * **Standardized Electrophysiological Battery:**
    1. **Resting-State:** 5 minutes alternating Eyes Open / Eyes Closed.
    2. **Passive Naturalistic Viewing:** Watching engaging animated movie clips (e.g., *Despicable Me*, *The Present*) to evaluate naturalistic neural tracking.
    3. **Surround Suppression & Contrast Grating Task:** Visual cortical processing and E/I balance.
    4. **Auditory Oddball / MMN Paradigm:** Pre-attentive sensory discrimination and P300.
    5. **Sequence Learning & Motor Inhibition Tasks.**
* **Format:** BIDS-compliant raw and preprocessed data (`.raw`, `.mff`, `.set`).
* **Access Status:** Free open registration (Data Use Agreement via Child Mind Institute portal).

---

## 3. King Abdulaziz University: Autism Spectrum Disorder EEG Dataset

* **Repository:** [KAU ASD EEG Dataset](https://data.mendeley.com/) / Mirror on Zenodo
* **Institution:** King Abdulaziz University, Jeddah, Saudi Arabia.
* **Subjects:**
  * Children with clinically confirmed Autism Spectrum Disorder ($N = 30$, CARS score $> 30$) and typically developing age-matched control children ($N = 30$).
* **Recording Specifications:**
  * **Montage:** 16-channel EEG system positioned according to the 10-20 layout.
  * **Sampling Rate:** $f_s = 256\text{ Hz}$.
  * **Conditions:** Relaxed state with dimmed lighting and passive visual exposure to calm geometric patterns vs. human social facial imagery.
* **Key Use Case:** Evaluating local gamma phase-locking and long-range coherence deficits in ASD.
* **Access Status:** Open Access (CC-BY).

---

## 4. OpenNeuro Accessions: `ds003826` and `ds003775`

* **Repository:** OpenNeuro
* **`ds003826`:** Pediatric resting-state and cognitive control EEG evaluating executive function and working memory across developmental cohorts.
* **`ds003775`:** Sensory processing and mismatch negativity in neurodevelopmental conditions.
* **Format:** BIDS standardized, BrainVision and EEGLAB sets.

---

## Summary Matrix of Neurodevelopmental Datasets

| Dataset | Target Condition | Subjects ($N$) | Channels | Sampling Rate | Format | Paradigm | Access Link |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **IEEE Dataport ADHD** | ADHD (Inattentive / Comb.) | 121 | 19 | 128 Hz | MATLAB (.mat) | Cartoon Counting Visual Attention | [IEEE Dataport ADHD](https://ieee-dataport.org/open-access/eeg-data-adhd-control-children) |
| **Child Mind Inst. HBN** | ADHD, ASD, Anxiety | 2,500+ | 128 | 500 Hz | BIDS / EGI | Resting + Movies + Oddball | [CMI HBN Portal](https://healthybrainnetwork.org/) |
| **KAU Autism Dataset** | Autism Spectrum (ASD) | 60 | 16 | 256 Hz | CSV / MAT | Relaxed + Social Visual Stimuli | [Mendeley KAU ASD](https://data.mendeley.com/) |
| **OpenNeuro ds003826** | Pediatric Cognition | 80+ | 64 | 500 Hz | BIDS | Executive Task Battery | [OpenNeuro ds003826](https://openneuro.org/datasets/ds003826) |
