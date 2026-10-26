# Neurobiology and EEG Biomarkers in Neurodegenerative Disorders and Dementia

## 1. Pathophysiological Landscape

Neurodegenerative diseases represent progressive, irreversible structural and functional decay of neuronal populations, synaptic contacts, and neural networks:
* **Alzheimer’s Disease (AD):** Extracellular amyloid-beta ($\text{A}\beta_{42}$) plaques, intracellular hyperphosphorylated tau neurofibrillary tangles (NFTs), extensive synaptic degradation, and degeneration of cholinergic projections from the basal forebrain (Nucleus Basalis of Meynert).
* **Mild Cognitive Impairment (MCI):** Amnestic (aMCI) and non-amnestic subtypes representing the prodromal symptomatic phase where cognitive decline is measurable but daily living activities remain largely preserved. $10\% - 15\%$ of aMCI patients convert to AD annually.
* **Frontotemporal Dementia (FTD):** Degeneration of frontal and temporal cortices associated with tau (*MAPT*) or TDP-43 inclusions, driving severe behavioral, executive, and language deficits while spatial orientation is initially preserved.
* **Parkinson’s Disease (PD):** Loss of dopaminergic neurons in the substantia nigra pars compacta ($SNpc$), intraneuronal $\alpha$-synuclein accumulations (Lewy bodies), and aberrant beta-band synchronization within the basal ganglia-thalamocortical motor loop.
* **Lewy Body Dementia (LBD):** Widespread cortical Lewy bodies causing fluctuating attentional alertness, recurrent visual hallucinations, and spontaneous parkinsonism.

---

## 2. Electrophysiological Biomarkers

EEG serves as an ultra-sensitive, non-invasive readout of cortical synaptic integrity, showing measurable abnormalities years before gross volumetric atrophy is detectable on structural MRI.

```
       HEALTHY AGING                          ALZHEIMER'S DISEASE
 ┌───────────────────────────┐           ┌───────────────────────────┐
 │ Fast rhythms dominant     │           │ Severe EEG Slowing        │
 │ • Alpha peak ~ 10.0 Hz    │  ──────>  │ • Delta & Theta increase  │
 │ • High spectral entropy   │           │ • Alpha peak drops < 8 Hz │
 │ • Intact long-range wPLI  │           │ • Disconnected networks   │
 └───────────────────────────┘           └───────────────────────────┘
```

### 2.1 The Cardinal Biomarker: Spectral Slowing
* **Shift in Power Spectral Density (PSD):**
  * **Increase in Low Frequencies:** Marked elevation of diffuse **Delta ($0.5 - 4\text{ Hz}$)** and **Theta ($4 - 8\text{ Hz}$)** power across temporal, parietal, and frontal leads.
  * **Decrease in High Frequencies:** Significant drop in **Alpha ($8 - 12\text{ Hz}$)** and **Beta ($13 - 30\text{ Hz}$)** power.
* **Spectral Ratio Biomarkers:**
  * **Theta-to-Alpha Ratio (TAR):** $\frac{\text{Power}_\theta}{\text{Power}_\alpha}$
  * **Global Slowing Ratio:** $\frac{\text{Power}_\delta + \text{Power}_\theta}{\text{Power}_\alpha + \text{Power}_\beta}$
  * Highly correlated with Mini-Mental State Examination (MMSE) scores and CSF biomarkers ($\text{A}\beta_{42}/\text{A}\beta_{40}$ ratio, p-tau181).

### 2.2 Peak Alpha Frequency (PAF) Deceleration
* **Clinical Metric:** The individual frequency within the $7 - 13\text{ Hz}$ band demonstrating maximum power spectral density over posterior occipito-parietal electrodes (`O1, O2, Oz, Pz`).
* **Trajectory across Progression:**
  * Healthy Young Adults: $10.0 - 11.5\text{ Hz}$
  * Healthy Older Adults ($>65\text{ yrs}$): $9.5 - 10.2\text{ Hz}$
  * Mild Cognitive Impairment (MCI): $8.5 - 9.0\text{ Hz}$
  * Mild AD: $7.8 - 8.4\text{ Hz}$
  * Moderate-to-Severe AD: $< 7.5\text{ Hz}$ (alpha peak completely merges into the theta band).

### 2.3 Loss of Cortical Complexity and Entropy
* **Nonlinear Dynamics:** The healthy human brain operates at a state of self-organized criticality, exhibiting high information content and signal variability. Synaptic death and reduced corticocortical connectivity reduce the dynamical degrees of freedom.
* **Key Metrics:**
  * **Multiscale Entropy (MSE):** Quantifies sample entropy across multiple coarse-grained time scales. In AD and FTD, MSE drops significantly at mid-to-long timescales.
  * **Permutation Entropy (PE) & Spectral Entropy (SpecEn):** Consistently reduced in AD patients compared to age-matched controls, reflecting loss of information processing capacity.

### 2.4 Breakdown of Functional Connectivity
* **The "Disconnection Syndrome":** Alzheimer's disrupts long-range axonal pathways connecting the Default Mode Network (DMN), frontoparietal control network, and hippocampus.
* **Electrophysiological Manifestation:**
  * Marked decrease in **Weighted Phase Lag Index (wPLI)**, **Phase Lag Index (PLI)**, and **Phase-Locking Value (PLV)** in the alpha and beta frequency bands, particularly between anterior (frontal) and posterior (parieto-occipital) electrode pairs.
  * In contrast, local, pathologically synchronized theta connectivity is often abnormally elevated.

### 2.5 Parkinson's Disease: Pathological Beta Oscillations
* **The Basal Ganglia Motor Loop:** In PD, dopamine deficiency leads to hyper-synchronization of medium spiny neurons and subthalamic nucleus (STN) projections in the **beta band ($13 - 30\text{ Hz}$)**.
* **Beta Bursts:** Pathological long-duration beta bursts over sensorimotor cortex (`C3, C4, Cz`) correlate directly with bradykinesia and muscle rigidity. Administration of L-DOPA or activation of STN-DBS selectively terminates these long beta bursts, restoring fluid motor initiation.

---

## 3. Summary of Neurodegenerative Biomarkers

| Clinical Entity | Primary EEG Biomarkers | Topographical Distribution | Clinical / Functional Significance |
| :--- | :--- | :--- | :--- |
| **Alzheimer's Disease (AD)** | Generalized Slowing ($\uparrow \delta, \uparrow \theta, \downarrow \alpha$), PAF Deceleration ($<8\text{ Hz}$), $\downarrow$ wPLI connectivity | Bilateral temporo-parietal and frontal | Cortical de-afferentation, cholinergic deficit, cognitive decline (MMSE) |
| **Mild Cognitive Impairment (MCI)** | Moderate $\uparrow \theta$, drop in PAF ($8.5 - 9.0\text{ Hz}$), reduced Multiscale Entropy | Temporal and parietal leads | Early prodromal stage; predicts annual conversion rate to AD |
| **Frontotemporal Dementia (FTD)** | Selective frontal delta/theta elevation, preserved posterior alpha rhythm | Distinctly localized to frontal leads (`Fp1, Fp2, F7, F3, Fz, F4, F8`) | Frontal lobe atrophy; correlates with disinhibition and executive dysfunction |
| **Lewy Body Dementia (LBD)** | Frontal-occipital slowing, profound variability in dominant frequency over seconds | Diffuse, posterior and temporal | Drives fluctuating alertness and visual hallucinations |
| **Parkinson's Disease (PD)** | Pathological beta bursting ($13 - 30\text{ Hz}$), $\uparrow$ sensorimotor PAC | Sensorimotor cortex (`C3, C4`) and basal ganglia | Correlates with rigidity/bradykinesia; suppressed by L-DOPA/DBS |
