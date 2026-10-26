# Neurobiology and EEG Biomarkers in Neurodevelopmental Disorders

## 1. Neurodevelopmental Circuitry & Pathophysiology

Neurodevelopmental disorders emerge during early brain maturation, reflecting alterations in synaptogenesis, axonal pruning, myelination, and excitation-inhibition (E/I) balance:
* **Attention Deficit Hyperactivity Disorder (ADHD):** Characterized by persistent inattention, hyperactivity, and impulsivity, driven by delayed maturation of prefrontal-striatal-cerebellar networks and dopaminergic/noradrenergic signaling hypofunction.
* **Autism Spectrum Disorder (ASD):** Characterized by persistent deficits in social communication, restricted interests, and repetitive behaviors, linked to atypical synaptic plasticity, macrocephaly/accelerated early brain overgrowth, and altered long-range vs. local cortical connectivity.
* **Developmental Dyslexia:** Specific learning disability of neurobiological origin affecting phonological processing, linked to impaired neural entrainment to low-frequency speech rhythms.

---

## 2. Core EEG Biomarkers in ADHD

```
         PREFRONTAL RESTING EEG POWER SPECTRUM
Power ^
      │    ┌──┐
      │   │    │  <-- Elevated Theta (4 - 8 Hz) in ADHD
      │   │    │
      │  │      └──┐
      │  │         │       ┌──┐
      │  │         │      │    │  <-- Attenuated Beta (13 - 30 Hz) in ADHD
      └─┴──────────┴─────┴──────┴────────> Frequency (Hz)
         0         4     8      13     30
         ==> High Theta-to-Beta Ratio (TBR = Theta / Beta)
```

### 2.1 The Theta-to-Beta Ratio (TBR)
* **Definition:** Calculated as the ratio of absolute or relative spectral power:
  $$\text{TBR} = \frac{\text{Power}_{\theta\ (4-8\text{ Hz})}}{\text{Power}_{\beta\ (13-30\text{ Hz})}}$$
  measured at the vertex/frontal midline (`Cz, Fz`).
* **Pathophysiological Foundation:**
  * Elevated theta reflects cortical hypoarousal and sluggish prefrontal metabolic activity.
  * Attenuated beta reflects diminished top-down executive and attentional control.
* **Clinical Significance:** In 2013, the US FDA cleared the **NEBA system** (Neuropsychiatric EEG-Based Assessment Aid) as a clinical diagnostic adjunct based on TBR. While high TBR is sensitive to inattentive ADHD subtypes, recent multi-center trials show heterogeneity, emphasizing the need for multi-feature machine learning combinations rather than TBR in isolation.

### 2.2 Event-Related Potentials: Inhibitory & Attentional Failures
* **N200 Deficit:** The frontocentral N200 (occurring 200–300 ms post-stimulus) indexes conflict monitoring in the anterior cingulate cortex (ACC). Children with ADHD exhibit markedly reduced N200 amplitudes during No-Go trials in Go/No-Go and Stop-Signal tasks.
* **P300 (P3b) Blunting:** Substantially reduced P300 amplitude during target detection tasks, reflecting deficient allocation of working memory and selective attentional resources.
* **Contingent Negative Variation (CNV):** Attenuated CNV slow-wave during cue-target anticipation, reflecting deficits in temporal preparation and motor readiness.

---

## 3. Core EEG Biomarkers in Autism Spectrum Disorder (ASD)

### 3.1 Atypical Connectivity Architecture: Local Over- vs. Long-Range Under-Connectivity
* **The Dual-Connectivity Model:**
  * **Local Hyper-Connectivity:** Abnormally increased phase synchronization and coherence over short cortical distances (e.g., adjacent frontal or temporal channels), reflecting excessive local synaptic arborization and impaired developmental pruning.
  * **Long-Range Hypo-Connectivity:** Significant attenuation in long-range functional connectivity (measured via $wPLI$ and phase-locking value) between frontal executive hubs and posterior sensory/social cortices (e.g., `Fz - Pz`, `F3 - O1`).

### 3.2 Aberrant Gamma-Band Dynamics ($30 - 80\text{ Hz}$)
* **The E/I Balance Hypothesis:** Rubenstein and Merzenich formulated that autism is driven by an elevated ratio of cortical excitation to inhibition.
* **Electrophysiological Readout:**
  * Elevated resting baseline gamma noise combined with blunted **evoked/induced gamma synchrony** during sensory processing tasks.
  * Impaired 40-Hz Auditory Steady-State Response (ASSR) phase-locking, directly linking to parvalbumin-positive interneuron channelopathy.

### 3.3 The N170 Face-Processing Deficit
* **Phenomenology:** The N170 is a negative ERP component peaking at 170 ms over occipito-temporal leads (`P7, P8`), specialized for human facial processing.
* **Autistic Hallmarks:**
  * **Delayed Latency:** N170 peak latency is significantly delayed in individuals with ASD when viewing human faces.
  * **Absence of the Face Inversion Effect:** In neurotypical individuals, upside-down faces dramatically delay and increase the N170. Individuals with ASD show minimal differentiation between upright and inverted faces, demonstrating non-expert, feature-based rather than holistic facial processing.

### 3.4 Reduced Nonlinear Entropy
* Multiscale Entropy (MSE) analysis shows significantly reduced entropy across fine-grained timescales in children with ASD, indexing restricted neurodynamic flexibility.

---

## 4. Summary Matrix of Neurodevelopmental Biomarkers

| Disorder | Electrophysiological Biomarker | Topography / Electrodes | Functional / Clinical Significance |
| :--- | :--- | :--- | :--- |
| **ADHD** | High Theta-to-Beta Ratio (TBR) | Vertex / Frontal (`Cz, Fz, FCz`) | Prefrontal hypoarousal; executive inattention |
| **ADHD** | Blunted No-Go N200 & P300 | Frontocentral (`FCz, Cz`) | Impaired motor response inhibition and conflict control |
| **ADHD** | High Intra-individual Variability (IIV) | Global / Trial-to-trial ERP | Temporal instability in sustained attentional networks |
| **ASD** | Long-Range Hypo-connectivity | Fronto-Parietal pairs (`Fz - Pz`) | Impaired coordination across distributed social networks |
| **ASD** | Local Hyper-connectivity | Frontal / Temporal clusters | Excessive short-range synaptic density; sensory overload |
| **ASD** | Delayed & Blunted N170 | Occipito-temporal (`P7, P8, PO7, PO8`) | Impaired facial decoding and social-emotional cognition |
| **ASD** | Reduced Evoked Gamma Oscillations | Auditory / Visual Cortices | Parvalbumin interneuron hypofunction; E/I imbalance |
| **Dyslexia**| Impaired Delta-Theta Phase Tracking | Primary Auditory (`T7, T8`) | Inability to parse syllabic speech envelope boundaries |
