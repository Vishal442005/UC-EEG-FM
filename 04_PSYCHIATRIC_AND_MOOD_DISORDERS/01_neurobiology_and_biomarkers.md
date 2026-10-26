# Neurobiology and EEG Biomarkers in Psychiatric and Mood Disorders

## 1. Clinical Context & Neural Circuitry

Psychiatric and affective disorders represent network-level disruptions across fronto-limbic, salience, and default mode networks (DMN):
* **Major Depressive Disorder (MDD) & Treatment-Resistant Depression (TRD):** Characterized by persistent depressed mood, anhedonia, psychomotor alterations, and cognitive deficits, rooted in fronto-striatal and cortico-limbic dysregulation (hypoactive dorsolateral prefrontal cortex - dlPFC, hyperactive subgenual anterior cingulate cortex - sgACC).
* **Bipolar Disorder (BD):** Cyclic fluctuations between depressive episodes and manic/hypomanic episodes marked by heightened goal-directed activity, flight of ideas, and reduced sleep necessity.
* **Schizophrenia (SZ) & Psychosis Spectrum:** Characterized by positive symptoms (hallucinations, delusions), negative symptoms (avolition, social withdrawal), and cognitive disorganization, driven by NMDA receptor hypofunction on parvalbumin-positive ($PV^+$) GABAergic interneurons and aberrant dopaminergic striatal signaling.
* **Obsessive-Compulsive Disorder (OCD) & Anxiety Disorders:** Hyperactivity within the cortico-striato-thalamo-cortical (CSTC) loop driving excessive threat appraisal and repetitive compulsions.

---

## 2. Core EEG Biomarkers in Major Depressive Disorder (MDD)

### 2.1 Frontal Alpha Asymmetry (FAA)
* **Theoretical Foundation:** Richard Davidson's Affective Style & Approach/Withdrawal Model:
  * Left prefrontal cortex (PFC) mediates approach-related behaviors, appetitive motivation, and positive affect.
  * Right prefrontal cortex mediates avoidance/withdrawal behaviors, threat detection, and negative affect.
* **Mathematical Definition:**
  $$\text{FAA} = \ln\left(\text{Power}_{F4, \alpha}\right) - \ln\left(\text{Power}_{F3, \alpha}\right)$$
  *(Alternatively computed across `F8 - F7` for lateralized ventral prefrontal activity).*
* **Clinical Pattern:**
  * Because alpha power is inversely related to cortical activation, **negative FAA** values indicate relatively higher right frontal activity and lower left frontal activity.
  * Depressed patients consistently exhibit negative FAA, reflecting anhedonia and motivational deficits.

### 2.2 Theta Cordance & Antidepressant Treatment Response
* **Definition:** Developed by Leuchter et al. (UCLA), cordance is a normalized metric combining absolute and relative EEG power across electrodes.
* **Predictive Biomarker:**
  * A decrease in prefrontal theta cordance within **48 hours to 1 week** after starting an SSRI/SNRI antidepressant accurately predicts clinical remission at week 8, long before behavioral improvements manifest.
* **Rostral Anterior Cingulate Cortex (rACC) Theta Activity:** Source-localized elevated resting theta power ($4 - 8\text{ Hz}$) within the rACC (Brodmann areas 24/32) identifies depressed patients who will respond positively to both pharmacotherapy (escitalopram) and repetitive Transcranial Magnetic Stimulation (rTMS).

---

## 3. Core EEG Biomarkers in Schizophrenia & Psychosis

```
       P50 Sensory Gating Paradigm (Paired-Click S1 - S2):
Healthy Control:   S1:  /\ (Normal)      S2:  _ (Suppressed, S2/S1 < 0.4)
Schizophrenia:     S1:  /\ (Normal)      S2:  /\ (Failed Gating, S2/S1 > 0.8)
```

### 3.1 P50 Auditory Sensory Gating Deficit
* **Paradigm:** Two identical auditory clicks ($S_1$ and $S_2$) presented with a 500-ms inter-stimulus interval.
* **Healthy Gating:** The brain habituates to the irrelevant repetitive stimulus, attenuating the $S_2$ P50 wave ($\frac{S_2}{S_1} \text{ ratio} < 0.40$).
* **Schizophrenic Breakdown:** Schizophrenia patients exhibit impaired sensory gating ($\frac{S_2}{S_1} > 0.80$), flooding the conscious cortex with sensory noise. This deficit is tied directly to $\alpha 7$ nicotinic acetylcholine receptor ($CHRNA7$) polymorphisms and hippocampal dysfunction.

### 3.2 Mismatch Negativity (MMN) Attenuation
* **Phenomenology:** An involuntary negative ERP component elicited 150–250 ms after an infrequent deviant auditory tone embedded within a repetitive stream of standard tones.
* **Pathophysiological Marker:** MMN amplitude is severely attenuated in chronic schizophrenia and first-episode psychosis. Because MMN generation depends on functional NMDA receptors on pyramidal neurons in primary auditory cortex and dlPFC, it serves as the prime electrophysiological surrogate of **NMDA receptor hypofunction**.

### 3.3 40-Hz Auditory Steady-State Response (ASSR) Deficit
* **Paradigm:** Continuous auditory click trains or amplitude-modulated tones presented at 40 Hz.
* **Deficit:** Patients with schizophrenia exhibit significantly decreased **40-Hz phase-locking factor (PLF)** and evoked power. This reflects failure of fast-spiking $PV^+$ GABAergic interneurons to generate synchronous gamma oscillations.

### 3.4 Resting-State EEG Microstate Dynamics
* **Canonical Microstates (A, B, C, D):** Quasi-stable spatial potential topographies persisting for 60–120 ms before abruptly transitioning to another configuration.
  * **Class A:** Associated with phonological / auditory processing network.
  * **Class B:** Associated with visual network.
  * **Class C:** Associated with the Salience Network (anterior insula, ACC).
  * **Class D:** Associated with the Central Executive / Dorsal Attention Network (right-lateralized fronto-parietal).
* **Schizophrenic Signature:**
  * **Increased Class C Presence:** Reflects unconstrained hyperactivity of the salience network, attributing aberrant salience to internally generated thoughts (hallucinations/delusions).
  * **Decreased Class D Presence:** Reflects executive cognitive breakdown.

---

## 4. Anxiety, OCD, and PTSD Biomarkers

### 4.1 Enhanced Error-Related Negativity (ERN / Ne) in OCD and Anxiety
* Patients with Obsessive-Compulsive Disorder (OCD) and Generalized Anxiety Disorder (GAD) exhibit abnormally **hyperactive, exaggerated ERN amplitudes** during response conflict tasks (Flanker, Stroop).
* Indicates hyperactive error monitoring and hyper-vigilance within the anterior cingulate cortex (ACC).

### 4.2 Late Positive Potential (LPP) and Trauma Conditioning in PTSD
* In Post-Traumatic Stress Disorder (PTSD), threat-related imagery elicits sustained, heightened LPPs over centro-parietal channels that fail to habituate over time, demonstrating deficient inhibitory fear extinction.

---

## 5. Summary Matrix of Psychiatric EEG Biomarkers

| Disorder | Key Electrophysiological Biomarker | Topography / Leads | Underlying Neural Mechanism |
| :--- | :--- | :--- | :--- |
| **Major Depression (MDD)** | Frontal Alpha Asymmetry (FAA $\downarrow$, Right $>$ Left) | Frontal (`F3, F4, F7, F8`) | Approach motivation deficit, anhedonia |
| **MDD Treatment Response** | Early Theta Cordance drop; rACC Theta Elevation | Frontocentral (`FCz, Fz`) / rACC | Frontal circuit plastic reorganization |
| **Schizophrenia (SZ)** | P50 Gating Deficit ($\frac{S_2}{S_1} \uparrow$) | Vertex (`Cz, FCz`) | Hippocampal / $\alpha 7$ nicotinic dysregulation |
| **Schizophrenia (SZ)** | Mismatch Negativity (MMN) Blunting | Frontocentral (`Fz, FCz`) | Cortical NMDA receptor hypofunction |
| **Schizophrenia (SZ)** | 40-Hz ASSR Gamma Phase-Locking drop | Frontal / Superior Temporal | $PV^+$ fast-spiking interneuron deficit |
| **Schizophrenia (SZ)** | Microstate Class C $\uparrow$, Class D $\downarrow$ | Whole Scalp (Global Topography) | Salience network hyperactivity |
| **OCD / Anxiety** | Exaggerated Error-Related Negativity (ERN) | Frontocentral (`FCz, Cz`) | ACC hyper-reactivity to perceived errors |
| **Bipolar vs. MDD** | Elevated resting Frontal Gamma & Fast Beta | Frontal / Temporal leads | Cortical hyperexcitability during mood switches |
