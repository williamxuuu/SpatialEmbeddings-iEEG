# Spatial/location encodings in multi-subject iEEG and EEG/MEG foundation models: prior comparisons and controls

Scope and method: about 22 tool calls (web searches, arXiv HTML fetches, and local pdftotext of the MAPA and Seegnificant PDFs), covering 2021 to November 2026. Each item is tagged:
- **[VERIFIED]**: I read the detail in the paper's full text (HTML or PDF).
- **[ABSTRACT/SNIPPET]**: the detail comes only from an abstract, landing page or search snippet.
- **[MEMORY]**: the detail comes from my prior knowledge and was not re-checked this session. The report writer should treat these as unconfirmed.

## Q0. BaRISTA and MAPA, in detail

### Takeaway
- **BaRISTA** compared three spatial scales, each used both as the encoding and as the masking unit: channel (discretized LPI coordinates), Destrieux parcel, and lobe. The data were Brain Treebank, 10 subjects. Parcel encoding was best on sentence-onset and speech tasks.
  - It had no shuffled or matched-count controls.
  - The data-amount sweep used only the single best configuration. So BaRISTA does not test whether the best scale changes with data density.
- **MAPA** ablated a single DKT region embedding (74 labels plus 1 reserved), crossed with a within-shaft RoPE. It used 7 pretraining subjects and 27.9 h of data.
  - It had no atlas-resolution comparison, no coordinate, ID or shuffled controls, and no data-amount sweep.

### Cited Findings
**BaRISTA (Oganesian, Hashemi, Shanechi; NeurIPS 2025; arXiv 2512.12135)**
- [VERIFIED] **Encoding mechanism.** Each channel's token gets a learnable spatial embedding E_j for its spatial category, added as S = B + E. The category is the channel, parcel or lobe, depending on the scale (Sec. 3.2). — [arXiv HTML](https://arxiv.org/html/2512.12135)
- [VERIFIED] **Channel scale is a coordinate code, not a per-electrode ID.** It uses LPI coordinates "(x,y,z) where each element is an integer between 0 to 200". Each axis has its own learned embedding table, and a channel's encoding is the sum of its three axis embeddings (appendix). — [arXiv HTML](https://arxiv.org/html/2512.12135)
- [VERIFIED] **Scales and atlases.**
  - Parcels are Destrieux atlas parcels, including annotated subcortical structures such as the hippocampus (Sec. 3.1).
  - Lobes come from the Desikan-Killiany atlas lobe mapping (via Klein & Tourville), plus non-lobe regions such as the cingulate.
  - Destrieux was chosen because its finer parcels "permit finer-grained analyses". — [arXiv HTML](https://arxiv.org/html/2512.12135)
- [VERIFIED] **Data density per subject (Table 9, after cleaning).**
  - Channels range from 72 to 205.
  - Parcels range from 19 to 47, for example S1 has 91 channels in 25 parcels, S7 has 205 in 47, and S9 has 72 in 19.
  - Lobes range from 4 to 9.
  - Electrodes per parcel are therefore about 3.5 to 4.4 on average. The paper reports no electrodes-per-region analysis. — [arXiv HTML](https://arxiv.org/html/2512.12135)
- [VERIFIED] **Data.** Brain Treebank: 10 subjects, 26 sessions, 2048 Hz. 17 sessions were used for pretraining, 2 for validation and 7 for test (Sec. 4.1). — [arXiv HTML](https://arxiv.org/html/2512.12135)
- [VERIFIED] **Design.** A 3×3 factorial: encoding scale × masking scale. Table 2 AUC values:

  | Task | Encoding | Mask channels | Mask parcels | Mask lobes |
  |---|---|---|---|---|
  | Sentence onset | Channels | 0.778 | 0.710 | 0.654 |
  | Sentence onset | Parcels | 0.862 | 0.861 | 0.841 |
  | Sentence onset | Lobes | 0.842 | 0.816 | 0.840 |
  | Speech vs non-speech | Channels | 0.764 | 0.668 | 0.652 |
  | Speech vs non-speech | Parcels | 0.869 | 0.866 | 0.845 |
  | Speech vs non-speech | Lobes | 0.841 | 0.823 | 0.840 |

  Random initialization scores about 0.68 (sentence onset) and 0.62 (speech). — [arXiv HTML](https://arxiv.org/html/2512.12135)
- [VERIFIED] **Statistics.** A two-way ANOVA found encoding significant at p < 1e-3. Masking was significant at p = 0.010 (sentence onset) and p = 0.037 (speech), and there was no significant interaction. The authors conclude that "the choice of spatial encoding, rather than spatial masking, has a larger impact on final downstream performance." — [arXiv HTML](https://arxiv.org/html/2512.12135)
- [VERIFIED] **Reconstruction (Table 3).**
  - Parcel encoding with channel masking: MSE 0.391, R² 0.609.
  - Channel encoding with channel masking: MSE 0.397, R² 0.603.
  - So region-level encoding keeps channel-level reconstruction. — [arXiv HTML](https://arxiv.org/html/2512.12135)
- [VERIFIED] **Data scaling.**
  - Pretraining used 5, 10, 25, 50 and 75% of the data, with epochs adjusted to keep the number of updates comparable (Fig. 5 / App. K.1, Fig. 9).
  - This was run **only for the best model (parcel encoding, channel masking)**. Performance "successfully scaled with more pretraining data".
  - There is **no scale × data-amount interaction**. — [arXiv HTML](https://arxiv.org/html/2512.12135)
- [VERIFIED] **Held-out subject.** Excluding the test subject from pretraining gave 0.841 (onset) and 0.852 (speech), against 0.862 and 0.869 when it was included (Table 4). — [arXiv HTML](https://arxiv.org/html/2512.12135)
- [VERIFIED] **Controls and analyses.**
  - The only control is random initialization, plus comparisons to PopT and Brant.
  - There are **no shuffled-region, random-grouping, matched-count, grid, or functional-parcellation controls**.
  - There is no soft assignment and no convergence or training-curve analysis.
  - Tasks: the main text has two (sentence onset, speech/non-speech). Volume and optical flow are added in App. K.
  - The authors list simpler sensory tasks as future work (Sec. 5). — [arXiv HTML](https://arxiv.org/html/2512.12135)
- [VERIFIED] **Interpretability.** Destrieux parcels are used to visualize linear-projection weights (App. G). Per-subject results are in App. H, Tables 11 and 12. — [arXiv HTML](https://arxiv.org/html/2512.12135)
- [ABSTRACT/SNIPPET] **Venue.** NeurIPS 2025 poster 118567. — [NeurIPS](https://neurips.cc/virtual/2025/poster/118567)

**MAPA (Tang, Spalding, Cogan; Duke; "Pretraining for Sample-Efficient Neural Interfaces"; NeurIPS 2026; arXiv 2609.13507)**
- [VERIFIED] **Architecture.** A vanilla MAE with a ViT-S/384 encoder (21.3M parameters) and two spatial encodings:
  1. A learned anatomical region embedding added to every token of a contact.
  2. RoPE over the integer contact index along a shaft, plus time.
  - Contacts attend only within their own shaft.
  - Masking is uniform at 75%.
  - Pretraining data: 27.9 h of Brain Treebank, 13 sessions, 7 subjects. Subjects 7 and 10 are held out entirely. — [arXiv PDF](https://arxiv.org/pdf/2609.13507)
- [VERIFIED] **Region table.**
  - 74 labels: 62 cortical DKT regions plus 12 subcortical structures, and a reserved 75th row for out-of-vocabulary contacts. Labels are read from the Brain Treebank localization table.
  - The table is initialized near zero and indexed only by label, never by subject, session or contact.
  - Pretraining covers 50 of the 74 regions. Four right-hemisphere regions appear only in evaluation and stay untrained (App. B).
  - The model card gives 75×384 = 28,800 parameters. — [arXiv PDF](https://arxiv.org/pdf/2609.13507); [HF model card](https://huggingface.co/bentang18/MAPA)
- [VERIFIED] **Ablation.** A 2×2 design (both / RoPE only / region only / neither), each one pretraining run with one seed. Table 5 gives mean macro-AUROC gain over the frontend baseline across 15 Neuroprobe tasks:

  | Regime | Both | RoPE only | Region only | Neither |
  |---|---|---|---|---|
  | Within-session | +0.0210 | +0.0125 | +0.0085 | −0.0130 |
  | Cross-session | +0.0249 | +0.0221 | +0.0116 | −0.0064 |
  | Cross-subject | +0.0211 | +0.0119 | +0.0184 | −0.0076 |

  - The cross-subject results are not significant after BH correction (q = 0.0625) because there are only 5 subjects.
  - The region embedding's marginal contribution on top of RoPE ("both" minus "RoPE only") is about +0.009, +0.003 and +0.009.
  - Against "neither", region alone adds about +0.022, +0.018 and +0.026.
  - So "+0.01 to 0.02" is correct relative to the frontend baseline, but the effect over the no-spatial MAE is larger. — [arXiv PDF](https://arxiv.org/pdf/2609.13507)
- [VERIFIED] **Notable result.** "With neither spatial encoding, the pretrained model falls below the baseline" in all regimes (0/6, 0/6 and 1/5 subjects improved).
  - On held-out subjects (Fig. 4), the four models "stay close through the first half of the encoder and separate in the deeper layers".
  - This is a representation-depth result, not a training-curve result. — [arXiv PDF](https://arxiv.org/pdf/2609.13507)
- [VERIFIED] **Cross-subject readout.** Contact features are averaged within each DKT region, and the readout is fit on the regions shared by the anchor and test subjects (3 to 7 per pair).
  - The region atlas is therefore also part of the evaluation alignment, a confound to keep in mind. — [arXiv PDF](https://arxiv.org/pdf/2609.13507)
- [VERIFIED] **What MAPA does not do.**
  - No other atlas or resolution, no coordinates, no per-channel ID, no shuffled labels.
  - No data-amount or subject-count sweep for the spatial ablation, and no soft assignment.
  - No analysis of the learned region embeddings, and no convergence curves.
  - The Neuroprobe leaderboard values MAPA reports: MAPA 0.695 / 0.691 / 0.608 (within-session / cross-session / cross-subject), PopT 0.670 / 0.663 / 0.575, BrainBERT 0.626 / 0.633 / 0.547. — [arXiv PDF](https://arxiv.org/pdf/2609.13507)
- [VERIFIED] **MAPA's own summary of prior work.** Prior iEEG models "either did not encode spatial position [BrainBERT], or encoded it as a physical coordinate [6], as a learned embedding per channel [22], or as a coordinate in a template brain [15]". — [arXiv PDF](https://arxiv.org/pdf/2609.13507)

### Inferences
- Neither paper crosses spatial resolution with data amount or subject count. BaRISTA's sweep is single-configuration, and MAPA has no sweep. The "optimal resolution scales with electrodes per region" proposal is therefore untested in these two papers.
- BaRISTA's "channel" condition is a discretized coordinate code, not a per-electrode ID. Its poor score therefore cannot be read as "per-electrode IDs fail"; it is "fine coordinates with per-axis lookup fail". Your planned per-electrode ID vectors and Fourier-coordinate conditions would separate these explanations.
- MAPA's "neither encoding falls below baseline" result fits the idea that, without spatial identity, MAE pretraining on pooled subjects learns weak features. It is consistent with, but not evidence for, an optimization/plateau story.

### Gaps
- BaRISTA's exact atlas-label assignment rules, such as handling of white matter and unknown labels, were not visible in the fetched text.
- Neither paper reports multiple seeds for the spatial comparisons. MAPA explicitly uses one run per model with seed 33.

## Q1. Which iEEG/EEG/MEG models use location encodings, of what type, and did any ablate spatial resolution?

### Takeaway
- The encodings in use fall into a few families:
  - none (BrainBERT, Brant)
  - continuous coordinates with sinusoids or RBFs (PopT, Seegnificant, REVE, NeurIPT, SEEGformer)
  - learned per-name or per-slot channel embeddings (LaBraM, EEGPT, BIOT)
  - convolutional or conditional position encodings (CBraMod, DIVER-1)
  - predefined anatomical region groupings (CSBrain, Du-IN, BaRISTA, MAPA)
  - data-driven functional embeddings (Javadzadeh et al. 2025)
- **BaRISTA is the only paper I found that compares multiple anatomical granularities.** No paper compares several atlases at matched or varied region counts.

### Cited Findings
- [VERIFIED] **PopT (Chau, Wang et al.; ICLR 2025).**
  - Position encoding: sinusoidal encodings of each electrode's Left, Posterior and Inferior coordinates (iEEG) or XYZ (EEG), concatenated with an ensemble-membership encoding and summed with the channel embedding.
  - Table 3 ablation without position encoding, at 90 electrodes (full → ablated, ROC-AUC): pitch 0.74 → 0.62, volume 0.87 → 0.76, sentence onset 0.90 → 0.81, speech 0.93 → 0.83. All drops significant (Dunnett, p < 0.05).
  - There is no comparison against atlas or region encodings. DKT is used only post hoc to group attention weights (App. H, r = 0.4 with significant-electrode fraction).
  - Data scaling: 2, 5, 10, 25 and 50% of pretraining data (Fig. 7), but not crossed with the position-encoding choice.
  - Data: Brain Treebank, 10 subjects, 1,688 electrodes; plus TUEG/TUAB for EEG. — [arXiv v4](https://arxiv.org/html/2406.03044v4)
- [VERIFIED] **Seegnificant (Mentzelopoulos et al.; NeurIPS 2024; arXiv 2411.10458).**
  - Data: 21 subjects, 29 sessions, response-time regression.
  - Position encoding: MNI coordinates, each axis discretized into n bins, with m Gaussian RBFs per bin midpoint, concatenated to a 3·m·n vector and linearly projected.
  - The ablation of the position encoding is reported in Fig. 6A. The full text I read gives numeric ΔR² only for subject-specific heads (about 0.18) and spatial attention (0.10).
  - [ABSTRACT/SNIPPET] A search snippet says removing the position encoding cost about 0.02 R² per subject, not significant by Wilcoxon rank-sum. I did not verify this against the figure. — [arXiv PDF](https://arxiv.org/pdf/2411.10458); [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/c473b9c8897f50203fa23570687c6b30-Paper-Conference.pdf)
- [ABSTRACT/SNIPPET] **SEEGformer (eBioMedicine 2025).**
  - Encodes 3D MRI-derived coordinates through a shared learnable linear projection.
  - In its ablation, the spatial encoding did not help seizure detection but did help epileptogenic-zone localization. This is a task-dependence data point. — [Lancet eBioMedicine](https://www.thelancet.com/journals/ebiom/article/PIIS2352-3964(25)00535-3/fulltext)
- [VERIFIED] **Du-IN (Zheng et al.; NeurIPS 2024).**
  - Data: 12 subjects, sEEG, 61-word classification at about 62.7% top-1.
  - Builds region-level tokens by fusing channels within speech-relevant regions (vSMC, STG).
  - Ablations credit the region-level fusion.
  - These models are subject-specific, not a multi-subject spatial-encoding comparison. [The details come from a search summary, so they are ABSTRACT-level.] — [arXiv](https://arxiv.org/abs/2405.11459)
- [ABSTRACT/SNIPPET] **DIVER-1 (Han et al.; arXiv 2512.19097).**
  - An iEEG/EEG foundation model with "any-variate attention" and spatio-temporal conditional position encodings, so it assumes no fixed montage.
  - Its scaling-law study varies data, model size, compute and subject count. I found no spatial-encoding × scale analysis. — [arXiv](https://arxiv.org/html/2512.19097v3)
- [ABSTRACT/SNIPPET] **BrainWave (Yuan, Yang et al.; arXiv 2402.10251).**
  - Pretrained on invasive and non-invasive recordings, about 40k h from about 16k people.
  - Its spatial handling was not checked. — [arXiv](https://arxiv.org/abs/2402.10251)
- [VERIFIED] **EEG models' spatial mechanisms, as characterized by the 2026 probing paper.**
  - BIOT: learned channel-slot tokens.
  - LaBraM: embeddings indexed by electrode name.
  - CBraMod: channel order and convolutional position encoding (ACPE), no electrode identity.
  - CSBrain: groups electrodes into anatomical regions.
  - REVE: continuous 3D coordinates. — [arXiv 2610.29367](https://arxiv.org/html/2610.29367)
- [ABSTRACT/SNIPPET] **REVE (NeurIPS 2025).**
  - A 4D Fourier positional encoding (x, y, z, time) with a learnable component and Gaussian noise on electrode positions.
  - Pretrained on about 25,000 subjects. I found no PE-ablation numbers. — [arXiv](https://arxiv.org/html/2510.21585v1)
- [ABSTRACT/SNIPPET] **EEGPT** uses a learnable channel-embedding codex keyed by electrode name. — [Channel adaptation benchmark](https://arxiv.org/html/2604.23091v1)
- [ABSTRACT/SNIPPET] **EEG-X** argues that learnable channel embeddings "lack explicit spatial awareness" and replaces them with a coordinate-based location embedding. — [arXiv](https://arxiv.org/html/2511.08861)
- [ABSTRACT/SNIPPET] **CortexBridge (arXiv 2610.01124).** For scalp EEG, it maps montage channels onto atlas ROI centroids, using coordinate-based geometric bias in cross-attention, in front of a frozen foundation model. — [arXiv](https://arxiv.org/pdf/2610.01124)
- [ABSTRACT/SNIPPET] **Neuroprobe (arXiv 2509.21671).**
  - Benchmark on Brain Treebank: 10 subjects, 40 h, 15 tasks.
  - Linear baselines are very strong.
  - The package exposes electrode coordinates, but the benchmark does not evaluate spatial encodings. — [arXiv](https://arxiv.org/html/2509.21671v1)
- [ABSTRACT/SNIPPET] **iMINDBench (Chau, Hashemi, … Shanechi, Yue; arXiv 2609.18104).**
  - 15 tasks across three movie-watching iEEG datasets.
  - Up to 25× more cross-subject or cross-institution supervised data gives only small or task-dependent gains.
  - No spatial-encoding evaluation was visible on the landing page. — [arXiv](https://arxiv.org/abs/2609.18104)
- [MEMORY] Models I did not check this session:
  - BrainBERT (Wang et al., ICLR 2023) is single-channel with no spatial encoding. MAPA confirms "did not encode spatial position" [VERIFIED via MAPA].
  - Brant (NeurIPS 2023) and Brant-2 treat channels without anatomical location encodings.
  - MtM (Zhang et al., NeurIPS 2024) is Neuropixels with region-aware masking and session embeddings, not iEEG.
  - H2DiLR and MIBRAIN were not found in my searches. I have no verified information on their spatial encodings.

### Inferences
- The field is split between continuous coordinates (PopT, Seegnificant, REVE) and categorical regions (BaRISTA, MAPA, CSBrain, Du-IN), with essentially no head-to-head comparison in iEEG beyond BaRISTA's coordinate-lookup-vs-parcel-vs-lobe comparison.
- A multi-atlas comparison (lobes, Harvard-Oxford, DKT/Destrieux, Schaefer-400) on the same backbone appears novel. Confidence is moderate-high: I ran about 6 targeted searches and read BaRISTA, MAPA and PopT in full.

### Gaps
- H2DiLR, MIBRAIN, Brant-2 and MtM details were not verified.
- REVE, CSBrain and LaBraM ablation tables were not read.
- MEG foundation models (BrainOmni, MEG-GPT and source-space-parcellation models) were not covered by my searches.

## Q2. Did any vary training data, subjects or electrode density and report how the best spatial encoding changes?

### Takeaway
- **I found none.**
- BaRISTA and PopT vary pretraining data only for their single chosen configuration.
- DIVER-1 varies subject count, but not the spatial encoding.
- MAPA has no sweep.

### Cited Findings
- BaRISTA's data sweep (5 to 75%) covers only the parcel-encoding / channel-masking model [VERIFIED]. — [arXiv HTML](https://arxiv.org/html/2512.12135)
- PopT's data sweep (2 to 50%) covers only the full model, and the position-encoding ablation is at full data [VERIFIED]. — [arXiv v4](https://arxiv.org/html/2406.03044v4)
- MAPA uses a fixed 7 subjects and 27.9 h, with no sweep [VERIFIED]. — [arXiv PDF](https://arxiv.org/pdf/2609.13507)
- DIVER-1 ran a scaling study over data, parameters, compute and subject count, but did not cross it with spatial encoding [ABSTRACT/SNIPPET]. — [arXiv](https://arxiv.org/html/2512.19097v3)
- BaRISTA reports parcel counts per subject but no electrodes-per-region analysis [VERIFIED]. — [arXiv HTML](https://arxiv.org/html/2512.12135)

### Inferences
- The "optimal resolution scales with electrodes per region" hypothesis appears novel for iEEG/EEG. Confidence is moderate-high for iEEG (the key papers were read in full) and moderate for EEG/MEG (abstract-level only).

### Gaps
- I did not check Google Scholar citing pages of BaRISTA or MAPA. A very recent (October–November 2026) preprint could exist.

## Q3. Matched-region-count controls: atlas vs grid/cubes vs shuffled labels vs data-driven parcellation

### Takeaway
- No iEEG paper I found uses shuffled-region, random-grouping, grid/cube, or matched-count controls against atlas encodings.
- The closest are:
  - test-time permutation of positional codes in EEG models (a probing paper)
  - electrode-position shuffling in CoCoT-EEG
  - learned functional embeddings in place of anatomy (Javadzadeh et al.)

### Cited Findings
- BaRISTA's only control is random initialization [VERIFIED]. — [arXiv HTML](https://arxiv.org/html/2512.12135)
- **Probing Spatial and Spectral Grounding in EEG Foundation Models (arXiv 2610.29367).** Permuting positional codes at test time with the readout frozen (PhysioNet-MI, balanced accuracy):

  | Model | Clean | Permuted |
  |---|---|---|
  | REVE | 0.455 | 0.340 |
  | CSBrain | 0.440 | 0.337 |
  | LaBraM | 0.354 | 0.343 |
  | BIOT | 0.276 | 0.275 (near chance) |

  This is a test-time intervention, not a train-time shuffled-label control [VERIFIED]. — [arXiv](https://arxiv.org/html/2610.29367)
- CoCoT-EEG reports that shuffling electrode positions "does not consistently degrade performance" [ABSTRACT/SNIPPET]. — [arXiv](https://arxiv.org/pdf/2607.09543)
- **Functional embeddings (Javadzadeh et al.; ICLR 2026 submission; arXiv 2510.27090).**
  - Data: 20 subjects, basal ganglia–thalamic LFP.
  - A Siamese contrastive encoder learns a subject-agnostic functional identity per electrode.
  - The authors argue that MNI coordinates "often fail to capture true functional similarity".
  - No quantitative comparison against MNI or atlas encodings was visible [ABSTRACT/SNIPPET]. — [arXiv](https://arxiv.org/abs/2510.27090)

### Inferences
- A train-time shuffled-region control at matched region count, together with grid/cube partitions at matched counts, would separate "anatomy matters" from "any shared categorical grouping across subjects helps". This appears novel. Confidence is moderate-high for iEEG.

### Gaps
- The CoCoT-EEG and functional-embedding full texts were not read.

## Q4. Probing learned per-channel ID embeddings for anatomical structure

### Takeaway
- No iEEG paper I found tests whether learned per-electrode ID embeddings recover physical distance or atlas regions.
- In EEG, a 2026 probing paper measured distance-dependent influence. It found topographic locality in REVE and LaBraM, but not in CSBrain, CBraMod or BIOT.
- Its method is a perturbation analysis, not a direct embedding-geometry versus distance regression.

### Cited Findings
- **Probing paper (arXiv 2610.29367), SPA2.** Noise was injected at 10 dB SNR into one channel and the influence on other channels measured against distance.
  - REVE ρ ≈ −0.83 and LaBraM ρ ≈ −0.81, with LaBraM peaking in intermediate layers.
  - CSBrain and CBraMod showed little distance structure, and BIOT was roughly flat.
  - Identity displacement: BIOT 0.687, REVE 0.272 (PhysioNet-MI) [VERIFIED]. — [arXiv](https://arxiv.org/html/2610.29367)
- PopT compares attention-derived connectivity with coherence-based connectivity (per-subject r = 0.42 to 0.66) and with DKT-region significance (r = 0.4). It does not analyze distance [VERIFIED]. — [arXiv v4](https://arxiv.org/html/2406.03044v4)
- Javadzadeh et al. report that learned functional embeddings form "region-consistent clusters", but give no quantitative atlas-agreement metric [ABSTRACT/SNIPPET]. — [arXiv](https://arxiv.org/abs/2510.27090)
- MAPA does not analyze its learned DKT embeddings [VERIFIED]. — [arXiv PDF](https://arxiv.org/pdf/2609.13507)

### Inferences
- Showing that learned per-electrode IDs in sEEG recover inter-electrode distance or atlas membership (for example with RSA against Euclidean distance, or atlas-label decoding from the ID vectors) appears novel for iEEG. Confidence is moderate.

### Gaps
- Analogous analyses on spike-sorted unit embeddings (POYO, NDT-style models) or in fMRI were outside scope and not searched.

## Q5. Soft, probabilistic or weighted region assignment

### Takeaway
- No iEEG/EEG foundation model I found uses soft or probabilistic region embeddings.
- Probabilistic electrode labeling exists as a localization tool (ELAS) but is not used as a model input.
- A multi-atlas soft-ROI fusion exists for fMRI decoding.

### Cited Findings
- The ELAS toolbox assigns iEEG contacts to cortical areas probabilistically. In 14 patients, 72.5% of cortical contacts were assignable. It is a localization tool, not a model encoding [ABSTRACT/SNIPPET]. — [PubMed](https://pubmed.ncbi.nlm.nih.gov/31437467/)
- "Unified Multimodal Brain Decoding via Cross-Subject Soft-ROI Fusion" (arXiv 2512.20249, fMRI) uses multi-atlas soft functional parcellations with a learned per-voxel gate over atlases [ABSTRACT/SNIPPET]. — [arXiv](https://arxiv.org/html/2512.20249v1)
- MAPA uses hard labels, one DKT label per contact with a reserved out-of-vocabulary row [VERIFIED]. BaRISTA also uses one parcel and one lobe per channel [VERIFIED]. — [MAPA](https://arxiv.org/pdf/2609.13507); [BaRISTA](https://arxiv.org/html/2512.12135)

### Inferences
- Soft or distance-weighted region embeddings for sEEG contacts, which matter for contacts near boundaries and in white matter, appear novel in iEEG. Confidence is moderate: two targeted searches.

### Gaps
- Not searched: EEG source-space or MEG models that use soft parcellation weights.

## Q6. Spatial information helping optimization (plateau escape, faster convergence) rather than final accuracy

### Takeaway
- I found no paper reporting that spatial encodings help escape loss plateaus or speed convergence in iEEG/EEG foundation models.
- MAPA's finding that no-spatial MAE pretraining falls below the frontend baseline, with layer-wise divergence, is the closest related result. It concerns representation quality, not training dynamics.

### Cited Findings
- **MAPA.**
  - The "neither" model falls below baseline in all regimes: −0.013, −0.006 and −0.008.
  - In held-out subjects the models "stay close through the first half of the encoder and separate in the deeper layers".
  - No loss curves are reported [VERIFIED]. — [arXiv PDF](https://arxiv.org/pdf/2609.13507)
- Seegnificant reports training time (the 2D-attention variant took about 5.5× longer to train). That speedup comes from the attention factorization, not the position encoding [VERIFIED]. — [arXiv PDF](https://arxiv.org/pdf/2411.10458)
- BaRISTA reports update counts only, with no convergence analysis [VERIFIED]. — [arXiv HTML](https://arxiv.org/html/2512.12135)
- A targeted search ("spatial positional encoding faster convergence loss plateau EEG iEEG") returned no relevant result [search]. — [search: STAMP](https://arxiv.org/pdf/2511.10848), [CoCoT-EEG](https://arxiv.org/pdf/2607.09543)

### Inferences
- An optimization-dynamics framing of spatial encodings in pooled-subject iEEG pretraining appears novel. Confidence is moderate: one targeted search, plus full reads of BaRISTA, MAPA and PopT.

### Gaps
- Appendix training curves in REVE, LaBraM and CBraMod were not checked.

## Q7. Does the best spatial resolution or encoding depend on the task?

### Takeaway
- Only indirect evidence exists:
  - BaRISTA found parcels best on both language/auditory tasks it tested, and deferred sensory tasks to future work.
  - SEEGformer found spatial encoding helped localization but not detection.
  - An EEG positional-encoding benchmark found the best encoding is task-dependent.
  - MAPA's per-task results show near-chance visual tasks either way.
- No paper crosses resolution with a localized-versus-distributed task taxonomy.

### Cited Findings
- BaRISTA: parcel encoding was best on both sentence onset and speech. Simpler sensory tasks are proposed as future work [VERIFIED]. — [arXiv HTML](https://arxiv.org/html/2512.12135)
- Yuce & Stober (arXiv 2605.29754) benchmarked five positional encodings in CBraMod. "No single strategy consistently outperforms across tasks": spherical PE is good for motor imagery and worse for emotion recognition, while ACPE is more consistent [ABSTRACT/SNIPPET]. — [arXiv](https://arxiv.org/abs/2605.29754)
- SEEGformer: spatial encoding gave no gain for seizure detection but a significant gain for epileptogenic-zone localization [ABSTRACT/SNIPPET]. — [eBioMedicine](https://www.thelancet.com/journals/ebiom/article/PIIS2352-3964(25)00535-3/fulltext)
- MAPA per-task cross-subject AUROC (frontend → MAPA):
  - onset 0.745 → 0.815
  - speech 0.719 → 0.779
  - frame brightness 0.517 → 0.495
  - face number 0.503 → 0.515
  - Per-task results are not broken down by spatial ablation [VERIFIED]. — [arXiv PDF](https://arxiv.org/pdf/2609.13507)

### Inferences
- A study that crosses resolution with task locality (for example, auditory/onset tasks in STG versus distributed language features such as GPT-2 surprisal or part of speech in Neuroprobe) would be novel. Confidence is moderate-high for iEEG.

### Gaps
- Per-task spatial-ablation breakdowns for MAPA, and BaRISTA's App. K results by scale, were not visible in what I read.
