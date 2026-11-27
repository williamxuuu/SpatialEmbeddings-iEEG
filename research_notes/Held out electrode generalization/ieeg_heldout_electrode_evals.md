# Held-out electrode / channel / shaft / region generalization in iEEG decoding and foundation models (through Nov 2026)

Method (2026-11-26):
- About 28 tool calls. Sources were web searches, arXiv abstract and HTML fetches, and full-text PDF greps.
- I downloaded 14 arXiv PDFs and converted them with pdftotext, then grepped each for "held-out/unseen/zero-shot/leave-one/electrode/channel/region/shaft" and for referencing terms. The 14 are: BrainBERT, BrainWave, Du-IN, PopT, Seegnificant, MVPFormer, Neuroprobe, Javadzadeh 2510.27090, BaRISTA, DIVER-1, CORTEG, MAPA, iMINDBench and PHASE.
- I also reused verified notes in `../Spatial encoding resolution prior work/ieeg_eeg_models.md` and `../Density resolution deep dive/preemption.md`.

Legend:
- **[FULL]**: the relevant methods text was read or grepped in the full paper.
- **[ABS]**: only the abstract or landing page was read.
- **[SNIP]**: search-engine snippet only.
- **[MEM]**: from memory, not re-checked this session.

## Q1. Which iEEG papers hold out electrodes, shafts, channels or regions, and how was the split made?

### Takeaway
- **Almost no iEEG foundation or decoding model is evaluated on electrodes held out within a person.** The standard generalization unit is the *subject* or the *session*, and sometimes the *institution*:
  - Subject: BrainBERT, PopT, BaRISTA, MAPA, Seegnificant, MVPFormer, CORTEG, Neuroprobe cross-subject.
  - Session: Neuroprobe, BaRISTA, MAPA.
  - Institution: PHASE, BrainWave, iMINDBench.
- A new subject brings new electrodes by construction. But in these tests the model is always fine-tuned or probed on labeled data from those new electrodes, so it is never asked to transfer to an electrode without data from it.
- The only papers I found with an explicit within-person held-out channel or held-out region test are the Javadzadeh et al. functional-embedding papers (arXiv 2510.27090; NeurIPS 2026 workshop). That work uses a DBS basal-ganglia/thalamus cohort, not Brain Treebank. They report:
  - held-out channels: 49.18% ± 12.11% k-NN region accuracy for the joint encoder (52.84% in the workshop version);
  - held-out region: VO fully masked and reconstructed from GPi, STN and VIM. Functional coordinates beat MNI (p ≈ 0.002).
- BaRISTA's "masked channel reconstruction" is a weaker variant. Channels are masked one at a time, but they come from subjects whose channels were all seen in pretraining, and the data are Laplacian-referenced.

### Cited Findings

**Javadzadeh et al. — functional embeddings (arXiv 2510.27090, ICLR 2026 submission) [FULL] and NeurIPS 2026 workshop "Learning Data-Driven Shared Coordinates for Cross-Subject Intracranial Recordings" [ABS]**
- **Data.** 20 dystonia DBS patients with sEEG in GPi, STN, VO, VA, VIM, PPN and SNr. 442.86 electrode-hours in total. — [arXiv PDF](https://arxiv.org/pdf/2510.27090)
- **Encoder.** A CNN trained contrastively on 10-s LFP segments into a 32-D embedding. Region labels define the positive and negative pairs. There are two objectives, PSC (pairwise Siamese) and MSC. — [arXiv PDF](https://arxiv.org/pdf/2510.27090)
- **Split units.**
  - "Held-out time segments from electrodes seen during training" vs a "stricter held-out channel evaluation (regions with > 3 channels)".
  - Time splits are chronological 70/15/15 per session.
  - The held-out channel is embedded by running the trained encoder on its own LFP. No region label is given to the encoder. — [arXiv PDF](https://arxiv.org/pdf/2510.27090); [NeurIPS listing](https://neurips.cc/virtual/2026/157293)
- **Results, per-subject encoders (k-NN region accuracy).** Held-out time 75.78% ± 17.90%; held-out channels 45.79% ± 18.44%, with some subjects near 70%. — [arXiv PDF](https://arxiv.org/pdf/2510.27090)
- **Results, joint multi-subject encoder.** Held-out time 80.71% ± 11.41%; held-out channels 49.18% ± 12.11%. This is about 5% better than per-subject CNNs. — [arXiv PDF](https://arxiv.org/pdf/2510.27090)
- **Authors' explanation.** Channel-level generalization is lower than time-segment generalization, "likely due to the limited number of electrodes per region within a subject". — [arXiv PDF](https://arxiv.org/pdf/2510.27090)
- **Workshop version.** Reports 52.84% held-out-channel k-NN accuracy with the selected objective. It does not state the split fraction or k. — [NeurIPS 2026 workshop](https://neurips.cc/virtual/2026/157293)
- **Held-out region test.**
  - "All channels from a target region (here, VO) were withheld, signals from the remaining regions (GPi, STN, VIM) were encoded as source tokens, and the model queried the withheld VO channels using their coordinate system."
  - Coordinates compared: MNI vs Functional-1 (PSC) vs Functional-2 (MSC), on 11 subjects (limited by MNI availability).
  - Functional-2 beat MNI with p ≈ 0.002. Functional-1 vs MNI was a non-significant trend. Raw r values are not given in the text.
  - The model has no subject-ID tokens or subject heads.
  - MNI failure mode: "the four VO electrodes often shared nearly identical or uncertain MNI coordinates", so the MNI model produced near-identical waveforms for each VO target. — [arXiv PDF](https://arxiv.org/pdf/2510.27090)
- **Workshop version of the region test.** Pooled transformers conditioned on region labels vs MNI vs learned coordinates. The learned coordinates significantly improved cross-region waveform prediction in both hemispheres. — [NeurIPS listing](https://neurips.cc/virtual/2026/157293)
- **Not reported.** The re-referencing scheme (preprocessing lists only anti-aliasing, notch and z-scoring), any sweep of embedding dimension (fixed at 32), and any data-amount or subject-count sweep. — [arXiv PDF](https://arxiv.org/pdf/2510.27090)
- **Caveat on the VO test.** Every VO channel is the query, so the VO region is "held out" at evaluation only, within a time split. Training sessions of the same subjects contain VO channels. It is a held-out-in-time masked-region reconstruction, not a region never seen in training. [Inferred from the 70/15/15 chronological split of the same sessions — [arXiv PDF](https://arxiv.org/pdf/2510.27090)]

**BaRISTA (Oganesian, Hashemi, Shanechi; NeurIPS 2025; arXiv 2512.12135) [FULL]**
- **Masked channel reconstruction.** "During evaluation, we mask out one channel at a time and report the average reconstruction MSE and R² across all masked channels for the 7 held-out test sessions". All test sessions come from the 10 pretraining subjects, so the channels themselves were seen in pretraining.
  - Parcel encoding with channel masking: R² 0.609.
  - Channel (LPI-coordinate) encoding with channel masking: R² 0.603. — [arXiv](https://arxiv.org/pdf/2512.12135)
- **Referencing.** "remove corrupted/noisy channels before Laplacian re-referencing the rest, excluding channels with insufficient neighbors", and ventricle channels are removed. Laplacian referencing subtracts the shaft neighbours' mean, so a masked channel's signal shares components with its unmasked neighbours. — [arXiv](https://arxiv.org/pdf/2512.12135)
- **Held-out subject.** All sessions of the test subject are excluded from pretraining. Results: 0.841 / 0.852 vs 0.862 / 0.869 included. Downstream fine-tuning still uses the held-out subject's labeled channels. — [arXiv HTML](https://arxiv.org/html/2512.12135)

**BrainBERT (Wang et al., ICLR 2023; arXiv 2302.14367) [FULL]**
- **Data.** Pretraining used 19 of 26 sessions. "All ten subjects are represented in the pretraining data." 1,249 Laplacian-re-referenceable electrodes, about 74% of all electrodes, 4,551 electrode-hours. — [arXiv PDF](https://arxiv.org/pdf/2302.14367)
- **"Held-out electrodes" is held out in time, not in electrode identity.**
  - The paper claims improvement "on held-out data and held-out electrodes". Evaluation is per electrode on the 7 held-out sessions, using the top-10 electrodes by linear decodability.
  - Fig. 3 shows "the 947 held-out electrodes".
  - I could not confirm from the text whether those electrodes were absent from pretraining. Pretraining includes all subjects and every Laplacian-capable electrode, so most likely they were not. — [arXiv PDF](https://arxiv.org/pdf/2302.14367)
- **Hold-one-subject-out.** The best 10 electrodes of the held-out subject are used, and the difference from the included-subject model is "negligible". BrainBERT is a single-channel model with **no spatial encoding**, so electrode novelty is trivially handled. — [arXiv PDF](https://arxiv.org/pdf/2302.14367)

**PopT (Chau et al., ICLR 2025; arXiv 2406.03044) [FULL via HTML]**
- **Splits.** Hold-one-subject-out pretraining: "missing a subject from pretraining does not significantly affect the downstream results" (Fig. 6). There is no held-out-electrode experiment. — [arXiv HTML](https://arxiv.org/html/2406.03044v4)
- **Electrode subsets.** Downstream subsets are the top-k electrodes by individual linear decodability (main results 90; scaling 1 to 50). Random subsets are in App. F. — [arXiv HTML](https://arxiv.org/html/2406.03044v4)
- **Spatial encoding.** Sinusoidal LPI coordinates. Removing it drops AUC by about 0.1 on all tasks (e.g., sentence onset 0.90 → 0.81). — [arXiv HTML](https://arxiv.org/html/2406.03044v4)

**Neuroprobe (arXiv 2509.21671) [FULL via HTML]**
- **Splits.** Within-session (2-fold, contiguous blocks), cross-session, and cross-subject (train on Subject 2, Trial 4; test on other subjects). No electrode or region hold-out split. — [arXiv HTML](https://arxiv.org/html/2509.21671v2)
- **Lite.** 6 subjects × 2 trials. At most 120 electrodes per subject, chosen in whole-probe groups ranked by linear decoding. At most 3,500 datapoints per task. — [arXiv HTML](https://arxiv.org/html/2509.21671v2)
- **Referencing.** Laplacian for the Laplacian-STFT linear, CNN, MLP, GNN and BrainBERT baselines. — [arXiv HTML](https://arxiv.org/html/2509.21671v2)
- **Cross-subject alignment for non-PopT models.** Activity is averaged within the 34 Desikan-Killiany regions, using only regions shared by both subjects. — [arXiv HTML](https://arxiv.org/html/2509.21671v2)

**MAPA (Tang, Spalding, Cogan; NeurIPS 2026; arXiv 2609.13507) [FULL]**
- **Splits.** Subjects 7 and 10 are excluded from pretraining. Cross-subject evaluation fits on an anchor subject and tests on others, using DKT-region-averaged features from shared regions (3 to 7 per pair). — [arXiv PDF](https://arxiv.org/pdf/2609.13507)
- **Untrained region rows.** 4 of the 74 DKT rows appear only at evaluation, so their embeddings stay untrained. This is a de facto unseen-region case, but it is not analysed. — [arXiv PDF](https://arxiv.org/pdf/2609.13507)
- **Referencing.** MAPA uses an array mean re-reference and states: "No excluded contact enters an evaluation, not even through the array mean re-reference." This is explicit leakage hygiene for excluded contacts. — [arXiv PDF](https://arxiv.org/pdf/2609.13507)

**Seegnificant (Mentzelopoulos et al., NeurIPS 2024; arXiv 2411.10458) [FULL]**
- **Electrodes.** Bipolar montage of adjacent same-shank contacts. Each bipolar pair is a "virtual electrode" located at the midpoint.
- **Splits.** Leave-one-subject-out, 21 subjects. The left-out subject's data are used for fine-tuning. No held-out electrodes. — [arXiv PDF](https://arxiv.org/pdf/2411.10458)

**Du-IN (arXiv 2405.11459) [FULL grep]**
- Bipolar (or Laplacian) re-reference, with subject-specific models. No electrode hold-out found. — [arXiv PDF](https://arxiv.org/pdf/2405.11459)

**DIVER-1 (arXiv 2512.19097) [FULL grep]**
- Held-out *benchmarks* (Neuroprobe, MAYO) and held-out pretraining loss. Laplacian is the default after an ablation (Table 13). No electrode or region hold-out. — [arXiv PDF](https://arxiv.org/pdf/2512.19097)

**MVPFormer (ICLR 2026; arXiv 2506.20354) [FULL grep]**
- Unseen *subjects* (Kappa 0.61 across 50 unseen SWEC subjects; experiments with 10 or 40 unseen subjects).
- MVPFormer's implicit channel handling needs no electrode location. No electrode hold-out. — [arXiv PDF](https://arxiv.org/pdf/2506.20354)

**BrainWave (arXiv 2402.10251) [FULL grep]**
- Cross-subject and cross-institution zero-shot transfer, e.g., FNUSA → Mayo 93.82%. No electrode hold-out found. — [arXiv PDF](https://arxiv.org/pdf/2402.10251)

**PHASE (arXiv 2609.36087) [FULL grep]**
- Held-out participants and institutions (52 held-out Omni-iEEG participants, MAYO and FNUSA held-out cohorts). Uses the benchmark's Laplacian scheme. No electrode hold-out. — [arXiv PDF](https://arxiv.org/pdf/2609.36087)

**iMINDBench (arXiv 2609.18104) [FULL grep]**
- Fixed per-subject splits, held-out Brain Treebank subjects, and a multi-institution scaling study. Laplacian re-referencing throughout. No electrode or region hold-out. — [arXiv PDF](https://arxiv.org/pdf/2609.18104)

**CORTEG (arXiv 2605.10337) [FULL via HTML]**
- **Splits.** Leave-one-subject-out Stage 1 pooling, then Stage 2 fine-tuning on the held-out subject's training split. No held-out-electrode split. — [arXiv HTML](https://arxiv.org/html/2605.10337)
- **KNNSoftFourier adapter.** MNI xyz → softmax over the 142 frozen EEG channel embeddings, plus Fourier features (F = 32 log-spaced frequencies per axis).
- **Adapter ablations.** Random XYZ 0.479 vs real 0.554 (finger r); zero XYZ 0.236. Fourier-only 0.515; soft-only 0.443.
- **No sweeps** of k, anchors or Fourier frequency range.
- **Leave-one-channel-out.** Used only as a post hoc importance analysis. — [arXiv HTML](https://arxiv.org/html/2605.10337)

**Brant / Brant-2 [not checked this session]**
- I did not fetch these. [MEM] Brant is channel-independent, with no spatial encoding and per-channel tasks. I have no evidence of an electrode hold-out. Verify before citing.

**Spatially Masked Regression (Ostadsharif Memar & Dehghani, arXiv 2606.11415) [ABS]**
- Reconstructs each electrode's timeseries from the remaining electrodes, excluding a configurable neighbourhood around the target. Applied to iEEG and scalp EEG.
- Finds "substantial residual predictability when local neighbors are excluded".
- Cross-subject transfer is "markedly stronger in scalp EEG than in iEEG", so iEEG relationships are more local and subject-specific.
- This is the only paper found that systematically varies an *exclusion radius* around held-out targets. It is not a spatial-encoding model. — [arXiv abs](https://arxiv.org/abs/2606.11415)

**Scalp-EEG adjacent work (not iEEG) [ABS/SNIP]**
- HEAR evaluates on a dataset whose electrodes are disjoint from the pretraining layout. — [arXiv 2510.12515](https://arxiv.org/pdf/2510.12515)
- DIVER-0 is channel-permutation-equivariant, designed for unseen montages. — [arXiv 2507.14141](https://arxiv.org/html/2507.14141)
- A montage-robustness study finds that embedding stability and probe performance diverge. — [arXiv 2609.36288](https://arxiv.org/html/2609.36288v1)
- Device Passport crosses channel-embedding *type* with the number of fine-tuning subjects. — see [preemption notes](../Density%20resolution%20deep%20dive/preemption.md), arXiv 2607.00249

### Inferences
- **A true within-person held-out electrode test with a spatial encoding is essentially unoccupied territory** in iEEG foundation-model work. By "true" I mean: the electrode is absent from training, its location encoding is given, and it has no labeled or unlabeled data from itself. Javadzadeh comes closest, but it embeds the held-out channel *from its own signal*, so it is not location-only, and it uses a DBS cohort.
- **Laplacian re-referencing is near-universal** (BrainBERT, BaRISTA, Neuroprobe, DIVER-1, iMINDBench, PHASE). Bipolar referencing is used by Seegnificant and Du-IN. Under either scheme a held-out contact's signal is mixed with its shaft neighbours'. Only MAPA explicitly guards against reference leakage of excluded contacts. Any held-out-contact design must hold out whole shafts, or re-reference after splitting.
- BaRISTA's masked-channel result (parcel ≈ channel encoding for reconstruction) cannot distinguish "location encoding generalizes" from "Laplacian neighbours leak the target".

### Gaps
- I could not confirm whether BrainBERT's "947 held-out electrodes" were excluded from pretraining. The text implies they were not.
- I could not read the OpenReview PDFs for FunctionalMap (mvhZgiG0fa) or UniPopT (JcIG5NkCvk), which are ICLR 2027 submissions. OpenReview returned a bot-check page, so their split units are unknown.
- Brant, Brant-2 and Neuro-MoBRE were not checked in full text.
- I did not find the SMR mask radii or whether shaft neighbours are excluded; I read the abstract only.

## Q2. Has anyone swept spatial-encoding granularity crossed with data amount or density in a held-out-electrode setting?

### Takeaway
- **No.** No iEEG (or EEG) paper I found sweeps location-encoding resolution crossed with data amount, electrodes per region or number of subjects, in a held-out-electrode or any other setting.
- The nearest pieces, none of them crossed:
  - BaRISTA varies scale (channel/parcel/lobe) at fixed data, and data fraction only for its best configuration.
  - Javadzadeh attributes weak held-out-channel accuracy to few electrodes per region, without testing it.
  - DIVER-1 sweeps subject diversity, not resolution.
  - CORTEG fixes its Fourier and kNN resolution.

### Cited Findings
- BaRISTA: a 3×3 grid of encoding scale × masking scale. Parcel encoding was best (onset 0.862 vs channel 0.778). The data sweep (5 to 75%) used only parcels/channels. — [arXiv HTML](https://arxiv.org/html/2512.12135)
- Javadzadeh: held-out-channel accuracy is limited "likely due to the limited number of electrodes per region within a subject". There is no sweep, and the embedding is fixed at 32-D. — [arXiv PDF](https://arxiv.org/pdf/2510.27090)
- MAPA: one DKT table (74 labels), with 4 rows never trained. No granularity or subject sweep. — [arXiv PDF](https://arxiv.org/pdf/2609.13507)
- CORTEG: fixed F = 32 frequencies and k = 8. No resolution ablation. — [arXiv HTML](https://arxiv.org/html/2605.10337)
- Seegnificant: the MNI encoding uses an RBF basis with n bins × m RBFs per axis, a tunable resolution that is not swept in the main text. — [arXiv PDF](https://arxiv.org/pdf/2411.10458)
- Earlier pre-emption search (2026-11-20): "No paper found crosses resolution of the location encoding with the number of training subjects, in any modality." — [local notes](../Density%20resolution%20deep%20dive/preemption.md)

### Inferences
- Crossing resolution (region, parcel or cube size) with density (electrodes per cell, subjects, hours) under held-out electrodes or held-out shafts is novel as far as this search reaches. The bias-variance reading has indirect support in Javadzadeh's per-region-count explanation and in the MNI-collapse anecdote (identical coordinates for 4 VO contacts give identical predictions, i.e., too coarse or noisy a location code).

### Gaps
- Google Scholar cited-by for BaRISTA and MAPA was not checked. NeurIPS 2026 workshop papers are poorly indexed. The ICLR 2027 OpenReview full texts were inaccessible.

## Q3. Speech/BCI work decoding from new, held-out or dropped electrodes

### Takeaway
- Speech-BCI work handles electrode variability mainly through participant-level transfer, with MNI and ROI encodings or latent alignment, and through channel-dropout robustness tests. I found no speech paper that decodes from specific electrodes withheld from training in a participant who was seen in training.

### Cited Findings
- **SwinTW (Chen et al., J Neural Eng 2025; bioRxiv 2024.03.11.584533)**
  - Encoding: relative positional bias from an MLP on MNI xyz and their differences, plus a learned per-ROI embedding. Unknown-location electrodes get MNI (0, 0, 0) and an "Unknown" ROI.
  - Cohort: 52 participants (43 ECoG, 9 sEEG-only).
  - Unseen participants: mean PCC 0.765 across 43.
  - Electrode subsets: grid-only 0.825; grid plus extra electrodes 0.838 (n = 39, P = 0.00025); sEEG-only 0.798 (n = 9).
  - Split wording conflicts: the abstract says leave-one-out, the Methods describe 5-fold CV split by sex.
  - No electrode-level hold-out. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11773629/); [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.03.11.584533v2.full)
- **Sparse sEEG speech-state detection (PMC13194996, 2026)**
  - Setup: 4 participants, 1D CNN, no spatial encoding.
  - Random channel dropout of 10/30/50% (20 seeds): no significant accuracy change (per-participant ANOVA p = 0.684 / 0.055 / 0.068 / 0.206).
  - Excluding classical language regions: accuracy stayed above chance (p < 0.001) but dropped for 3 of 4 participants.
  - It is unclear whether the model was retrained on the reduced sets. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13194996/)
- **Cross-patient μECoG speech (Spalding/Cogan lab, bioRxiv 2025.08.21.671516)**: latent dynamics aligned and pooled across patients for cross-patient speech BCIs. [SNIP] — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.08.21.671516v2.full)
- **Imagined speech ECoG (Commun Biol 2024)**: cross-participant transfer was best within the same brain region. [SNIP] — [Nature](https://www.nature.com/articles/s42003-024-06518-6)
- **Speech-BCI review (arXiv 2603.12279)**: group-model accuracy plateaus at 3 to 4 subjects, and transfer improves with similarity of electrode coverage between training and inference subjects. [SNIP] — [arXiv](https://arxiv.org/pdf/2603.12279)
- **Intracortical cross-subject alignment**: linear alignment between T12 (128 channels) and T15 (256 channels). [SNIP] — [JNE](https://iopscience.iop.org/article/10.1088/1741-2552/ae8576)
- **Intracortical session-specific alignment layers**: ALIGN, an adversarial method for electrode shifts. — [arXiv 2603.18299](https://arxiv.org/pdf/2603.18299)
- **Intracortical end-to-end decoding**: a session-specific linear layer before a shared encoder. — [arXiv 2605.24313](https://arxiv.org/pdf/2605.24313)

### Inferences
- The speech results suggest that coverage overlap is a key driver of transfer: same-region transfer is best, and gains plateau at 3 to 4 subjects. This parallels the local "soft regions = coverage effect" finding. It argues for reporting the fraction of held-out electrodes whose spatial cell is occupied in training, at each resolution.

### Gaps
- I did not find any speech study that varies the spatial-encoding resolution (for example, the SwinTW ROI vs MNI ablation at several granularities). SwinTW's own encoding ablations were not extracted.
- Older ECoG transfer work (e.g., Peterson et al. HTNet 2021, Makin/Chang transfer) was not re-checked this session. [MEM] HTNet projected ECoG onto common ROIs for cross-participant transfer. Verify before citing.
