# Generalization to Unseen Channels / Electrodes / Recording Sites in Non-iEEG Neural-Data Models (scalp EEG, MEG, spiking, LFP)

Scope note: compiled 2026-11-26 from about 20 search/fetch calls. Each claim says how it was read:
- **[FT]**: read from the full text (arXiv HTML) during this session.
- **[ABS]**: read from the abstract or landing page only.
- **[SNIP]**: taken from a search-engine snippet. Lower confidence; verify before quoting.
- **[BG]**: background knowledge not re-verified this session. The URL is given, but the claim should be checked.

Intracranial (sEEG/ECoG) models are covered by another researcher. They appear here only as cross-references.

---

## 1. Channel-agnostic / position-encoded EEG/MEG foundation models and their unseen-channel tests

### Takeaway
Almost every 2025-2026 "any-montage" EEG/MEG foundation model gets montage flexibility by computing channel embeddings from 3D electrode coordinates (Fourier features plus an MLP) or from sensor physics, instead of looking them up in a learned table. Rigorous held-out-electrode tests are still rare. The clearest are:
- NOMAD-EEG: unseen electrodes such as CPz and POz, plus a position-shuffle control.
- HEAR: disjoint layout split.
- BrainOmni: unseen MEG/EEG devices.
- REVE: unseen bipolar TUEV montages, with no per-montage numbers.

Most "unseen montage" claims are really fine-tuning on a new montage, not zero-shot inference on new electrodes.

### Cited Findings

**REVE (Oct 2025; 4D Fourier PE)**
- **How the encoding works [FT]:**
  - Each token's (x, y, z, t) is expanded into Fourier features with nfreq ∈ {3, 4, 5} frequencies per dimension.
  - All frequency combinations across the four dimensions are combined (Cartesian product), giving nfreq⁴ terms, or 2·nfreq⁴ after sine and cosine.
  - nfreq = 4 for Small/Base and 5 for Large. The hidden dimension is matched to that output size.
  - The final PE is LayerNorm(Fourier + a learned Linear-GELU-LayerNorm projection of the coordinates).
  - [REVE arXiv 2510.21585](https://arxiv.org/html/2510.21585)
- **No ablation of frequency count [FT]:** the frequency count was tied to model size, so the "resolution" of the Fourier code was never varied on its own. [REVE](https://arxiv.org/html/2510.21585)
- **Masking settings [FT]:** 55% mask ratio, 3 cm spatial masking radius, 3 s temporal radius, 10% whole-channel dropout, and a 4 cm dropout radius. Spatially structured masking acts as an implicit channel-interpolation pretext task. [REVE](https://arxiv.org/html/2510.21585)
- **Table 18, mask ratio (small model, balanced accuracy averaged over 8 datasets) [FT]:**

  | Mask ratio | Frozen, random | Frozen, block | Fine-tuned, random | Fine-tuned, block |
  |---|---|---|---|---|
  | 0.25 | 0.523 | 0.513 | 0.612 | 0.602 |
  | 0.55 | 0.550 | 0.558 | 0.643 | 0.665 |
  | 0.75 | 0.519 | 0.546 | 0.606 | 0.655 |

  [REVE](https://arxiv.org/html/2510.21585)
- **Table 19, component ablations (Base model, balanced accuracy averaged over PhysionetMI and Mental Arithmetic) [FT]:**

  | Variant | Average balanced accuracy |
  |---|---|
  | Full model (baseline) | 0.707 |
  | Learnable PE | 0.701 |
  | MLP-only 4D PE | 0.677 |
  | Without position noise | 0.660 |
  | Without dropout block masking | 0.662 |
  | Without temporal block masking | 0.685 |

  - The paper's footnote: learnable PE "matches the baseline, but does not allow for the extension to larger time windows or unseen spatial configurations."
  - Removing **position noise** (jittering electrode coordinates in training) caused the largest drop. Its magnitude is not given in the text that was read.
  - [REVE](https://arxiv.org/html/2510.21585)
- **Table 21, channel-count robustness (REVE-Base, balanced accuracy) [FT]:**
  - PhysionetMI L-R: 0.824 at 64 channels, 0.808 at 32, 0.787 at 16, 0.781 at 8, 0.728 at 4, 0.700 at 2, 0.660 at 1.
  - Speech task: 0.565 at 64 channels, 0.490 at 32, 0.469 at 16, 0.294 at 8, 0.258 at 4, 0.228 at 2, 0.209 at 1.
  - [REVE](https://arxiv.org/html/2510.21585)
- **Unseen-montage claim [FT]:** REVE claims generalization to TUEV bipolar montages never seen in pretraining. No separate per-montage held-out numbers were found in the text. [REVE](https://arxiv.org/html/2510.21585)

**NOMAD-EEG (Oct 2026; diffusion, position-only electrode identity) — the most direct "position is the only clue" design found**
- **How the encoding works [FT]:**
  - Each electrode is identified only by its 3D scalp coordinate.
  - Fourier features: sine and cosine of 2^k·π·p for k = 0..7 on each axis, giving 48 dimensions, projected to width 256.
  - "The positional features enter the attention input only; the residual stream carries the signal features." Position is re-injected in every block.
  - Bipolar channels are placed at the midpoint of their two electrodes.
  - One fixed coordinate scale is used, so an electrode has the same coordinate in every montage.
  - [NOMAD-EEG arXiv 2610.35246](https://arxiv.org/html/2610.35246v1)
- **Unseen-electrode tests [FT]:**
  - BCIAUT-P300: 2 of its 8 electrodes (CPz, POz) are absent from pretraining. NOMAD-EEG had AUROC 0.718, best of 5 models.
  - EEGMMIDB: 45 of its 64 positions were never seen in pretraining. Frozen balanced accuracy was 0.442 (chance 0.25), best among frozen models.
  - [NOMAD-EEG](https://arxiv.org/html/2610.35246v1)
- **Channel-dropout test [FT]:** on Mumtaz, a probe trained on 19 electrodes was applied unchanged to random subsets, with no imputation.
  - Mean AUROC: 0.942 with 19 electrodes, 0.919 with 14, 0.919 with 10, 0.891 with 5. The worst subset was 0.877.
  - Balanced accuracy at 5 electrodes was 0.696.
  - [NOMAD-EEG](https://arxiv.org/html/2610.35246v1)
- **Position-reliance control (rare and useful) [FT]:**
  - Permuting electrode order changed no predictions, because the model is permutation-equivariant.
  - Shuffling positions while keeping the signals changed 17.5% of predictions and lowered AUROC from 0.942 to 0.909.
  - So the model uses position, but only moderately.
  - [NOMAD-EEG](https://arxiv.org/html/2610.35246v1)
- **Scaling [FT]:** NOMAD-EEG-Large (67.6 M parameters, 6,984 h of data, 1.9 epochs) was not significantly better than Base (12 M) on any benchmark. [NOMAD-EEG](https://arxiv.org/html/2610.35246v1)
- **No granularity ablation [FT]:** no ablation of the number of Fourier frequencies was found in the first ~100k characters of the paper. [NOMAD-EEG](https://arxiv.org/html/2610.35246v1)

**HEAR (Oct 2025)**
- **How the encoding works [FT]:**
  - 3D coordinates from a global dictionary of 1,132 electrodes pass through a learnable spatial MLP, broadcast over time and added to the tokens.
  - A second MLP maps pairwise coordinate differences to per-head attention biases.
  - No Fourier features or resolution parameter are described.
  - [HEAR arXiv 2510.12515](https://arxiv.org/html/2510.12515)
- **Layout-transfer test [FT]:**
  - BCI-IV-2B electrodes are split into two non-overlapping layouts. The model is fine-tuned on one and tested on the other.
  - The paper calls this "zero-shot", but the model is fine-tuned on the source layout.
  - The exact split is unclear, because the dataset section lists only C3, Cz and C4.
  - [HEAR](https://arxiv.org/html/2510.12515)
- **Layout-transfer results [FT]:**

  | Model | L1→L2 accuracy | L2→L1 accuracy |
  |---|---|---|
  | HEAR-tiny | 0.7190 | 0.7071 |
  | EEGPT | 0.6292 | 0.6292 |

  [HEAR](https://arxiv.org/html/2510.12515)
- **Ablations [FT]:** the spatial embedding was ablated as a block. There was no ablation of frequency, resolution or neighbour density. [HEAR](https://arxiv.org/html/2510.12515)

**LUNA (Oct 2025)**
- **How the encoding works [FT]:**
  - NeRF-style sinusoidal encoding of normalized 3D coordinates, followed by an MLP. The number of frequencies is not stated.
  - Q learned queries cross-attend over channels (Q = 4/6/8), so the model handles any channel count.
  - [LUNA arXiv 2510.22257](https://arxiv.org/html/2510.22257)
- **Pretraining data [FT]:** TUEG + Siena, about 21.9k h, covering only "three dominant layouts". [LUNA](https://arxiv.org/html/2510.22257)
- **SEED-V, an unseen 62-channel montage [FT]:**

  | Model | Balanced accuracy |
  |---|---|
  | CBraMod | 40.91% |
  | LaBraM-Base | 39.76% |
  | LUNA-Large | 39.18% |
  | LUNA-Huge | 39.00% |
  | LUNA-Base | 37.30% |

  - The authors attribute the gap to "reliance on positional encodings learned during pre-training" and limited montage diversity. This is a speculative explanation with no PE ablation.
  - The setup involves fine-tuning, so calling it "zero-shot" is questionable.
  - [LUNA](https://arxiv.org/html/2510.22257)

**DIVER-0 (Jul 2025)**
- **Design [FT]:**
  - Fully channel-equivariant. There are no channel embeddings or coordinates.
  - Channel identity enters only through learnable same-channel vs cross-channel attention-bias scalars. RoPE is used for time only.
  - The authors admit this "sacrifices detailed channel-specific discrimination."
  - [DIVER-0 arXiv 2507.14141](https://arxiv.org/html/2507.14141)
- **Permutation test (FACED) [FT]:** balanced accuracy stayed at 59.0–59.6% across all four intact/permuted pretrain × finetune combinations. [DIVER-0](https://arxiv.org/html/2507.14141)
- **No held-out-electrode test [FT]:** pretraining used 19 channels, while FACED has 32 and PhysioNet-MI has 64. That montage difference is the only evidence offered for handling new montages. [DIVER-0](https://arxiv.org/html/2507.14141)

**BrainOmni (NeurIPS 2025; EEG + MEG)**
- **Sensor Encoder [SNIP/ABS]:** encodes each sensor's 3D position, orientation and type (EEG, gradiometer, magnetometer) instead of channel names. [BrainOmni arXiv 2505.18185](https://arxiv.org/pdf/2505.18185); [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/file/3aef4a18c2646b9d11d9ab4d9bec72c8-Paper-Conference.pdf)
- **Unseen devices [SNIP/ABS]:**
  - Reported "zero-shot" reconstruction on entirely unseen EEG and MEG devices.
  - Pretraining used about 1,997 h of EEG and 656 h of MEG.
  - The PDF could not be parsed, so the evaluation design and numbers were not verified.
  - [BrainOmni](https://arxiv.org/pdf/2505.18185)

**Yuce & Stober, positional-encoding benchmark (May 2026) [ABS]**
- **Design:** five PE strategies compared inside the CBraMod backbone, using linear probing and fine-tuning on motor imagery and emotion recognition. [arXiv 2605.29754](https://arxiv.org/abs/2605.29754)
- **Results:**
  - No single strategy wins on every task.
  - Spherical PE (SPE) is strong for motor imagery but weak for emotion.
  - ACPE is the most consistent across tasks.
  - The abstract describes no unseen-channel test, and only SPE and ACPE are named there.
  - [arXiv 2605.29754](https://arxiv.org/abs/2605.29754)

**Avramidis, Tashev & Emmanouilidou, "Probing Spatial and Spectral Grounding in EEG Foundation Models" (Oct 2026) [ABS]**
- **Design:** perturbation probes of channel identity and topographic locality, applied to 5 published EEG foundation models on motor-imagery and seizure data. [arXiv 2610.29367](https://arxiv.org/abs/2610.29367)
- **Results:**
  - Electrode identity can be encoded in the embeddings without influencing classification.
  - Permuting **LaBraM's positional codes** heavily displaced the embeddings but cut motor-imagery accuracy by only **1.2 points**.
  - Only REVE beat a spectral baseline on a simulated clinical-marker task.
  - [arXiv 2610.29367](https://arxiv.org/abs/2610.29367)

**CoCoT-EEG (Jul 2026) [ABS]**
- Contrastive multiscale-convolution transformer. It handles "heterogeneous electrode configurations", but the abstract gives no details on its channel/position encoding and no unseen-channel test. [arXiv 2607.09543](https://arxiv.org/abs/2607.09543)

**Steglich et al., "Representational and Functional Robustness to Electrode Montages in EEG Foundation Models" (Sep 2026) [FT]**
- **Design:**
  - Models: SingLEM, CBraMod, LUNA and REVE (REVE with ROI pooling or global attention pooling).
  - Linear probes are trained on embeddings from an N-channel montage and tested on an M-channel montage.
  - Resting state: HBN 128-channel montage, probes tested at 64 and 32 channels.
  - Motor imagery: 62-channel montage, probes tested at 32 and 16 channels.
  - Channel subsets were chosen by farthest-point sampling.
  - [arXiv 2609.36288](https://arxiv.org/html/2609.36288)
- **Results:**
  - Cross-montage drops were minimal for CBraMod and REVE-ROI and pronounced for LUNA and REVE-attention-pooling. The values are in figures only.
  - Full-montage motor-imagery AUROC: CBraMod 0.71, REVE 0.69, SingLEM 0.67, LUNA 0.62.
  - There was no PE ablation. The authors say the differences reflect whole pretrained systems, not any single mechanism.
  - [arXiv 2609.36288](https://arxiv.org/html/2609.36288)

**Other related designs**
- **Montage-agnostic adapter [SNIP]:** for frozen models, channel embeddings are interpolated from a reference set using 3D coordinates. Evaluated on 8 corpora. [J. Neural Eng. 2026](https://iopscience.iop.org/article/10.1088/1741-2552/ae6142/meta); [PubMed](https://pubmed.ncbi.nlm.nih.gov/41997170/)
- **OmniEEG-Bench [SNIP]:** zero-masks random channels at 20–80% during linear probing. This tests missing sensors, not new positions. [arXiv 2606.00815](https://arxiv.org/html/2606.00815v1)
- **EEG-Bench [SNIP]:** keeps only the pretraining channels and zero-pads any that are missing, so it never tests unseen positions. [EEG-Bench](https://www.researchgate.net/publication/398560481_EEG-Bench_A_Benchmark_for_EEG_Foundation_Models_in_Clinical_Applications)

**Earlier models [BG]: channel handling, not re-verified this session**
- **LaBraM:** learned spatial embeddings looked up by 10-20 channel name, so a truly new electrode name has no embedding. [arXiv 2405.18765](https://arxiv.org/abs/2405.18765)
- **BIOT:** learned channel-token embeddings, flattened per channel. [arXiv 2305.10351](https://arxiv.org/abs/2305.10351)
- **CBraMod:** ACPE, a convolutional conditional PE whose behaviour depends on channel ordering and neighbourhood rather than coordinates. [arXiv 2412.07236](https://arxiv.org/abs/2412.07236)
- **EEGPT:** learned channel embeddings over a fixed 58-electrode vocabulary. [EEGPT, NeurIPS 2024](https://openreview.net/forum?id=lvS2b8CjG5)
- In all four, unseen electrode names require new or learned embeddings, which is consistent with the coordinate-based designs being promoted for montage transfer.

### Inferences
- In EEG, the coordinate-based approach is now standard, but the evidence for zero-shot generalization to new electrodes is thin.
  - Only NOMAD-EEG tests genuinely unseen positions with a position-shuffle control.
  - REVE's ablation shows only that learned and Fourier PE are equal on seen montages. Its stated advantage on unseen montages rests on argument, not a measured held-out test.
- REVE's position-noise result (removing jitter costs about 0.05 balanced accuracy, the largest Table 19 effect) suggests that regularizing the coordinate input matters more than the PE family. This parallels a smoothness / effective-resolution argument for iEEG, though no study frames it that way.
- Scalp EEG is spatially low-pass (volume conduction), so a coarse coordinate code may be enough. That may explain why position permutations barely hurt (LaBraM −1.2 points; NOMAD AUROC −0.033). Transfer of these conclusions to iEEG, where spatial specificity is much finer, should not be assumed.

### Gaps
- I could not verify the following this session:
  - MMM (topology-agnostic EEG with region-level latent tokens).
  - CSBrain.
  - "Dong et al."
  - Dedicated channel-interpolation pretext-task papers.
  - The full text and numbers of BrainOmni's unseen-device results (PDF unreadable).
  - The remainder of REVE and NOMAD-EEG beyond ~100k characters.
- The three Yuce & Stober PE strategies not named in the abstract are still unknown, as are its numbers. Only the abstract was read.

---

## 2. Spiking / Neuropixels / LFP models: how a never-seen unit or channel is handled

### Takeaway
Spiking models almost all **re-learn identity embeddings for new units or sessions (few-shot, gradient-based) rather than inferring them zero-shot from position**. Examples:
- POYO / POYO+ learn new unit-embedding rows and a session embedding.
- NDT2 uses session/subject context tokens.
- NEDS and BrainWideBench rely on calibration per target session.

Zero-shot "unseen neuron" work runs the other way. It infers the neuron's identity or location (region, cell type, 3D coordinate) from its activity, as in NuCLR and the LFP localization models, rather than using position as an input. NDT3 explicitly lists sensitivity to input-sensor variability as a failure mode.

### Cited Findings
- **POYO (NeurIPS 2023) [SNIP/FT-pdf snippet]:** for a new session, "freeze all existing weights of the model and simply add new rows to the UnitEmbed(·) lookup table for each of the new units, as well as a new session embedding" ("unit identification"), trained by gradient descent. [POYO arXiv 2310.16046](https://arxiv.org/pdf/2310.16046); [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/file/8ca113d122584f12a6727341aaf58887-Paper-Conference.pdf)
- **POYO+ [SNIP]:** an unseen unit has no embedding row. The approach is to continue pretraining from the checkpoint on that session, creating and fitting its unit and session embeddings. [as described in NeuroLens arXiv 2610.02864](https://arxiv.org/html/2610.02864)
- **NeuroLens (2026) [SNIP]:** for held-out days, freezes the backbone and read-out and trains only the new unit/session embedding rows plus a new decoder head (lr 3e-4, weight decay 0.05, batch size 16, at most 500 epochs), reusing POYO's few-shot recipe. [arXiv 2610.02864](https://arxiv.org/html/2610.02864)
- **BrainWideBench (Sep 2026; IBL, 276 regions, 139 mice) [ABS + SNIP]:**
  - Benchmarks pretraining and zero-shot transfer to unseen animals.
  - Pretraining beats single-session baselines, but the gain depends on how well the objective matches the task.
  - Pretrained models such as POYO+ still need target-session calibration: unit/session embeddings or session-specific stitchers.
  - Snippet: NDT2 and NEDS were at or below a single-session linear baseline on most tasks.
  - [arXiv 2609.22064](https://arxiv.org/abs/2609.22064); [PDF](https://arxiv.org/pdf/2609.22064)
- **NDT2 [SNIP]:** a spatiotemporal transformer over channel-patch tokens plus learned session/subject/task context embeddings. A new session needs a new context embedding, i.e. fine-tuning. [NDT2 NeurIPS 2023 PDF (as returned by search)](https://papers.neurips.cc/paper_files/paper/2023/file/fe51de4e7baf52e743b679e3bdba7905-Paper-Conference.pdf)
- **NDT3 [SNIP]:** pretrained on up to 2,000 h of monkey and human intracortical data. The authors report failures in "robustness to input sensor variability" and output stereotypy. [source as returned by search, PMC11302672](https://pmc.ncbi.nlm.nih.gov/articles/PMC11302672)
  - Caveat: the link between this PMC ID and NDT3 was not confirmed.
- **NEDS (2025) [SNIP]:**
  - Pretrained on 73 IBL animals and fine-tuned on 10 held-out animals, it beat POYO+ and NDT2.
  - Its latents predict each session's brain regions without explicit training, though this was shown at session level, not per neuron.
  - [arXiv 2504.08201](https://arxiv.org/html/2504.08201v2)
- **NuCLR (NeurIPS 2025) [ABS]:**
  - Contrastive, permutation-equivariant learning of neuron identity from population context.
  - Brain region and cell type are linearly decodable.
  - It shows "strong zero-shot generalization to unseen animals", and performance improves with more pretraining animals.
  - The abstract does not say electrode position is used as an input.
  - [arXiv 2512.01199](https://arxiv.org/abs/2512.01199)
- **LFP localization (Neuropixels/microelectrode, NeurIPS 2025) [SNIP]:** self-supervised pretraining on raw single-channel LFP. It reports zero-shot brain-region prediction in novel subjects. [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/119571)
- **LFPformer (NeurIPS 2026) [ABS]:**
  - Extends wav2vec 2.0 with probe geometry (channel spacing plus estimated insertion trajectory) as a self-supervisory signal, without needing histology.
  - Evaluated on 3D coordinate regression and region classification across labs, species and probe types.
  - Its localization errors are "too large to replace histology". No numbers were on the page.
  - [NeurIPS 2026 poster 150561](https://neurips.cc/virtual/2026/poster/150561)
- **Learned data-driven coordinates (cross-reference, intracranial) [SNIP]:** a 32-D per-channel coordinate is computed from LFP alone by a contrastive CNN over 20 subjects, with 52.84% held-out-channel k-NN accuracy. [NeurIPS 2026 listing](https://neurips.cc/virtual/2026/157293)
  - Cross-reference only; this is in the iEEG researcher's scope.
- **Functional embeddings for multi-area sEEG (cross-reference, intracranial) [SNIP]:** zero-shot transfer to held-out channels. [arXiv 2510.27090](https://arxiv.org/abs/2510.27090)
  - Cross-reference only; this is in the iEEG researcher's scope.
- **Not found:** no source for "Javadzadeh et al." by name. The two works above may be what was meant, but this is unconfirmed.

### Inferences
- In spiking models, unit identity is treated as a nuisance variable that is learned per session. No major spiking foundation model found here derives a new unit's embedding from its probe depth or atlas coordinate.
- Activity-derived identity, as in NuCLR and functional-embedding work, is the main zero-shot alternative. It is the mirror image of feeding position in as input.
- This suggests two design options for iEEG:
  - Hybrid: initialize a new channel's embedding from coordinates, then refine it with a few-shot gradient step (POYO-style).
  - Infer the channel embedding from the signal itself.

### Gaps
- Neuroformer: not checked.
- Whether POYO+ / POYO-3 or any IBL model ever conditions unit embeddings on probe/atlas coordinates: none found.
- NDT3's exact failure analysis numbers: not retrieved.
- Full BrainWideBench zero-shot protocol and numbers: not retrieved.

---

## 3. Held-out-channel designs where position is the only clue; results vs encoding granularity or neighbour density

### Takeaway
**No study found outside iEEG varies position-encoding granularity against data amount and reports an optimum**:
- Granularity here means Fourier frequency count, coordinate resolution or region size.
- REVE ties nfreq (4 or 5) to model size without isolating it.
- NOMAD-EEG fixes 8 octaves.
- LUNA and HEAR never ablate frequency count.

Neighbour density has only been varied indirectly, through channel-subsampling curves (REVE Table 21; NOMAD-EEG Mumtaz; Steglich et al. farthest-point subsets). These show smooth degradation: graceful for motor imagery, steep for speech.

### Cited Findings
- **Position-only held-out electrode designs:**
  - NOMAD-EEG: CPz/POz unseen in BCIAUT-P300 (AUROC 0.718); 45 of 64 EEGMMIDB positions unseen (balanced accuracy 0.442); position-shuffle control (AUROC 0.942→0.909, 17.5% of predictions flipped). [NOMAD-EEG](https://arxiv.org/html/2610.35246v1)
  - HEAR: disjoint-layout transfer, about 0.72 / 0.71 accuracy vs EEGPT 0.63. [HEAR](https://arxiv.org/html/2510.12515)
- **Granularity:**
  - REVE: nfreq ∈ {3, 4, 5}, assigned by model size, with no separate ablation. [REVE](https://arxiv.org/html/2510.21585)
  - NOMAD-EEG: k = 0..7 octaves fixed, with no ablation found. [NOMAD-EEG](https://arxiv.org/html/2610.35246v1)
- **Density / subsampling curves:**
  - REVE-Base, PhysionetMI: 0.824 at 64 channels, 0.781 at 8, 0.660 at 1.
  - REVE-Base, Speech: 0.565 at 64 channels, 0.294 at 8, 0.209 at 1.
  - [REVE](https://arxiv.org/html/2510.21585)
  - NOMAD-EEG, Mumtaz (AUROC): 0.942 at 19 electrodes, 0.919 at 10, 0.891 at 5. [NOMAD-EEG](https://arxiv.org/html/2610.35246v1)
  - Steglich et al.: farthest-point subsets of 10–128 channels. Within-subject cosine similarity starts near 0.75 at small subsets (near 0.5 for CBraMod) and rises toward 1 as channels are added. [arXiv 2609.36288](https://arxiv.org/html/2609.36288)
- **Related granularity knobs:**
  - REVE's masking radii (3 cm spatial, 4 cm dropout) set a spatial scale for the pretext task. Only the mask ratio, not the radius, appears in Table 18. [REVE](https://arxiv.org/html/2510.21585)
  - Steglich et al. found that ROI pooling made models more montage-robust than global attention pooling (REVE-ROI vs REVE-A). This is region-size-like aggregation, but region size itself was not swept. [arXiv 2609.36288](https://arxiv.org/html/2609.36288)

### Inferences
- A sweep of density × resolution (e.g. Fourier frequency count or atlas parcel size × number of training subjects or channels) would fill a gap that also exists in the scalp EEG/MEG literature. No precedent with an optimum was found.
- Scalp evidence on spatial scale exists only as fixed design choices: the 3–4 cm REVE masking radii and the 2^7·π top frequency in NOMAD-EEG. These could serve as baselines.

### Gaps
- MMM's region-count choices and BrainOmni's sensor-encoder ablations may contain granularity-like results but were not verified.
- The tails of the REVE and NOMAD-EEG appendices (the remaining ~35–60k characters each) were not read.

---

## 4. Reported failure modes: model ignores position, volume-conduction shortcut, montage leakage

### Takeaway
The best-documented failure is **position is encoded but not used**: permuting LaBraM's positional codes costs about 1.2 points, and shuffling NOMAD-EEG positions costs about 0.03 AUROC. Other documented failures:
- Sensitivity to montages absent from pretraining (LUNA on SEED-V).
- Attention-pooling readouts that are fragile across montages (REVE-A).
- Poor robustness to sensor variability in spiking models (NDT3).

No source found here explicitly documents a volume-conduction shortcut or montage-to-dataset leakage in a held-out-channel test.

### Cited Findings
- **Position ignored:**
  - LaBraM positional-code permutation: embeddings heavily displaced, but motor-imagery accuracy only −1.2 points. "Electrode identity can be encoded without affecting classification." [arXiv 2610.29367](https://arxiv.org/abs/2610.29367)
  - NOMAD-EEG position shuffle: AUROC −0.033, 17.5% of predictions changed. [NOMAD-EEG](https://arxiv.org/html/2610.35246v1)
  - DIVER-0 discards channel identity entirely and stays at about 59% balanced accuracy on FACED regardless of permutation. On that task, explicit channel identity was apparently not needed. [DIVER-0](https://arxiv.org/html/2507.14141)
- **Unseen-topology sensitivity:** LUNA trails CBraMod on SEED-V (37.3–39.2% vs 40.9% balanced accuracy). The authors attribute this to learned-PE reliance and pretraining montages limited to "three dominant layouts". [LUNA](https://arxiv.org/html/2510.22257)
- **Readout fragility:** with matched-sample probes, cross-montage drops were pronounced for LUNA and REVE-A and minimal for CBraMod and REVE-ROI. [arXiv 2609.36288](https://arxiv.org/html/2609.36288)
- **Task-dependent PE choice:** SPE is good for motor imagery and poor for emotion recognition, while ACPE is more consistent. [arXiv 2605.29754](https://arxiv.org/abs/2605.29754)
- **Spiking:**
  - NDT3 authors cite failure on "robustness to input sensor variability". [PMC11302672 (as returned by search)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11302672)
  - Pretrained NDT2/NEDS were at or below single-session linear baselines on many BrainWideBench tasks [SNIP]. [arXiv 2609.22064](https://arxiv.org/pdf/2609.22064)
- **Montage leakage by design:** EEG-Bench zero-pads to the pretraining channel set, so the montage is fixed and unseen positions are never tested. [EEG-Bench](https://www.researchgate.net/publication/398560481_EEG-Bench_A_Benchmark_for_EEG_Foundation_Models_in_Clinical_Applications)

### Inferences
- The fact that position is barely used is consistent with volume conduction in scalp EEG. Neighbouring channels carry largely redundant signal, so the model can do well while ignoring exact position. This is an inference; no paper here tests it.
- For a held-out-electrode iEEG design, a position-shuffle control (NOMAD-EEG style) and a code-permutation probe (Avramidis-style) are cheap, published precedents for showing the model actually uses position.

### Gaps
- No explicit study was found of a volume-conduction shortcut (e.g. predicting a held-out channel by copying its nearest neighbour) in scalp EEG/MEG.
- No explicit study was found of montage leakage, where montage identity predicts dataset or label.
- A "negative-control protocol for clinical EEG benchmarks" is cited in Steglich et al.'s references but was not retrieved.
