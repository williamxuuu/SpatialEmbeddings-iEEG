# Learned identity/positional embeddings: (a) spontaneous recovery of geometry, (b) role in optimization (plateaus, symmetry breaking)

Scope note: "Verified" means I read the claim in the source's own text (abstract, HTML or ar5iv full text) in this session. "Snippet" means it came from search-result summaries only. "Impression" means it comes from my prior knowledge and was not checked in this session. Impressions are kept out of Cited Findings and listed under Inferences or Gaps.

## (a) Do learned identity/positional embeddings, given no geometry, recover spatial structure, and how is that probed?

### Takeaway
Yes, in vision and language this is well documented. Randomly initialised learned position embeddings come to encode the input topology. ViT's patch embeddings recover 2D row/column structure, and GPT-2's 1D embeddings recover absolute position. The standard probes are: cosine-similarity maps between embeddings, linear-regression probes onto the true coordinate, logistic probes for pairwise order, and SVD/PCA spectra. In neural data, POYO+, NEDS and PHASE report that brain region or cell type can be decoded from learned representations without any anatomical labels. However, I found no iEEG paper that probes per-electrode learned ID embeddings (one vector per channel) against MNI coordinates or physical distance. That specific analysis looks open.

### Cited Findings
**Vision: ViT (Dosovitskiy et al., ICLR 2021). Verified, ar5iv full text.**
- At initialisation, "the position embeddings at initialization time carry no information about the 2D positions of the patches". After training, the model "learns to encode distance within the image in the similarity of position embeddings, i.e. closer patches tend to have more similar position embeddings. Further, the row-column structure appears; patches in the same row/column have similar embeddings." Appendix D adds "a sinusoidal structure is sometimes apparent for larger grids." The probe is a per-patch cosine-similarity map of learned 1D position embeddings (Fig. 7 centre, ViT-L/32). — [Dosovitskiy et al. 2020, arXiv 2010.11929](https://ar5iv.labs.arxiv.org/html/2010.11929)
- Ablation (Appendix D.4, Table 8; ViT-B/16, ImageNet 5-shot linear): no positional embedding 0.6138, learned 1D 0.6421, learned 2D 0.6400, relative 0.6403. The paper concludes "little to no difference between different ways of encoding positional information" once some PE is present. The large gap is between no PE and any PE, and hand-crafted 2D geometry adds nothing over learned 1D IDs. This is the closest vision analogue to "learned electrode ID vs given coordinates". — [same](https://ar5iv.labs.arxiv.org/html/2010.11929)

**Language: Wang & Chen, EMNLP 2020, "What Do Position Embeddings Learn?". Verified, ar5iv full text.**
- Linear-regression probe from embedding to absolute position index, MAE: sinusoid 0.0, GPT-2 1.03, RoBERTa 6.06, BERT 34.14. — [Wang & Chen 2020](https://ar5iv.arxiv.org/html/2010.04903)
- Logistic-regression probe for the relative order of a position pair, error rate: GPT-2 1.56%, sinusoid 5.08%, RoBERTa 7.23%, BERT 19.72%. — [same](https://ar5iv.arxiv.org/html/2010.04903)
- Cosine-similarity heatmaps: GPT-2 shows periodic structure. In BERT, embeddings "are similar to the positions nearby but have no explainable patterns in long-term relations", with a discontinuity at position 128 that the authors attribute to two-stage pretraining. SVD shows BERT's embeddings need many components, which the authors read as carrying information beyond position. They conclude that the training objective shapes what is learned: MLM encoders learn mostly local position, autoregressive GPT-2 learns absolute position. — [same](https://ar5iv.arxiv.org/html/2010.04903); [ACL Anthology](https://aclanthology.org/2020.emnlp-main.555); code [MiuLab/PE-Study](https://github.com/MiuLab/PE-Study)
- Skip-position attack (indices multiplied by 4): GPT-2 Wikitext-103 perplexity 25.83 to 25.74, BERT 12.45 to 323.12. — [same](https://ar5iv.arxiv.org/html/2010.04903)

**Transformers without any PE still learn position. Snippet.**
- Causal LMs with no positional encoding (NoPE) still encode positional information, because causal masking leaks it across stacked layers. — [Haviv et al., "Transformer Language Models without Positional Encodings Still Learn Positional Information"](https://www.researchgate.net/publication/372921406_Transformer_Language_Models_without_Positional_Encodings_Still_Learn_Positional_Information)
- NoPE can outperform explicit PEs on length generalisation. — [Kazemnejad et al. 2023, arXiv 2305.19466](https://arxiv.org/pdf/2305.19466)
- Caveat for the sEEG model: a bidirectional encoder over a set of electrodes has no causal mask, so this leakage route does not exist there (see Inferences).

**Abrupt learning: positional-embedding structure appears after the loss drop. Verified, arXiv HTML.**
- In BERT-style matrix completion (Gopalani, Lubana & Hu 2024), learned positional embeddings form clusters that reflect the matrix's row/column structure. "The clustering appears only after the sudden drop (step 20000 and after)", and "the positional embeddings show clustering some time after the sudden drop in loss has occurred at step 15000." Geometry in the learned PEs therefore lags the loss transition. Probe PE geometry at several checkpoints, not only at the end. — [Gopalani et al. 2024, arXiv 2410.22244](https://arxiv.org/html/2410.22244)

**Neural population models**
- POYO (Azabou et al., NeurIPS 2023): each spike token is a learnable unit embedding plus a spike time. The unit embedding is the object to probe. Snippet. — [arXiv 2310.16046](https://arxiv.org/pdf/2310.16046.pdf)
- POYO+ (Azabou et al., ICLR 2025): reports that latent embeddings carry meaningful structure about brain region and cell sub-type even though the model was never trained to separate them. Snippet; the exact probing method was not checked. — [POYO+ (NSF PAR)](https://par.nsf.gov/servlets/purl/10652733); [project page](https://poyo-plus.github.io)
- A sparse-autoencoder interpretability study of POYO+ latents on Allen Brain Observatory data. Snippet. — [arXiv 2506.14014](https://arxiv.org/pdf/2506.14014v1.pdf)
- NEDS: learned neuron embeddings are "highly predictive of brain region" without training on region labels. Snippet. — [arXiv 2504.08201](https://arxiv.org/pdf/2504.08201.pdf)
- NuCLR ("Know Thyself by Knowing Others"): learns neuron identity from population context, then linearly decodes cell type and region from frozen representations. Snippet. — [arXiv 2512.01199](https://arxiv.org/html/2512.01199v1)

**iEEG**
- PHASE (Zhang, ..., Roychowdhury; arXiv 2609.36087). Verified, HTML.
  - The model uses no electrode coordinates and no channel IDs. PHASE-T encodes each channel independently with a shared encoder, so it learns no per-electrode embedding.
  - Anatomy is probed with linear probes on frozen features (Omni-iEEG, macro-F1): Lobe-5 0.4822 vs BrainBERT 0.4054; Region-12 0.3398 vs BrainBERT 0.2592.
  - Region-averaged frozen embeddings matched their counterpart region across institutions (UCLA n=32, Detroit n=135) in 14 of 16 cases, against about 2 expected by chance (permutation p = 5e-4). Region pairs closest in embedding space are anatomical neighbours (motor/somatosensory, parietal/occipital), noted qualitatively only.
  - The paper does not correlate embedding distance with physical distance for anatomy.
  - This is an "activity-derived embedding encodes anatomy" result, not a "learned ID embedding encodes anatomy" result.
  - [PHASE](https://arxiv.org/html/2609.36087v2)
- Population Transformer (PopT; Chau et al., ICLR 2025) adds each electrode's 3D position to its vector, so geometry is given, not learned. I did not find a PE-removal ablation in the snippets. — [PopT arXiv 2406.03044](https://arxiv.org/html/2406.03044v4)
- A sEEG "functional embeddings" paper (ICLR 2026 submission) learns a subject-agnostic functional identity per electrode with contrastive learning and reports region-structured embeddings. Snippet. — [arXiv 2510.27090](https://arxiv.org/pdf/2510.27090)
- Scalp EEG: learned channel-index embeddings are reported to be montage-specific and not transferable across montages. Workarounds interpolate learned channel embeddings from 3D coordinates. Snippet. — [Montage robustness, arXiv 2609.36288](https://arxiv.org/pdf/2609.36288); [IOP JNE frozen-FM montage-agnostic](https://iopscience.iop.org/article/10.1088/1741-2552/ae6142)
- A CBraMod study compares learnable PE, spherical PE, asymmetric conditional PE and no PE. Snippet; results not read. — [arXiv 2605.29754](https://arxiv.org/html/2605.29754v1)

### Inferences
- A defensible probe battery for per-electrode ID embeddings, adapted from the methods above:
  1. Cosine-similarity matrix vs Euclidean MNI distance, tested with an RSA-style Mantel/Spearman test and a within-subject permutation null.
  2. Cross-validated linear (ridge) regression from embedding to MNI x/y/z, the analogue of Wang & Chen's absolute-position probe.
  3. Logistic probes for hemisphere, lobe and region, the analogue of the PHASE probes.
  4. Pairwise "which is closer" order probes, the analogue of Wang & Chen's relative-order probe.
  5. Probes at multiple checkpoints, since Gopalani et al. show PE structure emerging after the loss drop.
- Key confound in sEEG: contacts on the same shaft are both physically adjacent and highly correlated in signal. An ID embedding can recover "same shaft" from signal correlation alone. Controls:
  - Partial out the same-shaft indicator.
  - Test the distance relation between shafts only.
  - Compare against a baseline computed from raw signal correlation, to separate "embedding learned anatomy" from "embedding learned signal correlation".
- Multi-subject case: learned IDs are per-subject, so there is no shared coordinate frame. Geometry can only be tested within subject. Cross-subject anatomy alignment would need region-averaged matching as in PHASE.
- Impression (not verified this session): NDT2 (Ye et al., NeurIPS 2023) uses learned context/session/subject embeddings, and I recall no anatomical probe of them. word2vec-style neuron embeddings exist in the connectomics/calcium-imaging literature, but I did not locate a specific iEEG electrode-embedding equivalent.

### Gaps
- I found no paper that probes learned per-electrode ID embeddings in iEEG/sEEG against anatomy or physical distance. This appears to be a genuine gap. Search was limited, so a targeted check of BrainBERT, Brant, Brant-2 and DIVER-1 ([arXiv 2512.19097](https://arxiv.org/abs/2512.19097)) appendices is advisable.
- I did not verify the exact POYO+ method (which embedding was probed: unit embeddings or session latents, and with which classifier).
- No verified EEG paper shows learned channel-index embeddings recovering 10-20 topography, though it is plausible. Only snippets on montage-specificity were found.

## (b) Does positional/identity/side information help optimization (escaping plateaus, breaking symmetry, faster convergence) separately from the final representation?

### Takeaway
There is solid evidence of three kinds:
- Transformer training often sits on a plateau tied to a symmetric or collapsed state (near-parallel token representations, uniform attention) and escapes abruptly.
- Positional embeddings are among the slowest components to escape in at least one setting.
- Explicit PEs "significantly facilitate pretraining convergence" in LLMs and can be removed after training with little loss (DroPE). This is the closest published analogue to "spatial info only early".

The general frame for "side information at training time only" is learning using privileged information (LUPI) and generalized distillation. I found no neural-population or EEG paper showing that channel/location embeddings mainly speed convergence. The early-only and late-only spatial-information test appears novel in this domain.

### Cited Findings
**Plateaus and their mechanism in transformers**
- Gopalani & Hu, "What Happens During the Loss Plateau?" (NeurIPS 2025). Verified, abstract.
  - During the plateau the model forms a partial solution with strong repetition bias and shows "internal representation collapse, where hidden states across different tokens become nearly parallel."
  - "Slow learning of optimal attention maps" is the bottleneck.
  - Directly intervening on attention significantly changes plateau duration.
  - The same signatures appear in early pretraining of Pythia and OLMo.
  - [arXiv 2506.13688](https://arxiv.org/abs/2506.13688)
- Snippet from the same paper: during the early plateau the model shows permutation symmetry in first-layer weight vectors, and leaving that symmetric state produces the loss drop; token hidden states often have cosine similarity above 0.9. — [ar5iv 2506.13688](https://ar5iv.labs.arxiv.org/html/2506.13688)
- Gopalani, Lubana & Hu, "Abrupt Learning in Transformers: A Case Study on Matrix Completion" (NeurIPS 2024). Verified, arXiv HTML.
  - Each component was trained in isolation: "randomly initialize them and freeze the weights of other components to their values at the final step of training".
  - "Positional embeddings show the longest plateau in loss". In that run the sudden drop for PEs alone occurred just before step 40000, versus about step 15000 for full training.
  - "There is no loss plateau for token embeddings and MLP layers."
  - Before the drop the model copies the input; after it, the model predicts the masked entries.
  - This component-isolation protocol transfers directly to testing whether electrode ID embeddings are the bottleneck.
  - [arXiv 2410.22244](https://arxiv.org/html/2410.22244); [NeurIPS 2024 PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/630d293833e09e1ecd892a898a20b074-Paper-Conference.pdf)
- "Distinct mechanisms underlying in-context learning in transformers" (2026). Snippet.
  - Analytical model with RoPE that identifies two symmetry-breaking biases steering abrupt learning.
  - Predicts that formation time scales with sequence length.
  - [arXiv 2604.12151](https://arxiv.org/html/2604.12151)
- Olsson et al., "In-context Learning and Induction Heads" (Anthropic, 2022).
  - A "phase change" early in training in which induction heads form, visible as a bump in the loss curve.
  - The circuit relies on a "previous token head", so it depends on positional information. This dependence is my inference from the mechanism; the paper does not run a PE ablation, as far as the snippet shows.
  - Snippet. [Transformer Circuits](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html); [arXiv 2209.11895](https://arxiv.org/abs/2209.11895)
- Boix-Adserà et al., "Transformers learn through gradual rank increase" (NeurIPS 2023).
  - Proven for diagonal weights with small initialisation: training proceeds in stages with plateaus near saddle points, and the rank of the weight change grows by at most one per stage.
  - Observed empirically in vision and language transformers without those assumptions.
  - Snippet/abstract. [arXiv 2306.07042](https://arxiv.org/abs/2306.07042)
- Saxe, McClelland & Ganguli (ICLR 2014). Abstract-level.
  - Deep linear networks show "long plateaus followed by rapid transitions to lower error solutions".
  - Conserved quantities arise "from symmetries in the error function".
  - Initialisation strongly affects plateau length: random orthogonal initialisation gives depth-independent learning times.
  - [arXiv 1312.6120](https://arxiv.org/abs/1312.6120)
- "Breaking Symmetry When Training Transformers" (2024): without positional encoding, causal attention is needed to break the permutation symmetry, and residual connections help convergence. Snippet; not read. — [arXiv 2402.05969](https://arxiv.org/pdf/2402.05969)

**PEs aid convergence; PE only during training (closest analogue to "spatial info early only")**
- Gelberg, Eguchi, Akiba & Cetin, "Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings" (DroPE, ICLR 2026). Snippet plus the Sakana write-up.
  - Explicit PEs "significantly facilitate pretraining convergence by baking in an important inductive bias that is difficult to recover from data alone".
  - NoPE models are "notoriously unstable to train from scratch".
  - DroPE trains with RoPE, drops it, and recalibrates for under 1% of pretraining compute (with QKNorm).
  - Afterwards the models extrapolate zero-shot with no loss of in-context capability.
  - This is a published "PE early, removed late" experiment supporting PE as mainly an optimization aid.
  - [arXiv 2512.12167](https://arxiv.org/abs/2512.12167); [Sakana blog](https://sakana.ai/drope/); [ICLR 2026](https://iclr.cc/virtual/2026/poster/10009468)
- An associative-memory analysis reports that convergence slowed after removing positional embeddings. Snippet. — [arXiv 2505.19488](https://arxiv.org/pdf/2505.19488)
- ViT: no PE costs about 2.8 points of 5-shot accuracy (0.614 vs 0.642). Verified. — [ar5iv 2010.11929](https://ar5iv.labs.arxiv.org/html/2010.11929)
  - This is a final-performance effect. ViT does not separate optimization from representation.

**Side information at training time only (LUPI)**
- Vapnik & Vashist introduced learning using privileged information: extra features available only at training time. — [Lopez-Paz et al. summary](https://arxiv.org/pdf/1511.03643)
- Lopez-Paz, Bottou, Schölkopf & Vapnik (ICLR 2016), "generalized distillation": a teacher trained on privileged features, a student trained on regular features plus soft labels. — [arXiv 1511.03643](https://arxiv.org/abs/1511.03643)
- Provodin, van den Akker & Pechenizkiy (2024) argue there is little theoretical justification for when LUPI helps. — [arXiv 2408.14319](https://arxiv.org/pdf/2408.14319v1.pdf)

**Neural population / EEG models**
- I found no paper reporting that channel or location embeddings in neural or EEG models mainly speed convergence or rescue plateaued runs. PHASE avoids channel identity entirely (verified above). PopT gives 3D coordinates, but I did not see a convergence ablation.

### Inferences
- The observed loss = ln 2 plateau fits the symmetric/collapsed-state picture. A set encoder with no channel identity is permutation-equivariant over electrodes, so a model might first have to break that symmetry from the signal alone. Spatial (or ID) tokens break it at initialisation. This connects Gopalani & Hu's collapse/symmetry account and the "Breaking Symmetry" paper, but it is my synthesis, not a result shown for iEEG.
- Diagnostics borrowed from Gopalani & Hu to run on plateaued vs escaped runs:
  - Mean pairwise cosine similarity of token hidden states (representation collapse).
  - Attention entropy (near-uniform attention).
  - Repetition (outputs constant across inputs).
- Experiments suggested by the literature:
  1. Spatial info early then dropped (DroPE-style, with a short recalibration).
  2. Spatial info added only after N steps, to see whether a plateaued run escapes on injection.
  3. Random fixed per-electrode tags with no geometry. If these rescue runs as well as true coordinates, the benefit is symmetry breaking or identity, not anatomy.
  4. Gopalani-style component isolation: train only the ID/spatial embeddings with other components frozen.
  5. Shuffled-coordinate control: wrong but consistent geometry.
- The random-tag vs true-coordinate contrast is the cleanest test of "optimization aid vs anatomical prior". I did not find it done in the neural literature.
- Impressions (unverified this session, check before citing):
  - Hoffmann et al. 2023, "Eureka-Moments in Transformers" (arXiv 2310.12956): attribute plateaus to small softmax/attention gradients, and report that a temperature/normalisation fix shortens them.
  - Reddy (ICLR 2024): on abrupt induction-head formation in an in-context classification task.
  - Kirsch et al. 2022 (general-purpose in-context learning): report plateaus reduced by larger batch size.

### Gaps
- No verified neural/EEG/iEEG source separates the optimization effect of spatial embeddings from their representational effect.
- No source found that injects PE only late to rescue a plateaued run. DroPE covers only the early-then-drop direction, and the "Breaking Symmetry" paper's full content was not read.
- "Positional encoding dropout" as a regularizer was not specifically found. Only DroPE turned up.
- The PE-removal ablation in PopT and the convergence curves in POYO/NDT2 were not checked.
