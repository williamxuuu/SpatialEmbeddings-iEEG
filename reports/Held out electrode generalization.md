# Held-out shafts expose what location codes really learn

Nobody has run the proposed experiment. No iEEG decoding or foundation-model paper found trains on some of a person's electrodes and then decodes from electrodes it never saw, using only a location code (MNI coordinates, atlas region or grid cube) to tell the model where they are. No paper in any neural modality crosses the resolution of that code with data density (electrodes per region, subjects) in a held-out-electrode setting. The field's generalization unit is the subject, session or institution. Even "unseen subject" tests fine-tune or probe on labeled data from the new electrodes, so the model never transfers to an electrode it has no data from. The nearest work comes in three pieces, each missing something:

- **Javadzadeh et al.** ran held-out-channel and held-out-region tests in a deep-brain-stimulation (DBS) cohort. Their channel embeddings come from each channel's own signal rather than from location alone, and the "held-out" region is masked only at evaluation time.
- **SuperEEG** predicts held-out electrodes with a Gaussian process in MNI space. Its kernel width is fixed and was never cross-validated, and it reconstructs raw voltage rather than decoding.
- **NOMAD-EEG** tests scalp electrode positions absent from pretraining, but scalp EEG is spatially blurred.

Holding out shafts also changes the meaning of the project's main within-person result. Fine cubes win within a person because they act as electrode identity. A held-out shaft has no identity to look up: at fine resolution its contacts fall into cubes no training electrode occupies. The resolution sweep therefore becomes a contest between coverage and specificity, which is exactly the bias-variance question the project wants to ask, provided the design avoids the leaks others hit. The most urgent is the project's own common-average reference (CAR). It mixes every held-out channel into every training channel unless it is recomputed from training channels only.

## Only one iEEG paper holds out channels, and it cheats on location

Every major iEEG model found holds out units larger than an electrode, and every one lets the model see data from the test electrodes:

| Unit held out | Models |
|---|---|
| Subject | BrainBERT, PopT, BaRISTA, MAPA, Seegnificant, MVPFormer, CORTEG |
| Session | Neuroprobe, BaRISTA, MAPA |
| Institution | PHASE, BrainWave, iMINDBench |

PopT reports that "missing a subject from pretraining does not significantly affect the downstream results", but downstream decoders are still trained on that subject's electrodes ([PopT](https://arxiv.org/html/2406.03044v4)). Seegnificant leaves subjects out and then fine-tunes on them ([Seegnificant](https://arxiv.org/pdf/2411.10458)). CORTEG does the same, and uses leave-one-channel-out only as a post-hoc importance analysis ([CORTEG](https://arxiv.org/html/2605.10337)).

Neuroprobe, the benchmark the project uses, has within-session, cross-session and cross-subject splits but **no electrode or region hold-out split** ([Neuroprobe](https://arxiv.org/html/2509.21671v2)).

BrainBERT's claim of improvement on "held-out electrodes" means held out in time. Pretraining included every Laplacian-capable electrode of all ten subjects ([BrainBERT](https://arxiv.org/pdf/2302.14367)). BaRISTA's masked-channel reconstruction (R² 0.609 for parcel encoding vs 0.603 for coordinate encoding) also masks channels the model saw in pretraining. It runs on Laplacian-referenced data, in which a masked contact's signal is mixed with its unmasked shaft neighbours. The result therefore cannot separate "the location code generalizes" from "the neighbours leak the target" ([BaRISTA](https://arxiv.org/pdf/2512.12135)).

The single genuine precedent is the functional-embedding work of **Javadzadeh et al.** It covers 20 dystonia DBS patients with sEEG in the basal ganglia and thalamus.

- **Held-out-channel test.** A contrastive CNN maps 10-s LFP segments to 32-D embeddings. Held-out channels reach **49.18% ± 12.11% k-NN region accuracy**, against 80.71% for held-out time segments from seen channels. The authors attribute the gap to "the limited number of electrodes per region within a subject" ([arXiv 2510.27090](https://arxiv.org/pdf/2510.27090)). That is a density explanation they never tested, and it is the hypothesis this project can test.
- **Held-out-region test.** All VO channels were withheld and reconstructed from GPi, STN and VIM. Learned functional coordinates beat MNI coordinates (p ≈ 0.002, 11 subjects). The MNI failure is instructive: "the four VO electrodes often shared nearly identical or uncertain MNI coordinates", so the model produced near-identical waveforms for every VO target ([arXiv 2510.27090](https://arxiv.org/pdf/2510.27090)). A workshop version reports 52.84% held-out-channel accuracy, and finds that learned coordinates beat region labels and MNI coordinates in pooled transformers ([NeurIPS 2026 workshop](https://neurips.cc/virtual/2026/157293)).

Two features keep this from pre-empting the project. First, the held-out channel is embedded from *its own signal*, so location is not the only clue. Second, the VO channels appear in the same subjects' training sessions under a chronological 70/15/15 split, so the region is held out at evaluation, not in training. The paper also reports no referencing scheme, no sweep of embedding dimension and no data sweep.

**MAPA** holds an accidental unseen-region case: 4 of its 74 DKT rows appear only at evaluation and stay untrained. The paper does not analyse them ([MAPA](https://arxiv.org/pdf/2609.13507)). One abstract-level paper, Spatially Masked Regression, varies an exclusion radius around held-out targets. It finds "substantial residual predictability when local neighbors are excluded", and finds iEEG relationships more local and subject-specific than scalp EEG ([arXiv 2606.11415](https://arxiv.org/abs/2606.11415)). It uses no spatial encoding.

On the second half of the question, **no study sweeps location-code granularity against data amount in any held-out setting**:

- BaRISTA varied encoding scale at fixed data and ran its data sweep only on the best configuration ([BaRISTA](https://arxiv.org/html/2512.12135)).
- MAPA uses one DKT table.
- CORTEG fixes its Fourier frequencies (F = 32) and k = 8 ([CORTEG](https://arxiv.org/html/2605.10337)).
- Seegnificant's RBF basis has a tunable resolution that its main text never sweeps ([Seegnificant](https://arxiv.org/pdf/2411.10458)).

## EEG, MEG and spiking models dodge the question differently

In scalp EEG, montage flexibility now comes from computing channel embeddings from 3-D coordinates instead of looking them up by channel name. Rigorous unseen-position tests are still rare.

**NOMAD-EEG** is the cleanest "position is the only clue" design. Electrodes are identified only by Fourier features of their scalp coordinates. Results on electrode positions absent from pretraining:

- BCIAUT-P300, where CPz and POz are unseen: AUROC 0.718, best of five models.
- EEGMMIDB, where 45 of 64 positions are unseen: frozen balanced accuracy 0.442 against chance 0.25.

It also ran the control almost nobody else runs. Shuffling positions while keeping the signals lowered AUROC only from **0.942 to 0.909** and changed 17.5% of predictions ([NOMAD-EEG](https://arxiv.org/html/2610.35246v1)).

Other models show similarly weak reliance on position:

- Permuting LaBraM's positional codes displaced its embeddings heavily but cost only **1.2 points** of motor-imagery accuracy ([Avramidis et al.](https://arxiv.org/abs/2610.29367)).
- DIVER-0 discards channel identity altogether and stays near 59% on FACED under every permutation ([DIVER-0](https://arxiv.org/html/2507.14141)).
- HEAR fine-tunes on one electrode layout and tests on a disjoint one, scoring 0.72 against EEGPT's 0.63 ([HEAR](https://arxiv.org/html/2510.12515)).

REVE gives the most transferable lesson. In its ablation, removing *position noise* (jittering coordinates during training) dropped average balanced accuracy from **0.707 to 0.660**, a larger drop than swapping its Fourier position encoding (PE) for a learnable one (0.701). Its footnote concedes that learnable PE "does not allow for the extension to … unseen spatial configurations" ([REVE](https://arxiv.org/html/2510.21585)). Regularizing the coordinate input mattered more than the choice of PE family. That is an effective-resolution effect, although no paper frames it that way.

None of these models ablates its Fourier frequency count against data. REVE ties the count to model size, and NOMAD-EEG fixes eight octaves. Scalp EEG is also spatially low-pass, so its weak reliance on position should not be assumed to carry over to sEEG.

Spiking models avoid the problem by re-learning identity. POYO freezes the network and adds "new rows to the UnitEmbed(·) lookup table for each of the new units" by gradient descent ([POYO](https://arxiv.org/pdf/2310.16046)). POYO+ and NeuroLens follow the same few-shot recipe ([NeuroLens](https://arxiv.org/html/2610.02864)). BrainWideBench finds that pretrained spiking models still need calibration on the target session ([BrainWideBench](https://arxiv.org/abs/2609.22064)).

The zero-shot alternative runs the other way: it *infers* identity or region from activity. Examples are NuCLR ([NuCLR](https://arxiv.org/abs/2512.01199)) and LFP-based localization models, whose errors are still "too large to replace histology" ([LFPformer](https://neurips.cc/virtual/2026/poster/150561)). No spiking foundation model found derives a new unit's embedding from its atlas coordinate.

In speech BCIs, SwinTW gives electrodes with unknown location the coordinate (0, 0, 0) and an "Unknown" ROI. It transfers to unseen participants with mean PCC 0.765, but it holds out no electrodes within a person ([SwinTW](https://pmc.ncbi.nlm.nih.gov/articles/PMC11773629/)).

The adjacent literature therefore offers three strategies for a new electrode: compute its embedding from its coordinate, re-learn it from labeled data, or infer it from its signal. Only the first matches the project's question. The other two are useful comparison arms.

## Interpolation physics says cm-scale prediction must ride on coarse structure

Classical spatial statistics sets a hard ceiling. Kriging on dense µECoG found that predicting fields to within 10% error requires **sub-millimetre spacing at 75–300 Hz in every species and cortical area**. At 1.52 mm spacing, only **42.7% of human pSTG fields** were predictable, against 91.8% in motor cortex ([Trumpis et al. 2021](https://rogersgroup.northwestern.edu/files/2021/jnekriging.pdf)).

Macro-ECoG agrees. Gamma and high-gamma envelope and phase are "largely uncorrelated (<90%) at 4 mm", and correlation drops faster across a sulcus than within a gyrus ([Muller et al. 2016](https://escholarship.org/content/qt3sk9z4jc/qt3sk9z4jc.pdf?t=p9fgq8)).

The largest study, McCarty et al. (71 patients, 8752 sites), measured high-gamma full width at half maximum (FWHM):

| Recording | FWHM |
|---|---|
| sEEG gray matter, CAR | **7.14 mm** |
| sEEG gray matter, bipolar | **3.83 mm** |
| sEEG white matter | 11.3 mm |

([McCarty et al. 2022](https://www.eneuro.org/node/3797693.full.print)). Neighbouring sEEG contacts sit 3.5–5 mm apart ([Li et al. 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6495648)), and each samples a sphere of about 5 mm radius ([Mercier et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC10190110)). A held-out shaft is usually several millimetres or more from the nearest training contact, which is beyond the range where high-gamma time series are shared. If a held-out shaft decodes successfully at cm scale, the success must come from coarse functional organization, not from interpolating the local field.

**SuperEEG** is the closest classical analogue to "decode an unseen electrode from location". A Gaussian RBF at each electrode, pooled across 88 patients in MNI space, predicted held-out electrodes with **mean r = 0.51 when the correlation model came from other patients, against r = 0.32 from the same patient's other electrodes** ([Owen et al. 2020](https://rcastoragev2.blob.core.windows.net/b66b54c86f2ff95fdb0d403e12afadc1/PMC7472198.pdf)). Two details bear on this project.

- **The bandwidth choice is the project's trade-off, settled by hand.** The authors chose λ = 20 by heuristic, to balance "spatial resolution (which implies a small value of λ)" against generalizing "without dense electrode coverage (which implies a large value of λ)". They added that it "could in theory be further optimized, e.g., using cross validation."
- **Density did not predict accuracy.** Local electrode density, measured within 20 MNI units, was unrelated to accuracy in one dataset (r = 0.05) and slightly negative in the other (r = −0.21).

A large fixed kernel plausibly flattens any density dependence, but that is an inference, not a finding. SuperEEG also reconstructs z-scored raw voltage, which contains the reference-shared and volume-conducted components that a decoding model should not get credit for.

Group-iEEG mapping chooses resolution by convention, never by validation:

- Recording zones with a geodesic radius in SB-MEMA ([Kadipasaoglu et al. 2014](https://pubmed.ncbi.nlm.nih.gov/25019677/)).
- 38 atlas regions in Frauscher's normative atlas ([Frauscher 2018](https://scholars.duke.edu/publication/1600668)).
- 82 Lausanne regions with site as a mixed effect ([Woodhouse et al.](https://arxiv.org/html/2404.17952v2)).
- Density-map kernels of 2–5 mm, "approximately equal to electrode length" ([Frontiers Neuroinform 2017](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2017.00010/full)).

No iEEG paper found chooses bin or kernel size by cross-validation or as a function of electrode density.

Nonparametric theory predicts what such a sweep should show. Partitioning estimators are consistent when cells shrink while the count per cell grows (h → 0, n·h^d → ∞) ([Györfi lecture notes](https://www.szit.bme.hu/~gyorfi/booknonpar.pdf)). Balancing bias of order h² against variance of order 1/(n·h^d) gives h\* ∝ n^(−1/5) in 3-D. Doubling the electrode count should therefore shrink the optimal cube side by only about 13%. The resolution axis needs a wide density range to show any movement at all.

## The pitfalls cluster around references, neighbours and the test input

The table summarizes each pitfall's mechanism, evidence and severity for this design. The paragraphs below give the details.

| Pitfall | Mechanism | Evidence | Severity for this design |
|---|---|---|---|
| CAR leakage | Every held-out channel enters every training channel at weight 1/N | Mercier 2017; McCarty 2022 | **High**: current recipe uses CAR |
| Laplacian/bipolar at single-contact hold-out | Neighbours n±1 carry the held-out contact at weight −½ | Li 2018; BrainBERT | Low if whole shafts are hidden |
| Volume conduction / near shafts | Neighbouring shafts can pass within ~5 mm; white matter carries diffuse signal | Mercier 2017 | Medium |
| Random vs blocked CV | Nearby points act as pseudoreplicates | Ploton 2020; Roberts 2017 | Medium |
| Unseen cells | Fine cubes on held-out shafts have no training occupants | MAPA; Javadzadeh | **High**: determines the result |
| Localisation error | Sub-mm to ~2 mm error; 82% label agreement | Taylor 2021 | Sets a ~5 mm floor |
| Test-time input | Training shafts can carry the decision | No iEEG precedent | **High** |
| Temporal / stimulus leakage | Same time points share stimulus-driven variance | Neuroprobe; Kapoor 2023 | Medium |
| Electrode pre-selection | Neuroprobe ranks electrodes by linear decodability | Neuroprobe | Medium |
| Few subjects | 6–10 subjects; clustered contacts | Li & Tao 2026 | High for claims |

**Re-referencing is the pitfall that most directly threatens the current recipe.** Neuroprobe's baselines use a shaft Laplacian: each contact minus the mean of its two neighbours on the same shaft ([Neuroprobe](https://arxiv.org/html/2509.21671v2); [BrainBERT](https://arxiv.org/pdf/2302.14367)). This never crosses shafts, so whole-shaft hold-out is clean under Laplacian or bipolar referencing, while single-contact hold-out is not. CAR is the opposite case. Every training channel's signal contains every held-out channel at weight 1/N. CAR also introduces anti-correlations between channels ([Mercier et al. 2017](https://cerco.cnrs.fr/wp-content/uploads/2019/04/mercier_m_17_1.pdf)) and nearly doubles the apparent high-gamma spread relative to bipolar (7.14 vs 3.83 mm) ([McCarty et al. 2022](https://www.eneuro.org/node/3797693.full.print)). Inter-channel correlation falls from r = 0.31 under monopolar referencing to 0.06 under the Laplacian ([Li et al. 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6495648)). MAPA is the only paper found that states explicit hygiene: "No excluded contact enters an evaluation, not even through the array mean re-reference" ([MAPA](https://arxiv.org/pdf/2609.13507)). No paper quantifies how much reference mixing inflates held-out-channel scores, so the project should measure it.

**Volume conduction and spatial autocorrelation** call for blocked validation and a distance axis. White-matter contacts record volume-conducted gray-matter activity whose amplitude falls roughly as 1/distance². Only 42% of contacts with a gray-matter centroid are entirely surrounded by gray matter ([Mercier et al. 2017](https://cerco.cnrs.fr/wp-content/uploads/2019/04/mercier_m_17_1.pdf)). Ecology shows what happens when this is ignored. Random CV on a forest-biomass map gave R² = 0.53, spatial CV gave 0.14, and buffered leave-one-out fell to virtually zero beyond the roughly 120 km autocorrelation range. The authors' rule is that "an observation point cannot serve as spatially independent validation of nearby training data points" ([Ploton et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7486894)). Roberts et al. add the opposite caveat: blocking can push the test into extrapolation and overstate interpolation error ([Roberts et al. 2017](https://www.biometrie.uni-freiburg.de/mitarbeiter/dormann/roberts-et-al-2017-ecography.pdf)). For this project, a held-out shaft may pass through a cube that no training contact occupies, and the analysis must keep that case apart from the others.

**Localisation error sets a resolution floor.** Within a subject, automated contact localisation is accurate to about 0.4–0.6 mm against manual raters ([PMC5973981](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5973981/); [PMC10197594](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10197594/)). Atlas and manual labels agreed for only **82% of more than 17,000 SEEG contacts**, and on adjudication the atlas was right 48% of the time, the expert 36% and neither 16% ([Taylor et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8408609)). The earlier report attributed this figure to Brainstorm; it comes from Taylor et al. No source quantifies the error that MNI normalisation adds for sEEG. Combined with the ~5 mm sampling radius, cubes much finer than ~5 mm cannot carry anatomy. A win at that scale signals identity.

**The test input can hide everything.** If held-out shafts are fed together with training shafts, a population model can score well while ignoring the new channels. No iEEG paper found tests on held-out channels alone. Neuroprobe's cross-subject baseline sidesteps the issue by averaging within shared Desikan-Killiany regions and dropping unshared ones ([Neuroprobe](https://arxiv.org/html/2509.21671v2)).

**Temporal leakage is a separate risk.** Neuroprobe's within-session split uses two contiguous halves with no stated gap. Its default cross-session split is described as one that "ensures completely that no data-contamination due to auto-correlation has occurred" ([Neuroprobe](https://arxiv.org/html/2509.21671v2)). A shaft-only hold-out at the same time points lets training shafts carry the stimulus-locked timecourse. That falls under Kapoor and Narayanan's "non-independence" and "preprocessing on train+test" leakage categories ([Kapoor & Narayanan 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856)).

**Electrode selection is a further trap.** Neuroprobe Lite chooses up to 120 electrodes per subject, in whole-probe groups ranked by linear decodability. Held-out shafts drawn from that set are pre-selected to be decodable.

**Few-subject statistics** will limit every claim. A leakage-audited EEG benchmark with 16 participants found that no model survived correction across a 13-model family, and seed variance was substantial ([Li & Tao 2026](https://arxiv.org/abs/2605.00865)). With six subjects, an exact sign-flip test cannot reach p < 0.031.

## How novel the design is, and where reviewers will push

| Component | Status | Closest prior work |
|---|---|---|
| Within-person whole-shaft hold-out with location-only codes | **Novel** in iEEG | Javadzadeh (signal-derived embeddings, DBS, held out at evaluation only); BaRISTA masked channels (seen channels) |
| Cube size × electrodes-per-region sweep under hold-out | **Novel** in any neural modality | SuperEEG fixed λ; BaRISTA single-config data sweep; Javadzadeh untested density explanation |
| Accuracy vs distance to nearest training contact | **Novel** for iEEG at cm scale | Ploton buffered CV (ecology); Trumpis kriging (µECoG, mm scale) |
| Shuffled-cube control under hold-out | **Novel** in iEEG | NOMAD-EEG test-time position shuffle (scalp) |
| Reference-leakage quantification | **Novel** (no measurement found) | MAPA hygiene statement |

Reviewers will raise three challenges. The first is "this is just SuperEEG with a transformer". The reply is that SuperEEG reconstructs voltage with one hand-set kernel, while this project decodes task labels and sweeps the kernel against density. The second is that "the fine-resolution collapse is trivial because empty cells carry no information". That challenge is correct, and the design must answer it by conditioning on coverage (below). The third is that scalp-EEG models barely use position. The reply is that sEEG high-gamma is local at the 4–7 mm scale, so position should matter more in sEEG. The project's within-person gap of about 0.04 AUC between true and shuffled 20 mm cubes is already evidence that it does.

## Concrete design recommendations

**Unit and reference.** Hold out whole shafts, and switch the main analysis from CAR to the shaft Laplacian. Under the Laplacian, whole-shaft hold-out is leak-free by construction, provided that end contacts never borrow a neighbour across a shaft and that skipped contacts stay on the same shaft. Keep CAR as a secondary arm in two variants: (a) CAR over all channels, which reproduces the leak, and (b) CAR recomputed from training channels only, with that same training-channel average subtracted from the test channels. The difference between (a) and (b) is the first published measurement of reference leakage in held-out-channel decoding. Fit z-scoring and spectrogram normalisation on training channels and training time only, and drop or separately report white-matter test contacts.

**Splits.** Cross the shaft hold-out with a temporal hold-out, preferably Neuroprobe's cross-session (different-movie) split. Then a test sample is new in both space and time. Within each subject, draw many random shaft-split assignments, stratified so that held-out shafts are not systematically the most decodable ones. Draw them from all shafts rather than from Neuroprobe Lite's pre-ranked 120, or report the selection explicitly. Add a buffered variant that also removes training contacts within X ∈ {0, 5, 10, 20} mm of any test contact. X = 10 mm clears the CAR high-gamma FWHM, and X = 20 mm reaches the scale of the project's own 20 mm cubes.

**Test input.** Score three conditions:

- (a) training shafts only;
- (b) training plus held-out shafts;
- (c) held-out shafts only.

The headline result is (c), and (b − a) is the marginal value of the new shafts. Run (c) under four location assignments for the held-out shafts: true cubes, consistently shuffled cubes, a constant "unknown" code, and Gaussian coordinate jitter at σ = 2–5 mm. The shuffle is a fixed permutation of cell labels, so held-out contacts receive the embedding of some other *trained* cell. This makes the shuffle the clean held-out analogue of the project's within-person 0.04-AUC control, and the jitter tests sensitivity to localisation error.

**Unseen cells.** Decide, and report, what an unoccupied cube gets: its random initialization, an "unknown" row, or the code of its nearest occupied cell. Note that weight decay drives untrained rows toward zero. Then add at least one encoding that degrades gracefully, so that the resolution curve measures more than coverage. Options include a hierarchical code (cube plus parent cube plus lobe, with dropout toward the parent), soft or kernel-weighted cube assignment, or Fourier coordinates with a swept maximum frequency. REVE's position-noise result says training-time coordinate jitter is cheap and strong regularization ([REVE](https://arxiv.org/html/2510.21585)).

**Sweep axes and reporting.** Sweep cube size from 5 mm up to whole-lobe scale. Nothing below 5 mm: localisation and sampling radius make finer cubes identity codes. Use the 20 mm cubes as the anchor, since their within-person advantage over shuffled cubes is already established. Vary density in two independent ways. The within-person way subsamples the training shafts. The cross-person way adds other subjects' electrodes to the same cubes, which tests SuperEEG's finding that other patients' electrodes predict better than the patient's own (r = 0.51 vs 0.32). Express the x-axis as training electrodes per occupied cell, and report three quantities for every cell of the sweep:

- the fraction of held-out contacts whose cube is occupied in training;
- accuracy for occupied-cube and empty-cube contacts separately;
- accuracy against distance to the nearest training contact.

The theoretical prediction to test is h\* ∝ n^(−1/5), so make the density range span at least a factor of 8.

**Baselines and comparison arms.** Include:

- a cube-code-only model with no signal (a pure location prior);
- an audio-envelope baseline for the speech/non-speech task;
- a signal-inferred embedding arm in the style of Javadzadeh, which uses unlabeled test-shaft signal;
- a few-shot arm in the style of POYO, which re-learns the held-out shaft's rows from a small labeled budget.

Together these place location-only transfer between "no location" and "calibrated identity".

**Statistics.** Average over splits and seeds within each subject first. Then run an exact sign-flip or permutation test across subjects, and treat the whole resolution × density grid as one model family for multiplicity correction. For within-subject chance levels, use label permutations with circular time shifts, so that temporal autocorrelation is preserved.

## Conclusion

The held-out-shaft design turns an apparent negative, "finer is always better within a person", into a positive test. Within a person, resolution only buys identity. Under hold-out, it must buy transferable anatomy, and the optimum should land where cubes are coarse enough to be occupied and fine enough to stay functionally homogeneous. The project's earlier finding that soft regions mostly reduce "unknown" electrodes predicts that coverage, not anatomy, will drive much of the curve. The design wins if it can show a resolution optimum that survives after conditioning on cell occupancy. That result would separate a coverage effect from a specificity effect, which no SuperEEG-style kernel or BaRISTA-style atlas comparison has done.

The biggest practical risk is not novelty but leakage that flatters the hypothesis. Under the current CAR recipe, held-out shafts are partly present in training. Fed alongside training shafts at the same time points, they can look decodable while contributing nothing. If all three conditions are scored, Laplacian referencing is used and the split crosses space with time, the experiment becomes the first clean measurement of how far an iEEG location code generalizes beyond the electrodes it memorized. A negative result, location-only transfer near chance at every resolution, would also matter: it would mean the "spatial" benefit reported across iEEG foundation models is identity and symmetry-breaking rather than anatomy.
