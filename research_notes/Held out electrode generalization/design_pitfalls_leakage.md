# Design pitfalls and leakage risks for held-out-shaft generalization in sEEG

Context: Brain Treebank / Neuroprobe (2048 Hz sEEG, movie watching, speech vs non-speech), models using MNI-coordinate cube codes, whole sEEG shafts hidden during training and used for testing.
Access legend: **[FT]** full text read (or the relevant full-text section); **[ABS]** abstract or landing page only; **[SNIP]** search-result snippet only, not verified against the paper. Compiled 2026-11-26.

## 1. Re-referencing leakage (CAR, bipolar, Laplacian/shaft reference)

### Takeaway
Neuroprobe's baselines use shaft-Laplacian re-referencing: each contact minus the mean of its two same-shaft neighbours. That operation stays inside one shaft, so **whole-shaft hold-out is clean under Laplacian or bipolar referencing, and single-contact hold-out is not.** CAR, or any reference averaged over all channels, carries every held-out channel into every training channel at weight 1/N. It should be recomputed from training channels only, or avoided. I found no paper that measures how much reference mixing inflates held-out-channel scores, so this is a gap to test ourselves.

### Cited Findings
- **Neuroprobe [FT, HTML v2, first ~100k chars]:** Laplacian re-referencing is used for the linear, CNN/MLP, BrainBERT, GNN and PopulationTransformer baselines. PopT uses only electrodes that can be Laplacian-rereferenced. BrainBERT gets non-rereferenced data for electrodes that cannot be. Electrodes "were selected in groups from complete probes to retain flexibility for re-referencing techniques". The cap is 120 electrodes per subject, chosen in order of highest linear decoding performance. The part I read does not say which neighbours define the Laplacian, or whether re-referencing happens before or after subsetting. — [Neuroprobe arXiv 2509.21671 (HTML)](https://arxiv.org/html/2509.21671v2)
- **Pitfall from that selection rule:** Neuroprobe picks electrodes by linear decoding performance. If held-out shafts are chosen from that pre-ranked set, the test shafts are already the most decodable ones, which is a selection effect. — same source
- **Laplacian definition in this lineage:** BrainBERT subtracts "the mean signal from the two adjacent electrodes on the same shaft". Only electrodes that have such neighbours enter pretraining. [SNIP] — [BrainBERT arXiv 2302.14367](https://arxiv.org/pdf/2302.14367)
- A later Brain Treebank analysis (subjects 3, 7, 10) uses V_n^ref = V_n − (V_{n−1}+V_{n+1})/2 and keeps only contacts with both consecutively numbered neighbours after corrupted contacts are removed. [SNIP] — [arXiv 2610.40511](https://arxiv.org/pdf/2610.40511)
- I could not confirm the Brain Treebank dataset paper's own re-referencing statement; my search returned only its abstract. [ABS] — [Brain Treebank arXiv 2411.08343](https://arxiv.org/abs/2411.08343v1)
- **Li et al. 2018, "Optimal referencing for SEEG recordings", NeuroImage 183:327–335 [FT]:**
  - Data: 15 subjects, 161 shafts, 2151 contacts, contacts 2 mm long at 3.5 mm centre-to-centre spacing.
  - Schemes compared: monopolar (white-matter pair), CAR, gray/white-matter reference (GWR), electrode shaft reference (ESR, the mean of all contacts on the shaft), bipolar, and Laplacian (the mean of the two adjacent contacts).
  - Mean inter-channel correlation: monopolar r = 0.31 ± 0.04, Laplacian r = 0.06 ± 0.01. Ranking from worst to best: monopolar, GWR, CAR, ESR, bipolar, Laplacian. The Laplacian is lower than all others (p < 0.001). The paper gives no numeric correlations for CAR, ESR or bipolar.
  - CAR yielded a larger fraction of task-related channels than the Laplacian. Broadband gamma task-related fraction: monopolar 37.1 ± 7.4%, Laplacian 27.0 ± 4.7%.
  - Recommendation: use the Laplacian for local population activity.
  - [PMC6495648](https://pmc.ncbi.nlm.nih.gov/articles/PMC6495648)
- Note on the citation: the author list is Li, Jiang, Paraskevopoulou, Wang, Xu, Wu, Chen, Zhang, Schalk. Kahana and Brunner are not authors. — [PubMed 30121338](https://www.pubmed.gov/30121338/)
- **Mercier et al. 2017, NeuroImage 147:219–232 [FT, PDF text]:**
  - Commonly used references (subdermal, white-matter average, global average, local montage) change inter-electrode correlation "drastically".
  - Subdermal and white/gray-matter-average references give strong correlations. These are stronger when few contacts enter the average; one example had 7 white-matter contacts.
  - The global average gives a similar spatial pattern at lower magnitude, centred at zero, which implies added negative correlations. CAR therefore creates anti-correlations between channels.
  - "Re-referencing ... could drastically impact signal content and consequently the results of any given analysis."
  - [Mercier 2017 PDF](https://cerco.cnrs.fr/wp-content/uploads/2019/04/mercier_m_17_1.pdf)
- **Mercier et al. 2022 guidelines (NeuroImage 260:119438) [partial FT, first ~100k of 739k chars; the re-referencing §3.3 was not reached]:** distinguish the online reference from the offline re-reference. Each sEEG contact samples "a sphere of approximately 5 mm radius". — [PMC10190110](https://pmc.ncbi.nlm.nih.gov/articles/PMC10190110)
- Arnulfo et al. 2015 [SNIP]: bipolar referencing "may both confuse signals from distinct sources and suppress true neuronal interactions", while monopolar picks up volume conduction. A closest-white-matter reference was "not affected by large-scale volume conduction". — [UniGe record](https://unige.iris.cineca.it/handle/11567/868932)

### Inferences
- **Laplacian or bipolar with whole-shaft hold-out:** no training-channel signal contains a held-out contact, because the neighbours are always on the same shaft. Leakage through the reference is zero by construction.
  - Check that the end contacts of a held-out shaft do not borrow a neighbour from another shaft.
  - Check that a "neighbour" skipped over a removed contact is still on the same shaft.
- **Single-contact hold-out under Laplacian:** the neighbours n±1 are training channels whose references contain the held-out contact at weight −1/2. That is a direct leak. Hide whole shafts, or at minimum hide contacts n−1, n and n+1 together.
- **CAR, or the "average" re-reference often applied to raw Brain Treebank before Laplacian:** recompute the average over training channels only. Apply that same training-only average to the test channels, which mirrors fitting preprocessing on train only (Kapoor and Narayanan's "preprocessing on train+test" category, below). Alternatively, report Laplacian-only results.
- **Correlation-graph models:** the GNN's "eight nearest neighbours" graph is a functional correlation graph. If it is computed on data that includes held-out shafts, that is also a leak.

### Gaps
- No study found that quantifies how much CAR or reference mixing inflates held-out-channel decoding.
- Neuroprobe's exact Laplacian neighbour rule, and the handling of shaft ends and missing contacts, should be read from the code (`neuroprobe` repo). The text I read did not state them.

## 2. Volume conduction and same-shaft correlation

### Takeaway
Adjacent sEEG contacts are 3.5–5 mm apart, which is inside the ~5 mm radius each contact samples. Monopolar signals are therefore strongly correlated along a shaft, white-matter contacts record volume-conducted gray-matter activity, and passive spread plateaus after about 4–5 mm. Holding out a single contact while its shaft-mates stay in training is close to pseudo-replication, so whole-shaft hold-out is the right unit. Even with whole shafts held out, a neighbouring shaft can sit within 5–10 mm.

### Cited Findings
- **Contact geometry:** contacts 2 mm long at 3.5 mm spacing (Li 2018); 2.4 mm long at 5 mm spacing (Mercier 2017). — [Li 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6495648); [Mercier 2017](https://cerco.cnrs.fr/wp-content/uploads/2019/04/mercier_m_17_1.pdf)
- **Monopolar correlation:** mean inter-channel r is 0.31 under monopolar reference and 0.06 under Laplacian. A Subject 1 example went from 0.66 to 0.38. — [Li 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6495648)
- **White-matter contacts (Mercier 2017):**
  - White-matter contacts record "a mixture of activity, with part arising from the volume conduction (zero time delay) of activity from nearby gray matter".
  - White-matter amplitude falls with distance to gray matter, and the fit is best as 1/distance² (r ≥ 0.68).
  - Passive gray-matter spread "plateaued after 4–5 mm".
  - White-matter signal can correlate with distant gray matter. Only 17% of such pairs (passive watching) showed a common nearby gray-matter driver.
  - [Mercier 2017](https://cerco.cnrs.fr/wp-content/uploads/2019/04/mercier_m_17_1.pdf)
- **Tissue mix (Mercier 2017):** of contacts with centroid in gray matter, only 42% were entirely surrounded by gray matter. About 60% of all contacts touched both gray and white matter. Slightly more contacts sat entirely in white matter than entirely in gray matter. — same source
- **Sampling volume:** each sEEG contact samples about a 5 mm-radius sphere. — [Mercier 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC10190110)
- **Bipolar vs other references [SNIP]:** a 2021 bioRxiv study found bipolar referencing significantly lowered spatial spread (FWHM) for sEEG compared with a white-matter reference or CAR. — [bioRxiv 2021.10.22.465519](https://biorxiv.org/content/10.1101/2021.10.22.465519v1.full.pdf)

### Inferences
- Treat the shaft as the minimum hold-out unit. Also report performance as a function of the distance from each test contact to the nearest training contact. Neighbouring shafts can pass within about 5 mm of each other, so a "distance-buffered" hold-out (Section 3) is the stronger control.
- Report gray-matter and white-matter test contacts separately. White-matter contacts carry diffuse, volume-conducted signal, which a model can decode without any location specificity.

### Gaps
- I found no numbers for correlation versus distance between contacts on different shafts in Brain Treebank. This can be computed directly from the data: a correlation-versus-Euclidean-distance variogram after Laplacian referencing.

## 3. Spatial autocorrelation in cross-validation (spatial block CV)

### Takeaway
In ecology, random CV on spatially autocorrelated data inflated R² from about 0.14 to 0.53, and buffered CV showed predictive power falling to near zero once test points were beyond the autocorrelation range. The translation to our setting: whole-shaft hold-out is "blocked CV". The block or buffer should exceed the measured spatial correlation range of the signal, and models should be compared against a coordinates-only baseline.

### Cited Findings
- **Ploton et al. 2020, Nat Commun 11:4540 [FT]:**
  - Random 10-fold CV: R² = 0.53. Spatial 44-fold CV: R² = 0.14, with RMSPE 77.5 against 82 for a null model predicting the training mean.
  - Buffered leave-one-out: R² ≈ 0.50 with no buffer, about 0.15 at roughly 50 km, and virtually null beyond 100 km.
  - Biomass autocorrelation range is about 120 km. Clusters were built at 150 km by complete linkage, and even that was judged not quite large enough.
  - Recommendations: treat points within the autocorrelation range as pseudoreplicates, do not rely on residual autocorrelation checks alone, and compare against purely spatial baselines.
  - [PMC7486894](https://pmc.ncbi.nlm.nih.gov/articles/PMC7486894)
- **Roberts et al. 2017, Ecography 40:913–929 [ABS/SNIP]:** use block CV wherever dependence structure exists, even when residuals look independent. Ignoring structure gives "serious underestimation of predictive error". Blocking can also force extrapolation by restricting the predictor ranges available for training, which overestimates interpolation error. — [Roberts 2017 PDF](https://www.biometrie.uni-freiburg.de/mitarbeiter/dormann/roberts-et-al-2017-ecography.pdf)
- **blockCV R package (Valavi et al.) [ABS]:** provides tools for choosing spatial block size from the autocorrelation range. — [CRAN blockCV](https://search.r-project.org/CRAN/packages/blockCV/blockCV.pdf)
- **Survey on iEEG generalization [SNIP]:** "a patient-level split therefore does not by itself establish generalisation to unseen regions or implantation configurations." — [arXiv 2608.26998](https://arxiv.org/pdf/2608.26998)

### Inferences
- **Electrode translation, part 1 (unit and buffer):** the hold-out unit is the shaft. The buffer is the distance from test contacts to the nearest training contact.
  - Plot performance against that distance, the analogue of Ploton's Fig. 5a.
  - A "buffered" variant also removes training contacts within X mm of any test contact, with X swept over 0, 5, 10 and 20 mm.
- **Electrode translation, part 2 (extrapolation):** Roberts' caveat applies. A held-out shaft may sample a region (MNI cube) absent from training, which makes the test extrapolation rather than interpolation. Report test contacts by whether their cube code was seen in training.
- **Baseline:** include a "coordinates-only" or "cube-code-only" baseline, plus a shuffled-coordinate control. This mirrors Ploton's purely spatial baseline.

### Gaps
- Roberts' explicit block-size rule was not verified in full text.

## 4. Localisation and MNI registration error, and label disagreement

### Takeaway
Contact localisation within a subject is accurate to sub-millimetre to about 2 mm against manual reference. Neither manual localisation nor the error added by MNI normalisation is ground truth, and I found no source that quantifies the normalisation error. Atlas versus expert labels agree for about 82% of contacts. Given a ~5 mm sampling radius and 3.5–5 mm spacing, cube codes finer than about 5 mm are unlikely to be meaningful.

### Cited Findings
- **Automated contact segmentation:** mean absolute error 0.38 mm against manual segmentation, sensitivity 98.81%, PPV 95.01%, across 23 cases. [SNIP] — [PMC5973981](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5973981/)
- **Model-based localisation:** within 0.56 mm of experienced manual raters. [SNIP] — [PMC10197594](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10197594/)
- **CT metal artefact:** about 2.0 mm error attributed to high-contrast CT noise from metal artefacts. [SNIP, source attribution uncertain among the results] — [PubMed 30128278](https://pubmed.ncbi.nlm.nih.gov/30128278/)
- **Brain shift:** grids can shift up to 7 mm after craniotomy. This concerns ECoG and is less relevant to sEEG. [SNIP] — [PMC5333374](https://pmc.ncbi.nlm.nih.gov/articles/PMC5333374)
- **Taylor et al. 2021, Epilepsia Open [ABS]:**
  - Atlas (USCBrain via BrainSuite) versus manual labels agreed for 82% of more than 17,000 SEEG contacts in 103 patients.
  - Expert adjudication of disagreements: atlas correct 48%, manual correct 36%, both wrong 16%.
  - Note: this figure is **not** Brainstorm's own number.
  - [PMC8408609](https://pmc.ncbi.nlm.nih.gov/articles/PMC8408609)
- **Mercier 2017:** contact classification used a 3×3×3 mm neighbourhood "based on ... imaging voxel resolution of 1 mm³, estimated co-registration errors, and electrode contact size". — [Mercier 2017](https://cerco.cnrs.fr/wp-content/uploads/2019/04/mercier_m_17_1.pdf)
- **Mercier 2022:** "iEEG spatial specificity is in the range of millimeters". There is no consensus on native versus normalised-space analysis. — [PMC10190110](https://pmc.ncbi.nlm.nih.gov/articles/PMC10190110)

### Inferences
- Resolution floor: if resolution-sweep results show gains for cube sizes below about 5 mm, treat them with suspicion. They may reflect contact identity, meaning memorising individual contacts, rather than anatomy.
- Robustness test: jitter MNI coordinates at test time (σ = 2–5 mm) and report how performance degrades.

### Gaps
- No quantitative source found for MNI-normalisation error of sEEG contacts across subjects.
- I did not find the Brain Treebank localisation pipeline's error.

## 5. Test-time input design (do training channels mask held-out ones?)

### Takeaway
If held-out shafts are fed together with training shafts at test time, a multi-channel model can score well while ignoring the held-out channels entirely. I found no iEEG paper that explicitly tests with held-out channels only. EEG work such as HEAR evaluates on disjoint unseen electrodes.

### Cited Findings
- **HEAR (EEG):** pretraining covers one electrode layout, and "evaluation is performed on a disjoint set of unseen electrodes" across 11 channel subsets. [SNIP] — [arXiv 2510.12515](https://arxiv.org/pdf/2510.12515)
- **Masked latent reconstruction for iEEG:** dissociates spatial encoding from spatial masking. [SNIP] — [arXiv 2512.12135](https://arxiv.org/pdf/2512.12135)
- **Neuroprobe cross-subject baseline:** averages activity per Desikan-Killiany region and keeps only regions shared by each subject pair. This is a region-level rather than channel-level transfer, and it drops unseen regions. — [Neuroprobe HTML](https://arxiv.org/html/2509.21671v2)

### Inferences
Recommended controls:
- Feed held-out channels only at test time.
- Run three conditions: (a) training channels only, (b) training plus held-out channels, (c) held-out only. Credit the held-out contribution as (b − a), and as (c) on its own.
- Run an occlusion or permutation analysis on held-out channels inside the full input.
- Shuffle the coordinates assigned to held-out channels: if performance is unchanged, the spatial code is not being used.

### Gaps
- No iEEG paper found that reports this ablation. PopulationTransformer's channel-ablation behaviour was not verified.

## 6. Temporal leakage and stimulus confounds

### Takeaway
Neuroprobe's within-session split is two contiguous 50/50 blocks of the same movie with no stated gap. Its cross-session (different-movie) split is the default and is described as eliminating autocorrelation contamination. For shaft hold-out, keep the temporal split as well. Otherwise the test shaft's own time-locked signals at the same time points are mirrored in the training shafts' data. Speech vs non-speech labels are correlated with acoustics, which is a stimulus confound.

### Cited Findings
- **Neuroprobe splits:**
  - Within-session: 2-fold, 50/50, contiguous "non-overlapping segments of the movie". No temporal gap is stated. 2-fold and 5-fold results correlate at r = 0.982.
  - Cross-session: train on one movie, test on a different movie. This "ensures completely that no data-contamination due to auto-correlation has occurred", and it is the default split.
  - Cross-subject: train on Subject 2, Trial 4 only.
  - [Neuroprobe HTML](https://arxiv.org/html/2509.21671v2)
- **Windows and labels:**
  - Windows are 1 s from word onset. Results are robust to start jitter between −0.375 and −0.125 s.
  - Classes are rebalanced. Neuroprobe Lite caps tasks at 3500 class-balanced datapoints.
  - "Speech" = whether any speech is present in the interval.
  - Benchmark subset: 6 subjects × 2 trials.
  - [Neuroprobe HTML](https://arxiv.org/html/2509.21671v2)
- **EEG partitioning study (Del Pup et al. 2025):** more than 100,000 models trained. Subject-based CV is recommended, and non-nested CV is prone to leakage and favours larger models that overfit validation data. [ABS; inflation numbers not in abstract] — [arXiv 2505.13021](https://arxiv.org/abs/2505.13021)
- **Leakage taxonomy (Kapoor and Narayanan 2023, Patterns 4:100804):**
  - Leakage found in 22 papers across 17 fields, affecting 294 papers.
  - Eight leakage types, including preprocessing on train+test, temporal leakage, and non-independence of train/test.
  - Corrected civil-war ML models were no better than logistic regression.
  - [PMC10499856](https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856)

### Inferences
- Combine shaft hold-out with temporal hold-out: test on held-out shafts × held-out time blocks, ideally a different movie. Shaft-only hold-out on the same time points lets the model learn the stimulus timecourse from training shafts, which share stimulus-driven variance with the held-out shafts.
- If 1 s windows from word onsets overlap, place a gap of at least the window length at fold boundaries.
- Fit normalisation statistics, such as z-scoring and spectrogram scaling, on training channels and training time only.
- Control the speech/non-speech audio confound with an audio-envelope-only baseline.

### Gaps
- I found no published inflation estimate specific to overlapping word-onset windows in Brain Treebank.

## 7. Statistics with few subjects

### Takeaway
The independent unit is the subject, of which Neuroprobe has 6 to 10, not the electrode or the window. Contacts are clustered within shafts and shafts within subjects. Use subject-level inference: average over many shaft hold-out splits within each subject, then test across subjects with exact sign-flip or permutation tests, and correct for the number of models compared.

### Cited Findings
- **Neuroprobe statistics:** reports mean ± s.e.m. across subjects and trials. The part I read describes no formal significance tests. — [Neuroprobe HTML](https://arxiv.org/html/2509.21671v2)
- **Leakage-audited EEG benchmark (Li and Tao 2026):**
  - Participant-level inference with LOSO evaluation, 16 participants.
  - Best balanced accuracy was 21.474% against 20% chance (95% participant-bootstrap CI 19.526–23.482%).
  - No model survived correction across the 13-model family. Seed variance was substantial.
  - [arXiv 2605.00865](https://arxiv.org/abs/2605.00865)
- **Mercier 2022:** electrode-selection procedures "inevitably" introduce selection bias. — [PMC10190110](https://pmc.ncbi.nlm.nih.gov/articles/PMC10190110)

### Inferences
- **Sign-flip test resolution:** with n = 6 subjects, an exact sign-flip test has only 2⁶ = 64 permutations, so the minimum two-sided p is 2/64 ≈ 0.031. With n = 10, there are 1024 permutations.
- **Repeated splits:** use many random shaft hold-out splits per subject, then average within subject before testing. Treat the splits as reducing variance, not as adding samples.
- **Variance and multiplicity:** report seed variance. Correct across the density × resolution sweep, which is a model family in Li and Tao's sense.
- **Within-subject chance:** use label-permutation nulls with block or circular shifts to preserve temporal autocorrelation.

### Gaps
- No iEEG-specific source found on effective sample size for clustered electrodes.
- Mercier 2022's statistics section was not reached within the fetch limit.
