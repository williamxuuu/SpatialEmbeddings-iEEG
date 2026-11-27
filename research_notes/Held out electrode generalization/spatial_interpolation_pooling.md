# Spatial interpolation and spatial pooling of iEEG signals across electrode positions (classical statistics)

Researcher notes, compiled 2026-11-26. Access level for each source:
- **[FT]** = I read the full text (PDF extracted locally or full HTML).
- **[ABS]** = I saw only the abstract, a search snippet, or a secondary description. Treat these numbers as less certain.

PMC (pmc.ncbi.nlm.nih.gov) blocked automated fetches behind a browser check, so I got full texts from publisher or mirror PDFs where I could.

---

## Q1. Kriging / GP / virtual-electrode methods for predicting activity at unmeasured locations, and accuracy vs distance

### Takeaway
Only two lines of work apply formal spatial statistics to "predict an unseen electrode":
- **Kriging on dense µECoG** (Trumpis et al. 2021). This is within one array, within one person, at sub-mm to ~1.5 mm spacing. It finds that ≤10% error prediction of broadband/high-frequency fields needs **sub-millimetre spacing**, far denser than clinical ECoG/sEEG.
- **GP regression in MNI space, pooled across patients** (SuperEEG; Owen et al. 2020). It uses a fixed RBF width λ = 20 MNI units, not cross-validated, and reports held-out-electrode correlations of r ≈ 0.5 on raw voltage.

I found no paper that reports prediction accuracy as a curve of distance to the nearest training electrode at clinical (cm) spacings for high-gamma.

### Cited Findings
**Trumpis et al. 2021, J Neural Eng 18:036011, "Sufficient sampling for kriging prediction of cortical potential in rat, monkey, and human µECoG"** [FT]
- Method: spatial covariance modelling and geostatistical kriging on "thousands of 500 ms µECoG snapshots in human, monkey, and rat". They define a "probably approximately correct (PAC) spacing": the spacing needed to predict fields at ≤10% error in 95% of observations. — [Trumpis 2021 PDF](https://rogersgroup.northwestern.edu/files/2021/jnekriging.pdf); [PubMed](https://pubmed.ncbi.nlm.nih.gov/33326943/)
- Covariance family: Matérn. In simulation at 1 mm pitch, prediction MSE was monotonic in both the range and smoothness parameters. Changing pitch had a similar effect to changing noise: "By reducing pitch to 0.5 mm, approximately the same fields were predictable in 20% noise as were predictable in 0% noise and 1 mm pitch." — [Trumpis 2021](https://rogersgroup.northwestern.edu/files/2021/jnekriging.pdf)
- Main result: across five bands from 4–7 Hz to 75–300 Hz, PAC spacing was sub-millimetre for rat auditory cortex and anesthetized human posterior STG. "At 75–300 Hz, sub-millimeter PAC spacing was required in all species and cortical areas." — [Trumpis 2021](https://rogersgroup.northwestern.edu/files/2021/jnekriging.pdf)
- Ideal (noiseless) PAC spacing by area and band — [Trumpis 2021](https://rogersgroup.northwestern.edu/files/2021/jnekriging.pdf):
  - Rat auditory cortex: ~570–1050 µm across bands.
  - NHP and human motor cortex, alpha/beta/gamma: 1–1.5 mm.
  - Human pSTG: ~600–1000 µm.
  - High-frequency band (HFB): 600–850 µm "for all cortical areas and species".
- Effect of noise: PAC spacing was on average 115 µm smaller than ideal for intraoperative/semi-chronic recordings, and 365 µm smaller in the long-term chronic case because of higher thermal noise. — [Trumpis 2021](https://rogersgroup.northwestern.edu/files/2021/jnekriging.pdf)
- Human results (Duke intraoperative LCP arrays subsampled to 1.52 mm) — [Trumpis 2021](https://rogersgroup.northwestern.edu/files/2021/jnekriging.pdf):
  - PAC spacing: 829 µm in pSTG vs 1.43 mm in motor cortex.
  - Predictability coverage at the actual 1.52 mm spacing: 91.8% in motor cortex vs only 42.7% in pSTG.
- Half-correlation lengths from device-independent covariance kernels: minimum ~1.15 mm in HFB; maximum in alpha of ~3 mm (pSTG) and ~4.25 mm (motor). — [Trumpis 2021](https://rogersgroup.northwestern.edu/files/2021/jnekriging.pdf)
- Methodological point: "Spatially independent noise lowers normalized correlation values at all inter-electrode distances." Correlogram-based length scales therefore depend on the device and SNR, while the kriging covariance model separates field from noise. The paper also notes that kriging, as the best linear unbiased predictor, "tends to perform well compared to deterministic interpolators such as splines or inverse-weighted prediction", and that kriging error was often *smaller than the recording noise*, so correlated oversampling denoises. — [Trumpis 2021](https://rogersgroup.northwestern.edu/files/2021/jnekriging.pdf)
- The authors' own novelty claim: "the first study to rigorously apply optimal linear prediction methods to interpolate field potential based on spatial statistics". — [Trumpis 2021](https://rogersgroup.northwestern.edu/files/2021/jnekriging.pdf)

**Owen, Muntianu, Heusser, Daly, Scangos & Manning 2020, Cerebral Cortex 30(10):5333, "A Gaussian Process Model of Human Electrocorticographic Data" (SuperEEG)** [FT]
- Model:
  - A Gaussian RBF is placed at each electrode location to approximate the volume-conduction fall-off.
  - Electrode-by-electrode correlation matrices are pooled across patients in MNI space.
  - Activity at unobserved locations is reconstructed by GP conditioning.
  - Reconstruction is of z-scored voltage, not absolute voltage.
  - Sources: [Owen 2020 (OUP)](https://academic.oup.com/cercor/article/30/10/5333/5851264); [PMC PDF mirror](https://rcastoragev2.blob.core.windows.net/b66b54c86f2ff95fdb0d403e12afadc1/PMC7472198.pdf)
- Kernel width choice (verbatim logic): "we sought to maximize spatial resolution (which implies a small value of λ) while also maximizing the algorithm's ability to generalize to any location … without dense electrode coverage (which implies a large value of λ). Here, we set λ = 20, guided in part by our prior related work … and in part by examining the brain coverage with non-zero weights … We note that this value could in theory be further optimized, e.g., using cross validation." This is an explicit coverage-vs-resolution trade-off, settled heuristically rather than by CV. — [Owen 2020](https://rcastoragev2.blob.core.windows.net/b66b54c86f2ff95fdb0d403e12afadc1/PMC7472198.pdf)
- Data: Dataset 1 has 6876 electrodes from 88 patients. — [Owen 2020](https://academic.oup.com/cercor/article/30/10/5333/5851264)
- Held-out-electrode accuracy, Dataset 1 — [Owen 2020](https://rcastoragev2.blob.core.windows.net/b66b54c86f2ff95fdb0d403e12afadc1/PMC7472198.pdf):
  - Across-patient model (correlation model learned from all other patients): mean r = 0.51.
  - Within-patient model (the same patient's other electrodes): mean r = 0.32.
  - The across-patient model was significantly better: t(66) = 9.17. This replicated in Dataset 2: t(77) = 11.25.
- Accuracy by training scope: highest for across-patient models restricted to the same experiment (r = 0.68), then across experiments (r = 0.61), lowest across tasks (r = 0.47). — [Owen 2020](https://rcastoragev2.blob.core.windows.net/b66b54c86f2ff95fdb0d403e12afadc1/PMC7472198.pdf)
- Electrode density, defined as the proportion of other patients' electrodes within 20 MNI units: it did **not** reliably predict reconstruction accuracy in Dataset 1 (r = 0.05, P = 0.70) and was slightly *negatively* correlated in Dataset 2 (r = −0.21, P = 0.05). Voxelwise sampling density agreed across the two datasets at r = 0.6. — [Owen 2020](https://rcastoragev2.blob.core.windows.net/b66b54c86f2ff95fdb0d403e12afadc1/PMC7472198.pdf)
- Accuracy is also reported per frequency band (δ through broadband) in Fig. 4. I did not extract the per-band means. — [Owen 2020](https://rcastoragev2.blob.core.windows.net/b66b54c86f2ff95fdb0d403e12afadc1/PMC7472198.pdf)

### Inferences
- Trumpis shows that, *within a person*, high-frequency fields decorrelate on a ~1 mm scale. At clinical spacings (sEEG contacts ~3.5–5 mm apart; ECoG 4–10 mm), pure spatial interpolation of the high-gamma *time series* should therefore fail for most fields. Any held-out-electrode success at cm scale must come from shared, coarse-scale or functional structure, not from interpolating local field structure.
- SuperEEG's across > within result, and its null or negative density–accuracy relation, should be read with caution:
  - It reconstructs raw z-scored voltage, which includes low-frequency, reference-shared and volume-conducted components.
  - λ = 20 was fixed and not cross-validated.
  - My inference: a large fixed kernel would flatten any density dependence. This is a point to test, not a finding.
- No paper I found gives "accuracy vs distance to the nearest training electrode" for iEEG at cm scale. Building that curve, ideally with shaft-blocked or patient-blocked CV, would be a novel contribution.

### Gaps
- Per-band SuperEEG numbers, especially high gamma (γH) and broadband, were not extracted from Fig. 4.
- I did not find work applying kriging or GPs to sEEG specifically, or source-space interpolation for sEEG (e.g., beamformer "virtual electrodes" mapped back to unrecorded contacts) with held-out validation. I did not search this deeply.
- No result on spatial super-resolution (learned upsampling) of iEEG was retrieved in this pass.

---

## Q2. How far apart can iEEG electrodes be and still share function? (spatial correlation length; reference effects)

### Takeaway
High-gamma/broadband correlation falls off steeply:
- **Macro ECoG (Muller 2016):** high gamma is largely uncorrelated (<0.9) already at the minimum 4 mm spacing.
- **µECoG (Trumpis 2021):** half-correlation length is ~1.15 mm.
- **Large mixed sample (McCarty 2022, 71 patients):** high-gamma FWHM is ~6.5–7 mm under CAR for both ECoG and sEEG gray matter, shrinking to **~3.8 mm with bipolar sEEG**. White-matter contacts and common references broaden the apparent zone.

All of these are within-person time-series correlations. I found no direct estimate of the *across-person* distance over which high-gamma *functional selectivity* is shared.

### Cited Findings
**Muller, Hamilton, Edwards, Bouchard & Chang 2016, J Neural Eng 13:056013, "Spatial resolution dependence on spectral frequency in human speech cortex electrocorticography"** [FT]
- Setup: 5 patients, 256-channel high-density grids with 4 mm centre-to-centre spacing and 1.17 mm exposed diameter, over STG and vSMC, during speech listening and production. Referencing was **common average (CAR) over 16-channel banks**. — [Muller 2016 (eScholarship PDF)](https://escholarship.org/content/qt3sk9z4jc/qt3sk9z4jc.pdf?t=p9fgq8); [PubMed](https://pubmed.ncbi.nlm.nih.gov/27578414/)
- Main result: "both the envelope and phase of gamma and high gamma frequencies (30–150 Hz) are largely uncorrelated (<90%) at 4 mm, the smallest spacing". Lower frequencies stayed correlated over longer distances. — [Muller 2016](https://escholarship.org/content/qt3sk9z4jc/qt3sk9z4jc.pdf?t=p9fgq8)
- Curve fitting — [Muller 2016](https://escholarship.org/content/qt3sk9z4jc/qt3sk9z4jc.pdf?t=p9fgq8):
  - Models tested: exponential, Lorentzian, Gaussian and power-law fall-offs, each with ≤2 parameters plus a constant noise floor c, fit to pairs ≤20 mm apart.
  - Analytic amplitude: best fit by a Lorentzian, y = (1−c)B²/(B²+x²)+c.
  - Phase: theta and alpha were best fit by an exponential; beta to high gamma by a Lorentzian.
  - "The steepness of correlation falloff values are conserved across patients and cortical regions, but differ by frequency band."
  - The numeric B values are only in Fig. 7 and I did not extract them.
- Activity and anatomy effects — [Muller 2016](https://escholarship.org/content/qt3sk9z4jc/qt3sk9z4jc.pdf?t=p9fgq8):
  - High-gamma correlations differed significantly between task and rest at all distances <20 mm.
  - Correlations were higher within a gyrus than across a sulcus (STG–MTG) or across the Sylvian fissure (STG–frontal).
  - So Euclidean distance alone understates the drop in correlation across sulci.

**McCarty, Woolnough, Mosher, Seymour & Tandon 2022, eNeuro 9(2) ENEURO.0492-21.2022, "The Listening Zone of Human Electrocorticographic Field Potential Recordings"** [FT, via eNeuro print view]
- Data and method:
  - 71 patients: 18 with subdural electrodes (SDE), 47 sEEG, 6 high-density sEEG.
  - 8752 sites: 1842 SDE, 6757 sEEG (2916 gray matter, 2649 white matter), 153 hdsEEG.
  - Pearson correlation of band-limited voltage for pairs within 30 mm, computed 0–1000 ms post-stimulus.
  - FWHM = 2 × the distance where r = 0.5.
  - The paper says "75 patients" in one place and 71 in the Methods.
  - Sources: [McCarty 2022 eNeuro](https://www.eneuro.org/node/3797693.full.print); [PubMed](https://pubmed.ncbi.nlm.nih.gov/35410871/); [bioRxiv](https://www.biorxiv.org/content/10.1101/2021.10.22.465519v1.full)
- FWHM, broadband high gamma (70–150 Hz) — [McCarty 2022](https://www.eneuro.org/node/3797693.full.print):
  - SDE (CAR): 6.6 ± 2.5 mm.
  - sEEG gray matter (CAR): 7.14 ± 1.7 mm. Not significantly different from SDE.
  - hdsEEG (CAR): 6.5 ± 1.4 mm.
  - sEEG white matter: 11.3 ± 3.2 mm.
- FWHM, theta — [McCarty 2022](https://www.eneuro.org/node/3797693.full.print):
  - SDE: 13.0 ± 2.3 mm.
  - sEEG gray matter: 12.1 ± 1.8 mm.
  - sEEG white matter: 20.2 ± 4.3 mm.
- Reference effects for gray-matter sEEG — [McCarty 2022](https://www.eneuro.org/node/3797693.full.print):

  | Reference | High-gamma FWHM | Theta FWHM |
  |---|---|---|
  | CAR | 7.14 mm | 12.1 mm |
  | White-matter reference | 7.62 ± 1.8 mm | 14.4 ± 3.8 mm |
  | **Bipolar** | **3.83 ± 0.45 mm** | **7.19 ± 1.6 mm** |

  The reference effect was significant for sEEG only, not for SDE or hdsEEG.
- The authors describe "preserved locality of BHG across electrode scale". They report no same-shaft vs across-shaft comparison. — [McCarty 2022](https://www.eneuro.org/node/3797693.full.print)

**µECoG density studies**
- Trumpis 2021: half-correlation length ~1.15 mm (HFB) to ~3–4.25 mm (alpha), from covariance kernels with noise separated out [FT]. The same paper summarises earlier estimates [FT] — [Trumpis 2021](https://rogersgroup.northwestern.edu/files/2021/jnekriging.pdf):
  - A parametric STG model giving 1.5 mm for gamma/high gamma and >4 mm for theta.
  - Microwire-ECoG coherency in motor cortex giving 0.8–1 mm at 150 Hz and 2.13–3.45 mm at low frequencies.
  - Raw STG correlograms crossing r = 1/2 between 1 and 3 mm.
- Duraivel et al. 2023, Nat Commun, "High-resolution neural recordings improve the accuracy of speech decoding" [ABS]:
  - LCP µECoG with ~1.3–1.7 mm spacing over speech motor cortex, 4 intraoperative patients.
  - The abstract reports 57× higher spatial resolution and 48% higher SNR than macro-ECoG/SEEG, and 35% better decoding.
  - The preprint reportedly gives r ≈ 0.6 at separations <2 mm. I saw this only in a search summary and did not verify it.
  - Sources: [Nature PDF](https://www.nature.com/articles/s41467-023-42555-1.pdf); [DOAJ abstract](https://doaj.org/article/145648d56b284884badee79ef4520f18); [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.05.19.492723.full.pdf)

**Reference / montage effects in sEEG high-gamma mapping** [ABS]
- Bipolar and Laplacian montages reduce the signal deflections at deep white-matter and extra-parenchymal contacts that appear under CAR, while preserving gray-matter high-gamma augmentation. — [Frontiers Neurol 2024 / PMC11521834](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11521834/); [PubMed 39479009](https://pubmed.ncbi.nlm.nih.gov/39479009)

### Inferences
- For a held-out-electrode model, the "neighbour radius" that carries *time-series* information is about 2–4 mm for bipolar high gamma and about 7 mm under CAR. Much of the extra reach under CAR or white-matter referencing is shared reference or volume-conducted signal, not shared function.
- Two implications follow:
  1. Same-shaft neighbours (typically 3.5–5 mm apart) sit *inside* the CAR listening zone. Random-electrode CV therefore leaks information, and shaft-blocked CV is needed. This links to Q4.
  2. Reported cross-electrode prediction accuracy will depend strongly on the reference scheme. CAR inflates correlations at short and moderate range.
- Because correlation drops across sulci (Muller 2016), geodesic or surface distance, or a parcellation boundary, may be a better kernel metric than Euclidean MNI distance.

### Gaps
- No study I found measures *across-person* spatial correlation of functional selectivity (e.g., how similar task tuning is between electrodes from different people as a function of MNI distance). This is the quantity that matters for cross-subject pooling.
- I did not obtain Muller's numeric B values or Chiang et al. µECoG correlation-length numbers.
- No sEEG same-shaft vs cross-shaft correlation comparison was found.

---

## Q3. Group-level iEEG mapping methods that pool electrodes across people, and how they choose spatial resolution

### Takeaway
Group iEEG methods pool by one of:
- **Surface-based recording zones** with geodesic spread plus mixed-effects models (Kadipasaoglu/Tandon SB-MEMA).
- **Atlas parcels** (Frauscher 2018: 38 regions; Bernabei 2022 MNI Open iEEG Atlas; Woodhouse 2024: 82 Lausanne regions with site as a mixed effect).
- **Fixed-radius MNI kernels** (SuperEEG 20 MNI units; density maps at roughly electrode size).

In every case I found, the kernel or region size is set by convention, electrode geometry, or coverage heuristics. **I found no iEEG paper that chooses bin or kernel size by cross-validation or as an explicit function of electrode density.**

### Cited Findings
**Kadipasaoglu et al. 2014, NeuroImage 101:215–224, "Surface-based mixed effects multilevel analysis of grouped human electrocorticography" (SB-MEMA)** [ABS]
- The method combines registration to individual cortical topology, surface-based co-registration, and a mixed-effects multilevel analysis. It is designed to handle variable cortical anatomy, sparse coverage, and within- and between-subject variance. Applied to a face-recognition dataset with n = 22, it gave results more consistent with individual data and meta-analyses than existing techniques. — [PubMed 25019677](https://pubmed.ncbi.nlm.nih.gov/25019677/); [ResearchGate](https://www.researchgate.net/publication/263899420_Surface-Based_Mixed_Effects_Multilevel_Analysis_of_Grouped_Human_Electro-Corticography)
- The 2015 perspective "Development of grouped icEEG for the study of cognitive processing" describes the same pipeline. It says the mixed-effects step corrects for unsampled cortical regions (missing data). — [Frontiers Psychol 2015](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2015.01008/full)
- Recording-zone size [ABS, from search summaries of later papers; not verified in the primary text]:
  - Each electrode gets a geodesic recording zone, reported as an exponentially decaying geodesic radius that does not cross sulci.
  - One application used a 10 mm zone; another used 1.5 mm, matching electrode dimensions.
  - The radius is therefore study-specific, not a fixed method constant.
  - Source: [Frontiers 2015](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2015.01008/pdf). Treat as unverified.

**Atlas / parcel pooling** [ABS]
- Frauscher et al. 2018, Brain: atlas of normal awake iEEG from three centres; spectra computed for 38 brain regions; 540 recordings met criteria. Electrodes were registered to common stereotactic space. — [Duke Scholars](https://scholars.duke.edu/publication/1600668); [Masaryk Univ record](https://is.muni.cz/publication/1451676/cs)
- Bernabei et al. 2022, Brain: normative atlas (MNI Open iEEG Atlas) built from 1,772 normal channels, used to compare each patient's regional activity and connectivity. — [PMC9630716](https://pmc.ncbi.nlm.nih.gov/articles/PMC9630716)
- Woodhouse et al. 2024/2025: multi-centre normative lifespan mapping, 502 subjects, contacts assigned to 82 Lausanne scale-36 regions, hospital site modelled as a mixed effect. — [arXiv 2404.17952](https://arxiv.org/html/2404.17952v2); [Brain Struct Funct](https://link.springer.com/10.1007/s00429-025-02988-4)

**Electrode-density / kernel maps** [ABS]
- A 2017 Frontiers in Neuroinformatics iEEG localisation/visualisation tool builds density maps with a Gaussian kernel at each electrode's MNI coordinate, with FWHM = 5 mm for ECoG and 2 mm for SEEG, "approximately equal to electrode length". The kernel is sized by electrode geometry, not by a statistical criterion. — [Frontiers Neuroinform 2017](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2017.00010/full)
- SuperEEG uses a 20 MNI unit neighbourhood both for the RBF width and for its density definition, chosen by coverage heuristics (see Q1). — [Owen 2020](https://rcastoragev2.blob.core.windows.net/b66b54c86f2ff95fdb0d403e12afadc1/PMC7472198.pdf)

**Analogy from fMRI** [ABS]: group-level fMRI work finds that the choice of smoothing kernel affects reliability, with the best FWHMs around 6–8 mm for voxel data. It is not directly transferable to sparse electrodes. — [PMC10346894](https://pmc.ncbi.nlm.nih.gov/articles/PMC10346894/)

### Inferences
- The field's resolution choices fall into three groups:
  - Geometric: kernel ≈ electrode size (2–5 mm).
  - Physiological: 10 mm geodesic zones, consistent with the ~7 mm CAR high-gamma FWHM in McCarty.
  - Coverage-driven: 20 mm RBF; atlases with 38–82 regions.
- None of these is tuned to the number of electrodes per bin. This leaves a gap: a cross-validated bandwidth (or parcellation granularity) as a function of electrode density.
- Mixed-effects pooling (SB-MEMA, Woodhouse's site effect) handles between-subject variance. Spatial bin size still sets the bias, from mixing functionally distinct tissue, against the variance, from few electrodes or few subjects per bin.

### Gaps
- The full SB-MEMA text was not retrieved, so I could not verify its exact kernel form, radius, or minimum-patients-per-node rule.
- No Frauscher or Bernabei per-region minimum-patient threshold was found.
- I did not cover iELVis, Brainstorm, RAVE or other toolboxes' default projection radii in this pass.
- No iEEG study reporting an optimal bin size, or cross-validated bandwidth vs density, was found. This seems to be a genuine gap in the literature, not just a gap in my search, though my search was not exhaustive.

---

## Q4. Theory: optimal bandwidth/partition size vs sample size, and spatial / blocked cross-validation for clustered units (shafts, patients)

### Takeaway
Classical nonparametric theory says partitioning and kernel estimators of a Lipschitz (p = 1) function in d dimensions achieve error rate n^(−2p/(2p+d)), with bin width shrinking as more data arrive. The ecology literature (Roberts 2017; Ploton 2020) shows that random CV on spatially autocorrelated data badly overstates predictive skill, and that **block CV** should be used when the goal is predicting new locations. Both apply directly to choosing bin size vs electrode density and to holding out whole shafts or patients.

### Cited Findings
**Nonparametric regression rates** [ABS / lecture notes]
- Partitioning, kernel and nearest-neighbour estimates achieve the minimax rate n^(−2p/(2p+d)) for smoothness class p. Györfi et al. 2002, *A Distribution-Free Theory of Nonparametric Regression*, is the standard reference. — [Kohler/Krzyżak/Walk preprint](https://www2.mathematik.tu-darmstadt.de/~kohler/preprint05_01.pdf); [Györfi lecture notes](https://www.szit.bme.hu/~gyorfi/booknonpar.pdf)
- Consistency conditions for cubic partitions: h → 0 and n·h^d → ∞, i.e., bins shrink, but slowly enough that each bin's sample count grows. — [Györfi lecture notes](https://www.szit.bme.hu/~gyorfi/booknonpar.pdf)
- The rate n^(−2p/(2p+d)) is not achievable by partitioning estimates for p > 1 in some classes. Kernel or local-polynomial estimators are needed there. — [arXiv 2312.14889](https://arxiv.org/html/2312.14889v3)

**Roberts et al. 2017, Ecography 40(8):913–929, "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure"** [ABS + snippets]
- Random CV ignores dependence and seriously underestimates predictive error.
- Block CV, blocking in space, time, random-effect groups or phylogeny, "is nearly universally more appropriate than random cross-validation if the goal is predicting to new data or predictor space". It is recommended wherever dependence exists, "even if no correlation structure is visible in the fitted model residuals".
- Caveat: blocking can induce extrapolation, by restricting the predictor ranges available in training, and so overestimate interpolation error.
- Sources: [Roberts 2017 PDF (Dormann lab)](https://www.biometrie.uni-freiburg.de/mitarbeiter/dormann/roberts-et-al-2017-ecography.pdf); [Wright State record](https://research.wright.edu/en/publications/cross-validation-strategies-for-data-with-temporal-spatial-hierar/)

**Ploton et al. 2020, Nat Commun 11:4540, "Spatial validation reveals poor predictive performance of large-scale ecological mapping models"** [ABS + snippets]
- Data: an inventory of 11.8 million trees in central Africa, modelled with a random forest.
- Non-spatial validation suggested the model predicted more than half of biomass variation. Spatial validation that accounted for spatial autocorrelation revealed "quasi-null predictive power".
- Mechanism: "an observation point cannot serve as spatially independent validation of nearby training data points."
- Sources: [PMC7486894](https://pmc.ncbi.nlm.nih.gov/articles/PMC7486894); [HAL full text](https://hal.inrae.fr/hal-02936567v1/document)

### Inferences
- **Bin size vs density.** My derivation, not taken from a source: balancing squared bias ~h² (Lipschitz) against variance ~1/(n·h^d) gives h* ∝ n^(−1/(d+2)). For d = 3 (MNI) that is h* ∝ n^(−1/5); for d = 2 (cortical surface) it is h* ∝ n^(−1/4).
  - Doubling the electrode count should shrink the optimal bin width by only about 13% (2^(−1/5) ≈ 0.87) in 3-D.
  - So the optimal parcellation granularity grows slowly with density: the number of bins scales roughly as n^(3/5) in 3-D.
  - This gives a testable prediction for any density × resolution sweep: the best Schaefer-k should increase sub-linearly with electrodes per person.
  - Caveat: iEEG sampling is highly non-uniform. Clinical coverage concentrates in temporal and frontal regions (SuperEEG density maps). Adaptive (data-dependent) partitions or variable-bandwidth kernels are theoretically preferable to a single global h.
- **Spatial CV for iEEG.** Contacts on the same shaft are ~3.5–5 mm apart, inside the CAR high-gamma listening zone of ~7 mm FWHM (McCarty 2022). Random-electrode holdout is therefore the iEEG analogue of the Ploton failure mode.
  - Holding out whole shafts, or blocks of at least ~1 FWHM, is the Roberts-style remedy for within-person prediction.
  - Holding out whole patients is the remedy for cross-person prediction.
  - Note Roberts' caveat: shaft-blocking may push the test into extrapolation, e.g. regions or depths not covered in training, so report both random and blocked CV.
- Using CV to *choose* the bandwidth only gives an unbiased choice if the CV folds are themselves blocked at the same unit as the deployment target (new shaft or new person). Otherwise CV will favour bandwidths that are too small.

### Gaps
- I did not retrieve the exact Györfi theorem statement for the partitioning-estimate rate in the Lipschitz case. The rate is from secondary citation.
- I found no paper applying block/spatial CV to iEEG electrode-prediction tasks, or quantifying leakage from same-shaft neighbours. This looks like an open methodological point.
- I did not cover spatial CV variants in depth, such as buffered leave-one-out, Valavi's blockCV package, or kNNDM / NNDM (Milà et al. 2022).
