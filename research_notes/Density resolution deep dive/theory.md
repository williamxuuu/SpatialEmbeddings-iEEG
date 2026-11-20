# Density x resolution: theory notes (2026-11-20)

Verification legend: **[V]** checked against the source text, or against a secondary source that quotes it, during this session. **[S]** checked only via an abstract or a search snippet. **[R]** recalled from memory and not checked. Treat [R] items as such before citing.

## 1. Partitioning (regression histogram) rate

- Györfi, Kohler, Krzyżak, Walk (2002), *A Distribution-Free Theory of Nonparametric Regression*, Springer. Ch. 4 is "Partitioning Estimates".
  - Bound [V, via secondary source: arXiv 1810.13259 §5.2.3, which paraphrases GKKW]. Assume a cubic partition of side h_n, compact support S, Var(Y|X) <= sigma^2, and |m(x1) - m(x2)| <= C||x1 - x2|| (Lipschitz). Then
    `E||m_n - m||^2 <= c_hat * (sigma^2 + sup_S |m|^2) / (n h_n^d) + d C^2 h_n^2`.
    Choosing `h_n ∝ ((sigma^2 + sup|m|^2)/C^2)^{1/(d+2)} n^{-1/(d+2)}` gives `O(C^{2d/(d+2)} n^{-2/(d+2)})`.
  - Theorem number: probably **Theorem 4.3** [R, NOT verified]. Other papers cite weak universal consistency as Theorem 4.2 [S], so 4.3 is the likely next theorem. I could not open the book. Check it before citing "Thm 4.3".
  - The minimax lower bound for the Lipschitz (p = 1) class is n^{-2/(d+2)}. It is in GKKW Ch. 3 [R, theorem number not verified]. Stone (1982) Ann. Stat. 10:1040 [R].
  - The sup|m|^2 term carries the **empty-cell** contribution, because the estimate predicts 0 in an empty cell. The proof bounds `sum_A mu(A)(1-mu(A))^n <= N_cells/(e n)` [R, standard argument]. So within-distribution missing mass has the same order, 1/(n h^d), as the variance. It changes the constant, not the rate.
- Scott (1979) Biometrika 66(3):605-610 [R]. Gives the 1-D density histogram width `h* = 3.49 sigma n^{-1/3}`. In general `h* = (6 / int f'^2)^{1/3} n^{-1/3}`.
- Freedman and Diaconis (1981) Z. Wahrscheinlichkeitstheorie verw. Geb. 57:453-476 [R]. Gives `h = 2 IQR n^{-1/3}`.
- Scott (1992; 2nd ed. 2015), *Multivariate Density Estimation*, Wiley [R]. The d-dimensional histogram has `h_k* ≈ 3.5 sigma_k n^{-1/(d+2)}` (normal reference) and AMISE `O(n^{-2/(d+2)})`. Note: the n^{-1/(d+2)} exponent is shared by density histograms and Lipschitz regression partitions. Both have bias^2 ∝ h^2 and variance ∝ 1/(n h^d).

## 2. Clustered data (electrodes nested in subjects)

- Kish (1965) *Survey Sampling* [R]. Design effect `deff = 1 + (m-1) ICC`, and `n_eff = n / deff`.
- Kernel and partition smoothing with correlated or clustered data [all R]:
  - Altman (1990) JASA 85:749-759. With positively correlated errors, CV/GCV choose bandwidths that are too small (undersmoothing).
  - Hart and Wehrly (1986) JASA 81:1080-1088. Covers repeated-measures kernel regression.
  - Opsomer, Wang, Yang (2001) Stat. Sci. 16:134-153. Review of nonparametric regression with correlated errors.
  - Lin and Carroll (2000) JASA 95:520-534. Clustered data. Working-independence kernel GEE is the most efficient, because as h -> 0 the within-cluster correlation drops out of the leading variance.
- Our case is the opposite regime: **the clustering is in X, not just in the noise**. A subject's electrodes sit on shafts, so a cell holds many electrodes from few subjects. The cell-mean variance is about `sigma_b^2 / S_A + sigma_w^2 / n_A`, where S_A is the number of distinct training subjects in cell A and sigma_b^2 is the between-subject variance of the target at fixed location. When sigma_b^2 dominates (likely, given registration error and individual anatomy), the effective n is **number of subjects** times per-subject cell occupancy, not the electrode count. Consequences:
  1. The density factor in the design is exactly 5x (S = 1 to 5) whatever the ICC, because m per subject is fixed.
  2. The design effect depends on h. Big cells have large m and large deff. Small cells have m -> 1 and deff -> 1. This flattens the variance curve in h and weakens the predicted shift.
  3. Model selection on random electrode splits (not subject splits) is biased toward small h (Altman's undersmoothing). Always select on held-out subjects.
- I found no theorem on the optimal partition width under covariate clustering. Treat it as an open gap [search-level].

## 3. Coverage / missing mass

- Good (1953) Biometrika 40:237-264 [R]. Missing-mass estimate `M_hat = N_1 / n`, where N_1 is the number of singletons.
- McAllester and Schapire (2000) COLT [R]. Convergence of Good-Turing estimates. Berend and Kontorovich (2013) [R]. Concentration of the missing mass.
- Expected missing mass `E M_n = sum_j p_j (1 - p_j)^n` [standard].
- **Use subjects as the sampling unit.** Let pi_j(h) be the probability that a subject has at least one electrode in cell j, and q_j the held-out electrode mass in that cell. Then `P(uncovered) ≈ sum_j q_j (1 - pi_j)^S`. With S <= 5 this term is large at small h and falls steeply with h. It adds a penalty of roughly `(L_unknown - L_known) * P_uncov(h, S)` that grows as h shrinks. It pushes h* coarser, more so at low S. It also adds S-dependence to the shift beyond the n^{-1/(d+2)} term.
- **Measure it directly**, with no training needed. For each (h, S), compute the fraction of held-out-subject electrodes that fall in cells no training subject occupies. This is the leave-one-subject-out Good-Turing analogue. Use it as a covariate.
- Combined analyses: the GKKW bound already folds in empty cells (see §1). I found no paper that treats occupancy and bias-variance jointly under covariate shift across groups [search-level gap].

## 4. Intrinsic dimension

- Kpotufe (2011) NeurIPS 24, "k-NN Regression Adapts to Local Intrinsic Dimension" [R]. Kpotufe and Dasgupta (2012) J. Comput. Syst. Sci. 78(5):1496-1515, "A tree-based regressor that adapts to intrinsic dimension" (RP-trees) [R]. Bickel and Li (2007), local polynomial regression on unknown manifolds, IMS Lecture Notes [R]. Scott and Nowak (2006) IEEE TIT, dyadic decision trees with box-counting dimension [R].
- Fixed-grid argument [derivation]: the number of occupied cells is about `h^{-d_eff}` (box-counting). The variance term becomes `N_occ(h) / n`, so `h* ∝ n^{-1/(d_eff + 2)}`.
- **d_eff depends on scale.** Pooled across subjects, shafts at different angles fill the volume, so d_eff ≈ 3 at h ≳ inter-shaft spacing. Within a subject at h ≲ 10 mm, d_eff ≈ 1 (shafts) or 2 (cortical ribbon). **Estimate it empirically** as the slope of log N_occ(h) vs log(1/h) on the pooled coordinates.

## 5. Empirical precedents

- **Dohmatob, Richard, Pinho, Thirion (2021)** NeuroImage 229:117706, "Brain topography beyond parcellations: Local gradients of functional maps" [V, full text from HAL hal-03122173]. Note the correct title.
  - Task: predict task-fMRI from rest-fMRI with a parcel-wise linear model on HCP (Ntrain = 200, Ntest = 200).
  - Fig. 8: R^2 vs number of parcels follows "a standard bias/variance compromise, with an optimal region near 200 parcels". It beats a whole-brain MLP for 20 < k < 1000. **All at Ntrain = 60 only.**
  - The authors give two reasons small parcels fail. (i) Too few voxels to fit. (ii) Small parcels are "more sensitive to between subject-variability, given that the parcellation scheme used here is fixed across subjects". Reason (ii) is our registration and anatomy floor.
  - Fig. 6 does vary N_train from 1 to 300. It shows the **gain from bagging multiple random parcellations**, not optimal cardinality vs N. The gain is about 0.07 R^2 at N = 11, above 0.03 at N = 31, and about 1.5% at hundreds of subjects. **They do NOT report optimal cardinality as a function of sample size.** Do not cite them for that. They do support "variance-dominated at small N" and "average over random/shifted partitions".
  - The discussion says there is no single optimal resolution across tasks.
- Guisan et al. (2007) Diversity and Distributions 13(3):332-340, "Sensitivity of predictive SDMs to change in grain size" [S, abstract]. Covers 50 species, 5 regions and 10 methods. A 10x coarsening "does not severely affect predictions". **"Larger sample sizes resulted in better models, which tended to be more sensitive to grain."** Grain effects appear only when the data's intrinsic location error is smaller than the coarse grain. This qualitatively matches the density x resolution interaction. It does not estimate an optimum vs n.
- Guisan et al. (2007) Ecol. Monogr. 77(4):615-630, "What matters for predicting the occurrences of trees" [S]. Covers 10 methods, 30 Swiss trees, and perturbations of location error, grain and sample size. Location error and small n hurt. Grain had little effect for most methods. Variation among species exceeded variation among methods.
- MAUP: Openshaw (1984) CATMOG 38. Fotheringham and Wong (1991) Environ. Plan. A 23:1025 [R]. Results depend on both zoning and scale. This argues for averaging over shifted grids.
- Multi-scale geo embeddings that avoid choosing one cell size: Mai et al. (2020) ICLR, Space2Vec [R].
- Precedent for the analysis method: Hoffmann et al. (2022), "Training Compute-Optimal LLMs" (Chinchilla), Approach 2 [R]. Fit a parabola to loss vs log N on each isoFLOP slice, take the vertex, then regress log N_opt on log C. This is the same design as fitting a vertex per S and regressing log h* on log S.
- Scaling or data collapse: Rosenfeld et al. (2020) ICLR, "A constructive prediction of the generalization error across scales" [R]. Kaplan et al. (2020) [R]. Physics finite-size scaling with a collapse quality metric: Bhattacharjee and Seno (2001) J. Phys. A 34:6375 [R].

## 6. Quantitative prediction for our design

Model: `R(h,S) = A h^2 + B / (S h^{d_eff}) + D * P_uncov(h,S) + floor`.

Predicted ratio h*(S2)/h*(S1) = (S2/S1)^{-1/(d+2)}, ignoring coverage:

| d_eff | 1→2 subj | 1→5 subj | shift for 5x (octaves of h) |
|---|---|---|---|
| 3 | 0.87 | 0.72 | 0.46 |
| 2 | 0.84 | 0.67 | 0.58 |
| 1 | 0.79 | 0.59 | 0.77 |

- Our grid steps are factors of 2 (80/40/20/10/5 mm, one octave each). **The predicted 1→5 shift is half to three-quarters of a grid step.** Argmax-on-grid will often show no shift. Estimate a continuous vertex instead.
- The coverage term adds extra shift at low S. If the pooled d_eff is about 3 and P_uncov is large at 5-10 mm, the S = 1 optimum may sit at 40-80 mm.
- A floor from registration error and individual anatomy (about 5-10 mm; see the Dohmatob and Guisan intrinsic-error points) truncates the bias benefit below that scale. h* cannot shift below it however much data there is.
- **Caveat:** if AUC is near chance (the pilot plateau), the curve is flat, the vertex is undefined, and nothing here is testable. First confirm that some h beats the no-spatial-embedding baseline.

### Analysis plan

1. Run a mixed model per held-out subject and seed: `AUC ~ x + x^2 + x:s + x^2:s + s + (1 | test_subj) + (1 | seed)`, with x = log2 h and s = log2 S.
   - The vertex is `x* = -(b_x + b_xs s) / (2 (b_x2 + b_x2s s))`.
   - Test the slope of x* on s against the theory value -1/(d+2), which is -0.2 to -0.33 octave per doubling.
   - Get CIs by bootstrapping test subjects (cluster bootstrap) and seeds.
   - Add intermediate h (e.g. 57, 28, 14, 7 mm, i.e. half-octaves) to sharpen the vertex.
   - The a priori test is the interaction `x:s`: AUC slope vs log h becomes more negative with more data.
2. Collapse test.
   - Plot AUC vs electrodes per occupied cell, n/N_occ(h), or vs `h * S^{1/(d+2)}`, with one curve per S.
   - Fit the exponent by minimising collapse residual (Bhattacharjee-Seno style). Check that it agrees with the independently measured d_eff.
   - Partial collapse is expected: only the variance branch depends on n h^d. Also try collapsing on the number of distinct subjects per cell.
3. Measure P_uncov(h,S) from coordinates and include it as a covariate. Alternatively, report AUC separately on covered vs uncovered test electrodes. This separates the coverage effect from the bias-variance shift.
4. Use random grid offsets (a few shifts per h) and average or bag. This reduces the MAUP/zoning variance (cf. Dohmatob Fig. 6) and smooths the curve, which makes the vertex estimate stabler.
5. Power: the expected Δx* is about 0.46 octave between S = 1 and S = 5. Run a simulation with the pilot noise level to set the number of seeds and test subjects.
