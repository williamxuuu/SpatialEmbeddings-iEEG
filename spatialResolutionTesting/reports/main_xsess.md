# Density x resolution

Outcome: test AUC at the selected (early-stopped) epoch. Gain = AUC(cube) - AUC(none), paired on person, density and seed, then averaged over seeds. Analysis fixed in the sweep file header before any run.

192 runs; 6 held-out people; seeds per cell: 8-8.

Mean gain over people (rows: cube size mm; columns: density level):

```
density     w10    w100     w30
h                              
80.0     0.0024  0.0939  0.0491
40.0     0.0050  0.1551  0.1200
20.0     0.0475  0.2261  0.1564
10.0     0.0490  0.2672  0.2181
5.0      0.0526  0.2756  0.2736
```

Mean coverage:

```
Empty DataFrame
Columns: []
Index: []
```

## H1 (primary): does finer resolution pay off more with more data?

Slope of gain per halving of cube size, per person:

```
density     w10    w100     w30
subject                        
btb01    0.0249  0.0370  0.0801
btb02    0.0286  0.0237  0.0601
btb03    0.0044  0.0509  0.0433
btb04    0.0107  0.0173  0.0326
btb07    0.0094  0.0846  0.0492
btb10    0.0088  0.0718  0.0629
```

**Interaction slope(w100) - slope(w10) = +0.0331 AUC per halving; permutation p = 0.0002** (density labels permuted within 48 person x seed blocks; prediction: > 0)

- population level: 5/6 people > 0; exact sign-flip p = 0.062 (n = 6 caps this at p >= 0.031)
- secondary: slope(w30) - slope(w10) = +0.0402; 6/6 > 0; p = 0.031
- secondary: slope(w100) - slope(w30) = -0.0072; 3/6 > 0; p = 0.594

## H2: how far does the best cube size move?

Pooled vertex regression, best cube size at each density: level 10: 5.0 mm, level 30: 5.0 mm, level 100: 5.0 mm

**dx*/ds = +0.00 octaves finer per doubling of the density level** (bootstrap over people 95% CI -0.08 to +0.01; P(> 0) = 0.03). Prediction: > 0; variance-only theory 0.2-0.33.

Descriptive, per density:

```
         argmax_h_mm  vertex_h_mm vertex_h_95ci_mm
density                                           
w10              5.0          5.0         5.0-11.2
w100             5.0          5.0          5.0-8.2
w30              5.0          5.0          5.0-5.0
```

Vertex shift w100 vs w10: +0.11 halvings finer (bootstrap 95% CI -0.60 to +1.14; P(shift > 0) = 0.33). Prediction: > 0.

## H3: does density matter once coverage is known?

Figure: main_xsess.png

## Controls (selected epoch; paired on person x seed)

Mean AUC over people and seeds (rows: encoding; columns: density):

```
density             w10     w30    w100
spatial                                
electrode_id     0.6382  0.8040  0.8801
grid10           0.5933  0.7616  0.8687
grid20           0.5917  0.6998  0.8277
grid20_shuffled  0.5558  0.6602  0.7910
grid40           0.5492  0.6635  0.7567
grid5            0.5969  0.8170  0.8772
grid80           0.5466  0.5926  0.6955
none             0.5442  0.5435  0.6016
```

Does LOCATION matter, or just shared tags?
- w10: grid20 - grid20_shuffled = +0.0359 (5/6 people > 0, sign-flip p = 0.062; 48 pairs)
- w30: grid20 - grid20_shuffled = +0.0396 (5/6 people > 0, sign-flip p = 0.094; 48 pairs)
- w100: grid20 - grid20_shuffled = +0.0366 (6/6 people > 0, sign-flip p = 0.031; 48 pairs)

Per-electrode tags vs nothing:
- w10: electrode_id - none = +0.0939 (6/6 people > 0, sign-flip p = 0.031; 48 pairs)
- w30: electrode_id - none = +0.2605 (6/6 people > 0, sign-flip p = 0.031; 48 pairs)
- w100: electrode_id - none = +0.2786 (6/6 people > 0, sign-flip p = 0.031; 48 pairs)

Best cube (by mean over people, per density) vs per-electrode tags:
- w10 (best = grid5): grid5 - electrode_id = -0.0413 (0/6 people > 0, sign-flip p = 0.031; 48 pairs)
- w30 (best = grid5): grid5 - electrode_id = +0.0131 (4/6 people > 0, sign-flip p = 0.594; 48 pairs)
- w100 (best = grid5): grid5 - electrode_id = -0.0029 (2/6 people > 0, sign-flip p = 0.656; 48 pairs)

Seed SD of AUC per (encoding, density, person): median 0.0444, 90th percentile 0.1244 (planned ~0.025).

## Training health

```
         runs  left_plateau  epochs_run  best_epoch  train_windows
density                                                           
w10        64          1.00        21.0        10.0         2100.0
w100       64          0.98        21.0        10.0        21000.0
w30        64          1.00        21.0        10.0         6300.0
```
