# Density x resolution

Outcome: test AUC at the selected (early-stopped) epoch. Gain = AUC(cube) - AUC(none), paired on person, density and seed, then averaged over seeds. Analysis fixed in the sweep file header before any run.

216 runs; 6 held-out people; seeds per cell: 12-12.

Mean gain over people (rows: cube size mm; columns: density level):

```
density    s100     s25     s50
h                              
80.0     0.0182  0.0040  0.0151
40.0     0.0542  0.0051  0.0034
20.0     0.0394  0.0102  0.0277
10.0     0.0571 -0.0055  0.0338
5.0      0.0447 -0.0083  0.0199
```

Mean coverage:

```
density   s100    s25    s50
h                           
80.0     1.000  0.945  1.000
40.0     0.945  0.715  0.901
20.0     0.766  0.355  0.609
10.0     0.365  0.106  0.203
5.0      0.074  0.025  0.043
```

## H1 (primary): does finer resolution pay off more with more data?

Slope of gain per halving of cube size, per person:

```
density    s100     s25     s50
subject                        
btb01    0.0236 -0.0031  0.0117
btb02   -0.0125 -0.0099 -0.0005
btb03    0.0130 -0.0114  0.0094
btb04    0.0035  0.0039  0.0003
btb07   -0.0030  0.0014 -0.0015
btb10    0.0090 -0.0022  0.0046
```

**Interaction slope(s100) - slope(s25) = +0.0091 AUC per halving; permutation p = 0.0090** (density labels permuted within 72 person x seed blocks; prediction: > 0)

- population level: 3/6 people > 0; exact sign-flip p = 0.250 (n = 6 caps this at p >= 0.031)
- secondary: slope(s50) - slope(s25) = +0.0075; 4/6 > 0; p = 0.125
- secondary: slope(s100) - slope(s50) = +0.0016; 4/6 > 0; p = 0.625

## H2: how far does the best cube size move?

Pooled vertex regression, best cube size at each density: level 25: 36.4 mm, level 50: 15.5 mm, level 100: 11.3 mm

**dx*/ds = +0.84 octaves finer per doubling of the density level** (bootstrap over people 95% CI -0.00 to +2.00; P(> 0) = 0.95). Prediction: > 0; variance-only theory 0.2-0.33.

Descriptive, per density:

```
         argmax_h_mm  vertex_h_mm vertex_h_95ci_mm
density                                           
s100            10.0         13.1         5.0-24.5
s25             20.0         36.4         5.0-80.0
s50             10.0          8.4         5.0-19.7
```

Vertex shift s100 vs s25: +1.68 halvings finer (bootstrap 95% CI -1.21 to +3.74; P(shift > 0) = 0.95). Prediction: > 0.

## H3: does density matter once coverage is known?

Leave-one-person-out R^2: gain ~ log2 h * coverage = -0.064; + density = -0.090 (adds -0.026). Small or negative: density acts through coverage.

Figure: heldout_shafts.png
