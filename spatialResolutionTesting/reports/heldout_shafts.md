# Held-out shafts: location-only transfer to unseen electrodes

Outcome: AUC on the held-out shafts only (second half of the test movie), selected epoch (picked on training shafts). Shaft Laplacian reference. Replicates = shaft split x seed; seed means per person, exact sign-flip over the 6 people. Analysis fixed in configs/sweeps/heldout_shafts.yaml.

1728 person-runs. Leak check (shafts both trained and held out): 0 (OK).

Mean held-out AUC (rows: encoding; columns: density):

```
density                s25     s50    s100
spatial                                   
grid10              0.5263  0.5433  0.5553
grid20              0.5420  0.5371  0.5376
grid40              0.5369  0.5128  0.5524
grid5               0.5235  0.5293  0.5429
grid80              0.5358  0.5245  0.5164
none                0.5318  0.5094  0.4982
schaefer400_gauss3  0.5437  0.5823  0.5758
schaefer400_hard    0.5251  0.5858  0.6046
```

Mean coverage (share of held-out code weight on cells seen in training):

```
density               s25    s50   s100
spatial                                
grid10              0.106  0.203  0.365
grid20              0.355  0.609  0.766
grid40              0.715  0.901  0.945
grid5               0.025  0.043  0.074
grid80              0.945  1.000  1.000
none                1.000  1.000  1.000
schaefer400_gauss3  0.500  0.733  0.908
schaefer400_hard    0.294  0.517  0.715
```

## R1 resolution x density (cubes)

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

## R2 does the location carry to unseen shafts? (test-time checks, same model)

- grid10 s25: true - shuffled_cells = +0.0043 (4/6 people > 0, p = 0.188)
- grid10 s50: true - shuffled_cells = +0.0153 (4/6 people > 0, p = 0.219)
- grid10 s100: true - shuffled_cells = +0.0286 (4/6 people > 0, p = 0.250)
- grid10 s25: true - unknown = +0.0042 (4/6 people > 0, p = 0.125)
- grid10 s50: true - unknown = +0.0281 (5/6 people > 0, p = 0.062)
- grid10 s100: true - unknown = +0.0300 (6/6 people > 0, p = 0.031)
- grid20 s25: true - shuffled_cells = +0.0219 (3/6 people > 0, p = 0.406)
- grid20 s50: true - shuffled_cells = +0.0340 (5/6 people > 0, p = 0.094)
- grid20 s100: true - shuffled_cells = +0.0112 (5/6 people > 0, p = 0.406)
- grid20 s25: true - unknown = +0.0069 (3/6 people > 0, p = 0.656)
- grid20 s50: true - unknown = +0.0168 (4/6 people > 0, p = 0.250)
- grid20 s100: true - unknown = +0.0193 (5/6 people > 0, p = 0.062)
- grid40 s25: true - shuffled_cells = +0.0240 (5/6 people > 0, p = 0.094)
- grid40 s50: true - shuffled_cells = +0.0179 (3/6 people > 0, p = 0.656)
- grid40 s100: true - shuffled_cells = +0.0616 (6/6 people > 0, p = 0.031)
- grid40 s25: true - unknown = +0.0257 (5/6 people > 0, p = 0.094)
- grid40 s50: true - unknown = +0.0075 (3/6 people > 0, p = 0.812)
- grid40 s100: true - unknown = +0.0434 (5/6 people > 0, p = 0.125)
- grid5 s25: true - shuffled_cells = +0.0027 (3/6 people > 0, p = 0.375)
- grid5 s50: true - shuffled_cells = +0.0047 (4/6 people > 0, p = 0.219)
- grid5 s100: true - shuffled_cells = +0.0137 (6/6 people > 0, p = 0.031)
- grid5 s25: true - unknown = +0.0017 (3/6 people > 0, p = 0.250)
- grid5 s50: true - unknown = +0.0088 (6/6 people > 0, p = 0.031)
- grid5 s100: true - unknown = +0.0155 (6/6 people > 0, p = 0.031)
- grid80 s25: true - shuffled_cells = +0.0189 (5/6 people > 0, p = 0.125)
- grid80 s50: true - shuffled_cells = +0.0124 (4/6 people > 0, p = 0.312)
- grid80 s100: true - shuffled_cells = -0.0060 (2/6 people > 0, p = 0.438)
- grid80 s25: true - unknown = +0.0139 (5/6 people > 0, p = 0.062)
- grid80 s50: true - unknown = +0.0091 (4/6 people > 0, p = 0.344)
- grid80 s100: true - unknown = +0.0083 (3/6 people > 0, p = 0.438)
- schaefer400_gauss3 s25: true - shuffled_cells = +0.0403 (5/6 people > 0, p = 0.062)
- schaefer400_gauss3 s50: true - shuffled_cells = +0.0754 (6/6 people > 0, p = 0.031)
- schaefer400_gauss3 s100: true - shuffled_cells = +0.0783 (6/6 people > 0, p = 0.031)
- schaefer400_gauss3 s25: true - unknown = +0.0222 (5/6 people > 0, p = 0.062)
- schaefer400_gauss3 s50: true - unknown = +0.0617 (6/6 people > 0, p = 0.031)
- schaefer400_gauss3 s100: true - unknown = +0.0689 (6/6 people > 0, p = 0.031)
- schaefer400_hard s25: true - shuffled_cells = +0.0071 (3/6 people > 0, p = 0.844)
- schaefer400_hard s50: true - shuffled_cells = +0.0689 (5/6 people > 0, p = 0.062)
- schaefer400_hard s100: true - shuffled_cells = +0.0709 (5/6 people > 0, p = 0.062)
- schaefer400_hard s25: true - unknown = +0.0036 (3/6 people > 0, p = 0.812)
- schaefer400_hard s50: true - unknown = +0.0693 (6/6 people > 0, p = 0.031)
- schaefer400_hard s100: true - unknown = +0.0820 (6/6 people > 0, p = 0.031)

True cubes vs no location (training-time), per density:
- grid10 s25: grid10 - none = -0.0055 (1/6 people > 0, sign-flip p = 0.125; 72 pairs)
- grid10 s50: grid10 - none = +0.0338 (5/6 people > 0, sign-flip p = 0.062; 72 pairs)
- grid10 s100: grid10 - none = +0.0571 (5/6 people > 0, sign-flip p = 0.094; 72 pairs)
- grid20 s25: grid20 - none = +0.0102 (3/6 people > 0, sign-flip p = 0.438; 72 pairs)
- grid20 s50: grid20 - none = +0.0277 (4/6 people > 0, sign-flip p = 0.219; 72 pairs)
- grid20 s100: grid20 - none = +0.0394 (6/6 people > 0, sign-flip p = 0.031; 72 pairs)
- grid40 s25: grid40 - none = +0.0051 (4/6 people > 0, sign-flip p = 0.562; 72 pairs)
- grid40 s50: grid40 - none = +0.0034 (3/6 people > 0, sign-flip p = 0.844; 72 pairs)
- grid40 s100: grid40 - none = +0.0542 (5/6 people > 0, sign-flip p = 0.062; 72 pairs)
- grid5 s25: grid5 - none = -0.0083 (2/6 people > 0, sign-flip p = 0.344; 72 pairs)
- grid5 s50: grid5 - none = +0.0199 (4/6 people > 0, sign-flip p = 0.250; 72 pairs)
- grid5 s100: grid5 - none = +0.0447 (5/6 people > 0, sign-flip p = 0.062; 72 pairs)
- grid80 s25: grid80 - none = +0.0040 (3/6 people > 0, sign-flip p = 0.688; 72 pairs)
- grid80 s50: grid80 - none = +0.0151 (5/6 people > 0, sign-flip p = 0.281; 72 pairs)
- grid80 s100: grid80 - none = +0.0182 (4/6 people > 0, sign-flip p = 0.188; 72 pairs)
- schaefer400_gauss3 s25: schaefer400_gauss3 - none = +0.0119 (5/6 people > 0, sign-flip p = 0.094; 72 pairs)
- schaefer400_gauss3 s50: schaefer400_gauss3 - none = +0.0729 (6/6 people > 0, sign-flip p = 0.031; 72 pairs)
- schaefer400_gauss3 s100: schaefer400_gauss3 - none = +0.0776 (6/6 people > 0, sign-flip p = 0.031; 72 pairs)
- schaefer400_hard s25: schaefer400_hard - none = -0.0067 (3/6 people > 0, sign-flip p = 0.719; 72 pairs)
- schaefer400_hard s50: schaefer400_hard - none = +0.0763 (5/6 people > 0, sign-flip p = 0.062; 72 pairs)
- schaefer400_hard s100: schaefer400_hard - none = +0.1064 (6/6 people > 0, sign-flip p = 0.031; 72 pairs)

Schaefer 400 soft - hard:
- s25: schaefer400_gauss3 - schaefer400_hard = +0.0186 (6/6 people > 0, sign-flip p = 0.031; 72 pairs)
- s50: schaefer400_gauss3 - schaefer400_hard = -0.0034 (4/6 people > 0, sign-flip p = 0.938; 72 pairs)
- s100: schaefer400_gauss3 - schaefer400_hard = -0.0288 (0/6 people > 0, sign-flip p = 0.031; 72 pairs)

## R3 coverage

- s25: gain ~ size + size^2 (+ coverage): best size 36.4 mm without, 17.1 mm with coverage (coverage slope +0.019; nan = no interior optimum)
- s50: gain ~ size + size^2 (+ coverage): best size 8.4 mm without, 0.6 mm with coverage (coverage slope +0.029; nan = no interior optimum)
- s100: gain ~ size + size^2 (+ coverage): best size 13.1 mm without, 0.0 mm with coverage (coverage slope +0.075; nan = no interior optimum)

Gain vs distance from held-out contacts to the nearest training contact (Spearman over person x split x seed, per cube size):
- grid10: rho = -0.261
- grid20: rho = -0.137
- grid40: rho = -0.164
- grid5: rho = -0.152
- grid80: rho = -0.068

## Other test inputs (same model)

```
                       auc                 train_shafts                 all_shafts                
density               s100     s25     s50         s100     s25     s50       s100     s25     s50
spatial                                                                                           
grid10              0.5553  0.5263  0.5433       0.8670  0.7458  0.8071     0.8468  0.7088  0.7879
grid20              0.5376  0.5420  0.5371       0.8432  0.7331  0.7872     0.8107  0.7079  0.7511
grid40              0.5524  0.5369  0.5128       0.7810  0.6889  0.7422     0.7656  0.6358  0.6922
grid5               0.5429  0.5235  0.5293       0.8820  0.7217  0.8145     0.8767  0.6836  0.8002
grid80              0.5164  0.5358  0.5245       0.6386  0.6599  0.6762     0.6128  0.6067  0.6477
none                0.4982  0.5318  0.5094       0.6140  0.6130  0.6100     0.6005  0.5661  0.5810
schaefer400_gauss3  0.5758  0.5437  0.5823       0.8541  0.7321  0.8019     0.8273  0.6976  0.7779
schaefer400_hard    0.6046  0.5251  0.5858       0.8537  0.6975  0.7999     0.8377  0.6740  0.7909
```

Noise: SD across shaft splits (seed-mean) median 0.0618; SD across seeds within a split median 0.0217.
