# Experiment 2 Phase 0: soft vs hard (v2 protocol)

Feasibility and effect-size check, not a significance test. Rules: sweep headers.

## Cross-session (phase0_soft_xsess)

84 runs. Validation AUC (last epoch), mean over seeds:

```
density                        full    half
spatial                                    
none                         0.5823  0.5828
schaefer1000_gauss3          0.7540  0.7809
schaefer1000_gauss3_randnbr  0.7550  0.7585
schaefer1000_hard            0.7356  0.7443
schaefer400_gauss3           0.7713  0.7493
schaefer400_gauss3_randnbr   0.7167  0.7348
schaefer400_hard             0.7372  0.7589
```

- [PASS] F0 schaefer400/full: 18/18 runs leave the plateau by epoch 10 (need >= 89%)
- [PASS] D1 soft - hard schaefer400/full: +0.0341 (seed SD 0.0534, 4/6 seeds > 0)
- [PASS] D2 soft - randnbr schaefer400/full: +0.0546 (seed SD 0.0287, 6/6 seeds > 0)
- [PASS] F0 schaefer400/half: 18/18 runs leave the plateau by epoch 10 (need >= 89%)
- [FAIL] D1 soft - hard schaefer400/half: -0.0095 (seed SD 0.0495, 2/6 seeds > 0)
- [PASS] D2 soft - randnbr schaefer400/half: +0.0145 (seed SD 0.0380, 5/6 seeds > 0)
- [PASS] F0 schaefer1000/full: 18/18 runs leave the plateau by epoch 10 (need >= 89%)
- [PASS] D1 soft - hard schaefer1000/full: +0.0184 (seed SD 0.0271, 4/6 seeds > 0)
- [FAIL] D2 soft - randnbr schaefer1000/full: -0.0010 (seed SD 0.0502, 2/6 seeds > 0)
- [PASS] F0 schaefer1000/half: 18/18 runs leave the plateau by epoch 10 (need >= 89%)
- [PASS] D1 soft - hard schaefer1000/half: +0.0366 (seed SD 0.0315, 6/6 seeds > 0)
- [PASS] D2 soft - randnbr schaefer1000/half: +0.0224 (seed SD 0.0247, 5/6 seeds > 0)
- [FAIL] D3 density schaefer400: soft - hard at half -0.0095 vs full +0.0341 (predict half > full)
- [PASS] D3 density schaefer1000: soft - hard at half +0.0366 vs full +0.0184 (predict half > full)
- [FAIL] D3 resolution: soft - hard schaefer1000 +0.0184 vs schaefer400 +0.0341 at full (predict schaefer1000 >)

(For information only, after the rules) test AUC per person, mean over seeds:

```
subject                              btb01  btb02  btb03  btb04  btb07  btb10
density spatial                                                              
full    none                         0.732  0.551  0.599  0.555  0.512  0.540
        schaefer1000_gauss3          0.890  0.842  0.855  0.756  0.631  0.696
        schaefer1000_gauss3_randnbr  0.889  0.807  0.836  0.757  0.734  0.697
        schaefer1000_hard            0.880  0.852  0.857  0.755  0.628  0.627
        schaefer400_gauss3           0.912  0.872  0.821  0.760  0.689  0.737
        schaefer400_gauss3_randnbr   0.831  0.763  0.844  0.735  0.675  0.641
        schaefer400_hard             0.853  0.858  0.829  0.730  0.625  0.678
half    none                         0.700  0.541  0.692  0.554  0.496  0.502
        schaefer1000_gauss3          0.880  0.863  0.824  0.740  0.759  0.748
        schaefer1000_gauss3_randnbr  0.887  0.822  0.847  0.713  0.746  0.715
        schaefer1000_hard            0.878  0.853  0.809  0.741  0.695  0.662
        schaefer400_gauss3           0.855  0.872  0.636  0.719  0.721  0.723
        schaefer400_gauss3_randnbr   0.851  0.840  0.709  0.712  0.713  0.701
        schaefer400_hard             0.866  0.849  0.761  0.726  0.732  0.729
```

## Cross-subject (phase0_soft_xsubj)

267 runs. Validation AUC (last epoch), mean over folds and seeds:

```
size                             k5
spatial                            
schaefer1000_gauss3          0.6087
schaefer1000_gauss3_nearest  0.5855
schaefer1000_hard            0.4802
schaefer1000_hard_nearest    0.5204
schaefer400_gauss3           0.5796
schaefer400_gauss3_nearest   0.6227
schaefer400_hard             0.5662
schaefer400_hard_nearest     0.5800
```

- [PASS] F0 schaefer400/k5: 144/144 runs leave the plateau by epoch 10 (need >= 75%)
- [PASS] D1 soft - hard schaefer400/k5: +0.0134 (per fold: ho01 +0.0830, ho02 -0.0538, ho03 +0.0110; 15/36 runs > 0; sign-flip p = 0.5661 over person x seed pairs, 0.750 over people)
- [PASS] D4 equal coverage schaefer400/k5: +0.0427 (per fold: ho01 +0.1580, ho02 -0.0776, ho03 +0.0477; 21/36 runs > 0; sign-flip p = 0.0904 over person x seed pairs, 0.750 over people)
- [PASS] F0 schaefer1000/k5: 123/123 runs leave the plateau by epoch 10 (need >= 75%)
- [PASS] D1 soft - hard schaefer1000/k5: +0.1286 (per fold: ho01 +0.2613, ho02 -0.0630, ho03 +0.1874; 27/36 runs > 0; sign-flip p = 0.0001 over person x seed pairs, 0.500 over people)
- [PASS] D4 equal coverage schaefer1000/k5: +0.0626 (per fold: ho01 +0.1767, ho02 -0.0514; 16/24 runs > 0; sign-flip p = 0.0538 over person x seed pairs, 1.000 over people)

Held-out weight on regions training never saw (mean):

```
size                            k5
spatial                           
schaefer1000_gauss3          0.172
schaefer1000_gauss3_nearest  0.000
schaefer1000_hard            0.390
schaefer1000_hard_nearest    0.004
schaefer400_gauss3           0.100
schaefer400_gauss3_nearest   0.000
schaefer400_hard             0.267
schaefer400_hard_nearest     0.005
```


(For information only, after the rules) test AUC of the held-out person, mean over seeds:

```
fold                               ho01   ho02   ho03
size spatial                                         
k5   schaefer1000_gauss3          0.660  0.552  0.695
     schaefer1000_gauss3_nearest  0.680  0.548    NaN
     schaefer1000_hard            0.389  0.606  0.463
     schaefer1000_hard_nearest    0.487  0.594  0.506
     schaefer400_gauss3           0.639  0.499  0.630
     schaefer400_gauss3_nearest   0.723  0.519  0.616
     schaefer400_hard             0.546  0.575  0.606
     schaefer400_hard_nearest     0.560  0.602  0.581
```

