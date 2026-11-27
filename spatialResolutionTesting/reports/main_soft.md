# Soft vs hard regions, within person (main protocol)

Outcome: test AUC at the selected (early-stopped) epoch. Contrasts paired on person x seed, seed mean per person, exact sign-flip over people. Analysis fixed in configs/sweeps/main_soft_xsess.yaml.

Mean AUC over people and seeds:

```
density                         w10     w30    w100
spatial                                            
electrode_id                 0.6382  0.8040  0.8801
grid10                       0.5933  0.7616  0.8687
grid20                       0.5917  0.6998  0.8277
none                         0.5442  0.5435  0.6016
schaefer1000_gauss3          0.5816  0.7470  0.8477
schaefer1000_gauss3_randnbr  0.5839  0.7208  0.8519
schaefer1000_hard            0.5904  0.7546  0.8398
schaefer400_gauss3           0.5751  0.7105  0.8511
schaefer400_gauss3_randnbr   0.5794  0.7098  0.8304
schaefer400_hard             0.5857  0.6924  0.8217
```

## schaefer400

S1 soft - hard:
- w10: schaefer400_gauss3 - schaefer400_hard = -0.0106 (1/6 people > 0, sign-flip p = 0.219; 48 pairs)
- w30: schaefer400_gauss3 - schaefer400_hard = +0.0180 (5/6 people > 0, sign-flip p = 0.125; 48 pairs)
- w100: schaefer400_gauss3 - schaefer400_hard = +0.0294 (4/6 people > 0, sign-flip p = 0.125; 48 pairs)

S2 soft - random neighbours:
- w10: schaefer400_gauss3 - schaefer400_gauss3_randnbr = -0.0043 (2/6 people > 0, sign-flip p = 0.344; 48 pairs)
- w30: schaefer400_gauss3 - schaefer400_gauss3_randnbr = +0.0007 (3/6 people > 0, sign-flip p = 0.969; 48 pairs)
- w100: schaefer400_gauss3 - schaefer400_gauss3_randnbr = +0.0208 (6/6 people > 0, sign-flip p = 0.031; 48 pairs)

S3 hard - per-electrode tags:
- w10: schaefer400_hard - electrode_id = -0.0525 (1/6 people > 0, sign-flip p = 0.062; 48 pairs)
- w30: schaefer400_hard - electrode_id = -0.1115 (0/6 people > 0, sign-flip p = 0.031; 48 pairs)
- w100: schaefer400_hard - electrode_id = -0.0584 (0/6 people > 0, sign-flip p = 0.031; 48 pairs)

soft - none:
- w10: schaefer400_gauss3 - none = +0.0308 (5/6 people > 0, sign-flip p = 0.062; 48 pairs)
- w30: schaefer400_gauss3 - none = +0.1670 (6/6 people > 0, sign-flip p = 0.031; 48 pairs)
- w100: schaefer400_gauss3 - none = +0.2496 (6/6 people > 0, sign-flip p = 0.031; 48 pairs)

## schaefer1000

S1 soft - hard:
- w10: schaefer1000_gauss3 - schaefer1000_hard = -0.0088 (2/6 people > 0, sign-flip p = 0.281; 48 pairs)
- w30: schaefer1000_gauss3 - schaefer1000_hard = -0.0076 (3/6 people > 0, sign-flip p = 0.594; 48 pairs)
- w100: schaefer1000_gauss3 - schaefer1000_hard = +0.0079 (4/6 people > 0, sign-flip p = 0.469; 48 pairs)

S2 soft - random neighbours:
- w10: schaefer1000_gauss3 - schaefer1000_gauss3_randnbr = -0.0022 (4/6 people > 0, sign-flip p = 0.969; 48 pairs)
- w30: schaefer1000_gauss3 - schaefer1000_gauss3_randnbr = +0.0262 (5/6 people > 0, sign-flip p = 0.188; 48 pairs)
- w100: schaefer1000_gauss3 - schaefer1000_gauss3_randnbr = -0.0041 (4/6 people > 0, sign-flip p = 0.844; 48 pairs)

S3 hard - per-electrode tags:
- w10: schaefer1000_hard - electrode_id = -0.0477 (1/6 people > 0, sign-flip p = 0.062; 48 pairs)
- w30: schaefer1000_hard - electrode_id = -0.0493 (0/6 people > 0, sign-flip p = 0.031; 48 pairs)
- w100: schaefer1000_hard - electrode_id = -0.0403 (1/6 people > 0, sign-flip p = 0.125; 48 pairs)

soft - none:
- w10: schaefer1000_gauss3 - none = +0.0374 (5/6 people > 0, sign-flip p = 0.062; 48 pairs)
- w30: schaefer1000_gauss3 - none = +0.2035 (6/6 people > 0, sign-flip p = 0.031; 48 pairs)
- w100: schaefer1000_gauss3 - none = +0.2462 (6/6 people > 0, sign-flip p = 0.031; 48 pairs)

Seed SD of AUC per (encoding, density, person): median 0.0421 (planned ~0.021).
