# New people, extended: resolution x density, soft vs hard

Outcome: AUC on the second half of the held-out person's recording, last epoch (fixed 20 epochs); cal_f = frozen model, linear readout fit on the first f of the first half. e100 includes results/main_xsubj. Analysis fixed in configs/sweeps/main_xsubj_ext.yaml.

576 runs.

## e30

Mean over people and seeds (unknown_w = share of the new person's code weight on <unknown>):

```
         zero_shot  cal_0.1  cal_0.25   cal_1  unknown_w
spatial                                                 
grid10      0.5627   0.6283    0.6389  0.6893     0.7571
grid20      0.5663   0.6237    0.6372  0.6977     0.3515
grid40      0.5678   0.6073    0.6212  0.6779     0.0818
grid5       0.5131   0.6089    0.6231  0.6683     0.9583
grid80      0.5394   0.6221    0.6403  0.6902     0.0000
none        0.5272   0.5827    0.5986  0.6346        NaN
```

C1 gain over none, sign-flip over people:

- zero_shot grid80: +0.0121 (4/6 people > 0, p = 0.312)
- zero_shot grid40: +0.0405 (4/6 people > 0, p = 0.188)
- zero_shot grid20: +0.0391 (4/6 people > 0, p = 0.219)
- zero_shot grid10: +0.0354 (3/6 people > 0, p = 0.250)
- zero_shot grid5: -0.0142 (1/6 people > 0, p = 0.438)
  best cube for zero_shot: grid40
- cal_0.1 grid80: +0.0395 (5/6 people > 0, p = 0.062)
- cal_0.1 grid40: +0.0246 (5/6 people > 0, p = 0.094)
- cal_0.1 grid20: +0.0411 (6/6 people > 0, p = 0.031)
- cal_0.1 grid10: +0.0456 (6/6 people > 0, p = 0.031)
- cal_0.1 grid5: +0.0263 (5/6 people > 0, p = 0.156)
  best cube for cal_0.1: grid10
- cal_0.25 grid80: +0.0417 (6/6 people > 0, p = 0.031)
- cal_0.25 grid40: +0.0225 (4/6 people > 0, p = 0.188)
- cal_0.25 grid20: +0.0385 (5/6 people > 0, p = 0.094)
- cal_0.25 grid10: +0.0403 (4/6 people > 0, p = 0.188)
- cal_0.25 grid5: +0.0244 (4/6 people > 0, p = 0.406)
  best cube for cal_0.25: grid80
- cal_1 grid80: +0.0556 (6/6 people > 0, p = 0.031)
- cal_1 grid40: +0.0433 (5/6 people > 0, p = 0.062)
- cal_1 grid20: +0.0631 (6/6 people > 0, p = 0.031)
- cal_1 grid10: +0.0547 (6/6 people > 0, p = 0.031)
- cal_1 grid5: +0.0338 (5/6 people > 0, p = 0.062)
  best cube for cal_1: grid20

## e100

Mean over people and seeds (unknown_w = share of the new person's code weight on <unknown>):

```
                     zero_shot  cal_0.1  cal_0.25   cal_1  unknown_w
spatial                                                             
grid10                  0.5848   0.6326    0.6483  0.7082     0.5630
grid20                  0.5975   0.6560    0.6659  0.7235     0.2579
grid40                  0.5657   0.6469    0.6358  0.6873     0.0752
grid5                   0.5774   0.6363    0.6491  0.6974     0.8816
grid80                  0.5406   0.6114    0.6155  0.6609     0.0000
none                    0.5451   0.5707    0.5935  0.6251        NaN
schaefer1000_gauss3     0.6241   0.6846    0.6901  0.7321     0.2025
schaefer1000_hard       0.5274   0.6166    0.6330  0.6886     0.4143
schaefer400_gauss3      0.5834   0.6854    0.6887  0.7361     0.1131
schaefer400_hard        0.5477   0.6683    0.6872  0.7391     0.2735
```

C1 gain over none, sign-flip over people:

- zero_shot grid80: -0.0045 (2/6 people > 0, p = 0.906)
- zero_shot grid40: +0.0206 (3/6 people > 0, p = 0.844)
- zero_shot grid20: +0.0524 (3/6 people > 0, p = 0.438)
- zero_shot grid10: +0.0397 (3/6 people > 0, p = 0.312)
- zero_shot grid5: +0.0323 (4/6 people > 0, p = 0.469)
- zero_shot schaefer1000_gauss3: +0.0790 (4/6 people > 0, p = 0.188)
- zero_shot schaefer1000_hard: -0.0177 (3/6 people > 0, p = 0.844)
- zero_shot schaefer400_gauss3: +0.0382 (5/6 people > 0, p = 0.281)
- zero_shot schaefer400_hard: +0.0025 (3/6 people > 0, p = 0.906)
  best cube for zero_shot: grid20
- cal_0.1 grid80: +0.0406 (5/6 people > 0, p = 0.094)
- cal_0.1 grid40: +0.0762 (6/6 people > 0, p = 0.031)
- cal_0.1 grid20: +0.0852 (6/6 people > 0, p = 0.031)
- cal_0.1 grid10: +0.0618 (3/6 people > 0, p = 0.250)
- cal_0.1 grid5: +0.0656 (5/6 people > 0, p = 0.156)
- cal_0.1 schaefer1000_gauss3: +0.1139 (6/6 people > 0, p = 0.031)
- cal_0.1 schaefer1000_hard: +0.0459 (4/6 people > 0, p = 0.188)
- cal_0.1 schaefer400_gauss3: +0.1146 (6/6 people > 0, p = 0.031)
- cal_0.1 schaefer400_hard: +0.0976 (5/6 people > 0, p = 0.062)
  best cube for cal_0.1: grid20
- cal_0.25 grid80: +0.0220 (3/6 people > 0, p = 0.594)
- cal_0.25 grid40: +0.0423 (4/6 people > 0, p = 0.312)
- cal_0.25 grid20: +0.0724 (4/6 people > 0, p = 0.125)
- cal_0.25 grid10: +0.0548 (4/6 people > 0, p = 0.156)
- cal_0.25 grid5: +0.0557 (4/6 people > 0, p = 0.219)
- cal_0.25 schaefer1000_gauss3: +0.0966 (5/6 people > 0, p = 0.094)
- cal_0.25 schaefer1000_hard: +0.0396 (4/6 people > 0, p = 0.312)
- cal_0.25 schaefer400_gauss3: +0.0952 (5/6 people > 0, p = 0.094)
- cal_0.25 schaefer400_hard: +0.0937 (5/6 people > 0, p = 0.062)
  best cube for cal_0.25: grid20
- cal_1 grid80: +0.0358 (5/6 people > 0, p = 0.062)
- cal_1 grid40: +0.0622 (5/6 people > 0, p = 0.062)
- cal_1 grid20: +0.0984 (6/6 people > 0, p = 0.031)
- cal_1 grid10: +0.0831 (6/6 people > 0, p = 0.031)
- cal_1 grid5: +0.0723 (6/6 people > 0, p = 0.031)
- cal_1 schaefer1000_gauss3: +0.1070 (6/6 people > 0, p = 0.031)
- cal_1 schaefer1000_hard: +0.0635 (5/6 people > 0, p = 0.125)
- cal_1 schaefer400_gauss3: +0.1110 (6/6 people > 0, p = 0.031)
- cal_1 schaefer400_hard: +0.1140 (6/6 people > 0, p = 0.031)
  best cube for cal_1: grid20

## C2 soft - hard (e100), paired on person x seed, sign-flip over people

- schaefer400 zero_shot: +0.0357 (4/6 people > 0, p = 0.312)
- schaefer400 cal_0.1: +0.0171 (5/6 people > 0, p = 0.281)
- schaefer400 cal_0.25: +0.0015 (3/6 people > 0, p = 0.938)
- schaefer400 cal_1: -0.0030 (3/6 people > 0, p = 0.812)
- schaefer1000 zero_shot: +0.0967 (3/6 people > 0, p = 0.250)
- schaefer1000 cal_0.1: +0.0680 (5/6 people > 0, p = 0.062)
- schaefer1000 cal_0.25: +0.0570 (5/6 people > 0, p = 0.156)
- schaefer1000 cal_1: +0.0435 (4/6 people > 0, p = 0.375)

Median seed SD per (encoding, density, person):

```
                             zero_shot  cal_0.1  cal_0.25  cal_1
density spatial                                                 
e100    grid10                   0.055    0.057     0.047  0.045
        grid20                   0.041    0.031     0.024  0.021
        grid40                   0.072    0.028     0.023  0.022
        grid5                    0.034    0.034     0.026  0.031
        grid80                   0.042    0.053     0.038  0.040
        none                     0.021    0.029     0.027  0.033
        schaefer1000_gauss3      0.058    0.053     0.046  0.032
        schaefer1000_hard        0.047    0.039     0.027  0.036
        schaefer400_gauss3       0.068    0.049     0.039  0.042
        schaefer400_hard         0.064    0.041     0.029  0.026
e30     grid10                   0.061    0.046     0.055  0.053
        grid20                   0.064    0.050     0.049  0.047
        grid40                   0.067    0.057     0.041  0.045
        grid5                    0.066    0.058     0.051  0.049
        grid80                   0.084    0.046     0.043  0.041
        none                     0.056    0.029     0.038  0.043
```
