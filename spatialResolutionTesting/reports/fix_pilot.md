# Fix pilot report

126 runs from `results/fix_pilot` (planned 126).

## Means per arm (over people and seeds)

                         sel   last  oracle  sel_epoch  track  sel<0.5  last<0.5   n
arm                                                                                 
grid20|d0              0.517  0.554   0.648      3.611 -0.109    0.500     0.500  18
grid20|d50             0.553  0.605   0.658      3.278 -0.219    0.389     0.389  18
none|d0                0.514  0.512   0.572      7.444 -0.082    0.389     0.389  18
schaefer1000_hard|d0   0.542  0.553   0.627      4.167  0.068    0.333     0.167  18
schaefer1000_hard|d50  0.572  0.583   0.643      7.500  0.149    0.278     0.111  18
schaefer400_hard|d0    0.562  0.577   0.647      4.167 -0.021    0.333     0.278  18
schaefer400_hard|d50   0.591  0.594   0.655      5.222  0.176    0.278     0.167  18 

## Per person, `sel` (mean over seeds)

fold                    ho01   ho02   ho03   ho04   ho07   ho10
arm                                                            
grid20|d0              0.527  0.508  0.549  0.602  0.484  0.434
grid20|d50             0.554  0.549  0.679  0.599  0.479  0.460
none|d0                0.427  0.607  0.589  0.523  0.501  0.438
schaefer1000_hard|d0   0.493  0.588  0.610  0.611  0.522  0.427
schaefer1000_hard|d50  0.516  0.593  0.726  0.585  0.503  0.507
schaefer400_hard|d0    0.706  0.628  0.567  0.565  0.422  0.482
schaefer400_hard|d50   0.582  0.601  0.886  0.571  0.468  0.437 

## Q1 selection: sel - last (mean, share > 0)

                        mean  <lambda_0>
arm                                     
grid20|d0             -0.037       0.500
grid20|d50            -0.051       0.611
none|d0                0.002       0.444
schaefer1000_hard|d0  -0.011       0.500
schaefer1000_hard|d50 -0.011       0.444
schaefer400_hard|d0   -0.015       0.444
schaefer400_hard|d50  -0.003       0.500 

## Q2 code dropout: d50 - d0, paired on person x encoding x seed

- sel: overall +0.032 (n=54, 63% > 0); by encoding grid20 +0.036, schaefer1000_hard +0.030, schaefer400_hard +0.029
- last: overall +0.032 (n=54, 63% > 0); by encoding grid20 +0.050, schaefer1000_hard +0.030, schaefer400_hard +0.017

## Q3 noise: seed SD of `sel` per (person, arm), median over people

                       sd_sel  sd_last
arm                                   
grid20|d0               0.021    0.038
grid20|d50              0.061    0.025
none|d0                 0.020    0.022
schaefer1000_hard|d0    0.048    0.057
schaefer1000_hard|d50   0.068    0.029
schaefer400_hard|d0     0.039    0.086
schaefer400_hard|d50    0.039    0.053 

Before the fix: 0.07-0.13 (last epoch, 5 training people, 40 epochs).

## Validation person per fold

fold
ho01    btb02
ho02    btb03
ho03    btb04
ho04    btb10
ho07    btb10
ho10    btb01
