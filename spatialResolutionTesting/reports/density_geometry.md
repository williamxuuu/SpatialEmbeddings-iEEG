# Electrode density vs spatial resolution (coordinates only)

Same assignment code and nested training subsets as the density_resolution sweep; mean over 6 held-out people (and 6 subset draws for k = 1, 2). k = training people.

## Regions training occupies

```
k                        1      2      5
encoding                                
lobes                  8.1   11.9   16.5
ho                    29.9   48.5   71.5
destrieux             32.4   56.8   93.2
schaefer100           24.0   40.6   65.0
schaefer200           36.0   62.8  106.8
schaefer400           48.8   87.5  159.3
schaefer1000          68.4  129.4  264.0
lobes_gauss3           9.9   14.2   18.5
ho_gauss3             36.7   57.6   82.3
destrieux_gauss3      46.0   75.2  111.2
schaefer100_gauss3    31.8   51.5   78.2
schaefer200_gauss3    50.6   83.8  134.3
schaefer400_gauss3    80.1  136.3  225.0
schaefer1000_gauss3  137.6  244.7  434.7
grid80                 5.5    7.1    8.0
grid57                11.2   16.6   22.8
grid40                15.5   24.3   37.3
grid28                25.2   42.3   71.0
grid20                41.7   73.4  135.7
grid14                59.1  110.4  222.8
grid10                79.4  155.7  341.3
grid5                128.3  262.0  645.2
```

## Median training electrodes per occupied region

```
k                       1     2      5
encoding                              
lobes                11.7  14.8   23.1
ho                    4.1   5.2    9.2
destrieux             3.5   4.1    6.6
schaefer100           5.1   6.3    9.8
schaefer200           3.7   4.5    6.0
schaefer400           2.6   3.1    4.0
schaefer1000          2.0   2.0    2.0
lobes_gauss3          8.3  10.2   19.2
ho_gauss3             3.4   4.4    7.8
destrieux_gauss3      2.1   2.7    4.9
schaefer100_gauss3    4.3   5.0    7.6
schaefer200_gauss3    2.8   3.3    4.5
schaefer400_gauss3    1.4   1.8    2.7
schaefer1000_gauss3   0.1   0.6    1.0
grid80               28.4  44.8  104.1
grid57                9.7  12.0   21.4
grid40                8.4  10.1   16.7
grid28                5.5   6.1    8.0
grid20                3.4   3.9    4.4
grid14                2.4   2.7    3.0
grid10                2.0   2.0    2.0
grid5                 1.0   1.0    1.0
```

## Share of occupied regions with < 3 training electrodes

```
k                       1     2     5
encoding                             
lobes                0.10  0.10  0.11
ho                   0.28  0.22  0.09
destrieux            0.35  0.31  0.19
schaefer100          0.20  0.15  0.09
schaefer200          0.26  0.25  0.20
schaefer400          0.45  0.41  0.31
schaefer1000         0.69  0.64  0.54
lobes_gauss3         0.26  0.26  0.21
ho_gauss3            0.39  0.33  0.20
destrieux_gauss3     0.55  0.48  0.35
schaefer100_gauss3   0.38  0.32  0.22
schaefer200_gauss3   0.47  0.42  0.33
schaefer400_gauss3   0.65  0.59  0.49
schaefer1000_gauss3  0.84  0.80  0.72
grid80               0.00  0.00  0.00
grid57               0.18  0.17  0.12
grid40               0.08  0.06  0.07
grid28               0.26  0.22  0.18
grid20               0.36  0.33  0.29
grid14               0.53  0.48  0.43
grid10               0.70  0.67  0.59
grid5                0.99  0.99  0.98
```

## Coverage: held-out weight on regions training saw

```
k                        1      2      5
encoding                                
lobes                0.460  0.683  0.963
ho                   0.318  0.551  0.892
destrieux            0.273  0.482  0.851
schaefer100          0.310  0.535  0.866
schaefer200          0.251  0.453  0.822
schaefer400          0.206  0.383  0.727
schaefer1000         0.142  0.273  0.586
lobes_gauss3         0.636  0.861  0.984
ho_gauss3            0.369  0.614  0.958
destrieux_gauss3     0.361  0.587  0.955
schaefer100_gauss3   0.370  0.604  0.933
schaefer200_gauss3   0.320  0.541  0.913
schaefer400_gauss3   0.293  0.507  0.887
schaefer1000_gauss3  0.237  0.428  0.797
grid80               0.569  0.804  1.000
grid57               0.525  0.758  0.974
grid40               0.430  0.654  0.925
grid28               0.345  0.554  0.851
grid20               0.256  0.436  0.742
grid14               0.156  0.291  0.619
grid10               0.087  0.188  0.437
grid5                0.017  0.044  0.118
```

## Coverage soft adds (gauss3 - hard)

```
k                 1      2      5
base                             
lobes         0.176  0.178  0.021
ho            0.051  0.062  0.066
destrieux     0.088  0.105  0.104
schaefer100   0.060  0.069  0.067
schaefer200   0.069  0.088  0.090
schaefer400   0.087  0.124  0.160
schaefer1000  0.095  0.155  0.212
```
