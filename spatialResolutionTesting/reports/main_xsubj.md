# New people: zero-shot and calibrated

Outcome: AUC on the second half of the held-out person's recording, last epoch (fixed 20 epochs). cal_f = model frozen, linear readout fit on the first f of the first half. Analysis fixed in configs/sweeps/main_xsubj.yaml.

144 runs.

Mean over people and seeds:

```
         zero_shot  cal_0.1  cal_0.25   cal_1
spatial                                      
grid10      0.5848   0.6326    0.6483  0.7082
grid20      0.5975   0.6560    0.6659  0.7235
grid40      0.5657   0.6469    0.6358  0.6873
none        0.5451   0.5707    0.5935  0.6251
```

Per person (seed mean):

zero_shot:
```
fold      ho01   ho02   ho03   ho04   ho07   ho10
spatial                                          
grid10   0.757  0.519  0.631  0.506  0.491  0.605
grid20   0.735  0.532  0.766  0.499  0.466  0.587
grid40   0.563  0.564  0.735  0.542  0.458  0.533
none     0.660  0.569  0.488  0.539  0.514  0.501
```

cal_0.1:
```
fold      ho01   ho02   ho03   ho04   ho07   ho10
spatial                                          
grid10   0.823  0.599  0.779  0.499  0.527  0.569
grid20   0.776  0.582  0.866  0.543  0.539  0.629
grid40   0.694  0.618  0.852  0.531  0.554  0.633
none     0.647  0.514  0.642  0.503  0.539  0.579
```

cal_0.25:
```
fold      ho01   ho02   ho03   ho04   ho07   ho10
spatial                                          
grid10   0.858  0.581  0.805  0.518  0.482  0.645
grid20   0.870  0.558  0.876  0.532  0.473  0.686
grid40   0.741  0.587  0.863  0.521  0.451  0.652
none     0.724  0.519  0.669  0.535  0.506  0.608
```

cal_1:
```
fold      ho01   ho02   ho03   ho04   ho07   ho10
spatial                                          
grid10   0.882  0.705  0.847  0.612  0.505  0.698
grid20   0.885  0.688  0.896  0.642  0.496  0.735
grid40   0.772  0.699  0.886  0.639  0.449  0.679
none     0.758  0.585  0.726  0.606  0.454  0.621
```

Gain over none, per person seed mean, sign-flip over people:

- zero_shot grid10: +0.0397 (3/6 people > 0, p = 0.312)
- zero_shot grid20: +0.0524 (3/6 people > 0, p = 0.438)
- zero_shot grid40: +0.0206 (3/6 people > 0, p = 0.844)
- cal_0.1 grid10: +0.0618 (3/6 people > 0, p = 0.250)
- cal_0.1 grid20: +0.0852 (6/6 people > 0, p = 0.031)
- cal_0.1 grid40: +0.0762 (6/6 people > 0, p = 0.031)
- cal_0.25 grid10: +0.0548 (4/6 people > 0, p = 0.156)
- cal_0.25 grid20: +0.0724 (4/6 people > 0, p = 0.125)
- cal_0.25 grid40: +0.0423 (4/6 people > 0, p = 0.312)
- cal_1 grid10: +0.0831 (6/6 people > 0, p = 0.031)
- cal_1 grid20: +0.0984 (6/6 people > 0, p = 0.031)
- cal_1 grid40: +0.0622 (5/6 people > 0, p = 0.062)

Share of runs below 0.5, and median seed SD per (encoding, person):

```
         below05_zero_shot  below05_cal_0.1  below05_cal_0.25  below05_cal_1  sd_zero_shot  sd_cal_0.1  sd_cal_0.25  sd_cal_1
spatial                                                                                                                      
grid10               0.250            0.139             0.222          0.083         0.055       0.057        0.047     0.045
grid20               0.250            0.028             0.250          0.083         0.041       0.031        0.024     0.021
grid40               0.361            0.056             0.250          0.167         0.072       0.028        0.023     0.022
none                 0.306            0.111             0.111          0.167         0.021       0.029        0.027     0.033
```
