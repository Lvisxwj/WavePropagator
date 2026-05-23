# Wave Equation

## 3d_wpo_pure

### maskA

#### train

模型: WaveMST_3D  参数量: 0.79M

[Epoch 001] Loss: 0.086775  Time: 424.9s  LR: 4.00e-04
         Test → PSNR: 23.75  SSIM: 0.5889
[Epoch 030] Loss: 0.023197  Time: 413.4s  LR: 3.91e-04
         Test → PSNR: 32.20  SSIM: 0.8985
[Epoch 060] Loss: 0.019582  Time: 417.7s  LR: 3.63e-04
         Test → PSNR: 33.08  SSIM: 0.9145
[Epoch 090] Loss: 0.017954  Time: 413.7s  LR: 3.19e-04
         Test → PSNR: 33.74  SSIM: 0.9260
[Epoch 120] Loss: 0.016984  Time: 414.2s  LR: 2.64e-04
         Test → PSNR: 34.17  SSIM: 0.9323
[Epoch 150] Loss: 0.016236  Time: 416.4s  LR: 2.03e-04
         Test → PSNR: 34.29  SSIM: 0.9374
[Epoch 180] Loss: 0.015611  Time: 416.1s  LR: 1.41e-04
         Test → PSNR: 34.44  SSIM: 0.9395
[Epoch 210] Loss: 0.015253  Time: 412.4s  LR: 8.49e-05
         Test → PSNR: 34.57  SSIM: 0.9387
[Epoch 240] Loss: 0.014886  Time: 411.0s  LR: 4.03e-05
         Test → PSNR: 34.66  SSIM: 0.9420
[Epoch 270] Loss: 0.014773  Time: 410.8s  LR: 1.14e-05
         Test → PSNR: 34.69  SSIM: 0.9429
[Epoch 300] Loss: 0.014783  Time: 411.7s  LR: 1.01e-06
         Test → PSNR: 34.69  SSIM: 0.9431


#### test

Checkpoint: result/show/viz/3dwpo_maskA_new
MASK_MODE: A
  Scene 01: PSNR=34.75  SSIM=0.9355  SAM=0.1262  FreqAmpErr=11.41777
  Scene 02: PSNR=34.99  SSIM=0.9345  SAM=0.1571  FreqAmpErr=11.43371
  Scene 03: PSNR=36.02  SSIM=0.9496  SAM=0.1012  FreqAmpErr=10.10106
  Scene 04: PSNR=42.75  SSIM=0.9770  SAM=0.1191  FreqAmpErr=2.60895
  Scene 05: PSNR=32.04  SSIM=0.9386  SAM=0.1133  FreqAmpErr=21.39419
  Scene 06: PSNR=34.16  SSIM=0.9507  SAM=0.1530  FreqAmpErr=13.14143
  Scene 07: PSNR=33.21  SSIM=0.9210  SAM=0.1159  FreqAmpErr=17.01350
  Scene 08: PSNR=32.26  SSIM=0.9472  SAM=0.1889  FreqAmpErr=20.83765
  Scene 09: PSNR=34.97  SSIM=0.9407  SAM=0.1206  FreqAmpErr=13.39008
  Scene 10: PSNR=31.91  SSIM=0.9370  SAM=0.1477  FreqAmpErr=24.73465

#### result
  平均:    PSNR=34.70  SSIM=0.9432  SAM=0.1343
           FreqAmpErr=14.60730  LowFreqErr=682.91354  HighFreqErr=9.37612


### maskB

#### train

模型: WaveMST_3D  参数量: 0.83M
[Epoch 001] Loss: 0.095819  Time: 477.7s  LR: 4.00e-04
         Test → PSNR: 22.01  SSIM: 0.4112
[Epoch 030] Loss: 0.026887  Time: 477.5s  LR: 3.91e-04
         Test → PSNR: 31.13  SSIM: 0.8697
[Epoch 060] Loss: 0.022500  Time: 480.5s  LR: 3.63e-04
         Test → PSNR: 32.37  SSIM: 0.9027
[Epoch 090] Loss: 0.020324  Time: 480.2s  LR: 3.19e-04
         Test → PSNR: 32.94  SSIM: 0.9135
[Epoch 120] Loss: 0.018922  Time: 478.4s  LR: 2.64e-04
         Test → PSNR: 33.28  SSIM: 0.9219
[Epoch 150] Loss: 0.018164  Time: 478.5s  LR: 2.03e-04
         Test → PSNR: 33.55  SSIM: 0.9261
[Epoch 180] Loss: 0.017397  Time: 473.5s  LR: 1.41e-04
         Test → PSNR: 33.67  SSIM: 0.9303
[Epoch 210] Loss: 0.016907  Time: 472.5s  LR: 8.49e-05
         Test → PSNR: 33.85  SSIM: 0.9321
[Epoch 240] Loss: 0.016703  Time: 472.0s  LR: 4.03e-05
         Test → PSNR: 33.87  SSIM: 0.9332
[Epoch 270] Loss: 0.016539  Time: 472.0s  LR: 1.14e-05
         Test → PSNR: 33.96  SSIM: 0.9347
[Epoch 300] Loss: 0.016484  Time: 473.0s  LR: 1.01e-06
         Test → PSNR: 33.96  SSIM: 0.9348

#### test

Checkpoint: result/model/2026_05_02_15_22_32_3d_wpo_pure/best.pth
MASK_MODE: B
  Scene 01: PSNR=34.24  SSIM=0.9285  SAM=0.1331  FreqAmpErr=13.04284
  Scene 02: PSNR=34.05  SSIM=0.9235  SAM=0.1771  FreqAmpErr=14.18050
  Scene 03: PSNR=35.17  SSIM=0.9417  SAM=0.1138  FreqAmpErr=11.45819
  Scene 04: PSNR=41.49  SSIM=0.9736  SAM=0.1382  FreqAmpErr=3.59612
  Scene 05: PSNR=31.63  SSIM=0.9307  SAM=0.1210  FreqAmpErr=23.60216
  Scene 06: PSNR=33.53  SSIM=0.9435  SAM=0.1716  FreqAmpErr=14.89385
  Scene 07: PSNR=32.50  SSIM=0.9099  SAM=0.1253  FreqAmpErr=20.08873
  Scene 08: PSNR=31.44  SSIM=0.9392  SAM=0.2053  FreqAmpErr=25.59832
  Scene 09: PSNR=34.52  SSIM=0.9337  SAM=0.1342  FreqAmpErr=13.46226
  Scene 10: PSNR=31.16  SSIM=0.9238  SAM=0.1704  FreqAmpErr=28.82691

#### result
  平均:    PSNR=33.97  SSIM=0.9348  SAM=0.1490
           FreqAmpErr=16.87499  LowFreqErr=776.79351  HighFreqErr=10.92671


## 3d_wpo_kg   maskD

### train
模型: WaveMST_KG  参数量: 0.79M
[Epoch 001] Loss: 0.087262  Time: 445.9s  LR: 4.00e-04
         Test → PSNR: 24.70  SSIM: 0.5844
[Epoch 030] Loss: 0.022891  Time: 441.9s  LR: 3.91e-04
         Test → PSNR: 32.16  SSIM: 0.8970
[Epoch 060] Loss: 0.019301  Time: 440.7s  LR: 3.63e-04
         Test → PSNR: 33.01  SSIM: 0.9084
[Epoch 090] Loss: 0.017692  Time: 438.2s  LR: 3.19e-04
         Test → PSNR: 33.84  SSIM: 0.9319
[Epoch 120] Loss: 0.016543  Time: 438.3s  LR: 2.64e-04
         Test → PSNR: 34.13  SSIM: 0.9372
[Epoch 150] Loss: 0.016059  Time: 447.1s  LR: 2.03e-04
         Test → PSNR: 34.18  SSIM: 0.9370
[Epoch 180] Loss: 0.015594  Time: 447.5s  LR: 1.41e-04
         Test → PSNR: 34.48  SSIM: 0.9407
[Epoch 210] Loss: 0.015178  Time: 447.4s  LR: 8.49e-05
         Test → PSNR: 34.57  SSIM: 0.9424
[Epoch 240] Loss: 0.014804  Time: 447.0s  LR: 4.03e-05
         Test → PSNR: 34.60  SSIM: 0.9431
[Epoch 270] Loss: 0.014490  Time: 448.2s  LR: 1.14e-05
         Test → PSNR: 34.68  SSIM: 0.9440
[Epoch 300] Loss: 0.014687  Time: 446.0s  LR: 1.01e-06
         Test → PSNR: 34.68  SSIM: 0.9442

### test
模型: WaveMST_KG  参数量: 0.79M
Checkpoint: result/model/2026_05_02_00_40_50_3d_wpo_kg/best.pth
MASK_MODE: D
  Scene 01: PSNR=34.71  SSIM=0.9358  SAM=0.1285  FreqAmpErr=11.75029
  Scene 02: PSNR=34.95  SSIM=0.9354  SAM=0.1566  FreqAmpErr=11.72723
  Scene 03: PSNR=36.17  SSIM=0.9505  SAM=0.1013  FreqAmpErr=9.71647
  Scene 04: PSNR=41.93  SSIM=0.9766  SAM=0.1234  FreqAmpErr=4.86246
  Scene 05: PSNR=32.17  SSIM=0.9396  SAM=0.1090  FreqAmpErr=20.80161
  Scene 06: PSNR=34.30  SSIM=0.9522  SAM=0.1493  FreqAmpErr=12.94826
  Scene 07: PSNR=33.21  SSIM=0.9223  SAM=0.1150  FreqAmpErr=16.96078
  Scene 08: PSNR=32.50  SSIM=0.9495  SAM=0.1796  FreqAmpErr=20.19082
  Scene 09: PSNR=34.96  SSIM=0.9431  SAM=0.1191  FreqAmpErr=13.05670
  Scene 10: PSNR=31.97  SSIM=0.9370  SAM=0.1459  FreqAmpErr=24.96716

### result
  平均:    PSNR=34.69  SSIM=0.9442  SAM=0.1328
           FreqAmpErr=14.69818  LowFreqErr=708.73853  HighFreqErr=9.26556

## 2d_wpo_smsa + maskA

### train
模型: WaveMST_Parallel  参数量: 1.26M
[Epoch 001] Loss: 0.090574  Time: 662.2s  LR: 4.00e-04
         Test → PSNR: 23.18  SSIM: 0.5155
[Epoch 030] Loss: 0.022866  Time: 657.0s  LR: 3.91e-04
         Test → PSNR: 32.12  SSIM: 0.8873
[Epoch 060] Loss: 0.019111  Time: 658.6s  LR: 3.63e-04
         Test → PSNR: 33.48  SSIM: 0.9214
[Epoch 090] Loss: 0.017365  Time: 661.2s  LR: 3.19e-04
         Test → PSNR: 33.96  SSIM: 0.9308
[Epoch 120] Loss: 0.016551  Time: 652.7s  LR: 2.64e-04
         Test → PSNR: 34.16  SSIM: 0.9324
[Epoch 150] Loss: 0.015560  Time: 646.5s  LR: 2.03e-04
         Test → PSNR: 34.40  SSIM: 0.9385
[Epoch 180] Loss: 0.015086  Time: 648.5s  LR: 1.41e-04
         Test → PSNR: 34.52  SSIM: 0.9413
[Epoch 210] Loss: 0.014652  Time: 654.0s  LR: 8.49e-05
         Test → PSNR: 34.65  SSIM: 0.9425
[Epoch 240] Loss: 0.014222  Time: 645.7s  LR: 4.03e-05
         Test → PSNR: 34.75  SSIM: 0.9434
[Epoch 270] Loss: 0.014401  Time: 643.3s  LR: 1.14e-05
         Test → PSNR: 34.79  SSIM: 0.9444
[Epoch 300] Loss: 0.014285  Time: 648.2s  LR: 1.01e-06
         Test → PSNR: 34.80  SSIM: 0.9447

### test
Checkpoint: result/model/2026_05_02_16_31_57_3d_wpo_smsa/best.pth
MASK_MODE: A
  Scene 01: PSNR=34.86  SSIM=0.9381  SAM=0.1269  FreqAmpErr=11.75079
  Scene 02: PSNR=35.13  SSIM=0.9371  SAM=0.1664  FreqAmpErr=10.86976
  Scene 03: PSNR=35.90  SSIM=0.9499  SAM=0.1038  FreqAmpErr=11.04607
  Scene 04: PSNR=42.69  SSIM=0.9770  SAM=0.1312  FreqAmpErr=2.52345
  Scene 05: PSNR=32.46  SSIM=0.9429  SAM=0.1112  FreqAmpErr=19.97846
  Scene 06: PSNR=34.19  SSIM=0.9513  SAM=0.1551  FreqAmpErr=13.03998
  Scene 07: PSNR=33.25  SSIM=0.9204  SAM=0.1164  FreqAmpErr=16.60695
  Scene 08: PSNR=32.34  SSIM=0.9473  SAM=0.1906  FreqAmpErr=19.64858
  Scene 09: PSNR=35.14  SSIM=0.9435  SAM=0.1219  FreqAmpErr=11.30359
  Scene 10: PSNR=32.14  SSIM=0.9390  SAM=0.1589  FreqAmpErr=23.75436

### result
  平均:    PSNR=34.81  SSIM=0.9447  SAM=0.1382
           FreqAmpErr=14.05220  LowFreqErr=671.07976  HighFreqErr=8.90930


# helmoholtz equation

## h2_phy

### train
模型: WaveMST_Phys  参数量: 0.79M
训练开始: 5.4.4:22
[Epoch 001] Loss: 0.096204  Time: 459.3s  LR: 4.00e-04
         Test → PSNR: 22.13  SSIM: 0.4865  SAM: 0.5796
                FreqAmpErr: 279.92651  LowFreqErr: 26642.01484  HighFreqErr: 73.57678
[Epoch 030] Loss: 0.025769  Time: 449.4s  LR: 3.91e-04
         Test → PSNR: 31.39  SSIM: 0.8772  SAM: 0.2236
                FreqAmpErr: 30.25820  LowFreqErr: 1308.43532  HighFreqErr: 20.25325
[Epoch 060] Loss: 0.021336  Time: 449.4s  LR: 3.63e-04
         Test → PSNR: 32.57  SSIM: 0.9069  SAM: 0.2062
                FreqAmpErr: 22.92777  LowFreqErr: 1074.52732  HighFreqErr: 14.69635
  ★ 新最优: PSNR=32.57  SSIM=0.9069  → result/model/2026_05_04_04_22_38_h2_alpha_phys/best.pth
[Epoch 090] Loss: 0.019436  Time: 449.4s  LR: 3.19e-04
         Test → PSNR: 33.04  SSIM: 0.9160  SAM: 0.1979
                FreqAmpErr: 21.34050  LowFreqErr: 1124.12057  HighFreqErr: 12.70847
[Epoch 120] Loss: 0.018160  Time: 451.4s  LR: 2.64e-04
         Test → PSNR: 33.56  SSIM: 0.9237  SAM: 0.1708
                FreqAmpErr: 18.79399  LowFreqErr: 912.01785  HighFreqErr: 11.80227
[Epoch 150] Loss: 0.017109  Time: 448.3s  LR: 2.03e-04
         Test → PSNR: 33.86  SSIM: 0.9308  SAM: 0.1689
                FreqAmpErr: 17.21024  LowFreqErr: 856.68985  HighFreqErr: 10.63919
[Epoch 180] Loss: 0.016725  Time: 448.1s  LR: 1.41e-04
         Test → PSNR: 34.05  SSIM: 0.9324  SAM: 0.1508
                FreqAmpErr: 16.46715  LowFreqErr: 802.74061  HighFreqErr: 10.31258
[Epoch 210] Loss: 0.016069  Time: 448.1s  LR: 8.49e-05
         Test → PSNR: 34.15  SSIM: 0.9355  SAM: 0.1450
                FreqAmpErr: 16.41197  LowFreqErr: 809.88608  HighFreqErr: 10.20104
[Epoch 240] Loss: 0.015783  Time: 447.6s  LR: 4.03e-05
         Test → PSNR: 34.34  SSIM: 0.9382  SAM: 0.1468
                FreqAmpErr: 15.70324  LowFreqErr: 746.75072  HighFreqErr: 9.98095
[Epoch 270] Loss: 0.015644  Time: 451.5s  LR: 1.14e-05
         Test → PSNR: 34.37  SSIM: 0.9387  SAM: 0.1447
                FreqAmpErr: 15.71863  LowFreqErr: 747.68019  HighFreqErr: 9.98919
[Epoch 300] Loss: 0.015796  Time: 456.2s  LR: 1.01e-06
         Test → PSNR: 34.37  SSIM: 0.9389  SAM: 0.1438
                FreqAmpErr: 15.66416  LowFreqErr: 746.83408  HighFreqErr: 9.94091

### test
Checkpoint: result/model/2026_05_04_04_22_38_h2_alpha_phys/best.pth
MASK_MODE: A
  Scene 01: PSNR=34.45  SSIM=0.9314  SAM=0.1284  FreqAmpErr=12.05761
  Scene 02: PSNR=34.55  SSIM=0.9278  SAM=0.1700  FreqAmpErr=12.77218
  Scene 03: PSNR=35.56  SSIM=0.9451  SAM=0.1096  FreqAmpErr=11.16526
  Scene 04: PSNR=42.16  SSIM=0.9751  SAM=0.1323  FreqAmpErr=3.26577
  Scene 05: PSNR=31.92  SSIM=0.9345  SAM=0.1183  FreqAmpErr=21.69596
  Scene 06: PSNR=33.79  SSIM=0.9460  SAM=0.1678  FreqAmpErr=14.17093
  Scene 07: PSNR=32.92  SSIM=0.9156  SAM=0.1195  FreqAmpErr=17.96074
  Scene 08: PSNR=32.18  SSIM=0.9429  SAM=0.2055  FreqAmpErr=22.22608
  Scene 09: PSNR=34.89  SSIM=0.9398  SAM=0.1314  FreqAmpErr=13.95142
  Scene 10: PSNR=31.47  SSIM=0.9315  SAM=0.1666  FreqAmpErr=26.68661

### result
  平均:    PSNR=34.39  SSIM=0.9390  SAM=0.1449
           FreqAmpErr=15.59526  LowFreqErr=738.85914  HighFreqErr=9.93389

## h1_pure

### train
模型: Helmholtzformer  参数量: 0.78M
训练开始: 5.4.17:36
[Epoch 001] Loss: 0.090783  Time: 403.4s  LR: 4.00e-04
         Test → PSNR: 23.12  SSIM: 0.5672  SAM: 0.4965
                FreqAmpErr: 241.63037  LowFreqErr: 21790.02622  HighFreqErr: 72.95990
[Epoch 030] Loss: 0.027524  Time: 402.5s  LR: 3.91e-04
         Test → PSNR: 30.85  SSIM: 0.8601  SAM: 0.2760
                FreqAmpErr: 34.02376  LowFreqErr: 1598.98296  HighFreqErr: 21.77401
  ★ 新最优: PSNR=30.85  SSIM=0.8601  → result/model/2026_05_04_17_36_59_h1_gamma_helm_pure/best.pth
[Epoch 060] Loss: 0.023339  Time: 398.5s  LR: 3.63e-04
         Test → PSNR: 31.92  SSIM: 0.8924  SAM: 0.2258
                FreqAmpErr: 25.73638  LowFreqErr: 1223.97811  HighFreqErr: 16.35712
[Epoch 090] Loss: 0.020896  Time: 398.1s  LR: 3.19e-04
         Test → PSNR: 32.53  SSIM: 0.9064  SAM: 0.2154
                FreqAmpErr: 23.21834  LowFreqErr: 1104.53871  HighFreqErr: 14.75428
[Epoch 120] Loss: 0.019723  Time: 398.4s  LR: 2.64e-04
         Test → PSNR: 32.83  SSIM: 0.9136  SAM: 0.2052
                FreqAmpErr: 21.65142  LowFreqErr: 1116.42731  HighFreqErr: 13.08204
[Epoch 150] Loss: 0.018715  Time: 398.0s  LR: 2.03e-04
         Test → PSNR: 33.19  SSIM: 0.9207  SAM: 0.1733
                FreqAmpErr: 19.15915  LowFreqErr: 922.50497  HighFreqErr: 12.08819
[Epoch 180] Loss: 0.018777  Time: 399.8s  LR: 1.41e-04
         Test → PSNR: 33.44  SSIM: 0.9243  SAM: 0.1746
                FreqAmpErr: 18.36804  LowFreqErr: 865.04492  HighFreqErr: 11.74066
  ★ 新最优: PSNR=33.44  SSIM=0.9243  → result/model/2026_05_04_17_36_59_h1_gamma_helm_pure/best.pth
[Epoch 210] Loss: 0.018135  Time: 405.2s  LR: 8.49e-05
         Test → PSNR: 33.52  SSIM: 0.9266  SAM: 0.1647
                FreqAmpErr: 18.12347  LowFreqErr: 842.66086  HighFreqErr: 11.66939
[Epoch 240] Loss: 0.017533  Time: 397.2s  LR: 4.03e-05
         Test → PSNR: 33.61  SSIM: 0.9282  SAM: 0.1546
                FreqAmpErr: 17.83185  LowFreqErr: 834.69184  HighFreqErr: 11.43786
[Epoch 270] Loss: 0.017373  Time: 396.6s  LR: 1.14e-05
         Test → PSNR: 33.67  SSIM: 0.9291  SAM: 0.1578
                FreqAmpErr: 17.71574  LowFreqErr: 820.22451  HighFreqErr: 11.43409
  ★ 新最优: PSNR=33.67  SSIM=0.9291  → result/model/2026_05_04_17_36_59_h1_gamma_helm_pure/best.pth
[Epoch 300] Loss: 0.017138  Time: 396.6s  LR: 1.01e-06
         Test → PSNR: 33.68  SSIM: 0.9293  SAM: 0.1572
                FreqAmpErr: 17.65252  LowFreqErr: 815.62635  HighFreqErr: 11.40636

### test
Checkpoint: result/model/2026_05_04_17_36_59_h1_gamma_helm_pure/best.pth
MASK_MODE: A
  Scene 01: PSNR=34.07  SSIM=0.9244  SAM=0.1355  FreqAmpErr=13.64788
  Scene 02: PSNR=33.62  SSIM=0.9143  SAM=0.1914  FreqAmpErr=15.47682
  Scene 03: PSNR=34.80  SSIM=0.9371  SAM=0.1243  FreqAmpErr=12.83371
  Scene 04: PSNR=40.81  SSIM=0.9689  SAM=0.1520  FreqAmpErr=5.09160
  Scene 05: PSNR=31.47  SSIM=0.9282  SAM=0.1281  FreqAmpErr=24.68901
  Scene 06: PSNR=33.25  SSIM=0.9387  SAM=0.1817  FreqAmpErr=15.69928
  Scene 07: PSNR=32.38  SSIM=0.9059  SAM=0.1245  FreqAmpErr=19.65659
  Scene 08: PSNR=31.54  SSIM=0.9314  SAM=0.2241  FreqAmpErr=24.57681
  Scene 09: PSNR=33.98  SSIM=0.9282  SAM=0.1343  FreqAmpErr=15.40456
  Scene 10: PSNR=30.88  SSIM=0.9169  SAM=0.1863  FreqAmpErr=30.02508

### result
  平均:    PSNR=33.68  SSIM=0.9294  SAM=0.1582
           FreqAmpErr=17.71013  LowFreqErr=820.94807  HighFreqErr=11.42277


## h2_main

### train
[Epoch 001] Loss: 0.096625  Time: 471.0s  LR: 4.00e-04
         Test → PSNR: 22.74  SSIM: 0.5046  SAM: 0.4813
                FreqAmpErr: 206.43914  LowFreqErr: 16870.06738  HighFreqErr: 76.00428
[Epoch 030] Loss: 0.028571  Time: 470.5s  LR: 3.91e-04
         Test → PSNR: 30.81  SSIM: 0.8464  SAM: 0.2898
                FreqAmpErr: 32.57573  LowFreqErr: 1330.19662  HighFreqErr: 22.41858
  ★ 新最优: PSNR=30.81  SSIM=0.8464  → result/model/2026_05_05_13_01_33_h2_gamma_main/best.pth
[Epoch 060] Loss: 0.023596  Time: 462.7s  LR: 3.63e-04
         Test → PSNR: 32.03  SSIM: 0.8862  SAM: 0.2362
                FreqAmpErr: 25.39943  LowFreqErr: 1047.98538  HighFreqErr: 17.39512
  ★ 新最优: PSNR=32.03  SSIM=0.8862  → result/model/2026_05_05_13_01_33_h2_gamma_main/best.pth
[Epoch 090] Loss: 0.021287  Time: 462.6s  LR: 3.19e-04
         Test → PSNR: 32.55  SSIM: 0.9025  SAM: 0.2045
                FreqAmpErr: 22.52912  LowFreqErr: 1015.32902  HighFreqErr: 14.75796
  ★ 新最优: PSNR=32.55  SSIM=0.9025  → result/model/2026_05_05_13_01_33_h2_gamma_main/best.pth
[Epoch 120] Loss: 0.019726  Time: 459.7s  LR: 2.64e-04
         Test → PSNR: 32.90  SSIM: 0.9099  SAM: 0.2093
                FreqAmpErr: 20.49131  LowFreqErr: 923.96586  HighFreqErr: 13.41934
  ★ 新最优: PSNR=32.90  SSIM=0.9099  → result/model/2026_05_05_13_01_33_h2_gamma_main/best.pth
[Epoch 150] Loss: 0.018680  Time: 460.3s  LR: 2.03e-04
         Test → PSNR: 33.01  SSIM: 0.9173  SAM: 0.1961
                FreqAmpErr: 21.78366  LowFreqErr: 1128.61237  HighFreqErr: 13.11994
[Epoch 180] Loss: 0.018019  Time: 459.8s  LR: 1.41e-04
         Test → PSNR: 33.41  SSIM: 0.9208  SAM: 0.1915
                FreqAmpErr: 18.58307  LowFreqErr: 866.53945  HighFreqErr: 11.94567
[Epoch 210] Loss: 0.017747  Time: 460.0s  LR: 8.49e-05
         Test → PSNR: 33.56  SSIM: 0.9254  SAM: 0.1801
                FreqAmpErr: 17.99159  LowFreqErr: 835.96961  HighFreqErr: 11.58885
[Epoch 240] Loss: 0.016925  Time: 459.4s  LR: 4.03e-05
         Test → PSNR: 33.67  SSIM: 0.9268  SAM: 0.1703
                FreqAmpErr: 17.72251  LowFreqErr: 823.01606  HighFreqErr: 11.41906
[Epoch 270] Loss: 0.016813  Time: 459.4s  LR: 1.14e-05
         Test → PSNR: 33.73  SSIM: 0.9281  SAM: 0.1714
                FreqAmpErr: 17.55713  LowFreqErr: 810.61717  HighFreqErr: 11.34944
  ★ 新最优: PSNR=33.73  SSIM=0.9281  → result/model/2026_05_05_13_01_33_h2_gamma_main/best.pth
[Epoch 300] Loss: 0.017172  Time: 459.6s  LR: 1.01e-06
         Test → PSNR: 33.73  SSIM: 0.9285  SAM: 0.1704
                FreqAmpErr: 17.56940  LowFreqErr: 820.65766  HighFreqErr: 11.28321

### test
Checkpoint: result/model/2026_05_05_13_01_33_h2_gamma_main/best.pth
MASK_MODE: A
  Scene 01: PSNR=34.13  SSIM=0.9264  SAM=0.1367  FreqAmpErr=13.03211
  Scene 02: PSNR=33.82  SSIM=0.9125  SAM=0.1968  FreqAmpErr=15.20400
  Scene 03: PSNR=34.89  SSIM=0.9339  SAM=0.1290  FreqAmpErr=11.98625
  Scene 04: PSNR=41.00  SSIM=0.9697  SAM=0.1647  FreqAmpErr=4.71780
  Scene 05: PSNR=31.29  SSIM=0.9245  SAM=0.1421  FreqAmpErr=25.05884
  Scene 06: PSNR=33.22  SSIM=0.9382  SAM=0.2037  FreqAmpErr=15.71008
  Scene 07: PSNR=32.37  SSIM=0.9058  SAM=0.1290  FreqAmpErr=20.01918
  Scene 08: PSNR=31.59  SSIM=0.9300  SAM=0.2539  FreqAmpErr=24.72383
  Scene 09: PSNR=34.06  SSIM=0.9269  SAM=0.1431  FreqAmpErr=14.58325
  Scene 10: PSNR=30.97  SSIM=0.9163  SAM=0.2088  FreqAmpErr=30.10112

### result
  平均:    PSNR=33.73  SSIM=0.9284  SAM=0.1708
           FreqAmpErr=17.51365  LowFreqErr=810.74090  HighFreqErr=11.30464
























