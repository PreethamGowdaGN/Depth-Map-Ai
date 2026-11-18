# Benchmark — 20 held-out synthetic scenes (320×240)

Degradation: point-sampled downsampling, noise σ=0.01 (×depth), 10% blob holes. Settings: {'noise_reduction': 0.0, 'edge_preservation': 0.6, 'hole_filling': True, 'use_rgb': True, 'use_deep': True, 'use_residual': True, 'use_edges': True, 'use_jbu_fusion': False, 'use_refinement': False}


## 2×

| Method | RMSE | MAE | PSNR | SSIM | δ<1.25 | Edge F1 | Boundary RMSE | Texture-copy idx | Latency (s) | Peak mem (MB) |
|---|---|---|---|---|---|---|---|---|---|---|
| Nearest neighbour | 0.2321 | 0.0568 | 26.58 | 0.8818 | 0.9936 | 0.7112 | 0.6938 | 0.0072 | 0.0241 | 2.14 |
| Bilinear | 0.1683 | 0.0458 | 29.43 | 0.9274 | 0.9919 | 0.9817 | 0.5025 | 0.0059 | 0.0252 | 2.14 |
| Bicubic | 0.1731 | 0.0535 | 29.17 | 0.9086 | 0.9916 | 0.9685 | 0.5136 | 0.0068 | 0.0241 | 2.14 |
| Guided filtering | 0.1680 | 0.0405 | 29.72 | 0.9665 | 0.9888 | 0.9426 | 0.5098 | 0.0054 | 0.0462 | 15.84 |
| Joint bilateral filtering | 0.1660 | 0.0367 | 29.84 | 0.9674 | 0.9911 | 0.9760 | 0.5021 | 0.0043 | 0.0760 | 7.71 |
| Deep learning (DepthNet) | 0.0773 | 0.0202 | 36.16 | 0.9853 | 0.9972 | 0.9642 | 0.2317 | 0.0045 | 0.3645 | 5.71 |
| Hybrid reconstruction | 0.0720 | 0.0212 | 36.71 | 0.9818 | 0.9981 | 0.9671 | 0.2107 | 0.0043 | 0.2045 | 8.08 |

## 4×

| Method | RMSE | MAE | PSNR | SSIM | δ<1.25 | Edge F1 | Boundary RMSE | Texture-copy idx | Latency (s) | Peak mem (MB) |
|---|---|---|---|---|---|---|---|---|---|---|
| Nearest neighbour | 0.3130 | 0.0768 | 23.96 | 0.8675 | 0.9881 | 0.6005 | 0.9194 | 0.0052 | 0.0100 | 1.91 |
| Bilinear | 0.2440 | 0.0691 | 26.14 | 0.9201 | 0.9822 | 0.9123 | 0.7129 | 0.0043 | 0.0099 | 1.91 |
| Bicubic | 0.2488 | 0.0796 | 25.95 | 0.8930 | 0.9828 | 0.8579 | 0.7189 | 0.0064 | 0.0103 | 1.91 |
| Guided filtering | 0.2160 | 0.0630 | 27.49 | 0.9458 | 0.9824 | 0.8694 | 0.6271 | 0.0084 | 0.0318 | 15.72 |
| Joint bilateral filtering | 0.2184 | 0.0535 | 27.44 | 0.9535 | 0.9866 | 0.9165 | 0.6359 | 0.0055 | 0.0579 | 7.48 |
| Deep learning (DepthNet) | 0.0923 | 0.0252 | 34.56 | 0.9832 | 0.9968 | 0.9576 | 0.2705 | 0.0044 | 0.1066 | 2.28 |
| Hybrid reconstruction | 0.0892 | 0.0260 | 34.84 | 0.9822 | 0.9975 | 0.9612 | 0.2576 | 0.0043 | 0.1887 | 7.78 |

## 8×

| Method | RMSE | MAE | PSNR | SSIM | δ<1.25 | Edge F1 | Boundary RMSE | Texture-copy idx | Latency (s) | Peak mem (MB) |
|---|---|---|---|---|---|---|---|---|---|---|
| Nearest neighbour | 0.4380 | 0.1190 | 20.99 | 0.8383 | 0.9753 | 0.4760 | 1.2908 | 0.0044 | 0.0067 | 1.85 |
| Bilinear | 0.3500 | 0.1117 | 22.93 | 0.8922 | 0.9655 | 0.6610 | 1.0218 | 0.0049 | 0.0066 | 1.85 |
| Bicubic | 0.3636 | 0.1276 | 22.62 | 0.8601 | 0.9659 | 0.6612 | 1.0494 | 0.0072 | 0.0065 | 1.85 |
| Guided filtering | 0.2966 | 0.1077 | 24.57 | 0.9090 | 0.9691 | 0.7351 | 0.8255 | 0.0124 | 0.0292 | 15.69 |
| Joint bilateral filtering | 0.2814 | 0.0840 | 25.13 | 0.9291 | 0.9787 | 0.8274 | 0.7857 | 0.0080 | 0.0601 | 7.42 |
| Deep learning (DepthNet) | 0.1250 | 0.0404 | 31.75 | 0.9744 | 0.9933 | 0.9403 | 0.3373 | 0.0047 | 0.1099 | 2.22 |
| Hybrid reconstruction | 0.1227 | 0.0398 | 31.93 | 0.9747 | 0.9941 | 0.9430 | 0.3307 | 0.0046 | 0.1881 | 7.70 |

**Summary:** hybrid vs bicubic RMSE reduction 63.9%, boundary RMSE reduction 65.0% (mean of 2×/4×/8×).
