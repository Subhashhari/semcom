# Generative JSCC results (DiT at 20 DDIM steps)

## PSNR (dB) (higher is better)

| run | -5 dB | +0 dB | +5 dB |
|---|---|---|---|
| cnn_cbr48_spec0_s0 | 14.314 | 21.212 | 21.795 |
| dit-S-v_cbr48_spec-5_s0 | 15.585 | 16.469 | 16.696 |
| dit-S-v_cbr48_spec0_s0 | 13.484 | 17.491 | 18.454 |
| dit-S-v_cbr48_spec5_s0 | 11.973 | 16.574 | 18.986 |
| dit-S-v_cbr48_told-5to5_s0 | 14.860 | 17.480 | 19.131 |
| **specialist envelope (analog)** | **15.585** | **17.491** | **18.986** |
| dit-S-v_cbr48_told-5to5_s0 vs envelope | -0.725 | -0.011 | +0.145 |

## LPIPS (lower is better)

| run | -5 dB | +0 dB | +5 dB |
|---|---|---|---|
| cnn_cbr48_spec0_s0 | 0.648 | 0.426 | 0.448 |
| dit-S-v_cbr48_spec-5_s0 | 0.365 | 0.347 | 0.350 |
| dit-S-v_cbr48_spec0_s0 | 0.421 | 0.266 | 0.253 |
| dit-S-v_cbr48_spec5_s0 | 0.475 | 0.279 | 0.209 |
| dit-S-v_cbr48_told-5to5_s0 | 0.386 | 0.269 | 0.208 |
| **specialist envelope (analog)** | **0.365** | **0.266** | **0.209** |
| dit-S-v_cbr48_told-5to5_s0 vs envelope | -0.020 | -0.003 | +0.001 |

## DINOv2 similarity (higher is better)

| run | -5 dB | +0 dB | +5 dB |
|---|---|---|---|
| cnn_cbr48_spec0_s0 | 0.023 | 0.161 | 0.198 |
| dit-S-v_cbr48_spec-5_s0 | 0.093 | 0.131 | 0.137 |
| dit-S-v_cbr48_spec0_s0 | 0.042 | 0.170 | 0.229 |
| dit-S-v_cbr48_spec5_s0 | 0.027 | 0.121 | 0.251 |
| dit-S-v_cbr48_told-5to5_s0 | 0.074 | 0.166 | 0.271 |
| **specialist envelope (analog)** | **0.093** | **0.170** | **0.251** |
| dit-S-v_cbr48_told-5to5_s0 vs envelope | -0.020 | -0.003 | +0.020 |

## FID (lower is better)

| run | -5 dB | +0 dB | +5 dB |
|---|---|---|---|
| cnn_cbr48_spec0_s0 | 302.704 | 207.497 | 216.398 |
| dit-S-v_cbr48_spec-5_s0 | 190.780 | 162.277 | 159.997 |
| dit-S-v_cbr48_spec0_s0 | 253.516 | 149.341 | 126.453 |
| dit-S-v_cbr48_spec5_s0 | 270.838 | 192.504 | 130.017 |
| dit-S-v_cbr48_told-5to5_s0 | 209.355 | 154.104 | 117.679 |
| **specialist envelope (analog)** | **190.780** | **149.341** | **130.017** |
| dit-S-v_cbr48_told-5to5_s0 vs envelope | -18.575 | -4.763 | +12.339 |

Gap rows are signed so that positive means the adaptive arm is better than the specialist.
