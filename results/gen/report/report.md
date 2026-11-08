# Generative JSCC results (DiT at 20 DDIM steps)

## Transmitted symbols (digital runs, test images)

| run | usage entropy / log2 M | kurtosis | mean energy per symbol |
|---|---|---|---|
| cnn_cbr48_qam16_spec0_s0 | 0.996 | 1.286 | 1.054 |
| dit-S-v_cbr48_qam16_blind-5to5_s0 | 0.987 | 1.430 | 0.895 |
| dit-S-v_cbr48_qam16_spec-5_s0 | 0.991 | 1.341 | 1.023 |
| dit-S-v_cbr48_qam16_spec0_s0 | 0.987 | 1.436 | 0.883 |
| dit-S-v_cbr48_qam16_spec5_s0 | 0.979 | 1.474 | 0.830 |
| dit-S-v_cbr48_qam16_told-5to5_s0 | 0.987 | 1.428 | 0.897 |
| dit-S-v_cbr48_qam4_spec0_s0 | 0.989 | 1.000 | 1.000 |
| dit-S-v_cbr48_qam64_spec0_s0 | 0.992 | 1.495 | 0.888 |

## PSNR (dB) (higher is better)

| run | -5 dB | +0 dB | +5 dB |
|---|---|---|---|
| cnn_cbr48_qam16_spec0_s0 | 14.643 | 20.794 | 21.501 |
| cnn_cbr48_spec0_s0 | 14.314 | 21.212 | 21.795 |
| dit-S-v_cbr48_blind-5to5_s0 | 14.919 | 17.517 | 19.115 |
| dit-S-v_cbr48_qam16_blind-5to5_s0 | 14.631 | 16.944 | 18.318 |
| dit-S-v_cbr48_qam16_spec-5_s0 | 15.203 | 16.180 | 16.396 |
| dit-S-v_cbr48_qam16_spec0_s0 | 14.020 | 16.926 | 17.741 |
| dit-S-v_cbr48_qam16_spec5_s0 | 12.531 | 16.351 | 18.230 |
| dit-S-v_cbr48_qam16_told-5to5_s0 | 14.585 | 16.960 | 18.403 |
| dit-S-v_cbr48_qam4_spec0_s0 | 13.999 | 16.525 | 17.293 |
| dit-S-v_cbr48_qam64_spec0_s0 | 13.785 | 17.172 | 18.059 |
| dit-S-v_cbr48_spec-5_s0 | 15.585 | 16.469 | 16.696 |
| dit-S-v_cbr48_spec0_s0 | 13.484 | 17.491 | 18.454 |
| dit-S-v_cbr48_spec5_s0 | 11.973 | 16.574 | 18.986 |
| dit-S-v_cbr48_told-5to5_s0 | 14.860 | 17.480 | 19.131 |
| **specialist envelope (analog)** | **15.585** | **17.491** | **18.986** |
| dit-S-v_cbr48_blind-5to5_s0 vs envelope | -0.667 | +0.026 | +0.129 |
| dit-S-v_cbr48_told-5to5_s0 vs envelope | -0.725 | -0.011 | +0.145 |
| **specialist envelope (qam16)** | **15.203** | **16.926** | **18.230** |
| dit-S-v_cbr48_qam16_blind-5to5_s0 vs envelope | -0.572 | +0.018 | +0.088 |
| dit-S-v_cbr48_qam16_told-5to5_s0 vs envelope | -0.618 | +0.034 | +0.174 |

## LPIPS (lower is better)

| run | -5 dB | +0 dB | +5 dB |
|---|---|---|---|
| cnn_cbr48_qam16_spec0_s0 | 0.626 | 0.441 | 0.442 |
| cnn_cbr48_spec0_s0 | 0.648 | 0.426 | 0.448 |
| dit-S-v_cbr48_blind-5to5_s0 | 0.386 | 0.267 | 0.209 |
| dit-S-v_cbr48_qam16_blind-5to5_s0 | 0.394 | 0.288 | 0.236 |
| dit-S-v_cbr48_qam16_spec-5_s0 | 0.380 | 0.356 | 0.357 |
| dit-S-v_cbr48_qam16_spec0_s0 | 0.407 | 0.285 | 0.271 |
| dit-S-v_cbr48_qam16_spec5_s0 | 0.458 | 0.294 | 0.234 |
| dit-S-v_cbr48_qam16_told-5to5_s0 | 0.395 | 0.287 | 0.232 |
| dit-S-v_cbr48_qam4_spec0_s0 | 0.433 | 0.307 | 0.278 |
| dit-S-v_cbr48_qam64_spec0_s0 | 0.410 | 0.279 | 0.263 |
| dit-S-v_cbr48_spec-5_s0 | 0.365 | 0.347 | 0.350 |
| dit-S-v_cbr48_spec0_s0 | 0.421 | 0.266 | 0.253 |
| dit-S-v_cbr48_spec5_s0 | 0.475 | 0.279 | 0.209 |
| dit-S-v_cbr48_told-5to5_s0 | 0.386 | 0.269 | 0.208 |
| **specialist envelope (analog)** | **0.365** | **0.266** | **0.209** |
| dit-S-v_cbr48_blind-5to5_s0 vs envelope | -0.020 | -0.001 | -0.000 |
| dit-S-v_cbr48_told-5to5_s0 vs envelope | -0.020 | -0.003 | +0.001 |
| **specialist envelope (qam16)** | **0.380** | **0.285** | **0.234** |
| dit-S-v_cbr48_qam16_blind-5to5_s0 vs envelope | -0.014 | -0.003 | -0.003 |
| dit-S-v_cbr48_qam16_told-5to5_s0 vs envelope | -0.014 | -0.002 | +0.002 |

## DINOv2 similarity (higher is better)

| run | -5 dB | +0 dB | +5 dB |
|---|---|---|---|
| cnn_cbr48_qam16_spec0_s0 | 0.017 | 0.130 | 0.161 |
| cnn_cbr48_spec0_s0 | 0.023 | 0.161 | 0.198 |
| dit-S-v_cbr48_blind-5to5_s0 | 0.074 | 0.168 | 0.269 |
| dit-S-v_cbr48_qam16_blind-5to5_s0 | 0.072 | 0.140 | 0.223 |
| dit-S-v_cbr48_qam16_spec-5_s0 | 0.090 | 0.129 | 0.133 |
| dit-S-v_cbr48_qam16_spec0_s0 | 0.047 | 0.155 | 0.201 |
| dit-S-v_cbr48_qam16_spec5_s0 | 0.026 | 0.100 | 0.207 |
| dit-S-v_cbr48_qam16_told-5to5_s0 | 0.074 | 0.147 | 0.225 |
| dit-S-v_cbr48_qam4_spec0_s0 | 0.047 | 0.131 | 0.173 |
| dit-S-v_cbr48_qam64_spec0_s0 | 0.046 | 0.159 | 0.212 |
| dit-S-v_cbr48_spec-5_s0 | 0.093 | 0.131 | 0.137 |
| dit-S-v_cbr48_spec0_s0 | 0.042 | 0.170 | 0.229 |
| dit-S-v_cbr48_spec5_s0 | 0.027 | 0.121 | 0.251 |
| dit-S-v_cbr48_told-5to5_s0 | 0.074 | 0.166 | 0.271 |
| **specialist envelope (analog)** | **0.093** | **0.170** | **0.251** |
| dit-S-v_cbr48_blind-5to5_s0 vs envelope | -0.020 | -0.002 | +0.018 |
| dit-S-v_cbr48_told-5to5_s0 vs envelope | -0.020 | -0.003 | +0.020 |
| **specialist envelope (qam16)** | **0.090** | **0.155** | **0.207** |
| dit-S-v_cbr48_qam16_blind-5to5_s0 vs envelope | -0.018 | -0.014 | +0.017 |
| dit-S-v_cbr48_qam16_told-5to5_s0 vs envelope | -0.016 | -0.008 | +0.018 |

## FID (lower is better)

| run | -5 dB | +0 dB | +5 dB |
|---|---|---|---|
| cnn_cbr48_qam16_spec0_s0 | 275.699 | 212.243 | 221.570 |
| cnn_cbr48_spec0_s0 | 302.704 | 207.497 | 216.398 |
| dit-S-v_cbr48_blind-5to5_s0 | 213.849 | 152.043 | 117.834 |
| dit-S-v_cbr48_qam16_blind-5to5_s0 | 214.216 | 162.303 | 133.178 |
| dit-S-v_cbr48_qam16_spec-5_s0 | 188.410 | 161.494 | 157.808 |
| dit-S-v_cbr48_qam16_spec0_s0 | 244.176 | 155.028 | 134.225 |
| dit-S-v_cbr48_qam16_spec5_s0 | 270.210 | 200.474 | 143.395 |
| dit-S-v_cbr48_qam16_told-5to5_s0 | 208.684 | 160.100 | 130.043 |
| dit-S-v_cbr48_qam4_spec0_s0 | 247.683 | 171.030 | 144.658 |
| dit-S-v_cbr48_qam64_spec0_s0 | 246.088 | 152.680 | 131.172 |
| dit-S-v_cbr48_spec-5_s0 | 190.780 | 162.277 | 159.997 |
| dit-S-v_cbr48_spec0_s0 | 253.516 | 149.341 | 126.453 |
| dit-S-v_cbr48_spec5_s0 | 270.838 | 192.504 | 130.017 |
| dit-S-v_cbr48_told-5to5_s0 | 209.355 | 154.104 | 117.679 |
| **specialist envelope (analog)** | **190.780** | **149.341** | **130.017** |
| dit-S-v_cbr48_blind-5to5_s0 vs envelope | -23.069 | -2.701 | +12.183 |
| dit-S-v_cbr48_told-5to5_s0 vs envelope | -18.575 | -4.763 | +12.339 |
| **specialist envelope (qam16)** | **188.410** | **155.028** | **143.395** |
| dit-S-v_cbr48_qam16_blind-5to5_s0 vs envelope | -25.806 | -7.275 | +10.217 |
| dit-S-v_cbr48_qam16_told-5to5_s0 vs envelope | -20.274 | -5.072 | +13.352 |

Gap rows are signed so that positive means the adaptive arm is better than the specialist.
