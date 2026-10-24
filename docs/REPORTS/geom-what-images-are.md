# What the images actually are

| quantity | value |
|---|---|
| `w == 640` | 65.23% |
| `w < 640` | 11.44% |
| `w > 640` | 23.33% |
| `h == 640` | 24.92% |
| `min(w,h) == 640` (downscaled upstream) | 85.06% |
| `min(w,h) < 640` (native small, never upscaled) | 14.94% |
| `min(w,h) > 640` | 0% |
| width p1 / p50 / p99 / max | 416 / 640 / 1183 / 4291 px |
| height p1 / p50 / p99 / max | 478 / 841 / 1418 / 8231 px |
| height where `w == 640` (n=26092): p5/p50/p95 | 640 / 903 / 1137 px |
| distinct widths / heights / aspect values (40k sample) | 1088 / 1316 / 4067 |

Top exact sizes (40k sample): `640×905` 2195 (5.5%), `640×640` 1901 (4.8%), `640×904` 1487, `640×903` 1261, `640×853` 1216, `853×640` 516, `1137×640` 508.
